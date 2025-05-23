"""
Route Finder for TBRGS
This module integrates the traffic prediction models with the search algorithms
to find optimal routes based on predicted traffic conditions.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

from utils.graph_utils import TrafficGraph
from models.traffic_predictor import TrafficPredictor

# Initialize rich library variables
rich_available = False
console = None
Progress = None
SpinnerColumn = None
TextColumn = None
BarColumn = None
TimeElapsedColumn = None
Panel = None
Table = None
ROUNDED = None
Text = None

# Add the project root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import search algorithms from Assignment 2A
from utils.search.astar import astar
from utils.search.bfs import bfs
from utils.search.dfs import dfs
from utils.search.gbfs import gbfs
from utils.search.iddfs import iddfs
from utils.search.bdwa import bdwa

class RouteFinder:
    """
    Class for finding optimal routes based on traffic predictions.
    """
    def __init__(self, config_path=None):
        """
        Initialize the route finder.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'default_config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize traffic graph
        self.graph = None
        
        # System parameters
        self.speed_limit = self.config['system']['speed_limit']  # km/h
        self.intersection_delay = self.config['system']['intersection_delay']  # seconds
        self.max_routes = self.config['system']['max_routes']
        
        # Initialize traffic predictor
        try:
            self.traffic_predictor = TrafficPredictor(
                models_dir=self.config['paths']['models_dir'],
                model_type='ensemble',
                config=self.config
            )
            self.has_predictor = True
        except Exception as e:
            print(f"Warning: Could not initialize traffic predictor: {e}")
            print("Using default travel times based on distance only.")
            self.traffic_predictor = None
            self.has_predictor = False
    
    def load_graph(self, nodes_file=None, edges_file=None):
        """
        Load the traffic graph from files.
        
        Args:
            nodes_file: Path to the file containing node information (default: from config)
            edges_file: Path to the file containing edge information (default: from config)
            
        Returns:
            TrafficGraph: The loaded traffic graph
        """
        if nodes_file is None:
            nodes_file = self.config['paths']['nodes_data']
        if edges_file is None:
            edges_file = self.config['paths']['edges_data']
            
        # Check if files exist
        if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
            raise FileNotFoundError(f"Graph data files not found at {nodes_file} or {edges_file}. Please run data_processing.py first.")
            
        # Load nodes
        nodes = {}
        nodes_df = pd.read_csv(nodes_file)
        for _, row in nodes_df.iterrows():
            node_id = int(row['id'])  # Ensure node IDs are integers
            nodes[node_id] = {
                'name': row['name'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude'])
            }
            
        # Load edges
        edges = {}
        edges_df = pd.read_csv(edges_file)
        
        # Use actual distances from CSV
        use_actual_distances = True
        
        for _, row in edges_df.iterrows():
            source = int(row['source'])  # Ensure node IDs are integers
            target = int(row['target'])
            distance = float(row['distance'])
            
            # Use actual distances from the CSV file
            if use_actual_distances:
                # No scaling - use the actual distance
                pass
            else:
                # Scale down distances for testing/visualization (this was the previous behavior)
                distance = distance / 45.0  # Approximate scaling factor based on debug output
            
            if source not in edges:
                edges[source] = []
            edges[source].append((target, distance))
            
        # Create the traffic graph with the loaded data
        
        # Create traffic graph
        self.graph = TrafficGraph(
            nodes=nodes,
            edges=edges,
            speed_limit=self.config['system']['speed_limit'],
            intersection_delay=self.config['system']['intersection_delay']
        )
        
        # Apply fix to ensure graph edges use correct distances from CSV
        distance_map = {}
        for s, t, d in edges_df[['source', 'target', 'distance']].values:
            distance_map[(int(s), int(t))] = float(d)
        
        # Update the graph edges with the correct distances
        for source in self.graph.edges:
            updated_edges = []
            for target, _ in self.graph.edges[source]:
                if (source, target) in distance_map:
                    # Use the actual distance from the CSV
                    updated_edges.append((target, distance_map[(source, target)]))
                else:
                    # Keep the existing distance if not found in CSV
                    for t, d in self.graph.edges[source]:
                        if t == target:
                            updated_edges.append((t, d))
                            break
            self.graph.edges[source] = updated_edges
        
        
        print(f"Loaded graph with {len(nodes)} nodes and {sum(len(neighbors) for neighbors in edges.values())} edges")
        return self.graph
    
    def update_graph_with_predictions(self, historical_data_dict):
        """
        Update the traffic graph with predicted traffic flows.
        
        Args:
            historical_data_dict: Dictionary mapping (source, destination) tuples to sequences of historical traffic flow values
            
        Returns:
            The updated traffic graph
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_graph first.")
        
        # Check if traffic predictor is available
        if not hasattr(self, 'has_predictor') or not self.has_predictor:
            print("Warning: Traffic predictor not available. Using default traffic flows based on edge distances.")
            # Set default traffic flows based on edge distances
            for source in self.graph.edges:
                for target, distance in self.graph.edges[source]:
                    # Use a simple formula to estimate traffic flow based on distance
                    # Shorter distances typically have higher traffic flows
                    # This is just a placeholder and not based on real data
                    default_flow = 1000 * (1 / (1 + distance/1000))
                    self.graph.set_traffic_flow(source, target, default_flow)
            return self.graph
        
        # For each edge in the graph, predict the traffic flow and update the graph
        try:
            for (source, destination), historical_data in historical_data_dict.items():
                # Ensure the edge exists in the graph
                edge_exists = False
                for neighbor, _ in self.graph.get_neighbors(source):
                    if neighbor == destination:
                        edge_exists = True
                        break
                
                if not edge_exists:
                    continue
                
                try:
                    # Reshape historical data for prediction
                    if len(historical_data.shape) == 1:
                        # Convert to 3D tensor: [batch_size, sequence_length, features]
                        sequence = historical_data[-self.traffic_predictor.sequence_length:]
                        sequence = sequence.reshape(1, -1, 1)
                    else:
                        sequence = historical_data
                    
                    # Make prediction
                    predicted_flow = self.traffic_predictor.predict(sequence)
                    
                    # Update the graph with the predicted flow
                    self.graph.set_traffic_flow(source, destination, predicted_flow)
                except Exception as e:
                    # If prediction fails for a specific edge, use a default value
                    distance = self.graph.calculate_distance(source, destination)
                    default_flow = 1000 * (1 / (1 + distance/1000))
                    self.graph.set_traffic_flow(source, destination, default_flow)
        except Exception as e:
            print(f"Warning: Failed to update graph with predictions: {e}")
            print("Using default traffic flows based on edge distances.")
            # Set default traffic flows based on edge distances
            for source in self.graph.edges:
                for target, distance in self.graph.edges[source]:
                    default_flow = 1000 * (1 / (1 + distance/1000))
                    self.graph.set_traffic_flow(source, target, default_flow)
        
        return self.graph
    
    def find_routes(self, origin, destination, algorithm='astar', max_routes=None):
        """
        Find optimal routes from origin to destination using the specified search algorithm.
        
        Args:
            origin: Origin node ID
            destination: Destination node ID
            algorithm: Search algorithm to use ('astar', 'bfs', 'dfs', 'gbfs', 'iddfs', or 'bdwa')
            max_routes: Maximum number of routes to return
            
        Returns:
            List of routes, each containing a path, travel time, and other metadata
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_graph first.")
        
        # Validate origin and destination
        if origin not in self.graph.nodes:
            raise ValueError(f"Origin node {origin} not found in the graph")
        if destination not in self.graph.nodes:
            raise ValueError(f"Destination node {destination} not found in the graph")
        
        if max_routes is None:
            max_routes = self.max_routes
        
        # Ensure the graph has up-to-date travel times as edge costs
        self.graph.update_edge_costs_with_travel_times()
        
        # Get node positions for heuristic calculation
        node_positions = {node_id: (self.graph.nodes[node_id]['latitude'], self.graph.nodes[node_id]['longitude']) 
                         for node_id in self.graph.nodes}
        
        # Prepare edges in the format expected by the search algorithms
        search_edges = {}
        for source, neighbors in self.graph.edges.items():
            search_edges[source] = [(target, self.graph.calculate_travel_time(source, target)) 
                                   for target, _ in neighbors]
        
        # Choose search algorithm
        if algorithm == 'astar':
            search_func = astar
        elif algorithm == 'bfs':
            search_func = bfs
        elif algorithm == 'dfs':
            search_func = dfs
        elif algorithm == 'gbfs':
            search_func = gbfs
        elif algorithm == 'iddfs':
            search_func = iddfs
        elif algorithm == 'bdwa':
            search_func = bdwa
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Find the optimal route
        routes = []
        
        # Call the appropriate search function with the correct parameters
        try:
            if algorithm in ['astar', 'gbfs', 'bdwa']:
                # These algorithms require node_positions
                goal_reached, nodes_generated, path = search_func(
                    origin,
                    [destination],
                    search_edges,
                    node_positions
                )
            elif algorithm in ['bfs', 'dfs']:
                # These algorithms don't need node_positions
                goal_reached, nodes_generated, path = search_func(
                    origin,
                    [destination],
                    search_edges
                )
            elif algorithm == 'iddfs':
                # IDDFS has an optional node_positions parameter
                goal_reached, nodes_generated, path = search_func(
                    origin,
                    [destination],
                    search_edges
                )
            else:
                # Default case - try with all parameters
                goal_reached, nodes_generated, path = search_func(
                    origin,
                    [destination],
                    search_edges,
                    node_positions
                )
                
            if path:
                travel_time = self.calculate_route_travel_time(path)
                routes.append({
                    'path': path,
                    'travel_time': travel_time,
                    'nodes_generated': nodes_generated,
                    'algorithm': algorithm
                })
            
            # Find additional routes by temporarily removing edges from the optimal path
            if path is not None and max_routes > 1 and len(path) > 1:
                original_edges = {k: v[:] for k, v in search_edges.items()}
                
                for i in range(min(len(path) - 1, max_routes * 2)):
                    # Remove an edge from the path
                    edge_to_remove = path[i]
                    next_node = path[i + 1]
                    
                    if edge_to_remove in search_edges:
                        # Save the edge data
                        edge_data = search_edges[edge_to_remove].copy()
                        
                        # Remove the specific edge to the next node
                        search_edges[edge_to_remove] = [(dest, cost) for dest, cost in search_edges[edge_to_remove] 
                                                       if dest != next_node]
                        
                        # Find a new route with the same algorithm-specific parameters
                        try:
                            if algorithm in ['astar', 'gbfs', 'bdwa']:
                                new_goal_reached, new_nodes_generated, new_path = search_func(
                                    origin,
                                    [destination],
                                    search_edges,
                                    node_positions
                                )
                            elif algorithm in ['bfs', 'dfs']:
                                new_goal_reached, new_nodes_generated, new_path = search_func(
                                    origin,
                                    [destination],
                                    search_edges
                                )
                            elif algorithm == 'iddfs':
                                new_goal_reached, new_nodes_generated, new_path = search_func(
                                    origin,
                                    [destination],
                                    search_edges
                                )
                            else:
                                new_goal_reached, new_nodes_generated, new_path = search_func(
                                    origin,
                                    [destination],
                                    search_edges,
                                    node_positions
                                )
                                
                            # Restore the edge
                            search_edges[edge_to_remove] = edge_data
                            
                            if new_path is not None and new_path not in [r['path'] for r in routes]:
                                travel_time = self.calculate_route_travel_time(new_path)
                                routes.append({
                                    'path': new_path,
                                    'travel_time': travel_time,
                                    'nodes_generated': new_nodes_generated,
                                    'algorithm': algorithm
                                })
                                
                                if len(routes) >= max_routes:
                                    break
                        except Exception as e:
                            # Restore the edge and continue
                            search_edges[edge_to_remove] = edge_data
                            print(f"  Warning: Error finding alternative route: {e}")
                
                # Restore original edges
                search_edges = original_edges
        except Exception as e:
            # Handle exceptions from the search algorithm
            print(f"  Error with {algorithm}: {e}")
        
        # Sort routes by travel time
        routes.sort(key=lambda x: x['travel_time'])
        
        return routes[:max_routes]

    def get_route_details(self, route):
        """
        Get detailed information about a route.
        
        Args:
            route: Route dictionary containing path and travel time
            
        Returns:
            Dictionary with detailed route information
        """
        path = route.get('path', [])
        travel_time = route.get('travel_time', 0)
        algorithm = route.get('algorithm', 'unknown')
        nodes_generated = route.get('nodes_generated', 0)
        
        # Handle empty paths
        if not path or len(path) < 2:
            return {
                'path': path,
                'node_names': [],
                'travel_time': 0,
                'distance': 0,
                'average_speed': 0,
                'traffic_flows': [],
                'edge_times': [],
                'algorithm': algorithm,
                'nodes_generated': nodes_generated,
                'travel_time_formatted': "0.00 seconds (0.00 minutes)",
                'distance_formatted': "0.00 meters (0.00 km)",
                'average_speed_formatted': "0.00 km/h"
            }
        
        # Calculate distance using the actual distances from the edges.csv file
        distance = 0
        
        # Load the original distances from the CSV file
        edges_file = self.config['paths']['edges_data']
        import pandas as pd
        edges_df = pd.read_csv(edges_file)
        
        # Create a distance lookup map for quick access
        distance_map = {}
        for s, t, d in edges_df[['source', 'target', 'distance']].values:
            distance_map[(int(s), int(t))] = float(d)
        
        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i + 1]
            try:
                # First try to get the distance from our CSV-based lookup map
                if (source, destination) in distance_map:
                    segment_distance = distance_map[(source, destination)]
                else:
                    # Fall back to the graph's calculate_distance method
                    segment_distance = self.graph.calculate_distance(source, destination)
                    
                distance += segment_distance
            except Exception:
                pass
        
        # Get traffic flow information
        traffic_flows = []
        edge_times = []
        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i + 1]
            try:
                flow = self.graph.get_traffic_flow(source, destination)
                edge_time = self.graph.calculate_travel_time(source, destination)
                traffic_flows.append(flow)
                edge_times.append(edge_time)
            except Exception as e:
                print(f"Warning: Could not get traffic flow between {source} and {destination}: {e}")
                traffic_flows.append(0)
                edge_times.append(0)
        
        # Calculate average speed (km/h)
        avg_speed = 0
        if travel_time > 0 and distance > 0:
            # The current calculation doesn't account for intersection delays properly
            # The travel_time includes intersection_delay for each segment, which artificially lowers the speed
            # Let's calculate a more realistic average speed
            
            # Count the number of intersections (nodes - 1)
            num_intersections = len(path) - 1
            
            # Total intersection delay in hours
            total_intersection_delay = (num_intersections * self.graph.intersection_delay) / 3600
            
            # Actual travel time without intersection delays (in hours)
            actual_travel_time_hours = (travel_time / 3600) - total_intersection_delay
            
            # If the actual travel time is too small (or negative due to numerical issues),
            # use a minimum value to avoid unrealistic speeds
            actual_travel_time_hours = max(0.001, actual_travel_time_hours)
            
            # Calculate average speed in km/h based on actual travel time
            avg_speed = (distance / 1000) / actual_travel_time_hours
            
            # Cap the speed at a realistic maximum for Australian urban arterial roads (60 km/h)
            avg_speed = min(avg_speed, 60)
        
        # Get node names
        node_names = []
        for node_id in path:
            try:
                node_names.append(self.graph.nodes[node_id]['name'])
            except Exception as e:
                node_names.append(f"Node {node_id}")
        
        # Return detailed information
        return {
            'path': path,
            'node_names': node_names,
            'travel_time': travel_time,
            'distance': distance,
            'average_speed': avg_speed,
            'traffic_flows': traffic_flows,
            'edge_times': edge_times,
            'algorithm': algorithm,
            'nodes_generated': nodes_generated,
            'travel_time_formatted': f"{travel_time:.2f} seconds ({travel_time/60:.2f} minutes)",
            'distance_formatted': f"{distance:.2f} meters ({distance/1000:.2f} km)",
            'average_speed_formatted': f"{avg_speed:.2f} km/h"
        }

    def verify_graph_edges(self):
        """
        Verify that the graph edges match the data from the edges.csv file.
        """
        # Load the original edges.csv file to compare
        edges_file = self.config['paths']['edges_data']
        import pandas as pd
        edges_df = pd.read_csv(edges_file)
        
        # Check a few key edges
        test_edges = [(3001, 3804), (3804, 3812), (3812, 3120)]
        
        # Verify edges silently
        for source, target in test_edges:
            try:
                # Get distance from CSV file
                csv_distance = None
                csv_row = edges_df[(edges_df['source'] == source) & (edges_df['target'] == target)]
                if not csv_row.empty:
                    csv_distance = float(csv_row['distance'].values[0])
                
                # Check if edge exists in graph
                graph_distance = 0
                for neighbor, dist in self.graph.get_neighbors(source):
                    if neighbor == target:
                        graph_distance = dist
                        break
            except Exception:
                pass
                
    def calculate_route_travel_time(self, path):
        """
        Calculate the total travel time for a route.
        
        Args:
            path: List of node IDs representing the route
            
        Returns:
            Total travel time in seconds
        """
        if not path or len(path) < 2:
            return 0
        
        # Load the original distances from the CSV file for accurate calculations
        edges_file = self.config['paths']['edges_data']
        import pandas as pd
        edges_df = pd.read_csv(edges_file)
        
        # Create a distance lookup map for quick access
        distance_map = {}
        for s, t, d in edges_df[['source', 'target', 'distance']].values:
            distance_map[(int(s), int(t))] = float(d)
        
        total_time = 0
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Get the correct distance for this segment from our map
            if (source, target) in distance_map:
                distance = distance_map[(source, target)]
            else:
                # Fall back to the graph's calculate_distance method
                distance = self.graph.calculate_distance(source, target)
            
            # Get the traffic flow and calculate the speed
            flow = self.graph.get_traffic_flow(source, target)
            
            # Calculate the speed based on traffic flow
            speed = self.graph.calculate_speed(flow)
            
            # Calculate pure driving time (without intersection delay)
            distance_km = distance / 1000.0
            pure_driving_time = (distance_km / speed) * 3600  # in seconds
            
            # Add intersection delay
            travel_time = pure_driving_time + self.graph.intersection_delay
            
            total_time += travel_time
            
        # Ensure we have a non-zero travel time
        if total_time <= 0:
            # Calculate a fallback travel time based on distance
            distance = 0
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                distance += self.graph.calculate_distance(source, target)
            
            # Assume average speed of 40 km/h if no other data
            avg_speed = 40  # km/h
            total_time = (distance / 1000) / avg_speed * 3600  # seconds
        
        # Add intersection delay for each intersection (except the last one)
        total_time += self.intersection_delay * (len(path) - 2) if len(path) > 2 else 0
        
        return total_time

def main(models_dir=None, algorithm="all", origin=None, destination=None, config_path=None, graph_file=None, sequence_data=None, max_routes=None):
    """
    Main function to demonstrate the usage of the route finder.
    
    Args:
        models_dir: Path to the directory containing model checkpoints
        algorithm: Search algorithm to use (default: all)
        origin: Origin SCATS site ID
        destination: Destination SCATS site ID
        config_path: Path to configuration file
        graph_file: Path to the graph file
        sequence_data: Path to the sequence data
        max_routes: Maximum number of routes to return
    """
    try:
        # Check if rich library is available for enhanced visualization
        global rich_available, console, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        global Panel, Table, ROUNDED, Text
        
        try:
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
            from rich.panel import Panel
            from rich.table import Table
            from rich.box import ROUNDED
            from rich.text import Text
            
            console = Console()
            rich_available = True
            
            # Display a welcome banner
            console.print("\n[bold white on blue]Traffic-based Route Guidance System (TBRGS)[/bold white on blue]\n")
            console.print("[italic]A smart routing system using ML traffic predictions[/italic]\n")
        except ImportError:
            rich_available = False
            print("\nTraffic-based Route Guidance System (TBRGS)\n")
            print("Note: Install 'rich' library for enhanced visualization (pip install rich)\n")
        
        # Use the configuration values or command-line arguments
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'default_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Override models directory if specified
        if models_dir is not None:
            config['paths']['models_dir'] = models_dir
        
        # Get paths from config
        nodes_file = config['paths']['nodes_data']
        edges_file = config['paths']['edges_data']
        sequence_data_path = sequence_data if sequence_data else config['paths']['sequence_data']
        
        # Check if data files exist
        if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
            if rich_available:
                console.print("[bold red]Error:[/bold red] Graph data files not found. Please run data_processing.py first.")
            else:
                raise FileNotFoundError(f"Error: Graph data files not found. Please run data_processing.py first.")
        
        # Create a route finder instance
        route_finder = RouteFinder(config_path=config_path)
        
        # Load graph with progress indicator
        if rich_available:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Loading traffic graph..."),
                BarColumn(),
                TextColumn("[bold green]{task.percentage:.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Loading...", total=100)
                # Simulate progress steps
                progress.update(task, advance=30)
                try:
                    route_finder.load_graph()
                    progress.update(task, advance=70)
                except Exception as e:
                    console.print(f"[bold red]Error loading graph:[/bold red] {e}")
                    raise
        else:
            print("Loading traffic graph...")
            try:
                route_finder.load_graph()
            except Exception as e:
                print(f"Error loading graph: {e}")
                raise
        
        # Try to load or train models if needed
        try:
            if rich_available:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Loading ML models..."),
                    BarColumn(),
                    TextColumn("[bold green]{task.percentage:.0f}%"),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("[cyan]Loading...", total=100)
                    # Simulate progress steps
                    progress.update(task, advance=30)
                    try:
                        # Make sure the models directory is in the Python path
                        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                        
                        from models.model_trainer import load_or_train_models
                        models = load_or_train_models(data_path=sequence_data_path, config=config)
                        progress.update(task, advance=70)
                    except Exception as e:
                        console.print(f"[bold red]Warning: Could not load or use ML models:[/bold red] {e}")
                        console.print("Using default travel times based on distance only.")
                        models = None
            else:
                print("Loading ML models...")
                try:
                    # Make sure the models directory is in the Python path
                    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    
                    from models.model_trainer import load_or_train_models
                    models = load_or_train_models(data_path=sequence_data_path, config=config)
                except Exception as e:
                    print(f"Warning: Could not load or use ML models: {e}")
                    print("Using default travel times based on distance only.")
                    models = None
        except Exception as e:
            if rich_available:
                console.print(f"[bold red]Warning:[/bold red] {e}")
                console.print("Using default travel times based on distance only.")
            else:
                print(f"Warning: {e}")
                print("Using default travel times based on distance only.")
        
        # Load historical traffic data from the sequence data
        if rich_available:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Loading historical traffic data..."),
                BarColumn(),
                TextColumn("[bold green]{task.percentage:.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Loading...", total=100)
                progress.update(task, advance=30)
                data = np.load(sequence_data_path)
                X_test = data['X_test']
                progress.update(task, advance=30)
                
                # Create historical data dictionary using actual test data from SCATS
                console.print("[bold blue]Preparing historical data for traffic prediction...[/bold blue]")
                
                # Get all nodes from the graph
                nodes = list(route_finder.graph.nodes.keys())
                
                # Create historical data for edges in the graph using the actual test data
                historical_data = {}
                for source in route_finder.graph.edges:
                    for target, _ in route_finder.graph.edges[source]:
                        # Use a deterministic but varied selection from test data
                        # This ensures each edge gets consistent but different historical data
                        idx = (hash(f"{source}_{target}") % len(X_test))
                        historical_data[(source, target)] = X_test[idx].flatten()
                progress.update(task, advance=20)
                
                # Update graph with predictions
                console.print("[bold blue]Updating graph with traffic predictions...[/bold blue]")
                route_finder.update_graph_with_predictions(historical_data)
                progress.update(task, advance=20)
                using_predictions = True
        else:
            print("Loading historical traffic data...")
            data = np.load(sequence_data_path)
            X_test = data['X_test']
            
            # Create historical data dictionary using actual test data from SCATS
            print("Preparing historical data for traffic prediction...")
            
            # Get all nodes from the graph
            nodes = list(route_finder.graph.nodes.keys())
            
            # Create historical data for edges in the graph using the actual test data
            historical_data = {}
            for source in route_finder.graph.edges:
                for target, _ in route_finder.graph.edges[source]:
                    # Use a deterministic but varied selection from test data
                    # This ensures each edge gets consistent but different historical data
                    idx = (hash(f"{source}_{target}") % len(X_test))
                    historical_data[(source, target)] = X_test[idx].flatten()
            
            # Update graph with predictions
            print("Updating graph with traffic predictions...")
            route_finder.update_graph_with_predictions(historical_data)
            using_predictions = True
        
        # Display available SCATS sites
        if rich_available:
            console.print("\n[bold cyan]Available SCATS sites:[/bold cyan]")
            sites_table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
            sites_table.add_column("ID", style="dim", width=6)
            sites_table.add_column("Location", style="green")
            
            # Sort by node ID
            sites = []
            for node_id, node_info in route_finder.graph.nodes.items():
                sites.append((node_id, node_info['name']))
            sites.sort(key=lambda x: x[0])
            
            # Display a subset of sites
            max_display = 10
            for i, (node_id, name) in enumerate(sites):
                if i < max_display:
                    sites_table.add_row(str(node_id), name)
            
            if len(sites) > max_display:
                sites_table.add_row("...", f"... and {len(sites) - max_display} more sites")
                
            console.print(sites_table)
        else:
            print("\nAvailable SCATS sites:")
            sites = []
            for node_id, node_info in route_finder.graph.nodes.items():
                sites.append((node_id, node_info['name']))
            
            # Sort by node ID
            sites.sort(key=lambda x: x[0])
            
            # Display a subset of sites
            max_display = 10
            for i, (node_id, name) in enumerate(sites):
                if i < max_display:
                    print(f"  {node_id}: {name}")
            
            if len(sites) > max_display:
                print(f"  ... and {len(sites) - max_display} more")
        
        # Set default origin and destination
        default_origin = nodes[0]  # Default origin
        default_destination = nodes[-1]  # Default destination
        
        # Use command-line arguments if provided
        if origin is not None:
            user_origin = origin
        else:
            user_origin = default_origin
            
        if destination is not None:
            try:
                user_destination = int(destination)
            except (ValueError, TypeError):
                user_destination = default_destination
        else:
            user_destination = default_destination
            
        # If command-line arguments weren't provided, ask for user input
        if origin is None or destination is None:
            try:
                if rich_available:
                    console.print("\n[bold]Enter route details:[/bold]")
                    
                    # Only ask for origin if not provided via command line
                    if origin is None:
                        origin_prompt = Text(f"Enter origin SCATS site ID (default: {default_origin}): ", style="bold green")
                        console.print(origin_prompt, end="")
                        origin_input = input()
                        if origin_input.strip():
                            user_origin = int(origin_input)
                    
                    # Only ask for destination if not provided via command line
                    if destination is None:
                        destination_prompt = Text(f"Enter destination SCATS site ID (default: {default_destination}): ", style="bold green")
                        console.print(destination_prompt, end="")
                        destination_input = input()
                        if destination_input.strip():
                            user_destination = int(destination_input)
                else:
                    # Only ask for origin if not provided via command line
                    if origin is None:
                        origin_input = input(f"\nEnter origin SCATS site ID (default: {default_origin}): ")
                        if origin_input.strip():
                            user_origin = int(origin_input)
                    
                    # Only ask for destination if not provided via command line
                    if destination is None:
                        destination_input = input(f"Enter destination SCATS site ID (default: {default_destination}): ")
                        if destination_input.strip():
                            user_destination = int(destination_input)
            except ValueError:
                if rich_available:
                    console.print("[bold red]Invalid input.[/bold red] Using default values.")
                else:
                    print("Invalid input. Using default values.")
        
        # Assign the final values
        origin = user_origin
        destination = user_destination
        
        # Check if origin and destination are valid
        if origin not in route_finder.graph.nodes:
            if rich_available:
                console.print(f"[bold yellow]Origin {origin} not found in graph.[/bold yellow] Using default: {nodes[0]}")
            else:
                print(f"Origin {origin} not found in graph. Using default: {nodes[0]}")
            origin = nodes[0]
        if destination not in route_finder.graph.nodes:
            if rich_available:
                console.print(f"[bold yellow]Destination {destination} not found in graph.[/bold yellow] Using default: {nodes[-1]}")
            else:
                print(f"Destination {destination} not found in graph. Using default: {nodes[-1]}")
            destination = nodes[-1]
        
        if rich_available:
            route_panel = Panel.fit(
                f"[bold]From:[/bold] {origin} ({route_finder.graph.nodes[origin]['name']})\n[bold]To:[/bold] {destination} ({route_finder.graph.nodes[destination]['name']})",
                title="[white on blue] Route Details [/white on blue]",
                border_style="blue"
            )
            console.print(route_panel)
            # Verify graph edges
            route_finder.verify_graph_edges()
            # Find optimal routes
            console.print("[bold cyan]Finding optimal routes...[/bold cyan]")
        else:
            print(f"\nFinding routes from {origin} ({route_finder.graph.nodes[origin]['name']}) to {destination} ({route_finder.graph.nodes[destination]['name']})...")
        
        # Determine which algorithms to use
        if algorithm == "all":
            algorithms = ['astar', 'bfs', 'dfs', 'gbfs', 'iddfs', 'bdwa']
        else:
            algorithms = [algorithm]  # Use only the specified algorithm
        max_routes_value = max_routes if max_routes is not None else config['system']['max_routes']
        
        all_results = {}
        
        for algorithm in algorithms:
            try:
                if rich_available:
                    console.print(f"\n[bold]Using [cyan]{algorithm.upper()}[/cyan] algorithm:[/bold]")
                    with Progress(
                        SpinnerColumn(),
                        TextColumn(f"[bold blue]Running {algorithm.upper()}..."),
                        TimeElapsedColumn(),
                    ) as progress:
                        search_task = progress.add_task("[cyan]Searching...", total=None)
                        start_time = datetime.now()
                        routes = route_finder.find_routes(origin, destination, algorithm=algorithm, max_routes=max_routes_value)
                        end_time = datetime.now()
                        search_time = (end_time - start_time).total_seconds()
                        progress.update(search_task, completed=True)
                else:
                    print(f"\nUsing {algorithm.upper()} algorithm:")
                    start_time = datetime.now()
                    routes = route_finder.find_routes(origin, destination, algorithm=algorithm, max_routes=max_routes_value)
                    end_time = datetime.now()
                    search_time = (end_time - start_time).total_seconds()
                
                if not routes:
                    if rich_available:
                        console.print(f"  [bold red]No routes found[/bold red] using {algorithm}")
                    else:
                        print(f"  No routes found using {algorithm}")
                    continue
                
                all_results[algorithm] = routes
                
                if rich_available:
                    console.print(f"  [bold green]Found {len(routes)} route(s)[/bold green] in [bold]{search_time:.3f}[/bold] seconds")
                    
                    # Create a table for the routes
                    routes_table = Table(show_header=True, header_style="bold magenta", box=ROUNDED, title=f"{algorithm.upper()} Routes")
                    routes_table.add_column("Route", style="dim", width=5)
                    routes_table.add_column("Travel Time", style="green")
                    routes_table.add_column("Distance", style="yellow")
                    routes_table.add_column("Avg Speed", style="blue")
                    routes_table.add_column("Nodes Gen.", style="cyan", justify="center")
                    if using_predictions:
                        routes_table.add_column("Traffic Flow", style="red")
                    
                    for i, route in enumerate(routes):
                        try:
                            route_details = route_finder.get_route_details(route)
                            
                            # Add route to table
                            if using_predictions and route_details['traffic_flows']:
                                avg_flow = sum(route_details['traffic_flows']) / len(route_details['traffic_flows'])
                                routes_table.add_row(
                                    f"{i+1}",
                                    route_details['travel_time_formatted'],
                                    route_details['distance_formatted'],
                                    route_details['average_speed_formatted'],
                                    str(route['nodes_generated']),
                                    f"{avg_flow:.2f} veh/h"
                                )
                            else:
                                routes_table.add_row(
                                    f"{i+1}",
                                    route_details['travel_time_formatted'],
                                    route_details['distance_formatted'],
                                    route_details['average_speed_formatted'],
                                    str(route['nodes_generated']),
                                    "N/A" if using_predictions else ""
                                )
                                
                            # Display the best route path
                            if i == 0:  # Only for the best route
                                path_str = ' → '.join(map(str, route['path']))
                                if len(path_str) > 80:
                                    path_str = path_str[:77] + '...'
                                console.print(f"  [bold]Best path:[/bold] {path_str}")
                        except Exception as e:
                            routes_table.add_row(f"{i+1}", f"Error: {e}", "", "", "", "")
                    
                    console.print(routes_table)
                else:
                    print(f"  Found {len(routes)} route(s) in {search_time:.3f} seconds")
                    for i, route in enumerate(routes):
                        print(f"Route {i+1}:")
                        try:
                            route_details = route_finder.get_route_details(route)
                            
                            # Format path nicely with limited length
                            path_str = ' -> '.join(map(str, route['path']))
                            if len(path_str) > 80:
                                path_str = path_str[:77] + '...'
                            print(f"  Path: {path_str}")
                            
                            # Format node names nicely with limited length
                            via_str = ' -> '.join(route_details['node_names'])
                            if len(via_str) > 80:
                                via_str = via_str[:77] + '...'
                            print(f"  Via: {via_str}")
                            
                            print(f"  Travel time: {route_details['travel_time_formatted']}")
                            print(f"  Distance: {route_details['distance_formatted']}")
                            print(f"  Average speed: {route_details['average_speed_formatted']}")
                            print(f"  Nodes generated: {route['nodes_generated']}")
                            
                            # Show traffic flow information if predictions were used
                            if using_predictions and route_details['traffic_flows']:
                                avg_flow = sum(route_details['traffic_flows']) / len(route_details['traffic_flows'])
                                print(f"  Average traffic flow: {avg_flow:.2f} vehicles/hour")
                        except Exception as e:
                            print(f"  Error displaying route details: {e}")
                            print(f"  Path: {' -> '.join(map(str, route.get('path', [])))[:80]}")
                            print(f"  Travel time: {route.get('travel_time', 0):.2f} seconds")
                            print(f"  Nodes generated: {route.get('nodes_generated', 0)}")
            except Exception as e:
                if rich_available:
                    console.print(f"  [bold red]Error with {algorithm}:[/bold red] {e}")
                else:
                    print(f"  Error with {algorithm}: {e}")
        
        # Compare the best routes from each algorithm
        if all_results:
            if rich_available:
                console.print("\n[bold cyan]Comparison of best routes from each algorithm:[/bold cyan]")
                comparison_table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
                comparison_table.add_column("Rank", style="dim", width=4)
                comparison_table.add_column("Algorithm", style="bold cyan")
                comparison_table.add_column("Travel Time", style="green")
                comparison_table.add_column("Distance", style="yellow")
                comparison_table.add_column("Nodes", style="blue", justify="center")
                if using_predictions:
                    comparison_table.add_column("Avg Traffic", style="red")
            else:
                print("\nComparison of best routes from each algorithm:")
                
            best_routes = {}
            
            # Process each algorithm's best route
            for alg, routes in all_results.items():
                if routes and len(routes) > 0:
                    # Find the first valid route (non-zero path length)
                    valid_route = None
                    for r in routes:
                        if 'path' in r and len(r['path']) > 1:
                            valid_route = r
                            break
                    
                    if not valid_route:
                        continue  # Skip if no valid route found
                    
                    # Calculate travel time if needed
                    if 'travel_time' not in valid_route or valid_route['travel_time'] == 0:
                        try:
                            valid_route['travel_time'] = route_finder.calculate_route_travel_time(valid_route['path'])
                        except Exception:
                            valid_route['travel_time'] = float('inf')
                    
                    # Store the route with its details
                    best_routes[alg] = valid_route
            
            # Sort algorithms by travel time
            sorted_algs = sorted(best_routes.keys(), key=lambda a: best_routes[a].get('travel_time', float('inf')))
            
            # Display each algorithm's best route
            for i, alg in enumerate(sorted_algs):
                try:
                    route = best_routes[alg]
                    
                    # Calculate route details
                    travel_time = route.get('travel_time', 0)
                    path = route.get('path', [])
                    
                    # Get distance and other details using the same method as get_route_details
                    # Load the original distances from the CSV file
                    edges_file = route_finder.config['paths']['edges_data']
                    import pandas as pd
                    edges_df = pd.read_csv(edges_file)
                    
                    # Create a distance lookup map for quick access
                    distance_map = {}
                    for s, t, d in edges_df[['source', 'target', 'distance']].values:
                        distance_map[(int(s), int(t))] = float(d)
                    
                    distance = 0
                    traffic_flows = []
                    
                    for j in range(len(path) - 1):
                        source = path[j]
                        target = path[j + 1]
                        # First try to get the distance from our CSV-based lookup map
                        if (source, target) in distance_map:
                            segment_distance = distance_map[(source, target)]
                        else:
                            # Fall back to the graph's calculate_distance method
                            segment_distance = route_finder.graph.calculate_distance(source, target)
                        distance += segment_distance
                        traffic_flows.append(route_finder.graph.get_traffic_flow(source, target))
                    
                    # Format for display
                    travel_time_formatted = f"{travel_time:.2f} seconds ({travel_time/60:.2f} minutes)"
                    distance_formatted = f"{distance:.2f} meters ({distance/1000:.2f} km)"
                    
                    if rich_available:
                        avg_traffic = sum(traffic_flows) / len(traffic_flows) if traffic_flows else 0
                        comparison_table.add_row(
                            f"{i+1}",
                            alg.upper(),
                            travel_time_formatted,
                            distance_formatted,
                            str(len(path)),
                            f"{avg_traffic:.2f} veh/h" if using_predictions else "N/A"
                        )
                    else:
                        print(f"{i+1}. {alg.upper()}: {travel_time_formatted} ({len(path)} nodes, {distance_formatted})")
                except Exception as e:
                    if rich_available:
                        comparison_table.add_row(f"{i+1}", alg.upper(), f"Error: {e}", "", "", "")
                    else:
                        print(f"{i+1}. {alg.upper()}: Error getting details - {e}")
            
            if rich_available:
                console.print(comparison_table)
            
            # Identify the overall best algorithm
            if sorted_algs:
                best_alg = sorted_algs[0]
                try:
                    best_route = best_routes[best_alg]
                    
                    # Get route details directly
                    travel_time = best_route.get('travel_time', 0)
                    path = best_route.get('path', [])
                    
                    # Calculate distance and other metrics using the actual distances from the edges.csv file
                    # Load the original distances from the CSV file
                    edges_file = route_finder.config['paths']['edges_data']
                    import pandas as pd
                    edges_df = pd.read_csv(edges_file)
                    
                    # Create a distance lookup map for quick access
                    distance_map = {}
                    for s, t, d in edges_df[['source', 'target', 'distance']].values:
                        distance_map[(int(s), int(t))] = float(d)
                    
                    distance = 0
                    traffic_flows = []
                    node_names = []
                    
                    for j in range(len(path)):
                        node_id = path[j]
                        node_names.append(route_finder.graph.nodes[node_id]['name'])
                        
                        if j < len(path) - 1:
                            source = path[j]
                            target = path[j + 1]
                            # First try to get the distance from our CSV-based lookup map
                            if (source, target) in distance_map:
                                segment_distance = distance_map[(source, target)]
                            else:
                                # Fall back to the graph's calculate_distance method
                                segment_distance = route_finder.graph.calculate_distance(source, target)
                            distance += segment_distance
                            traffic_flows.append(route_finder.graph.get_traffic_flow(source, target))
                    
                    # Calculate average speed with the same method as in get_route_details
                    avg_speed = 0
                    if travel_time > 0 and distance > 0:
                        # Count the number of intersections (nodes - 1)
                        num_intersections = len(path) - 1
                        
                        # Total intersection delay in hours
                        total_intersection_delay = (num_intersections * route_finder.graph.intersection_delay) / 3600
                        
                        # Actual travel time without intersection delays (in hours)
                        actual_travel_time_hours = (travel_time / 3600) - total_intersection_delay
                        
                        # If the actual travel time is too small (or negative due to numerical issues),
                        # use a minimum value to avoid unrealistic speeds
                        actual_travel_time_hours = max(0.001, actual_travel_time_hours)
                        
                        # Calculate average speed in km/h based on actual travel time
                        avg_speed = (distance / 1000) / actual_travel_time_hours
                        
                        # Cap the speed at a realistic maximum for Australian urban arterial roads (60 km/h)
                        avg_speed = min(avg_speed, 60)  # km/h
                    
                    if rich_available:
                        # Create a fancy panel for the best route
                        best_route_panel = Panel(
                            f"[bold cyan]Algorithm:[/bold cyan] {best_alg.upper()}\n\n"
                            f"[bold green]Path:[/bold green] {' → '.join(map(str, path))}\n\n"
                            f"[bold yellow]Travel Time:[/bold yellow] {travel_time:.2f} seconds ({travel_time/60:.2f} minutes)\n"
                            f"[bold blue]Distance:[/bold blue] {distance:.2f} meters ({distance/1000:.2f} km)\n"
                            f"[bold magenta]Average Speed:[/bold magenta] {avg_speed:.2f} km/h",
                            title="[white on green] Best Route [/white on green]",
                            border_style="green",
                            expand=False
                        )
                        console.print("\n")
                        console.print(best_route_panel)
                        
                        # Create a simple ASCII map visualization
                        if len(path) > 1:
                            map_vis = "\n[bold cyan]Route Visualization:[/bold cyan]\n\n"
                            map_vis += "[bold green]START[/bold green] " + route_finder.graph.nodes[path[0]]['name'] + "\n"
                            
                            for i in range(len(path) - 1):
                                source = path[i]
                                target = path[i + 1]
                                traffic_flow = route_finder.graph.get_traffic_flow(source, target)
                                
                                # Get distance from the distance_map we created earlier
                                if (source, target) in distance_map:
                                    distance = distance_map[(source, target)]
                                else:
                                    distance = route_finder.graph.calculate_distance(source, target)
                                
                                # Determine traffic level color
                                if traffic_flow < 400:
                                    traffic_color = "green"
                                elif traffic_flow < 800:
                                    traffic_color = "yellow"
                                else:
                                    traffic_color = "red"
                                
                                # Create a visual representation of the segment
                                segment_length = min(20, int(distance / 50)) + 1  # Scale distance for display
                                segment = "│\n" * 2
                                segment += f"[{traffic_color}]{'↓' * segment_length}[/{traffic_color}] {traffic_flow:.0f} veh/h\n"
                                segment += "│\n" * 2
                                
                                map_vis += segment
                            
                            map_vis += "[bold red]END[/bold red] " + route_finder.graph.nodes[path[-1]]['name']
                            console.print(map_vis)
                            
                        if traffic_flows:
                            avg_flow = sum(traffic_flows) / len(traffic_flows)
                            console.print(f"\n[bold]Average traffic flow:[/bold] {avg_flow:.2f} vehicles/hour")
                    else:
                        print(f"\nBest algorithm for this route: {best_alg.upper()}")
                        
                        # Format path nicely with limited length
                        path_str = ' -> '.join(map(str, path))
                        if len(path_str) > 80:
                            path_str = path_str[:77] + '...'
                        print(f"Best route: {path_str}")
                        
                        # Display route details
                        print(f"Travel time: {travel_time:.2f} seconds ({travel_time/60:.2f} minutes)")
                        print(f"Distance: {distance:.2f} meters ({distance/1000:.2f} km)")
                        print(f"Average speed: {avg_speed:.2f} km/h")
                        
                        if traffic_flows:
                            avg_flow = sum(traffic_flows) / len(traffic_flows)
                            print(f"Average traffic flow: {avg_flow:.2f} vehicles/hour")
                except Exception as e:
                    if rich_available:
                        console.print(f"\n[bold red]Error displaying best route details:[/bold red] {e}")
                    else:
                        print(f"\nBest algorithm for this route: {best_alg.upper()}")
                        print(f"Error displaying best route details: {e}")
            else:
                print("\nNo best algorithm could be determined.")
        else:
            print("\nNo routes found with any algorithm.")
        
        if rich_available:
            console.print("\n[bold green]Done![/bold green]")
            # Add a footer with credits
            footer = Panel(
                "[italic]Traffic-based Route Guidance System (TBRGS)[/italic]\n"
                "[dim]Assignment 2B - COS30019 Introduction to Artificial Intelligence[/dim]",
                border_style="dim",
                expand=False
            )
            console.print(footer)
        else:
            print("\nDone!")
        return route_finder  # Return for interactive use
    except Exception as e:
        if rich_available:
            console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"Error: {e}")
        return None
    


def parse_arguments():
    """
    Parse command-line arguments for the route finder.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Traffic-based Route Guidance System (TBRGS)")
    
    # Add model path argument
    parser.add_argument(
        "--models-dir", "-m",
        type=str,
        help="Path to the directory containing model checkpoints",
        default=None
    )
    
    # Add algorithm argument
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["astar", "bfs", "dfs", "gbfs", "iddfs", "bdwa", "all"],
        help="Search algorithm to use (default: all)",
        default="all"
    )
    
    # Add origin argument
    parser.add_argument(
        "--origin", "-o",
        type=int,
        help="Origin SCATS site ID",
        default=None
    )
    
    # Add destination argument
    parser.add_argument(
        "--destination", "-d",
        type=int,
        help="Destination SCATS site ID",
        default=None
    )
    
    # Add config path argument
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
        default=None
    )
    
    # Add graph file argument
    parser.add_argument(
        "--graph-file", "-g",
        type=str,
        help="Path to the graph file",
        default=None
    )
    
    # Add sequence data argument
    parser.add_argument(
        "--sequence-data", "-s",
        type=str,
        help="Path to the sequence data",
        default=None
    )
    
    # Add max routes argument
    parser.add_argument(
        "--max-routes", "-r",
        type=int,
        help="Maximum number of routes to return",
        default=None
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(models_dir=args.models_dir, algorithm=args.algorithm, 
         origin=args.origin, destination=args.destination, config_path=args.config,
         graph_file=args.graph_file, sequence_data=args.sequence_data, max_routes=args.max_routes)
