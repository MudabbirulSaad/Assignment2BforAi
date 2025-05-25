#!/usr/bin/env python3
"""
TBRGS SCATS Route Predictor

This module integrates the Route Predictor with the enhanced graph structure
and SCATS data to provide real-world traffic-based route guidance.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from functools import lru_cache

from app.core.logging import TBRGSLogger
from app.core.utils.graph import EnhancedGraph
from app.core.integration.route_predictor import RoutePredictor
from app.core.integration.travel_time_calculator import TravelTimeCalculator
from app.config.config import config

# Initialize logger
logger = TBRGSLogger.get_logger("integration.scats_route_predictor")

# Type aliases for clarity
SCATSID = str
NodeID = str
FlowValue = float
RouteInfo = Dict[str, Any]

class SCATSRoutePredictor:
    """
    SCATS Route Predictor that integrates with the enhanced graph structure
    and real SCATS data for traffic-based route guidance.
    """
    
    def __init__(self, graph_path: Optional[str] = None, prediction_model=None):
        """
        Initialize the SCATS Route Predictor.
        
        Args:
            graph_path: Path to the SCATS graph JSON file
                       If None, uses the default path from config
            prediction_model: Optional model for traffic flow predictions
        """
        # Load the SCATS graph
        if graph_path is None:
            graph_path = os.path.join(config.DATA_DIR, "processed", "scats_graph.json")
        
        self.graph = self._load_graph(graph_path)
        self.prediction_model = prediction_model
        self.travel_time_calculator = TravelTimeCalculator()
        
        # Create the Route Predictor
        self.route_predictor = RoutePredictor(
            self.graph,
            prediction_model=self.prediction_model,
            travel_time_calculator=self.travel_time_calculator
        )
        
        # Cache for route predictions
        self.route_cache = {}
        self.last_update_time = time.time()
        self.cache_timeout = 300  # 5 minutes
        
        logger.info(f"SCATS Route Predictor initialized with graph from {graph_path}")
    
    def _load_graph(self, graph_path: str) -> EnhancedGraph:
        """
        Load the SCATS graph from a JSON file.
        
        Args:
            graph_path: Path to the SCATS graph JSON file
            
        Returns:
            EnhancedGraph instance
        """
        try:
            # Load from JSON file
            with open(graph_path, 'r') as f:
                graph_dict = json.load(f)
            
            # Create graph from dictionary
            graph = EnhancedGraph.from_dict(graph_dict)
            
            logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {graph._count_edges()} edges from {graph_path}")
            return graph
        except Exception as e:
            logger.error(f"Error loading graph from {graph_path}: {e}")
            logger.info("Creating empty graph")
            return EnhancedGraph({})
    
    def get_routes(self, origin_scats: SCATSID, destination_scats: SCATSID,
                  prediction_time: Optional[Union[str, datetime]] = None,
                  max_routes: int = 5, confidence_level: float = 0.95,
                  algorithms: Optional[List[str]] = None) -> List[RouteInfo]:
        """
        Get optimal routes between SCATS sites based on predicted traffic conditions.
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time: Time for which to predict traffic
                If None, uses current time
            max_routes: Maximum number of routes to return
            confidence_level: Confidence level for prediction intervals (0.0-1.0)
            algorithms: List of routing algorithms to use
                If None, uses all available algorithms
                
        Returns:
            List of route information dictionaries, sorted by travel time
        """
        # Verify that the SCATS IDs exist in the graph
        if origin_scats not in self.graph.nodes:
            logger.error(f"Origin SCATS ID {origin_scats} not found in graph")
            return []
        
        if destination_scats not in self.graph.nodes:
            logger.error(f"Destination SCATS ID {destination_scats} not found in graph")
            return []
        
        # Use the Route Predictor to get routes
        routes = self.route_predictor.get_routes(
            origin_scats, destination_scats, prediction_time,
            max_routes, confidence_level, algorithms
        )
        
        # Add SCATS-specific information to the routes
        for route in routes:
            # Add SCATS site names
            scats_names = []
            for scats_id in route['scats_path']:
                if scats_id in self.graph.node_metadata:
                    name = self.graph.node_metadata[scats_id].get('name', scats_id)
                    scats_names.append(name)
                else:
                    scats_names.append(scats_id)
            
            route['scats_names'] = scats_names
        
        return routes
    
    @lru_cache(maxsize=100)
    def predict_route(self, origin_scats: SCATSID, destination_scats: SCATSID,
                      prediction_time_str: str, algorithm: str) -> RouteInfo:
        """
        Predict a route between SCATS sites with caching.
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time_str: Time string for which to predict traffic
            algorithm: Routing algorithm to use
            
        Returns:
            Route information dictionary
        """
        # Parse the prediction time
        try:
            prediction_time = datetime.strptime(prediction_time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.warning(f"Invalid prediction time format: {prediction_time_str}. Using current time.")
            prediction_time = datetime.now()
        
        # Get routes using the specified algorithm
        routes = self.get_routes(
            origin_scats, destination_scats, prediction_time, 
            max_routes=1, algorithms=[algorithm]
        )
        
        # Return the first route or an empty dictionary if no routes found
        return routes[0] if routes else {}
    
    def invalidate_cache(self) -> None:
        """
        Invalidate the route cache.
        """
        self.route_cache.clear()
        self.predict_route.cache_clear()
        self.last_update_time = time.time()
        logger.debug("Route cache invalidated")
    
    def update_traffic_flows(self, flows: Dict[SCATSID, FlowValue]) -> None:
        """
        Update traffic flows and edge weights in the graph.
        
        Args:
            flows: Dictionary mapping SCATS IDs to traffic flow values
        """
        # Update edge weights based on traffic flows
        for from_scats, flow in flows.items():
            if from_scats not in self.graph.nodes:
                continue
            
            # Update outgoing edges from this SCATS site
            for to_scats, weight in self.graph.get_neighbors(from_scats):
                # Calculate new travel time based on flow
                distance = self.graph.get_node_distance(from_scats, to_scats)
                
                if distance is None:
                    continue
                
                # Calculate speed based on flow using the flow-speed relationship
                # flow = -1.4648375 * (speed)^2 + 93.75 * (speed)
                # Solve for speed given flow
                if flow <= 0:
                    speed = 60.0  # Default to speed limit if flow is invalid
                else:
                    # Quadratic formula: ax^2 + bx + c = 0
                    # a = -1.4648375, b = 93.75, c = -flow
                    a = -1.4648375
                    b = 93.75
                    c = -flow
                    
                    # Calculate discriminant
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant < 0:
                        # No real solutions, use default speed
                        speed = 60.0
                    else:
                        # Two solutions, use the smaller one (realistic for traffic)
                        speed1 = (-b + (discriminant)**0.5) / (2*a)
                        speed2 = (-b - (discriminant)**0.5) / (2*a)
                        
                        # Choose the positive solution
                        if speed1 > 0:
                            speed = speed1
                        elif speed2 > 0:
                            speed = speed2
                        else:
                            # Both solutions are negative, use default speed
                            speed = 60.0
                        
                        # Cap speed at speed limit
                        speed = min(speed, 60.0)
                
                # Calculate travel time in seconds
                travel_time = (distance / speed) * 3600 if speed > 0 else float('inf')
                
                # Add intersection delay (30 seconds per intersection)
                travel_time += 30.0
                
                # Update edge weight
                self.graph.update_edge_weight(from_scats, to_scats, travel_time)
                
                # Update edge data
                edge_data = {
                    'flow': flow,
                    'speed': speed,
                    'distance': distance,
                    'travel_time': travel_time,
                    'last_updated': datetime.now().isoformat()
                }
                self.graph.set_edge_data(from_scats, to_scats, edge_data)
        
        # Invalidate route cache after updating edge weights
        self.invalidate_cache()
        logger.info(f"Updated traffic flows for {len(flows)} SCATS sites")
    
    def save_graph(self, output_path: Optional[str] = None) -> bool:
        """
        Save the current graph to a JSON file.
        
        Args:
            output_path: Path to save the graph
                        If None, uses the default path from config
                        
        Returns:
            True if successful, False otherwise
        """
        if output_path is None:
            output_path = os.path.join(config.DATA_DIR, "processed", "scats_graph.json")
        
        try:
            # Convert graph to dictionary
            graph_dict = self.graph.to_dict()
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)
            
            logger.info(f"Saved graph to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False


def main():
    """Main function to test the SCATS Route Predictor."""
    print("=" * 80)
    print("TBRGS SCATS Route Predictor Test")
    print("=" * 80)
    
    # Create a sample graph for testing
    # Note: We're creating a dictionary with node IDs mapping to coordinate tuples
    # This is the format expected by the EnhancedGraph constructor
    nodes = {
        '2000': (-37.85192, 145.09432),  # WARRIGAL_RD/TOORAK_RD
        '3002': (-37.81514, 145.02655),  # DENMARK_ST/BARKERS_RD
        '970': (-37.86730, 145.09151),   # WARRIGAL_RD/HIGH STREET_RD
        '2200': (-37.816540, 145.098047), # WARRIGAL_RD/WAVERLEY_RD
        '3001': (-37.814655, 145.022137), # GLENFERRIE_RD/BARKERS_RD
        '4035': (-37.81830, 145.05811)    # BARKERS_RD/BURKE_RD
    }
    
    # Create node metadata
    node_metadata = {
        '2000': {'name': 'WARRIGAL_RD/TOORAK_RD'},
        '3002': {'name': 'DENMARK_ST/BARKERS_RD'},
        '970': {'name': 'WARRIGAL_RD/HIGH STREET_RD'},
        '2200': {'name': 'WARRIGAL_RD/WAVERLEY_RD'},
        '3001': {'name': 'GLENFERRIE_RD/BARKERS_RD'},
        '4035': {'name': 'BARKERS_RD/BURKE_RD'}
    }
    
    # Create graph directly with nodes and metadata
    graph = EnhancedGraph(nodes)
    graph.node_metadata = node_metadata
    
    # Build fully connected graph with realistic travel times
    # Use a custom weight function that calculates travel time based on distance
    # Assume 60 km/h average speed and add 30 seconds intersection delay
    def weight_function(from_node, to_node, distance):
        # Calculate travel time in seconds: distance (km) / speed (km/h) * 3600 (s/h)
        travel_time = (distance / 60.0) * 3600
        # Add intersection delay
        travel_time += 30.0
        return travel_time
    
    graph.build_fully_connected_graph(weight_function)
    
    # Save graph to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        temp_path = tmp.name
    
    with open(temp_path, 'w') as f:
        json.dump(graph.to_dict(), f, indent=2)
    
    # Create SCATS Route Predictor
    predictor = SCATSRoutePredictor(graph_path=temp_path)
    
    # Define test origin and destination
    origin_scats = "2000"  # WARRIGAL_RD/TOORAK_RD
    destination_scats = "3002"  # DENMARK_ST/BARKERS_RD
    
    # Update traffic flows
    flows = {
        '2000': 100.0,  # Moderate traffic
        '3002': 50.0,   # Light traffic
        '970': 150.0,   # Heavy traffic
        '2200': 80.0,   # Moderate traffic
        '3001': 60.0,   # Light traffic
        '4035': 120.0   # Moderate-heavy traffic
    }
    
    predictor.update_traffic_flows(flows)
    
    # Get routes for current time
    routes = predictor.get_routes(origin_scats, destination_scats)
    
    if not routes:
        print("No routes found")
    else:
        print(f"\nFound {len(routes)} routes:")
        
        for i, route in enumerate(routes):
            print(f"\nRoute {i+1} ({route['algorithm']}):")
            print(f"  Travel Time: {route['travel_time']:.1f} seconds ({route['travel_time']/60:.1f} minutes)")
            print(f"  Distance: {route['distance']:.2f} km")
            print(f"  Average Speed: {route['average_speed']:.1f} km/h")
            
            if 'confidence_interval' in route and route['confidence_interval'] is not None:
                ci = route['confidence_interval']
                if 'lower_bound' in ci and 'upper_bound' in ci:
                    print(f"  Confidence Interval: {ci['lower_bound']/60:.1f} - {ci['upper_bound']/60:.1f} minutes " +
                         f"({ci['confidence_level']*100:.0f}% confidence)")
            
            print(f"  Path: {' -> '.join(route['scats_names'])}")
    
    # Test with different prediction times
    print("\nTesting different prediction times:")
    
    # Morning peak (8 AM)
    morning_peak = datetime.now().replace(hour=8, minute=0, second=0)
    print(f"\nMorning Peak ({morning_peak.strftime('%H:%M')})")
    
    # Update flows for morning peak
    peak_flows = {scats_id: flow * 2.0 for scats_id, flow in flows.items()}  # Double the flows for peak
    predictor.update_traffic_flows(peak_flows)
    
    routes = predictor.get_routes(origin_scats, destination_scats, morning_peak, max_routes=1)
    
    if routes:
        route = routes[0]
        print(f"  Travel Time: {route['travel_time']:.1f} seconds ({route['travel_time']/60:.1f} minutes)")
        print(f"  Average Speed: {route['average_speed']:.1f} km/h")
    
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
