#!/usr/bin/env python3
"""
TBRGS Routing Adapter Module

This module provides adapters for the Part A routing algorithms to work with the
Traffic-Based Route Guidance System. It handles the interface between the traffic
prediction system and the existing routing algorithms.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Tuple, Optional, Any

# Add the methods directory to the Python path
methods_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'methods')
if methods_dir not in sys.path:
    sys.path.append(methods_dir)

# Import the Graph class and input parser
try:
    from app.core.methods.graph import Graph
    from app.core.methods.input_parser import build_data
except ImportError:
    # Fallback to direct import if package import fails
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.methods.graph import Graph
    from core.methods.input_parser import build_data

# Initialize logger
from app.core.logging import TBRGSLogger
logger = TBRGSLogger.get_logger("integration.routing_adapter")


def load_algorithm(algorithm_name: str):
    """
    Dynamically load a routing algorithm from the methods directory.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'astar', 'bfs')
        
    Returns:
        function: The loaded algorithm function or None if not found
    """
    try:
        # First try package import
        module_name = f"app.core.methods.{algorithm_name}_search"
        module = importlib.import_module(module_name)
        
        # Get the main algorithm function (same name as the algorithm)
        algorithm_func = getattr(module, algorithm_name)
        logger.debug(f"Loaded algorithm {algorithm_name} from {module_name}")
        return algorithm_func
    except (ImportError, AttributeError) as e:
        # Try direct import
        try:
            module_path = os.path.join(methods_dir, f"{algorithm_name}_search.py")
            spec = importlib.util.spec_from_file_location(f"{algorithm_name}_search", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the main algorithm function
            algorithm_func = getattr(module, algorithm_name)
            logger.debug(f"Loaded algorithm {algorithm_name} from {module_path}")
            return algorithm_func
        except (ImportError, AttributeError, FileNotFoundError) as e2:
            logger.error(f"Could not load algorithm {algorithm_name}: {e2}")
            return None


class RoutingAdapter:
    """
    Adapter class for Part A routing algorithms.
    
    This class provides a consistent interface for the routing algorithms
    to work with the Traffic-Based Route Guidance System.
    """
    
    def __init__(self):
        """
        Initialize the routing adapter.
        """
        # Load algorithms
        self.algorithms = {}
        for algo_name in ['astar', 'bfs', 'dfs', 'gbfs', 'iddfs', 'bdwa']:
            algorithm = load_algorithm(algo_name)
            if algorithm:
                self.algorithms[algo_name] = algorithm
        
        logger.info(f"Routing Adapter initialized with {len(self.algorithms)} algorithms")
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get the names of available routing algorithms.
        
        Returns:
            List[str]: List of algorithm names
        """
        return list(self.algorithms.keys())
    
    def find_route(self, graph: Any, origin: str, destination: str, 
                  algorithm: str = 'astar') -> Dict[str, Any]:
        """
        Find a route using the specified algorithm.
        
        Args:
            graph: The routing graph
            origin: Origin node ID
            destination: Destination node ID
            algorithm: Name of the routing algorithm to use
            
        Returns:
            Dict[str, Any]: Route information dictionary
        """
        if algorithm not in self.algorithms:
            logger.warning(f"Unknown routing algorithm: {algorithm}")
            return {}
        
        # Get the algorithm function
        algorithm_func = self.algorithms[algorithm]
        
        # Prepare the graph for the algorithm
        # The Part A algorithms expect a specific format
        try:
            # Ensure we're using integers for edge costs to avoid type errors
            # Create a copy of the edges with integer costs
            integer_edges = {}
            for node, neighbors in graph.edges.items():
                integer_edges[node] = []
                for neighbor, cost in neighbors:
                    # Convert float costs to integers to avoid type errors
                    integer_edges[node].append((neighbor, int(cost)))
            
            # Call the algorithm - handle different parameter requirements
            if algorithm in ['bfs', 'dfs']:
                # These algorithms only take 3 parameters
                goal_reached, nodes_generated, path = algorithm_func(
                    origin, [destination], integer_edges)
            elif algorithm == 'iddfs':
                # IDDFS needs special handling for the max depth
                goal_reached, nodes_generated, path = algorithm_func(
                    origin, [destination], integer_edges, max_depth=100)
            else:
                # A*, GBFS, BDWA take all 4 parameters
                goal_reached, nodes_generated, path = algorithm_func(
                    origin, [destination], integer_edges, graph.nodes)
            
            if not goal_reached or not path:
                logger.warning(f"No route found using {algorithm} from {origin} to {destination}")
                return {}
            
            # Calculate the route cost using the original graph to maintain accuracy
            cost = 0
            for i in range(len(path) - 1):
                source = path[i]
                # Handle different edge formats
                if hasattr(graph, 'get_neighbors'):
                    # Use the graph's get_neighbors method if available
                    neighbors = graph.get_neighbors(source)
                    for dest, edge_cost in neighbors:
                        if dest == path[i + 1]:
                            cost += edge_cost
                            break
                else:
                    # Fall back to using the edges dictionary directly
                    for dest, edge_cost in graph.edges.get(source, []):
                        if dest == path[i + 1]:
                            cost += edge_cost
                            break
            
            # Create route information dictionary
            route_info = {
                'algorithm': algorithm,
                'path': path,
                'cost': cost,
                'nodes_generated': nodes_generated,
                'goal_reached': goal_reached
            }
            
            return route_info
        except Exception as e:
            logger.error(f"Error finding route with {algorithm}: {e}")
            return {}


# Create a singleton instance for easy import
adapter = RoutingAdapter()


def find_route(graph: Any, origin: str, destination: str, algorithm: str = 'astar') -> Dict[str, Any]:
    """
    Convenience function to find a route using the routing adapter.
    
    Args:
        graph: The routing graph
        origin: Origin node ID
        destination: Destination node ID
        algorithm: Name of the routing algorithm to use
        
    Returns:
        Dict[str, Any]: Route information dictionary
    """
    return adapter.find_route(graph, origin, destination, algorithm)


def get_available_algorithms() -> List[str]:
    """
    Convenience function to get the names of available routing algorithms.
    
    Returns:
        List[str]: List of algorithm names
    """
    return adapter.get_available_algorithms()


if __name__ == "__main__":
    # Test the routing adapter
    print("=" * 80)
    print("TBRGS Routing Adapter Test")
    print("=" * 80)
    
    # Create a simple test graph
    nodes = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 0),
        'D': (0, 2),
        'E': (2, 2)
    }
    
    edges = {
        'A': [('B', 1), ('C', 2)],
        'B': [('A', 1), ('C', 1), ('D', 2), ('E', 3)],
        'C': [('A', 2), ('B', 1), ('E', 2)],
        'D': [('B', 2), ('E', 1)],
        'E': [('B', 3), ('C', 2), ('D', 1)]
    }
    
    graph = Graph(nodes, edges)
    
    # Print available algorithms
    print(f"Available algorithms: {get_available_algorithms()}")
    
    # Test each algorithm
    for algorithm in get_available_algorithms():
        print(f"\nTesting {algorithm}:")
        route = find_route(graph, 'A', 'E', algorithm)
        
        if route:
            print(f"  Path: {' -> '.join(route['path'])}")
            print(f"  Cost: {route['cost']}")
            print(f"  Nodes generated: {route['nodes_generated']}")
        else:
            print(f"  No route found")
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
