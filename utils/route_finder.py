# utils/route_finder.py
# Interface for finding optimal routes using search algorithms
# Adapted from Assignment 2A's search.py

import heapq
from utils.graph_utils import TrafficGraph

class RouteFinder:
    def __init__(self, traffic_graph, max_routes=5):
        """
        Initializes the RouteFinder with a traffic graph.
        
        Args:
            traffic_graph: A TrafficGraph instance.
            max_routes: Maximum number of routes to return (default: 5).
        """
        self.graph = traffic_graph
        self.max_routes = max_routes
        
    def find_routes(self, origin, destination, method="AS"):
        """
        Finds optimal routes from origin to destination using the specified search method.
        
        Args:
            origin: The origin SCATS site ID.
            destination: The destination SCATS site ID.
            method: The search method to use (default: "AS" for A*).
                    Options: "BFS", "DFS", "GBFS", "AS", "CUS1", "CUS2"
                    
        Returns:
            A list of tuples (path, travel_time) representing the top-k routes.
        """
        # Import search algorithms
        from utils.search.bfs import bfs
        from utils.search.dfs import dfs
        from utils.search.gbfs import gbfs
        from utils.search.astar import astar
        from utils.search.iddfs import iddfs
        from utils.search.bdwa import bdwa
        
        # Update edge costs with current travel times
        self.graph.update_edge_costs_with_travel_times()
        
        # Select the appropriate search algorithm
        if method == "BFS":
            goal, count, path = bfs(origin, [destination], self.graph.edges)
        elif method == "DFS":
            goal, count, path = dfs(origin, [destination], self.graph.edges)
        elif method == "GBFS":
            goal, count, path = gbfs(origin, [destination], self.graph.edges, self.graph.nodes)
        elif method == "AS":
            goal, count, path = astar(origin, [destination], self.graph.edges, self.graph.nodes)
        elif method == "CUS1":
            goal, count, path = iddfs(origin, [destination], self.graph.edges)
        elif method == "CUS2":
            goal, count, path = bdwa(origin, [destination], self.graph.edges, self.graph.nodes)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # If no path was found, return an empty list
        if not path:
            return []
        
        # Calculate the total travel time for the path
        total_time = self.calculate_path_travel_time(path)
        
        # For now, we only have one path, but in the future, we'll implement
        # functionality to find multiple paths
        return [(path, total_time)]
    
    def find_multiple_routes(self, origin, destination, methods=None):
        """
        Finds multiple routes using different search algorithms.
        
        Args:
            origin: The origin SCATS site ID.
            destination: The destination SCATS site ID.
            methods: List of search methods to use. If None, uses all available methods.
                    
        Returns:
            A list of tuples (path, travel_time) representing the top-k routes.
        """
        if methods is None:
            methods = ["AS", "GBFS", "CUS2", "BFS", "CUS1", "DFS"]
        
        all_routes = []
        
        # Find routes using each method
        for method in methods:
            routes = self.find_routes(origin, destination, method)
            all_routes.extend(routes)
        
        # Remove duplicate paths
        unique_routes = []
        seen_paths = set()
        
        for path, time in all_routes:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                unique_routes.append((path, time))
        
        # Sort by travel time and take the top-k routes
        unique_routes.sort(key=lambda x: x[1])
        return unique_routes[:self.max_routes]
    
    def calculate_path_travel_time(self, path):
        """
        Calculates the total travel time for a path.
        
        Args:
            path: A list of SCATS site IDs representing a path.
            
        Returns:
            The total travel time in seconds.
        """
        total_time = 0
        
        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i + 1]
            travel_time = self.graph.calculate_travel_time(source, destination)
            total_time += travel_time
        
        return total_time
    
    def format_route_output(self, route, node_names=None):
        """
        Formats a route for display.
        
        Args:
            route: A tuple (path, travel_time).
            node_names: A dictionary mapping node IDs to human-readable names.
            
        Returns:
            A dictionary with formatted route information.
        """
        path, travel_time = route
        
        # Format travel time
        hours = int(travel_time // 3600)
        minutes = int((travel_time % 3600) // 60)
        seconds = int(travel_time % 60)
        
        formatted_time = ""
        if hours > 0:
            formatted_time += f"{hours} hour{'s' if hours != 1 else ''} "
        if minutes > 0:
            formatted_time += f"{minutes} minute{'s' if minutes != 1 else ''} "
        if seconds > 0 or (hours == 0 and minutes == 0):
            formatted_time += f"{seconds} second{'s' if seconds != 1 else ''}"
        
        # Format path
        formatted_path = []
        for node in path:
            if node_names and node in node_names:
                formatted_path.append(f"{node} ({node_names[node]})")
            else:
                formatted_path.append(str(node))
        
        return {
            "path": path,
            "formatted_path": formatted_path,
            "travel_time_seconds": travel_time,
            "formatted_time": formatted_time.strip()
        }
