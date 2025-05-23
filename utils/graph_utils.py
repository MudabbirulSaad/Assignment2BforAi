# utils/graph_utils.py
# Enhanced Graph class for the Traffic-based Route Guidance System (TBRGS)
# Adapted from Assignment 2A's graph.py

import math
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        """
        Initializes the graph with the provided nodes and edges.
        
        Args:
            nodes: A dictionary mapping node IDs to coordinate tuples (x, y).
            edges: A dictionary mapping source node IDs to a list of tuples (destination, cost).
        """
        self.nodes = nodes  # Store node information.
        self.edges = edges  # Store edge information (connections between nodes).

    def get_neighbors(self, node):
        """
        Retrieves the neighbors of a given node.
        
        Args:
            node: The node ID whose neighbors are required.
        
        Returns:
            A list of tuples (neighbor, cost). Returns an empty list if the node has no outgoing edges.
        """
        return self.edges.get(node, [])
    
    def calculate_euclidean_distance(self, node1, node2):
        """
        Calculates the Euclidean distance between two nodes.
        
        Args:
            node1: The ID of the first node.
            node2: The ID of the second node.
            
        Returns:
            The Euclidean distance between the nodes.
        """
        if node1 in self.nodes and node2 in self.nodes:
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return float('inf')


class TrafficGraph(Graph):
    def __init__(self, nodes, edges, speed_limit=60, intersection_delay=30):
        """
        Initializes the traffic graph with nodes, edges, and traffic-specific parameters.
        
        Args:
            nodes: A dictionary mapping SCATS site IDs to coordinate tuples (x, y).
            edges: A dictionary mapping source SCATS site IDs to a list of tuples (destination, distance).
            speed_limit: The speed limit in km/h (default: 60).
            intersection_delay: The average delay at each controlled intersection in seconds (default: 30).
        """
        super().__init__(nodes, edges)
        self.speed_limit = speed_limit  # km/h
        self.intersection_delay = intersection_delay  # seconds
        self.traffic_flows = {}  # Will store predicted traffic flows for each edge
    
    def set_traffic_flow(self, source, destination, flow):
        """
        Sets the predicted traffic flow for an edge.
        
        Args:
            source: The source SCATS site ID.
            destination: The destination SCATS site ID.
            flow: The predicted traffic flow (vehicles per hour).
        """
        if source not in self.traffic_flows:
            self.traffic_flows[source] = {}
        self.traffic_flows[source][destination] = flow
    
    def get_traffic_flow(self, source, destination):
        """
        Gets the predicted traffic flow for an edge.
        
        Args:
            source: The source SCATS site ID.
            destination: The destination SCATS site ID.
            
        Returns:
            The predicted traffic flow, or 0 if not set.
        """
        return self.traffic_flows.get(source, {}).get(destination, 0)
    
    def calculate_speed(self, flow):
        """
        Calculates the expected speed based on traffic flow.
        
        Args:
            flow: The traffic flow in vehicles per hour.
            
        Returns:
            The expected speed in km/h.
        """
        # Check if flow is below the threshold for free flow
        if flow <= 351:
            return self.speed_limit  # Free flow speed = speed limit
        
        # Simplified speed-flow relationship
        # As traffic flow increases, speed decreases
        # We'll use a piecewise linear relationship:
        
        # Maximum flow capacity is around 1500 vehicles/hour
        max_flow = 1500
        
        if flow <= max_flow:
            # For flows between 351 and 1500:
            # Speed decreases linearly from 60 km/h to 30 km/h
            flow_ratio = (flow - 351) / (max_flow - 351)
            speed = self.speed_limit - (flow_ratio * 30)  # Decrease by up to 30 km/h
        else:
            # For flows above 1500:
            # Speed decreases from 30 km/h down to 10 km/h as flow increases
            excess_ratio = min(1.0, (flow - max_flow) / 1000)  # Cap at 2500 vehicles/hour
            speed = 30 - (excess_ratio * 20)  # Decrease from 30 down to 10 km/h
        
        return max(10, min(speed, self.speed_limit))  # Ensure speed is between 10 and speed_limit
    
    def calculate_distance(self, source, destination):
        """
        Calculates the physical distance between two adjacent SCATS sites.
        
        Args:
            source: The source SCATS site ID.
            destination: The destination SCATS site ID.
            
        Returns:
            The distance in meters, or float('inf') if nodes are not connected.
        """
        # Get the distance between the nodes from the edge data
        distance = 0
        for neighbor, dist in self.get_neighbors(source):
            if neighbor == destination:
                distance = dist
                break
        
        if distance == 0:
            # If not directly connected, return infinity
            return float('inf')
            
        return distance
        
    def calculate_travel_time(self, source, destination):
        """
        Calculates the travel time between two adjacent SCATS sites.
        
        Args:
            source: The source SCATS site ID.
            destination: The destination SCATS site ID.
            
        Returns:
            The travel time in seconds.
        """
        # Get the distance between the nodes
        distance = self.calculate_distance(source, destination)
        
        if distance == float('inf'):
            return float('inf')  # Nodes are not connected
        
        # Get the traffic flow and calculate the speed
        flow = self.get_traffic_flow(source, destination)
        speed = self.calculate_speed(flow)
        
        # Calculate travel time: time = distance / speed
        # Convert distance from meters to km to match speed in km/h
        distance_km = distance / 1000.0
        travel_time_hours = distance_km / speed
        
        # Convert to seconds and add intersection delay
        travel_time_seconds = travel_time_hours * 3600 + self.intersection_delay
        
        return travel_time_seconds
    
    def update_edge_costs_with_travel_times(self):
        """
        Updates the edge costs with calculated travel times.
        This should be called after setting traffic flows and before running search algorithms.
        """
        updated_edges = {}
        
        for source, neighbors in self.edges.items():
            updated_neighbors = []
            for destination, _ in neighbors:
                travel_time = self.calculate_travel_time(source, destination)
                updated_neighbors.append((destination, travel_time))
            updated_edges[source] = updated_neighbors
        
        self.edges = updated_edges


def build_traffic_graph(nodes, edges, speed_limit=60, intersection_delay=30):
    """
    Creates a TrafficGraph instance from nodes and edges.
    
    Args:
        nodes: A dictionary mapping SCATS site IDs to coordinate tuples (x, y).
        edges: A dictionary mapping source SCATS site IDs to a list of tuples (destination, distance).
        speed_limit: The speed limit in km/h (default: 60).
        intersection_delay: The average delay at each controlled intersection in seconds (default: 30).
        
    Returns:
        A TrafficGraph instance.
    """
    return TrafficGraph(nodes, edges, speed_limit, intersection_delay)
