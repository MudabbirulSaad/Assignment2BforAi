#!/usr/bin/env python3
"""
TBRGS Enhanced Graph Module

This module defines an enhanced Graph class that supports:
1. SCATS site coordinates
2. Dynamic edge weight updates
3. Real-time graph updates
4. Performance optimizations
5. Graph validation

This is an extension of the basic Graph class from methods/graph.py,
adapted specifically for the TBRGS project requirements.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from app.core.logging import TBRGSLogger
from app.core.integration.geo_calculator import haversine_distance

# Initialize logger
logger = TBRGSLogger.get_logger("utils.graph")

# Type aliases for clarity
NodeID = str  # Graph node ID (typically SCATS site ID)
EdgeWeight = float  # Edge weight in seconds
Coordinate = Tuple[float, float]  # (latitude, longitude)
EdgeData = Dict[str, Any]  # Additional edge data (flow, speed, etc.)

class EnhancedGraph:
    """
    Enhanced Graph class with support for SCATS site coordinates and dynamic edge weights.
    """
    
    def __init__(self, nodes: Dict[NodeID, Coordinate], edges: Optional[Dict[NodeID, List[Tuple[NodeID, EdgeWeight]]]] = None):
        """
        Initialize the enhanced graph with nodes and optional edges.
        
        Args:
            nodes: Dictionary mapping node IDs to coordinate tuples (latitude, longitude)
            edges: Optional dictionary mapping source node IDs to a list of tuples (destination, cost)
                  If not provided, an empty edge dictionary will be created
        """
        self.nodes = nodes  # Store node information with coordinates
        self.edges = edges or {}  # Store edge information with weights
        
        # Additional data structures for enhanced functionality
        self.edge_data = {}  # Store additional edge data (flow, speed, regime, etc.)
        self.last_update_time = time.time()  # Track when the graph was last updated
        self.node_metadata = {}  # Store additional node metadata (SCATS site info, etc.)
        
        # Performance optimization: pre-compute distances between nodes
        self._distance_cache = {}
        
        # Validate the graph on initialization
        self._validate_graph()
        
        logger.info(f"Enhanced Graph initialized with {len(self.nodes)} nodes and {self._count_edges()} edges")
    
    def get_neighbors(self, node: NodeID) -> List[Tuple[NodeID, EdgeWeight]]:
        """
        Retrieve the neighbors of a given node.
        
        Args:
            node: The node ID whose neighbors are required
            
        Returns:
            List of tuples (neighbor, cost)
            Returns an empty list if the node has no outgoing edges
        """
        return self.edges.get(node, [])
    
    def get_edge_data(self, from_node: NodeID, to_node: NodeID) -> Optional[EdgeData]:
        """
        Get additional data for a specific edge.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            
        Returns:
            Dictionary with edge data or None if the edge doesn't exist
        """
        edge_key = (from_node, to_node)
        return self.edge_data.get(edge_key)
    
    def set_edge_data(self, from_node: NodeID, to_node: NodeID, data: EdgeData) -> None:
        """
        Set additional data for a specific edge.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            data: Dictionary with edge data
        """
        edge_key = (from_node, to_node)
        self.edge_data[edge_key] = data
    
    def update_edge_weight(self, from_node: NodeID, to_node: NodeID, new_weight: EdgeWeight) -> bool:
        """
        Update the weight of a specific edge.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            new_weight: New edge weight
            
        Returns:
            True if the edge was updated, False if the edge doesn't exist
        """
        if from_node not in self.edges:
            return False
        
        for i, (dest, _) in enumerate(self.edges[from_node]):
            if dest == to_node:
                self.edges[from_node][i] = (dest, new_weight)
                self.last_update_time = time.time()
                return True
        
        return False
    
    def add_node(self, node_id: NodeID, coordinates: Coordinate, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new node to the graph.
        
        Args:
            node_id: Node ID (typically SCATS site ID)
            coordinates: Tuple of (latitude, longitude)
            metadata: Optional dictionary with additional node metadata
        """
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists in the graph. Updating coordinates.")
        
        self.nodes[node_id] = coordinates
        
        if metadata:
            self.node_metadata[node_id] = metadata
        
        # Clear distance cache for this node
        self._clear_distance_cache(node_id)
        
        self.last_update_time = time.time()
    
    def add_edge(self, from_node: NodeID, to_node: NodeID, weight: EdgeWeight, 
                edge_data: Optional[EdgeData] = None) -> bool:
        """
        Add a new edge to the graph.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            weight: Edge weight
            edge_data: Optional dictionary with additional edge data
            
        Returns:
            True if the edge was added, False if the nodes don't exist
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            logger.warning(f"Cannot add edge: nodes {from_node} or {to_node} don't exist in the graph")
            return False
        
        # Initialize edges list for the source node if it doesn't exist
        if from_node not in self.edges:
            self.edges[from_node] = []
        
        # Check if the edge already exists
        for i, (dest, _) in enumerate(self.edges[from_node]):
            if dest == to_node:
                # Update existing edge
                self.edges[from_node][i] = (dest, weight)
                if edge_data:
                    self.set_edge_data(from_node, to_node, edge_data)
                return True
        
        # Add new edge
        self.edges[from_node].append((to_node, weight))
        
        if edge_data:
            self.set_edge_data(from_node, to_node, edge_data)
        
        self.last_update_time = time.time()
        return True
    
    def remove_node(self, node_id: NodeID) -> bool:
        """
        Remove a node and all its connected edges from the graph.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if the node was removed, False if it doesn't exist
        """
        if node_id not in self.nodes:
            return False
        
        # Remove the node
        del self.nodes[node_id]
        
        # Remove any metadata
        if node_id in self.node_metadata:
            del self.node_metadata[node_id]
        
        # Remove outgoing edges
        if node_id in self.edges:
            del self.edges[node_id]
        
        # Remove incoming edges
        for source in self.edges:
            self.edges[source] = [(dest, weight) for dest, weight in self.edges[source] if dest != node_id]
        
        # Remove edge data
        for edge_key in list(self.edge_data.keys()):
            if edge_key[0] == node_id or edge_key[1] == node_id:
                del self.edge_data[edge_key]
        
        # Clear distance cache for this node
        self._clear_distance_cache(node_id)
        
        self.last_update_time = time.time()
        return True
    
    def get_node_distance(self, node1: NodeID, node2: NodeID) -> Optional[float]:
        """
        Calculate the distance between two nodes using their coordinates.
        
        Args:
            node1: ID of the first node
            node2: ID of the second node
            
        Returns:
            float: Distance in kilometers or None if nodes don't exist
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            return None
        
        return haversine_distance(self.nodes[node1], self.nodes[node2])
        
    def has_edge(self, node1: NodeID, node2: NodeID) -> bool:
        """
        Check if there is an edge between two nodes.
        
        Args:
            node1: ID of the first node
            node2: ID of the second node
            
        Returns:
            bool: True if there is an edge from node1 to node2, False otherwise
        """
        if node1 not in self.edges:
            return False
            
        for neighbor, _ in self.edges[node1]:
            if neighbor == node2:
                return True
                
        return False
    
    def build_fully_connected_graph(self, default_weight_function=None) -> None:
        """
        Build a fully connected graph from the nodes.
        
        Args:
            default_weight_function: Optional function to calculate default edge weights
                                    Function signature: (from_node, to_node, distance) -> weight
                                    If not provided, distance in kilometers * 60 will be used
                                    (assuming 60 km/h average speed, converting to seconds)
        """
        if not default_weight_function:
            # Default function: distance in km * 60 seconds (assuming 60 km/h)
            default_weight_function = lambda from_node, to_node, distance: distance * 60
        
        # Create edges between all nodes
        for from_node in self.nodes:
            if from_node not in self.edges:
                self.edges[from_node] = []
            
            for to_node in self.nodes:
                if from_node != to_node:
                    distance = self.get_node_distance(from_node, to_node)
                    weight = default_weight_function(from_node, to_node, distance)
                    
                    # Check if edge already exists
                    edge_exists = False
                    for i, (dest, _) in enumerate(self.edges[from_node]):
                        if dest == to_node:
                            self.edges[from_node][i] = (dest, weight)
                            edge_exists = True
                            break
                    
                    if not edge_exists:
                        self.edges[from_node].append((to_node, weight))
        
        self.last_update_time = time.time()
        logger.info(f"Built fully connected graph with {len(self.nodes)} nodes and {self._count_edges()} edges")
    
    def get_connected_components(self) -> List[Set[NodeID]]:
        """
        Find all connected components in the graph.
        
        Returns:
            List of sets, where each set contains the node IDs in a connected component
        """
        # Initialize visited set and components list
        visited = set()
        components = []
        
        # Helper function for DFS
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            
            for neighbor, _ in self.get_neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Find all connected components
        for node in self.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components
    
    def is_connected(self) -> bool:
        """
        Check if the graph is connected (all nodes can reach all other nodes).
        
        Returns:
            True if the graph is connected, False otherwise
        """
        components = self.get_connected_components()
        return len(components) == 1
    
    def _validate_graph(self) -> bool:
        """
        Validate the graph structure.
        
        Returns:
            True if the graph is valid, False otherwise
        """
        # Check that all edge endpoints exist in the nodes dictionary
        for from_node, edges in self.edges.items():
            if from_node not in self.nodes:
                logger.warning(f"Invalid graph: source node {from_node} not in nodes dictionary")
                return False
            
            for to_node, _ in edges:
                if to_node not in self.nodes:
                    logger.warning(f"Invalid graph: destination node {to_node} not in nodes dictionary")
                    return False
        
        # Check for duplicate edges
        for from_node, edges in self.edges.items():
            destinations = [dest for dest, _ in edges]
            if len(destinations) != len(set(destinations)):
                logger.warning(f"Invalid graph: duplicate edges from node {from_node}")
                return False
        
        return True
    
    def _count_edges(self) -> int:
        """
        Count the total number of edges in the graph.
        
        Returns:
            Total number of edges
        """
        return sum(len(edges) for edges in self.edges.values())
    
    def _clear_distance_cache(self, node_id: Optional[NodeID] = None) -> None:
        """
        Clear the distance cache.
        
        Args:
            node_id: Optional node ID to clear cache only for this node
                    If None, the entire cache will be cleared
        """
        if node_id is None:
            self._distance_cache.clear()
        else:
            # Clear cache entries involving this node
            for key in list(self._distance_cache.keys()):
                if key[0] == node_id or key[1] == node_id:
                    del self._distance_cache[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'edge_data': self.edge_data,
            'node_metadata': self.node_metadata,
            'last_update_time': self.last_update_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedGraph':
        """
        Create a graph from a dictionary.
        
        Args:
            data: Dictionary representation of the graph
            
        Returns:
            EnhancedGraph instance
        """
        graph = cls(data['nodes'], data['edges'])
        graph.edge_data = data.get('edge_data', {})
        graph.node_metadata = data.get('node_metadata', {})
        graph.last_update_time = data.get('last_update_time', time.time())
        return graph
    
    @classmethod
    def from_scats_data(cls, scats_data: Dict[str, Dict[str, Any]]) -> 'EnhancedGraph':
        """
        Create a graph from SCATS site data.
        
        Args:
            scats_data: Dictionary mapping SCATS IDs to site data
                        Each site data should have 'latitude' and 'longitude' keys
            
        Returns:
            EnhancedGraph instance
        """
        # Extract nodes from SCATS data
        nodes = {}
        node_metadata = {}
        
        for scats_id, site_data in scats_data.items():
            if 'latitude' in site_data and 'longitude' in site_data:
                # Create coordinate tuple in the correct format
                lat = float(site_data['latitude'])
                lon = float(site_data['longitude'])
                nodes[scats_id] = (lat, lon)
                
                # Store the full site data as metadata
                node_metadata[scats_id] = site_data
        
        # Create graph
        graph = cls(nodes)
        graph.node_metadata = node_metadata
        
        return graph


# Example usage
if __name__ == "__main__":
    # Sample SCATS site data
    sample_scats_data = {
        '2000': {
            'name': 'WARRIGAL_RD/TOORAK_RD',
            'latitude': -37.85192,
            'longitude': 145.09432
        },
        '3002': {
            'name': 'DENMARK_ST/BARKERS_RD',
            'latitude': -37.81514,
            'longitude': 145.02655
        },
        '970': {
            'name': 'WARRIGAL_RD/HIGH STREET_RD',
            'latitude': -37.86730,
            'longitude': 145.09151
        }
    }
    
    # Create graph from SCATS data
    graph = EnhancedGraph.from_scats_data(sample_scats_data)
    
    # Build fully connected graph
    graph.build_fully_connected_graph()
    
    # Print graph information
    print(f"Graph has {len(graph.nodes)} nodes and {graph._count_edges()} edges")
    
    # Print distances between nodes
    for from_node in graph.nodes:
        for to_node in graph.nodes:
            if from_node != to_node:
                distance = graph.get_node_distance(from_node, to_node)
                print(f"Distance from {from_node} to {to_node}: {distance:.2f} km")
    
    # Check if graph is connected
    print(f"Graph is connected: {graph.is_connected()}")
