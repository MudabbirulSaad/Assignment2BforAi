#!/usr/bin/env python3
"""
TBRGS Edge Weight Updater Module

This module implements the replacement of static edge costs with predicted travel times
for the Traffic-Based Route Guidance System. It integrates traffic flow predictions
with the routing graph from Part A.

It provides functionality to:
1. Replace static costs with predicted travel times
2. Implement real-time weight updates based on traffic predictions
3. Add fallback to default weights when predictions are unavailable
4. Create weight validation and bounds checking
5. Implement caching for performance optimization
"""

import os
import json
import time
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import lru_cache
from app.config.config import config
from app.core.logging import TBRGSLogger
from app.core.integration.site_mapper import get_node_for_site, get_site_for_node, get_site_coordinate
from app.core.integration.geo_calculator import haversine_distance
from app.core.integration.flow_speed_converter import flow_to_speed
from app.core.integration.travel_time_calculator import TravelTimeCalculator

# Initialize logger
logger = TBRGSLogger.get_logger("integration.edge_weight_updater")

# Type aliases for clarity
NodeID = str  # Graph node ID
SCATSID = str  # SCATS site ID
EdgeWeight = float  # Edge weight in seconds
FlowValue = float  # Traffic flow in vehicles per 15 minutes
Coordinate = Tuple[float, float]  # (latitude, longitude)


class EdgeWeightUpdater:
    """
    Edge Weight Updater class for replacing static costs with predicted travel times.
    
    This class handles the integration of traffic flow predictions with the routing graph,
    replacing static edge costs with dynamic travel times based on current or predicted
    traffic conditions.
    
    Attributes:
        graph: The routing graph from Part A
        travel_time_calculator: Calculator for travel times between SCATS sites
        default_edge_cost: Default cost (seconds) for edges without predictions
        max_edge_cost: Maximum allowed edge cost (seconds)
        min_edge_cost: Minimum allowed edge cost (seconds)
        weight_cache: Cache for computed edge weights
        cache_timeout: Time (seconds) before cache entries expire
        last_update_time: Timestamp of the last weight update
    """
    
    def __init__(self, graph, travel_time_calculator=None, 
                 default_edge_cost: float = 60.0,
                 max_edge_cost: float = 600.0,
                 min_edge_cost: float = 5.0,
                 cache_size: int = 1024,
                 cache_timeout: float = 300.0):
        """
        Initialize the Edge Weight Updater.
        
        Args:
            graph: The routing graph from Part A
            travel_time_calculator: Calculator for travel times between SCATS sites
            default_edge_cost: Default cost (seconds) for edges without predictions
            max_edge_cost: Maximum allowed edge cost (seconds)
            min_edge_cost: Minimum allowed edge cost (seconds)
            cache_size: Size of the LRU cache for edge weights
            cache_timeout: Time (seconds) before cache entries expire
        """
        self.graph = graph
        
        # Create travel time calculator if not provided
        if travel_time_calculator is None:
            self.travel_time_calculator = TravelTimeCalculator()
        else:
            self.travel_time_calculator = travel_time_calculator
        
        # Set parameters
        self.default_edge_cost = default_edge_cost
        self.max_edge_cost = max_edge_cost
        self.min_edge_cost = min_edge_cost
        self.cache_timeout = cache_timeout
        
        # Initialize cache
        self.weight_cache = {}
        self.last_update_time = time.time()
        
        # Create LRU cache for get_edge_weight
        self.get_edge_weight = lru_cache(maxsize=cache_size)(self._get_edge_weight_uncached)
        
        logger.info(f"Edge Weight Updater initialized with default cost {default_edge_cost}s, " +
                   f"bounds [{min_edge_cost}s, {max_edge_cost}s], cache size {cache_size}")
    
    def update_edge_weights(self, flow_predictions: Dict[SCATSID, FlowValue]) -> int:
        """
        Update edge weights based on flow predictions.
        
        Args:
            flow_predictions: Dictionary mapping SCATS IDs to flow predictions
            
        Returns:
            int: Number of edges updated
        """
        # Clear the cache when updating weights
        self.get_edge_weight.cache_clear()
        self.weight_cache = {}
        self.last_update_time = time.time()
        
        # Track the number of edges updated
        updated_edges = 0
        
        # Update weights for each edge in the graph
        for source_node, edges in self.graph.edges.items():
            source_scats = get_site_for_node(source_node)
            
            for i, (target_node, _) in enumerate(edges):
                target_scats = get_site_for_node(target_node)
                
                # Skip if either source or target doesn't have a SCATS mapping
                if not source_scats or not target_scats:
                    continue
                
                # Get flow prediction for the source SCATS site
                flow = flow_predictions.get(source_scats)
                
                if flow is not None:
                    # Calculate travel time based on flow
                    travel_time_info = self.travel_time_calculator.calculate_travel_time(
                        source_scats, target_scats, flow=flow)
                    
                    # Extract the travel time value from the dictionary
                    travel_time = travel_time_info['travel_time']
                    
                    # Validate and bound the travel time
                    travel_time = self._validate_edge_weight(travel_time)
                    
                    # Update the edge weight
                    self.graph.edges[source_node][i] = (target_node, travel_time)
                    updated_edges += 1
                    
                    # Cache the computed weight
                    cache_key = (source_node, target_node)
                    self.weight_cache[cache_key] = (travel_time, time.time())
                    
                    # Get the traffic regime for logging
                    regime = travel_time_info.get('regime', 'unknown')
                    
                    logger.debug(f"Updated edge {source_node}->{target_node} with weight {travel_time:.2f}s " +
                                f"based on flow {flow:.2f} ({regime} regime)")
        
        logger.info(f"Updated {updated_edges} edge weights based on flow predictions")
        return updated_edges
    
    def _get_edge_weight_uncached(self, source_node: NodeID, target_node: NodeID) -> EdgeWeight:
        """
        Get the edge weight between two nodes (uncached version).
        
        Args:
            source_node: Source node ID
            target_node: Target node ID
            
        Returns:
            EdgeWeight: Edge weight in seconds
        """
        # Check if the edge exists in the graph
        for dest, weight in self.graph.get_neighbors(source_node):
            if dest == target_node:
                return weight
        
        # If edge doesn't exist, calculate a default weight
        source_scats = get_site_for_node(source_node)
        target_scats = get_site_for_node(target_node)
        
        if source_scats and target_scats:
            # Use travel time calculator with default flow
            travel_time_info = self.travel_time_calculator.calculate_travel_time(
                source_scats, target_scats)
            # Extract the travel time value from the dictionary
            travel_time = travel_time_info['travel_time']
            return self._validate_edge_weight(travel_time)
        
        # Fallback to default weight based on geographic distance
        source_coord = self._get_node_coordinate(source_node)
        target_coord = self._get_node_coordinate(target_node)
        
        if source_coord and target_coord:
            # Calculate distance in kilometers
            distance = haversine_distance(source_coord, target_coord)
            
            # Assume default speed of 40 km/h (11.11 m/s)
            # Convert to seconds: distance (km) / speed (km/s)
            travel_time = (distance / 40.0) * 3600.0
            
            # Add intersection delay
            travel_time += self.travel_time_calculator.intersection_delay
            
            return self._validate_edge_weight(travel_time)
        
        # Ultimate fallback
        return self.default_edge_cost
    
    def _get_node_coordinate(self, node_id: NodeID) -> Optional[Coordinate]:
        """
        Get the coordinate for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Optional[Coordinate]: Coordinate tuple or None if not available
        """
        # First try to get from graph nodes
        if hasattr(self.graph, 'nodes') and node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        
        # Then try to get from SCATS site mapping
        scats_id = get_site_for_node(node_id)
        if scats_id:
            return get_site_coordinate(scats_id)
        
        return None
    
    def _validate_edge_weight(self, weight: EdgeWeight) -> EdgeWeight:
        """
        Validate and bound an edge weight.
        
        Args:
            weight: Edge weight in seconds
            
        Returns:
            EdgeWeight: Validated and bounded edge weight
        """
        # Ensure weight is positive
        if weight <= 0:
            weight = self.default_edge_cost
        
        # Apply bounds
        weight = max(self.min_edge_cost, min(self.max_edge_cost, weight))
        
        return weight
    
    def is_cache_valid(self) -> bool:
        """
        Check if the cache is still valid.
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        current_time = time.time()
        return (current_time - self.last_update_time) < self.cache_timeout
    
    def invalidate_cache(self) -> None:
        """
        Invalidate the edge weight cache.
        """
        self.get_edge_weight.cache_clear()
        self.weight_cache = {}
        self.last_update_time = time.time()
        logger.debug("Edge weight cache invalidated")
    
    def get_all_edge_weights(self) -> Dict[Tuple[NodeID, NodeID], EdgeWeight]:
        """
        Get all current edge weights.
        
        Returns:
            Dict[Tuple[NodeID, NodeID], EdgeWeight]: Dictionary mapping edge tuples to weights
        """
        all_weights = {}
        
        for source_node, edges in self.graph.edges.items():
            for target_node, weight in edges:
                all_weights[(source_node, target_node)] = weight
        
        return all_weights
    
    def save_weights(self, output_path: str) -> None:
        """
        Save current edge weights to a JSON file.
        
        Args:
            output_path: Path to save the edge weights
        """
        # Convert edge weights to a serializable format
        weights_data = {}
        
        for source_node, edges in self.graph.edges.items():
            weights_data[source_node] = {}
            for target_node, weight in edges:
                weights_data[source_node][target_node] = weight
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(weights_data, f, indent=2)
            logger.info(f"Saved edge weights to {output_path}")
        except Exception as e:
            logger.error(f"Error saving edge weights: {e}")
    
    def load_weights(self, input_path: str) -> bool:
        """
        Load edge weights from a JSON file.
        
        Args:
            input_path: Path to the edge weights JSON file
            
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            with open(input_path, 'r') as f:
                weights_data = json.load(f)
            
            # Update the graph edges with loaded weights
            for source_node, targets in weights_data.items():
                if source_node not in self.graph.edges:
                    self.graph.edges[source_node] = []
                
                # Create a new edges list for this source node
                new_edges = []
                
                # Add edges from the loaded data
                for target_node, weight in targets.items():
                    new_edges.append((target_node, float(weight)))
                
                # Replace the edges for this source node
                self.graph.edges[source_node] = new_edges
            
            # Invalidate cache
            self.invalidate_cache()
            
            logger.info(f"Loaded edge weights from {input_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading edge weights: {e}")
            return False


# Create a function to get a default graph for testing
def create_test_graph():
    """
    Create a test graph for demonstration and testing.
    
    Returns:
        Graph: A test graph with sample nodes and edges
    """
    from app.core.integration.site_mapper import mapper
    
    # Create a simple graph class if not available
    if 'Graph' not in globals():
        class Graph:
            def __init__(self, nodes=None, edges=None):
                self.nodes = nodes or {}
                self.edges = edges or {}
            
            def get_neighbors(self, node):
                return self.edges.get(node, [])
    
    # Get SCATS site coordinates from the mapper
    nodes = {}
    edges = {}
    
    # Create nodes from SCATS sites
    for scats_id, coord in mapper.site_locations.items():
        node_id = f"N{scats_id}"
        nodes[node_id] = coord
        
        # Set up the mapping in the site mapper
        mapper.site_to_node_map[scats_id] = node_id
        mapper.node_to_site_map[node_id] = scats_id
    
    # Create some sample edges
    for node_id in nodes:
        edges[node_id] = []
        
        # Connect to 3 random nodes
        import random
        random.seed(42)  # For reproducibility
        
        other_nodes = list(nodes.keys())
        other_nodes.remove(node_id)
        
        for _ in range(min(3, len(other_nodes))):
            target = random.choice(other_nodes)
            other_nodes.remove(target)
            
            # Default cost of 60 seconds
            edges[node_id].append((target, 60.0))
    
    return Graph(nodes, edges)


if __name__ == "__main__":
    # Suppress logging for cleaner output
    import logging
    logging.getLogger('tbrgs').setLevel(logging.WARNING)
    
    print("=" * 80)
    print("TBRGS Edge Weight Updater Test")
    print("=" * 80)
    
    # Create a test graph
    test_graph = create_test_graph()
    print(f"Created test graph with {len(test_graph.nodes)} nodes and {sum(len(edges) for edges in test_graph.edges.values())} edges")
    
    # Create travel time calculator
    from app.core.integration.travel_time_calculator import TravelTimeCalculator
    travel_time_calculator = TravelTimeCalculator()
    
    # Create edge weight updater
    updater = EdgeWeightUpdater(test_graph, travel_time_calculator)
    
    # Create some sample flow predictions
    sample_predictions = {}
    
    # Get some sample SCATS IDs
    from app.core.integration.site_mapper import mapper
    sample_scats_ids = list(mapper.site_locations.keys())[:10]
    
    import random
    random.seed(42)  # For reproducibility
    
    for scats_id in sample_scats_ids:
        # Random flow between 0 and 1500 vehicles/hour
        flow = random.uniform(0, 1500) / 4  # Convert to vehicles/15min
        sample_predictions[scats_id] = flow
    
    print("\nSample Flow Predictions:")
    print("-" * 70)
    print(f"{'SCATS ID':8} | {'Flow (veh/15min)':16} | {'Flow (veh/hour)':14}")
    print("-" * 70)
    
    for scats_id, flow in sample_predictions.items():
        print(f"{scats_id:8} | {flow:16.2f} | {flow*4:14.2f}")
    
    # Update edge weights
    updated = updater.update_edge_weights(sample_predictions)
    print(f"\nUpdated {updated} edge weights based on flow predictions")
    
    # Print some sample updated weights
    print("\nSample Updated Edge Weights:")
    print("-" * 70)
    print(f"{'Source':8} | {'Target':8} | {'Weight (s)':10} | {'Speed (km/h)':12}")
    print("-" * 70)
    
    all_weights = updater.get_all_edge_weights()
    sample_edges = list(all_weights.items())[:10]
    
    for (source, target), weight in sample_edges:
        # Get SCATS IDs for the nodes
        source_scats = get_site_for_node(source)
        target_scats = get_site_for_node(target)
        
        if source_scats and target_scats:
            # Calculate the implied speed
            source_coord = get_site_coordinate(source_scats)
            target_coord = get_site_coordinate(target_scats)
            
            if source_coord and target_coord:
                distance = haversine_distance(source_coord, target_coord)
                speed_kmh = (distance / (weight / 3600)) if weight > 0 else 0
                
                print(f"{source:8} | {target:8} | {weight:10.2f} | {speed_kmh:12.2f}")
    
    # Test caching
    print("\nTesting Edge Weight Caching:")
    
    import time
    start_time = time.time()
    
    # Get the same edge weight multiple times
    if sample_edges:
        source, target = sample_edges[0][0]
        
        for _ in range(1000):
            weight = updater.get_edge_weight(source, target)
        
        end_time = time.time()
        print(f"Retrieved edge weight 1000 times in {(end_time - start_time)*1000:.2f} ms")
    
    # Test saving and loading weights
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        export_path = tmp.name
    
    updater.save_weights(export_path)
    
    # Create a new updater and load the weights
    new_updater = EdgeWeightUpdater(create_test_graph())
    new_updater.load_weights(export_path)
    
    # Verify the loaded weights
    print("\nLoaded Weights Verification:")
    original_weights = updater.get_all_edge_weights()
    loaded_weights = new_updater.get_all_edge_weights()
    
    print(f"Original edges: {len(original_weights)}")
    print(f"Loaded edges: {len(loaded_weights)}")
    
    # Check if weights match
    matches = sum(1 for edge in original_weights if edge in loaded_weights and 
                  abs(original_weights[edge] - loaded_weights[edge]) < 0.01)
    print(f"Matching weights: {matches} out of {len(original_weights)}")
    
    # Clean up the temporary file
    try:
        os.remove(export_path)
    except:
        pass
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
