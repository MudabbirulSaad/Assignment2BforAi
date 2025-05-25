#!/usr/bin/env python3
"""
TBRGS SCATS Graph Builder

This script builds a graph from the processed SCATS data and saves it for use in the routing system.
It demonstrates how to use the EnhancedGraph class with real SCATS data.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from app.core.logging import TBRGSLogger
from app.core.utils.graph import EnhancedGraph
from app.core.utils.data_loader import load_scats_data
from app.config.config import config

# Initialize logger
logger = TBRGSLogger.get_logger("utils.build_scats_graph")

def load_scats_site_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Load SCATS site metadata from the processed data.
    
    Returns:
        Dictionary mapping SCATS IDs to site metadata
    """
    # Path to the processed SCATS site metadata
    metadata_path = os.path.join(config.DATA_DIR, "processed", "scats_site_metadata.json")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(metadata)} SCATS sites")
        return metadata
    except Exception as e:
        logger.error(f"Error loading SCATS site metadata: {e}")
        # Return empty dictionary if file doesn't exist or has errors
        return {}

def build_graph_from_scats_data() -> EnhancedGraph:
    """
    Build a graph from the processed SCATS data.
    
    Returns:
        EnhancedGraph instance
    """
    # Load SCATS site metadata
    site_metadata = load_scats_site_metadata()
    
    if not site_metadata:
        logger.error("No SCATS site metadata found. Cannot build graph.")
        return None
    
    # Create graph from SCATS site metadata
    graph = EnhancedGraph.from_scats_data(site_metadata)
    
    # Build fully connected graph with default weights
    graph.build_fully_connected_graph()
    
    logger.info(f"Built graph with {len(graph.nodes)} nodes and {graph._count_edges()} edges")
    
    return graph

def save_graph(graph: EnhancedGraph, output_path: str) -> bool:
    """
    Save the graph to a JSON file.
    
    Args:
        graph: EnhancedGraph instance
        output_path: Path to save the graph
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert graph to dictionary
        graph_dict = graph.to_dict()
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        logger.info(f"Saved graph to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving graph: {e}")
        return False

def load_graph(input_path: str) -> EnhancedGraph:
    """
    Load a graph from a JSON file.
    
    Args:
        input_path: Path to the graph JSON file
        
    Returns:
        EnhancedGraph instance or None if loading fails
    """
    try:
        # Load from JSON file
        with open(input_path, 'r') as f:
            graph_dict = json.load(f)
        
        # Create graph from dictionary
        graph = EnhancedGraph.from_dict(graph_dict)
        
        logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {graph._count_edges()} edges from {input_path}")
        return graph
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        return None

def analyze_graph(graph: EnhancedGraph) -> None:
    """
    Analyze the graph and print statistics.
    
    Args:
        graph: EnhancedGraph instance
    """
    if not graph:
        logger.error("No graph to analyze")
        return
    
    # Basic statistics
    num_nodes = len(graph.nodes)
    num_edges = graph._count_edges()
    avg_edges_per_node = num_edges / num_nodes if num_nodes > 0 else 0
    
    print(f"Graph Statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Avg. Edges per Node: {avg_edges_per_node:.2f}")
    
    # Connectivity
    connected = graph.is_connected()
    components = graph.get_connected_components()
    
    print(f"  Connected: {connected}")
    print(f"  Connected Components: {len(components)}")
    
    if not connected:
        for i, component in enumerate(components):
            print(f"  Component {i+1}: {len(component)} nodes")
    
    # Distance statistics
    distances = []
    for from_node in list(graph.nodes.keys())[:10]:  # Limit to first 10 nodes for brevity
        for to_node in list(graph.nodes.keys())[:10]:
            if from_node != to_node:
                distance = graph.get_node_distance(from_node, to_node)
                distances.append(distance)
    
    if distances:
        min_distance = min(distances)
        max_distance = max(distances)
        avg_distance = sum(distances) / len(distances)
        
        print(f"  Min Distance: {min_distance:.2f} km")
        print(f"  Max Distance: {max_distance:.2f} km")
        print(f"  Avg Distance: {avg_distance:.2f} km")

def main():
    """
    Main function to build and save the SCATS graph.
    """
    print("=" * 80)
    print("TBRGS SCATS Graph Builder")
    print("=" * 80)
    
    # Build graph from SCATS data
    graph = build_graph_from_scats_data()
    
    if not graph:
        print("Failed to build graph. Check logs for details.")
        return
    
    # Analyze graph
    analyze_graph(graph)
    
    # Save graph
    output_dir = os.path.join(config.DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scats_graph.json")
    
    if save_graph(graph, output_path):
        print(f"Graph saved to {output_path}")
    else:
        print("Failed to save graph. Check logs for details.")
    
    print("=" * 80)
    print("Done")
    print("=" * 80)

if __name__ == "__main__":
    main()
