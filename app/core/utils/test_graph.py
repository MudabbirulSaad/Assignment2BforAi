#!/usr/bin/env python3
"""
Test script for the enhanced graph structure.
"""

import os
import sys
import json
from app.core.utils.graph import EnhancedGraph
from app.core.integration.geo_calculator import haversine_distance

def test_enhanced_graph():
    """
    Test the enhanced graph structure with SCATS site coordinates.
    """
    print("=" * 80)
    print("Testing Enhanced Graph with SCATS Site Coordinates")
    print("=" * 80)
    
    # Create nodes with coordinates
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
    
    # Create graph
    graph = EnhancedGraph(nodes)
    graph.node_metadata = node_metadata
    
    print(f"Created graph with {len(graph.nodes)} nodes")
    
    # Test node coordinates
    print("\nNode Coordinates:")
    for node_id, coords in graph.nodes.items():
        print(f"  {node_id}: {coords}")
    
    # Test distance calculation
    print("\nDistance Calculations:")
    for from_node in list(graph.nodes.keys())[:3]:  # Test first 3 nodes
        for to_node in list(graph.nodes.keys())[:3]:
            if from_node != to_node:
                # Calculate distance directly using haversine_distance
                direct_distance = haversine_distance(graph.nodes[from_node], graph.nodes[to_node])
                
                # Calculate distance using graph method
                graph_distance = graph.get_node_distance(from_node, to_node)
                
                print(f"  {from_node} to {to_node}:")
                print(f"    Direct: {direct_distance:.2f} km")
                print(f"    Graph:  {graph_distance:.2f} km")
    
    # Build fully connected graph
    print("\nBuilding fully connected graph...")
    graph.build_fully_connected_graph()
    print(f"Graph now has {graph._count_edges()} edges")
    
    # Test edge weights
    print("\nEdge Weights:")
    for from_node in list(graph.nodes.keys())[:2]:  # Test first 2 nodes
        print(f"  Edges from {from_node}:")
        for to_node, weight in graph.get_neighbors(from_node):
            print(f"    To {to_node}: {weight:.2f} seconds")
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_enhanced_graph()
