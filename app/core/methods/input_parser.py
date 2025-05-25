#!/usr/bin/env python3
"""
TBRGS Input Parser Module

This module provides functions to parse input data for the routing algorithms.
"""

def build_data(filename=None, nodes=None, edges=None, origin=None, destinations=None):
    """
    Build data structures for the routing algorithms.
    
    This function can either read data from a file or use provided data structures.
    
    Args:
        filename (str, optional): Path to the input file
        nodes (dict, optional): Dictionary mapping node IDs to coordinates
        edges (dict, optional): Dictionary mapping source node IDs to lists of (destination, cost) tuples
        origin (str, optional): Origin node ID
        destinations (list, optional): List of destination node IDs
        
    Returns:
        tuple: (nodes, edges, origin, destinations)
    """
    # If data is provided directly, use it
    if nodes is not None and edges is not None and origin is not None and destinations is not None:
        return nodes, edges, origin, destinations
    
    # Otherwise, try to read from file
    if filename:
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Parse the input file
            # This is a simplified version - adjust based on your actual file format
            nodes = {}
            edges = {}
            origin = None
            destinations = []
            
            # Example parsing logic - adjust as needed
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if parts[0] == 'NODE':
                    node_id = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    nodes[node_id] = (x, y)
                elif parts[0] == 'EDGE':
                    source = parts[1]
                    dest = parts[2]
                    cost = float(parts[3])
                    if source not in edges:
                        edges[source] = []
                    edges[source].append((dest, cost))
                elif parts[0] == 'ORIGIN':
                    origin = parts[1]
                elif parts[0] == 'DESTINATION':
                    destinations.append(parts[1])
            
            return nodes, edges, origin, destinations
        except Exception as e:
            print(f"Error reading input file: {e}")
            # Return empty data structures
            return {}, {}, None, []
    
    # If no data provided and no file, return empty data structures
    return {}, {}, None, []


if __name__ == "__main__":
    # Example usage
    def test_build_data():
        """
        Test the build_data function with sample data.
        """
        # Sample data
        sample_nodes = {
            'A': (0, 0),
            'B': (1, 1),
            'C': (2, 0)
        }
        
        sample_edges = {
            'A': [('B', 1), ('C', 2)],
            'B': [('A', 1), ('C', 1)],
            'C': [('A', 2), ('B', 1)]
        }
        
        sample_origin = 'A'
        sample_destinations = ['C']
        
        # Test with direct data
        nodes, edges, origin, destinations = build_data(
            nodes=sample_nodes,
            edges=sample_edges,
            origin=sample_origin,
            destinations=sample_destinations
        )
        
        print("Nodes:", nodes)
        print("Edges:", edges)
        print("Origin:", origin)
        print("Destinations:", destinations)
    
    # Run test
    test_build_data()
