#!/usr/bin/env python3
"""
TBRGS Graph Module

This module defines the Graph class, which stores the nodes and edges of the graph
and provides a method to retrieve neighbors of a given node.
"""

class Graph:
    def __init__(self, nodes, edges):
        """
        Initializes the graph with the provided nodes and edges.
        
        :param nodes: A dictionary mapping node IDs to coordinate tuples (x, y).
        :param edges: A dictionary mapping source node IDs to a list of tuples (destination, cost).
        """
        self.nodes = nodes  # Store node information.
        self.edges = edges  # Store edge information (connections between nodes).

    def get_neighbors(self, node):
        """
        Retrieves the neighbors of a given node.
        This function looks up the edges dictionary for the provided node.
        
        :param node: The node ID whose neighbors are required.
        :return: A list of tuples (neighbor, cost). Returns an empty list if the node has no outgoing edges.
        """
        return self.edges.get(node, [])


# The following integration example builds a Graph object using our input parser.
if __name__ == "__main__":
    # Example usage
    def build_graph_example():
        """
        Example of how to build a Graph object.
        """
        # Sample data
        nodes = {
            'A': (0, 0),
            'B': (1, 1),
            'C': (2, 0),
            'D': (0, 2)
        }
        
        edges = {
            'A': [('B', 1), ('C', 2)],
            'B': [('A', 1), ('C', 1), ('D', 2)],
            'C': [('A', 2), ('B', 1)],
            'D': [('B', 2)]
        }
        
        # Create graph
        graph_instance = Graph(nodes, edges)
        
        # Test
        print("Graph Nodes:", graph_instance.nodes)
        print("Graph Edges:", graph_instance.edges)
        print("Neighbors of A:", graph_instance.get_neighbors('A'))
        
    # Run example
    build_graph_example()
