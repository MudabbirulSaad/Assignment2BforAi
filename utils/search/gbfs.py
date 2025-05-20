# utils/search/gbfs.py
# Greedy Best-First Search implementation adapted from Assignment 2A

import math
from queue import PriorityQueue

def calculate_heuristic(current_node, goal_nodes, node_positions):
    """
    Calculate the heuristic value (Euclidean distance) to the closest goal.
    For GBFS, we only use the cost to reach the goal from the current node.
    
    Args:
        current_node: The node for which to calculate the heuristic
        goal_nodes: List of destination nodes
        node_positions: Dictionary mapping node IDs to their (x,y) coordinates
        
    Returns:
        float: The minimum Euclidean distance to any goal node
    """
    if not goal_nodes:
        return float('inf')
        
    # Find the minimum Euclidean distance to any goal
    min_distance = float('inf')
    for goal in goal_nodes:
        if goal in node_positions and current_node in node_positions:
            x1, y1 = node_positions[current_node]
            x2, y2 = node_positions[goal]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            min_distance = min(min_distance, distance)
    
    return min_distance

def gbfs(origin, destinations, edges, node_positions):
    """
    Greedy Best-First Search implementation.
    - Uses a priority queue to always expand the node with smallest heuristic
    - Heuristic is the Euclidean distance to the closest goal
    - Follows assignment requirements for node expansion order and tracking
    
    Args:
        origin: The starting node ID
        destinations: List of destination node IDs
        edges: Dictionary mapping source node IDs to lists of (destination, cost) tuples
        node_positions: Dictionary mapping node IDs to (x,y) coordinates
        
    Returns:
        tuple: (goal_reached, nodes_generated, path)
            - goal_reached: The ID of the destination node that was reached
            - nodes_generated: Number of nodes generated during search
            - path: List of node IDs representing the path from origin to goal
    """
    # Initialize data structures
    frontier = PriorityQueue()
    frontier.put((0, origin, [origin]))  # (priority, node, path)
    visited = set()
    nodes_generated = 0
    
    # For tracking equal-priority nodes to maintain chronological order
    entry_count = 0
    
    while not frontier.empty():
        # Get node with lowest heuristic value
        priority, current, path = frontier.get()
        
        if current in visited:
            continue
            
        visited.add(current)
        nodes_generated += 1
        
        # Check if we've reached a destination
        if current in destinations:
            return current, nodes_generated, path
        
        # Get all neighbors
        neighbors = []
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                # Calculate heuristic to goal
                h = calculate_heuristic(neighbor, destinations, node_positions)
                neighbors.append((neighbor, h))
        
        # Sort neighbors by heuristic value, then by node ID for tiebreaking
        neighbors.sort(key=lambda x: (x[1], x[0]))
        
        # Add neighbors to frontier
        for neighbor, h in neighbors:
            new_path = path + [neighbor]
            entry_count += 1
            # When all else is equal, nodes should be expanded in ascending order
            # Add a small fraction to the priority based on the node ID
            # Add another smaller fraction based on entry count for chronological order
            adjusted_priority = h + (neighbor * 1e-9) + (entry_count * 1e-10)
            frontier.put((adjusted_priority, neighbor, new_path))
    
    # No path found
    return None, nodes_generated, []
