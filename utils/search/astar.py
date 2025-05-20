# utils/search/astar.py
# A* Search implementation adapted from Assignment 2A

import math
from queue import PriorityQueue

def calculate_heuristic(current_node, goal_nodes, node_positions):
    """
    Calculate the heuristic value (Euclidean distance) to the closest goal.
    
    Args:
        current_node: The node for which to calculate the heuristic
        goal_nodes: List of destination node IDs
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

def get_path_cost(path, edges):
    """
    Calculate the total cost of a path based on edge costs.
    
    Args:
        path: List of node IDs representing a path
        edges: Dictionary mapping source node IDs to lists of (destination, cost) tuples
        
    Returns:
        float: The total cost of the path
    """
    total_cost = 0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        
        # Find the edge cost
        for neighbor, cost in edges.get(from_node, []):
            if neighbor == to_node:
                total_cost += cost
                break
                
    return total_cost

def astar(origin, destinations, edges, node_positions):
    """
    A* Search implementation.
    - Uses both the cost to reach the goal from the current node (heuristic)
      and the cost to reach the current node from the origin (path cost)
    - Follows assignment requirements for node expansion order
    
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
    open_set = PriorityQueue()
    open_set.put((0, 0, origin, [origin]))  # (f_score, g_score, node, path)
    closed_set = set()
    nodes_generated = 0
    
    # Keep track of the g_score (cost from start to node)
    g_scores = {origin: 0}
    
    # For breaking ties (chronological order)
    entry_count = 0
    
    while not open_set.empty():
        # Get node with lowest f_score
        f_score, g_score, current, path = open_set.get()
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        nodes_generated += 1
        
        # Check if we've reached a destination
        if current in destinations:
            return current, nodes_generated, path
        
        # Explore neighbors
        for neighbor, edge_cost in edges.get(current, []):
            if neighbor in closed_set:
                continue
                
            # Calculate the g_score for this neighbor
            tentative_g_score = g_score + edge_cost
            
            # If we've found a better path to this neighbor
            if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                # Update g_score
                g_scores[neighbor] = tentative_g_score
                
                # Calculate heuristic
                h_score = calculate_heuristic(neighbor, destinations, node_positions)
                
                # Calculate f_score = g_score + h_score
                f_score = tentative_g_score + h_score
                
                # Create new path
                new_path = path + [neighbor]
                
                # Add neighbor to open set with adjusted priority for tie-breaking
                entry_count += 1
                # When all else is equal, nodes should be expanded in ascending order
                # Add a small fraction based on node ID and entry count for tie-breaking
                adjusted_f_score = f_score + (neighbor * 1e-9) + (entry_count * 1e-10)
                
                open_set.put((adjusted_f_score, tentative_g_score, neighbor, new_path))
    
    # No path found
    return None, nodes_generated, []
