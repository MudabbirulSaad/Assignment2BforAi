# utils/search/bfs.py
# Breadth-First Search implementation adapted from Assignment 2A

from collections import deque

def bfs(origin, destinations, edges, node_positions=None):
    """
    Breadth-First Search implementation.
    - Expands nodes in ascending order when equal
    - Maintains chronological order for equal priority nodes
    - Tracks number of nodes generated
    
    Args:
        origin: The starting node ID
        destinations: List of destination node IDs
        edges: Dictionary mapping source node IDs to lists of (destination, cost) tuples
        
    Returns:
        tuple: (goal_reached, nodes_generated, path)
            - goal_reached: The ID of the destination node that was reached
            - nodes_generated: Number of nodes generated during search
            - path: List of node IDs representing the path from origin to goal
    """
    queue = deque([(origin, [origin])])  # Store node and its path
    visited = set()
    nodes_generated = 0
    
    while queue:
        current, path = queue.popleft()
        nodes_generated += 1
        
        if current in destinations:
            return current, nodes_generated, path
            
        # Get all neighbors and sort in ascending order
        neighbors = sorted(edges.get(current, []), key=lambda x: x[0])
        
        for neighbor, _ in neighbors:
            if neighbor not in visited and neighbor not in [n for n, _ in queue]:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
                
    return None, nodes_generated, []
