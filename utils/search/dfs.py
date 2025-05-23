# utils/search/dfs.py
# Depth-First Search implementation adapted from Assignment 2A

def dfs(origin, destinations, edges, node_positions=None):
    """
    Depth-First Search implementation.
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
    stack = [(origin, [origin])]  # Store node and its path
    visited = set()
    nodes_generated = 0

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
            
        visited.add(current)
        nodes_generated += 1

        if current in destinations:
            return current, nodes_generated, path

        # Get neighbors and sort in descending order (since we're using a stack)
        # This ensures we process smaller numbers first when expanding
        neighbors = sorted(edges.get(current, []), key=lambda x: x[0], reverse=True)
        
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))

    return None, nodes_generated, []
