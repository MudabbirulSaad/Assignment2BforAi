# utils/search/iddfs.py
# Iterative Deepening Depth-First Search implementation adapted from Assignment 2A

def reconstruct_path(came_from, goal):
    """
    Reconstructs the path from the origin to the goal using the came_from dictionary.
    
    Args:
        came_from: Dictionary mapping each node to its predecessor in the search.
        goal: The goal node where the search ended.
        
    Returns:
        List of node IDs representing the path from origin to goal.
    """
    path = [goal]
    while goal in came_from:
        goal = came_from[goal]
        path.append(goal)
    return path[::-1]  # Reverse the path to start at the origin

def depth_limited_search(current, destinations, edges, limit, visited, came_from, nodes_generated):
    """
    Performs a recursive depth-first search up to a fixed depth limit.
    
    Args:
        current: The current node being explored.
        destinations: List of destination node IDs.
        edges: Dictionary mapping nodes to lists of (neighbor, cost) tuples.
        limit: The remaining depth limit for the search.
        visited: Set of nodes already visited in this iteration.
        came_from: Dictionary storing the predecessor of each visited node.
        nodes_generated: Counter for the number of nodes generated.
        
    Returns:
        A tuple (goal_reached, nodes_generated, came_from)
            - goal_reached: The destination node if found; otherwise, None.
            - nodes_generated: Updated count of generated nodes.
            - came_from: The updated predecessor dictionary.
    """
    # Check if current node is a destination
    if current in destinations:
        return current, nodes_generated, came_from
    
    # If depth limit is reached, stop recursion
    if limit == 0:
        return None, nodes_generated, came_from

    # Sort neighbors in ascending order for consistent node expansion
    neighbors = sorted(edges.get(current, []), key=lambda t: t[0])
    
    for neighbor, _ in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            came_from[neighbor] = current
            nodes_generated += 1
            
            # Recurse with reduced depth limit
            result, nodes_generated, came_from = depth_limited_search(
                neighbor, destinations, edges, limit - 1, visited, came_from, nodes_generated)
                
            # If a destination was found, propagate it back up
            if result is not None:
                return result, nodes_generated, came_from
                
    return None, nodes_generated, came_from

def iddfs(origin, destinations, edges, node_positions=None, max_depth=50):
    """
    Iterative Deepening Depth-First Search (IDDFS) implementation.
    - Uses depth-limited search with increasing depth limits
    - Guarantees the shortest path in unweighted graphs
    - Memory-efficient compared to BFS
    
    Args:
        origin: The starting node ID
        destinations: List of destination node IDs
        edges: Dictionary mapping source node IDs to lists of (destination, cost) tuples
        max_depth: Maximum depth to search before giving up (default: 50)
        
    Returns:
        tuple: (goal_reached, nodes_generated, path)
            - goal_reached: The ID of the destination node that was reached
            - nodes_generated: Number of nodes generated during search
            - path: List of node IDs representing the path from origin to goal
    """
    total_nodes_generated = 1  # Count the origin node
    
    # Try increasing depth limits from 0 to max_depth
    for depth_limit in range(max_depth + 1):
        visited = set([origin])
        came_from = {}
        
        result, nodes_generated, came_from = depth_limited_search(
            origin, destinations, edges, depth_limit, visited, came_from, 0)
            
        total_nodes_generated += nodes_generated
        
        if result is not None:
            return result, total_nodes_generated, reconstruct_path(came_from, result)
    
    # No path found within max_depth
    return None, total_nodes_generated, []
