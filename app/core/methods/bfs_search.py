import sys
from collections import deque
from graph import Graph
from input_parser import build_data

def bfs(origin, destinations, edges):
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python bfs_search.py <filename>")
        return

    filename = sys.argv[1]
    
    # Use the common input parser
    node_pos, edges, origin, destinations = build_data(filename)

    # Create graph instance
    graph = Graph(node_pos, edges)
    
    # Run BFS
    goal, count, path = bfs(origin, destinations, edges)

    # Output in required format
    print(f"{filename} BFS")
    if path:
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found")

if __name__ == "__main__":
    main()
