import sys
import math
import heapq
from collections import defaultdict
from graph import Graph
from input_parser import build_data

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

class BidirectionalWeightedAStar:
    """
    Bidirectional Weighted A* implementation with dynamic weighting.
    
    This algorithm combines:
    1. Bidirectional search: simultaneous search from origin and destinations
    2. Dynamic weighting: adjusting epsilon based on search progress
    3. Advanced meeting criteria: sophisticated meeting point selection
    4. Consistent tie-breaking: for reproducible results
    """
    
    def __init__(self, origin, destinations, edges, node_positions):
        """
        Initialize the Bidirectional Weighted A* search.
        
        Args:
            origin: The starting node ID
            destinations: List of destination node IDs
            edges: Dictionary mapping source node IDs to lists of (destination, cost) tuples
            node_positions: Dictionary mapping node IDs to (x,y) coordinates
        """
        self.origin = origin
        self.destinations = destinations
        self.edges = edges
        self.node_positions = node_positions
        
        # Search parameters
        self.initial_weight = 2.0  # Initial epsilon weight for heuristic
        self.min_weight = 1.0      # Minimum epsilon weight (1.0 = regular A*)
        self.weight_decay = 0.95   # Decay factor for epsilon during search
        self.max_iterations = 1000  # Safety limit for search iterations
        
        # Statistics
        self.nodes_generated = 0
        
        # Forward search (from origin)
        self.forward_open = []     # Priority queue
        self.forward_closed = set()
        self.forward_g_scores = {}
        self.forward_parents = {}
        
        # Backward search (from destinations)
        self.backward_open = []    # Priority queue
        self.backward_closed = set()
        self.backward_g_scores = {}
        self.backward_parents = {}
        
        # Entry counter for tie-breaking
        self.entry_count = 0
        
        # Best meeting point so far
        self.best_meeting_node = None
        self.best_meeting_cost = float('inf')
    
    def search(self):
        """
        Execute the Bidirectional Weighted A* search algorithm.
        
        Returns:
            tuple: (goal_reached, nodes_generated, path)
                - goal_reached: The ID of the destination node that was reached
                - nodes_generated: Number of nodes generated during search
                - path: List of node IDs representing the path from origin to goal
        """
        # Initialize the forward search (from origin)
        self.forward_g_scores[self.origin] = 0
        self.forward_parents[self.origin] = None
        h_origin = calculate_heuristic(self.origin, self.destinations, self.node_positions)
        heapq.heappush(self.forward_open, (h_origin * self.initial_weight, 0, self.entry_count, self.origin))
        self.entry_count += 1
        self.nodes_generated += 1
        
        # Initialize the backward search (from each destination)
        for dest in self.destinations:
            self.backward_g_scores[dest] = 0
            self.backward_parents[dest] = None
            h_dest = calculate_heuristic(dest, [self.origin], self.node_positions)
            heapq.heappush(self.backward_open, (h_dest * self.initial_weight, 0, self.entry_count, dest))
            self.entry_count += 1
            self.nodes_generated += 1
        
        # Current weight for the heuristic (dynamically adjusted)
        current_weight = self.initial_weight
        
        # Main search loop with alternating forward and backward steps
        iterations = 0
        while self.forward_open and self.backward_open and iterations < self.max_iterations:
            iterations += 1
            
            # Alternating forward and backward steps
            if iterations % 2 == 1:
                # Forward step
                f_node = self._forward_search_step(current_weight)
                if f_node:
                    # Check if f_node is in backward closed set (meeting criteria)
                    self._check_meeting_criteria(f_node)
            else:
                # Backward step
                b_node = self._backward_search_step(current_weight)
                if b_node:
                    # Check if b_node is in forward closed set (meeting criteria)
                    self._check_meeting_criteria(b_node)
            
            # Dynamically adjust weight if needed (gradually reduce epsilon)
            if iterations % 10 == 0 and current_weight > self.min_weight:
                current_weight = max(current_weight * self.weight_decay, self.min_weight)
        
        # Check if we found a valid meeting point
        if self.best_meeting_node is not None:
            goal_reached = self._find_goal_reached()
            path = self._reconstruct_path(self.best_meeting_node)
            return goal_reached, self.nodes_generated, path
        
        # If we reach here, no path was found
        return None, self.nodes_generated, []
    
    def _forward_search_step(self, current_weight):
        """
        Perform one step of the forward search.
        
        Args:
            current_weight: The current epsilon weight for the heuristic
            
        Returns:
            The current node being processed or None if no more nodes to expand
        """
        if not self.forward_open:
            return None
        
        # Get node with lowest f-score from the forward open set
        _, g_score, _, current = heapq.heappop(self.forward_open)
        
        # Skip if already processed
        if current in self.forward_closed:
            return None
        
        # Add to closed set
        self.forward_closed.add(current)
        
        # Check if the current node is a destination
        if current in self.destinations:
            # Direct path found to a destination
            self._update_best_meeting(current, self.forward_g_scores[current])
            return current
        
        # Explore neighbors
        neighbors = sorted(self.edges.get(current, []), key=lambda x: x[0])  # Sort for consistent tie-breaking
        
        for neighbor, cost in neighbors:
            if neighbor in self.forward_closed:
                continue
                
            # Calculate tentative g-score
            tentative_g_score = g_score + cost
            
            # If we found a better path
            if neighbor not in self.forward_g_scores or tentative_g_score < self.forward_g_scores[neighbor]:
                # Update g-score and parent
                self.forward_g_scores[neighbor] = tentative_g_score
                self.forward_parents[neighbor] = current
                
                # Calculate h-score (weighted)
                h_score = calculate_heuristic(neighbor, self.destinations, self.node_positions)
                f_score = tentative_g_score + h_score * current_weight
                
                # Add to open set with entry count for tie-breaking
                heapq.heappush(self.forward_open, (f_score, tentative_g_score, self.entry_count, neighbor))
                self.entry_count += 1
                self.nodes_generated += 1
        
        return current
    
    def _backward_search_step(self, current_weight):
        """
        Perform one step of the backward search.
        
        Args:
            current_weight: The current epsilon weight for the heuristic
            
        Returns:
            The current node being processed or None if no more nodes to expand
        """
        if not self.backward_open:
            return None
        
        # Get node with lowest f-score from the backward open set
        _, g_score, _, current = heapq.heappop(self.backward_open)
        
        # Skip if already processed
        if current in self.backward_closed:
            return None
        
        # Add to closed set
        self.backward_closed.add(current)
        
        # Check if the current node is the origin
        if current == self.origin:
            # Direct path found from origin to this node
            self._update_best_meeting(current, self.backward_g_scores[current])
            return current
        
        # For backward search, we need to find nodes that can reach the current node
        # Since edges are stored as (source -> [(dest, cost), ...]), we need to find all sources
        # that have current as a destination
        incoming_edges = []
        for source, edges_list in self.edges.items():
            for dest, cost in edges_list:
                if dest == current:
                    incoming_edges.append((source, cost))
        
        # Sort for consistent tie-breaking
        incoming_edges.sort(key=lambda x: x[0])
        
        for neighbor, cost in incoming_edges:
            if neighbor in self.backward_closed:
                continue
                
            # Calculate tentative g-score
            tentative_g_score = g_score + cost
            
            # If we found a better path
            if neighbor not in self.backward_g_scores or tentative_g_score < self.backward_g_scores[neighbor]:
                # Update g-score and parent
                self.backward_g_scores[neighbor] = tentative_g_score
                self.backward_parents[neighbor] = current
                
                # Calculate h-score (weighted)
                h_score = calculate_heuristic(neighbor, [self.origin], self.node_positions)
                f_score = tentative_g_score + h_score * current_weight
                
                # Add to open set with entry count for tie-breaking
                heapq.heappush(self.backward_open, (f_score, tentative_g_score, self.entry_count, neighbor))
                self.entry_count += 1
                self.nodes_generated += 1
        
        return current
    
    def _check_meeting_criteria(self, node):
        """
        Check if the node represents a meeting point between forward and backward searches.
        If it's a better meeting point than what we've found so far, update our best meeting point.
        
        Args:
            node: The node to check
        """
        # Check if node is in both forward and backward closed sets
        if node in self.forward_closed and node in self.backward_closed:
            # Calculate the path cost through this meeting point
            forward_cost = self.forward_g_scores.get(node, float('inf'))
            backward_cost = self.backward_g_scores.get(node, float('inf'))
            total_cost = forward_cost + backward_cost
            
            # Update best meeting point if this is better
            if total_cost < self.best_meeting_cost:
                self.best_meeting_node = node
                self.best_meeting_cost = total_cost
    
    def _update_best_meeting(self, node, cost):
        """
        Update the best meeting point if the given node provides a better path.
        
        Args:
            node: The node to consider as a meeting point
            cost: The cost associated with this node
        """
        if cost < self.best_meeting_cost:
            self.best_meeting_node = node
            self.best_meeting_cost = cost
    
    def _find_goal_reached(self):
        """
        Determine which goal was reached in the final path.
        
        Returns:
            The goal node that was reached
        """
        # Reconstruct the path to identify which destination was reached
        if self.best_meeting_node in self.destinations:
            return self.best_meeting_node
        
        # Check backward path to find which destination led to the meeting point
        node = self.best_meeting_node
        while node in self.backward_parents and self.backward_parents[node] is not None:
            node = self.backward_parents[node]
            if node in self.destinations:
                return node
        
        # If no specific goal identified, return the first destination
        return self.destinations[0] if self.destinations else None
    
    def _reconstruct_path(self, meeting_node):
        """
        Reconstruct the path from origin to destination through the meeting node.
        
        Args:
            meeting_node: The node where forward and backward searches met
            
        Returns:
            list: The path from origin to destination
        """
        # Reconstruct forward path (origin to meeting node)
        forward_path = []
        current = meeting_node
        while current is not None:
            forward_path.append(current)
            current = self.forward_parents.get(current)
        forward_path.reverse()  # Reverse to get path from origin to meeting node
        
        # Reconstruct backward path (meeting node to destination)
        backward_path = []
        current = self.backward_parents.get(meeting_node)
        while current is not None:
            backward_path.append(current)
            current = self.backward_parents.get(current)
        
        # Combine paths (skip meeting node in backward path as it's already in forward path)
        complete_path = forward_path + backward_path
        
        return complete_path

def bdwa(origin, destinations, edges, node_positions):
    """
    Bidirectional Weighted A* with Dynamic Weighting implementation.
    - Uses simultaneous search from origin and destinations
    - Dynamically adjusts heuristic weight during search
    - Employs sophisticated meeting criteria
    - Includes consistent tie-breaking strategy
    
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
    # Create and run the bidirectional weighted A* search
    search = BidirectionalWeightedAStar(origin, destinations, edges, node_positions)
    return search.search()

def main():
    if len(sys.argv) != 2:
        print("Usage: python bdwa_search.py <filename>")
        return

    filename = sys.argv[1]
    
    # Use the common input parser
    node_pos, edges, origin, destinations = build_data(filename)

    # Create graph instance
    graph = Graph(node_pos, edges)
    
    # Run Bidirectional Weighted A* Search
    goal, count, path = bdwa(origin, destinations, edges, node_pos)

    # Output in required format
    print(f"{filename} CUS2")
    if path:
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found")

if __name__ == "__main__":
    main() 