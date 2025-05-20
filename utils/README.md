# Utility Functions

This directory contains utility functions for the Traffic-based Route Guidance System (TBRGS).

## Files

- `graph_utils.py`: Utilities for working with the road network graph, including the `TrafficGraph` class
- `data_utils.py`: Helper functions for data manipulation and SCATS data processing
- `route_finder.py`: Interface for finding optimal routes using search algorithms
- `search/`: Directory containing search algorithm implementations

## Integration Plan

This module integrates key components from Assignment 2A into our Assignment 2B implementation while maintaining a professional and organized codebase.

### Components Integrated

#### 1. Graph Representation
**Source:** `2A/graph.py`  
**Target:** `utils/graph_utils.py`

The Graph class has been adapted to:
- Use SCATS intersections as nodes
- Use travel time (based on ML predictions) as edge costs
- Include additional methods for working with the traffic network

#### 2. Search Algorithms
**Source:** `2A/methods/` directory  
**Target:** `utils/search/` directory

We've integrated all search algorithms:
- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- GBFS (Greedy Best-First Search)
- A* Search
- IDDFS (Iterative Deepening DFS)
- BDWA (Bidirectional Weighted A*)

Modifications made:
- Updated heuristic functions to use travel time
- Modified to return multiple paths (top-k)
- Optimized for the specific requirements of TBRGS

#### 3. Input Parsing
**Source:** `2A/input_parser.py`  
**Target:** `utils/data_utils.py`

The input parsing functions have been:
- Extended to handle SCATS data format
- Modified to create a graph of the Boroondara area
- Integrated with the ML prediction system

#### 4. Search Interface
**Source:** `2A/search.py`  
**Target:** `utils/route_finder.py`

The search interface has been:
- Converted from command-line to a programmatic API
- Enhanced to work with the GUI
- Modified to use ML-predicted travel times

### Directory Structure

```
utils/
├── search/
│   ├── __init__.py
│   ├── bfs.py
│   ├── dfs.py
│   ├── gbfs.py
│   ├── astar.py
│   ├── iddfs.py
│   └── bdwa.py
├── graph_utils.py
├── data_utils.py
└── route_finder.py
```

### Usage Example

```python
# Import necessary components
from utils.data_utils import build_traffic_network, load_traffic_flow_data
from utils.route_finder import RouteFinder

# Build the traffic network
traffic_graph = build_traffic_network('data/scats_sites.csv')

# Load traffic flow predictions from ML model
traffic_flows = load_traffic_flow_data('data/predicted_flows.csv')

# Update the graph with predicted traffic flows
for source, dest, flow in traffic_flows:
    traffic_graph.set_traffic_flow(source, dest, flow)

# Create route finder
route_finder = RouteFinder(traffic_graph, max_routes=5)

# Find routes from origin to destination
routes = route_finder.find_multiple_routes(origin=2000, destination=3002)

# Display routes
for i, (path, time) in enumerate(routes, 1):
    print(f"Route {i}: {' -> '.join(map(str, path))}")
    print(f"Estimated travel time: {time:.2f} seconds")
```
