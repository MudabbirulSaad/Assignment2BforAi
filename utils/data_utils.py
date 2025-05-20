# utils/data_utils.py
# Utilities for handling data in the Traffic-based Route Guidance System (TBRGS)
# Adapted from Assignment 2A's input_parser.py

import pandas as pd
import numpy as np
from utils.graph_utils import TrafficGraph

def load_scats_data(scats_file_path):
    """
    Loads SCATS site data from a CSV file.
    
    Args:
        scats_file_path: Path to the CSV file containing SCATS site data.
        
    Returns:
        A dictionary mapping SCATS site IDs to (x, y) coordinates.
    """
    try:
        # Load SCATS data
        scats_data = pd.read_csv(scats_file_path)
        
        # Extract SCATS site IDs and coordinates
        nodes = {}
        for _, row in scats_data.iterrows():
            site_id = int(row['TFM_ID'])
            x = float(row['X'])
            y = float(row['Y'])
            nodes[site_id] = (x, y)
        
        return nodes
    except Exception as e:
        print(f"Error loading SCATS data: {e}")
        return {}

def build_road_network(nodes, distance_threshold=1.0):
    """
    Builds a road network graph based on SCATS site locations.
    
    Args:
        nodes: A dictionary mapping SCATS site IDs to (x, y) coordinates.
        distance_threshold: Maximum distance for connecting nodes (in coordinate units).
        
    Returns:
        A dictionary mapping source node IDs to lists of (destination, distance) tuples.
    """
    edges = {}
    
    # Create edges between nodes that are within the distance threshold
    for source_id, source_coords in nodes.items():
        edges[source_id] = []
        
        for dest_id, dest_coords in nodes.items():
            if source_id != dest_id:
                # Calculate Euclidean distance
                x1, y1 = source_coords
                x2, y2 = dest_coords
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Connect nodes if they are within the threshold
                if distance <= distance_threshold:
                    edges[source_id].append((dest_id, distance))
    
    return edges

def load_traffic_flow_data(flow_file_path):
    """
    Loads traffic flow data from a CSV file.
    
    Args:
        flow_file_path: Path to the CSV file containing traffic flow data.
        
    Returns:
        A pandas DataFrame with traffic flow data.
    """
    try:
        return pd.read_csv(flow_file_path)
    except Exception as e:
        print(f"Error loading traffic flow data: {e}")
        return pd.DataFrame()

def load_sequence_data(sequence_file_path):
    """
    Loads sequence data for ML models from an NPZ file.
    
    Args:
        sequence_file_path: Path to the NPZ file containing sequence data.
        
    Returns:
        A tuple (X_train, y_train, X_test, y_test).
    """
    try:
        data = np.load(sequence_file_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        return None, None, None, None

def build_traffic_network(scats_file_path, distance_threshold=1.0, speed_limit=60, intersection_delay=30):
    """
    Builds a complete traffic network from SCATS data.
    
    Args:
        scats_file_path: Path to the CSV file containing SCATS site data.
        distance_threshold: Maximum distance for connecting nodes (in coordinate units).
        speed_limit: The speed limit in km/h (default: 60).
        intersection_delay: The average delay at each controlled intersection in seconds (default: 30).
        
    Returns:
        A TrafficGraph instance.
    """
    # Load SCATS site data
    nodes = load_scats_data(scats_file_path)
    
    # Build road network
    edges = build_road_network(nodes, distance_threshold)
    
    # Create TrafficGraph
    return TrafficGraph(nodes, edges, speed_limit, intersection_delay)

def get_scats_site_names(scats_file_path):
    """
    Gets human-readable names for SCATS sites.
    
    Args:
        scats_file_path: Path to the CSV file containing SCATS site data.
        
    Returns:
        A dictionary mapping SCATS site IDs to human-readable names.
    """
    try:
        # Load SCATS data
        scats_data = pd.read_csv(scats_file_path)
        
        # Extract SCATS site IDs and names
        site_names = {}
        for _, row in scats_data.iterrows():
            site_id = int(row['TFM_ID'])
            site_desc = row['SITE_DESC'] if 'SITE_DESC' in row else f"Site {site_id}"
            site_names[site_id] = site_desc
        
        return site_names
    except Exception as e:
        print(f"Error loading SCATS site names: {e}")
        return {}
