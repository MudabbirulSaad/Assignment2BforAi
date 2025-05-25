#!/usr/bin/env python3
"""
TBRGS SCATS Router

This script implements the Traffic-Based Route Guidance System using real SCATS data.
It loads SCATS site data and traffic flow data from the processed dataset,
builds a graph, and calculates optimal routes between SCATS sites.

This module integrates all three ML models (LSTM, GRU, CNN-RNN) for traffic prediction,
allowing for model selection and ensemble prediction capabilities.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Use relative imports
import sys
import os

# Add the parent directory to the path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.core.logging import TBRGSLogger
from app.core.utils.graph import EnhancedGraph
from app.core.integration.route_predictor import RoutePredictor, get_routes
from app.core.integration.geo_calculator import haversine_distance
from app.core.integration.flow_speed_converter import flow_to_speed
from app.core.integration.site_mapper import mapper as site_mapper, get_node_for_site, get_site_for_node
from app.core.ml.model_integration import model_integration, predict_traffic_flows
from app.core.ml.traffic_predictor import TrafficPredictor

# Initialize logger
logger = TBRGSLogger.get_logger("integration.scats_router")

class SCATSRouter:
    """
    SCATS Router for the Traffic-Based Route Guidance System.
    
    This class loads SCATS data, builds a graph, and calculates optimal routes
    between SCATS sites based on real traffic data. It integrates multiple ML models
    (LSTM, GRU, CNN-RNN) for traffic prediction and provides model selection and
    ensemble prediction capabilities.
    """
    
    def __init__(self, model_type: str = "GRU", use_ensemble: bool = False):
        """
        Initialize the SCATS Router.
        
        Args:
            model_type: Type of ML model to use (LSTM, GRU, CNN-RNN)
            use_ensemble: Whether to use ensemble prediction with all models
        """
        # Define paths to data files - using the actual dataset files
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset', 'processed')
        self.site_reference_path = os.path.join(self.data_dir, 'scats_site_reference.csv')
        self.traffic_data_path = os.path.join(self.data_dir, 'scats_traffic.csv')
        
        # Initialize data containers
        self.site_data = {}
        self.traffic_data = None
        self.graph = None
        self.route_predictor = None
        
        # ML model configuration
        self.model_type = model_type.upper()
        self.use_ensemble = use_ensemble
        
        # Configure model integration
        self._configure_model_integration()
        
        # Load data and initialize components
        self._load_site_data()
        self._load_traffic_data()
        self._build_graph()
        self._initialize_site_mapper()
        self._initialize_route_predictor()
        
        logger.info(f"SCATS Router initialized successfully with {self.model_type} model")
        if self.use_ensemble:
            logger.info("Ensemble prediction enabled")
    
    def _load_site_data(self):
        """
        Load SCATS site data from the processed dataset.
        """
        try:
            # Load site reference data
            df = pd.read_csv(self.site_reference_path)
            
            # Convert to dictionary
            for _, row in df.iterrows():
                scats_id = str(row['SCATS_ID'])
                
                # Extract site data
                site_data = {
                    'latitude': float(row['Latitude']),
                    'longitude': float(row['Longitude']),
                    'name': row['Location'],
                    'melway': row['CD_MELWAY'] if 'CD_MELWAY' in row else 'Unknown'
                }
                
                # Add any additional columns as metadata
                for col in df.columns:
                    if col not in ['SCATS_ID', 'Latitude', 'Longitude', 'Location', 'CD_MELWAY']:
                        site_data[col.lower()] = row[col]
                
                self.site_data[scats_id] = site_data
            
            logger.info(f"Loaded {len(self.site_data)} SCATS sites from {self.site_reference_path}")
        
        except Exception as e:
            logger.error(f"Error loading SCATS site data: {e}")
            raise
    
    def _load_traffic_data(self):
        """
        Load SCATS traffic data from the processed dataset.
        """
        try:
            # Load traffic data (sample to save memory)
            self.traffic_data = pd.read_csv(self.traffic_data_path, nrows=10000)
            
            # Convert SCATS_ID to string for consistency
            if 'SCATS_ID' in self.traffic_data.columns:
                self.traffic_data['SCATS_ID'] = self.traffic_data['SCATS_ID'].astype(str)
            
            logger.info(f"Loaded {len(self.traffic_data)} traffic records from {self.traffic_data_path}")
        
        except Exception as e:
            logger.error(f"Error loading SCATS traffic data: {e}")
            raise
    
    def _build_graph(self):
        """
        Build a graph from SCATS site data.
        """
        try:
            # Create graph from SCATS data
            self.graph = EnhancedGraph.from_scats_data(self.site_data)
            
            # Build connections between sites
            self._build_connections()
            
            logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {self.graph._count_edges()} edges")
        
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def _build_connections(self, max_distance: float = 5.0):
        """
        Build connections between SCATS sites based on proximity.
        
        Args:
            max_distance: Maximum distance in kilometers for connecting sites
        """
        # Get list of site IDs
        site_ids = list(self.site_data.keys())
        
        connections_built = 0
        unique_sites_connected = set()
        
        # First, try to connect sites based on proximity
        for i, site1 in enumerate(site_ids):
            for site2 in site_ids[i+1:]:
                # Get coordinates
                if site1 not in self.graph.nodes or site2 not in self.graph.nodes:
                    continue
                    
                coord1 = self.graph.nodes[site1]
                coord2 = self.graph.nodes[site2]
                
                # Calculate distance
                distance = haversine_distance(coord1, coord2)
                
                # Connect if within max_distance
                if distance <= max_distance:
                    # Calculate travel time based on distance
                    # Assume 40 km/h average speed and add 30 seconds intersection delay
                    travel_time = (distance / 40.0) * 3600 + 30.0
                    
                    # Add bidirectional edges
                    self.graph.add_edge(site1, site2, travel_time)
                    self.graph.add_edge(site2, site1, travel_time)
                    
                    connections_built += 2
                    unique_sites_connected.add(site1)
                    unique_sites_connected.add(site2)
        
        # Ensure the graph is fully connected
        if len(unique_sites_connected) < len(site_ids):
            logger.warning("Graph is not fully connected. Building additional connections...")
            
            # Create a minimum spanning tree to ensure connectivity
            remaining_sites = set(site_ids) - unique_sites_connected
            
            if unique_sites_connected:
                # Connect remaining sites to the closest connected site
                for site1 in remaining_sites:
                    if site1 not in self.graph.nodes:
                        continue
                        
                    # Find closest connected site
                    closest_site = None
                    min_distance = float('inf')
                    
                    for site2 in unique_sites_connected:
                        if site2 not in self.graph.nodes:
                            continue
                            
                        coord1 = self.graph.nodes[site1]
                        coord2 = self.graph.nodes[site2]
                        
                        distance = haversine_distance(coord1, coord2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_site = site2
                    
                    if closest_site:
                        # Calculate travel time
                        travel_time = (min_distance / 40.0) * 3600 + 30.0
                        
                        # Add bidirectional edges
                        self.graph.add_edge(site1, closest_site, travel_time)
                        self.graph.add_edge(closest_site, site1, travel_time)
                        
                        connections_built += 2
                        unique_sites_connected.add(site1)
            else:
                # No sites connected yet, connect the first site to all others
                if remaining_sites:
                    first_site = next(iter(remaining_sites))
                    remaining_sites.remove(first_site)
                    unique_sites_connected.add(first_site)
                    
                    for site2 in remaining_sites:
                        if site2 not in self.graph.nodes or first_site not in self.graph.nodes:
                            continue
                            
                        coord1 = self.graph.nodes[first_site]
                        coord2 = self.graph.nodes[site2]
                        
                        distance = haversine_distance(coord1, coord2)
                        travel_time = (distance / 40.0) * 3600 + 30.0
                        
                        # Add bidirectional edges
                        self.graph.add_edge(first_site, site2, travel_time)
                        self.graph.add_edge(site2, first_site, travel_time)
                        
                        connections_built += 2
                        unique_sites_connected.add(site2)
        
        logger.info(f"Built {connections_built} connections between {len(unique_sites_connected)} unique sites")
    

    
    def _initialize_site_mapper(self):
        """
        Initialize the site mapper with SCATS site data.
        """
        try:
            # Clear existing mappings
            site_mapper.site_to_node_map.clear()
            site_mapper.node_to_site_map.clear()
            
            # Create mappings between SCATS IDs and node IDs
            for scats_id, site_data in self.site_data.items():
                # In our implementation, SCATS IDs are used directly as node IDs
                # So we create a direct mapping
                site_mapper.site_to_node_map[scats_id] = scats_id
                site_mapper.node_to_site_map[scats_id] = scats_id
                
                # Also store the site coordinates
                site_mapper.site_locations[scats_id] = (site_data['latitude'], site_data['longitude'])
            
            logger.info(f"Initialized site mapper with {len(site_mapper.site_to_node_map)} SCATS site mappings")
            
        except Exception as e:
            logger.error(f"Error initializing site mapper: {e}")
            raise
    
    def _initialize_route_predictor(self):
        """
        Initialize the Route Predictor with ML model integration.
        """
        try:
            # Create a route predictor
            self.route_predictor = RoutePredictor(self.graph)
            
            # Patch the route predictor to use ML predictions
            self._patch_route_predictor()
            
            logger.info("Route Predictor initialized with ML model integration")
        
        except Exception as e:
            logger.error(f"Error initializing Route Predictor: {e}")
            raise
    
    def _patch_route_predictor(self):
        """
        Patch the route predictor to use ML predictions.
        """
        # Store the original method for fallback
        self.route_predictor._original_get_traffic_predictions = self.route_predictor._get_traffic_predictions
        
        # Replace with ML prediction method
        self.route_predictor._get_traffic_predictions = self._ml_get_traffic_predictions
        
        logger.info("Route predictor patched to use ML predictions")
    
    def _configure_model_integration(self):
        """
        Configure the ML model integration.
        """
        try:
            # Configure model integration
            if self.use_ensemble:
                model_integration.enable_ensemble(True)
                logger.info("Ensemble prediction enabled with weights: " + 
                           str(model_integration.ensemble_weights))
            else:
                model_integration.enable_ensemble(False)
                model_integration.set_active_model(self.model_type)
                logger.info(f"Active model set to {self.model_type}")
            
            # Check available models
            available_models = model_integration.get_available_models()
            logger.info(f"Available models: {available_models}")
            
            if not available_models:
                logger.warning("No ML models available. Using time-based predictions.")
        
        except Exception as e:
            logger.error(f"Error configuring model integration: {e}")
            logger.warning("Falling back to time-based predictions")
    
    def _ml_get_traffic_predictions(self, prediction_time: datetime) -> Dict[str, float]:
        """
        Get traffic flow predictions using ML models.
        
        Args:
            prediction_time: Time for which to predict traffic
            
        Returns:
            Dict[str, float]: Dictionary mapping SCATS IDs to predicted flows
        """
        try:
            # Get predictions from ML models
            predictions = predict_traffic_flows(prediction_time, model_type=self.model_type)
            logger.info(f"Generated {len(predictions)} ML predictions for {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return predictions
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            logger.info("Falling back to original prediction method")
            
            # Fall back to original method
            return self.route_predictor._original_get_traffic_predictions(prediction_time)
    
    def get_time_based_flow(self, hour: int) -> tuple:
        """
        Get traffic flow based on time of day.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            tuple: (base_flow, time_period)
        """
        if 7 <= hour < 9:
            # Morning peak (higher flow)
            base_flow = 500  # Vehicles per hour
            time_period = "Morning Peak (7-9 AM)"
        elif 16 <= hour < 19:
            # Evening peak (highest flow)
            base_flow = 600  # Vehicles per hour
            time_period = "Evening Peak (4-7 PM)"
        elif 9 <= hour < 16:
            # Midday (moderate flow)
            base_flow = 300  # Vehicles per hour
            time_period = "Midday (9 AM - 4 PM)"
        else:
            # Off-peak (low flow)
            base_flow = 150  # Vehicles per hour
            time_period = "Night/Off-Peak"
        
        return base_flow, time_period
    
    def update_traffic_predictions(self, prediction_time=None):
        """
        Update edge weights in the graph based on traffic predictions.
        
        Args:
            prediction_time: Time for which to predict traffic
                If None, uses the current time
        """
        try:
            # Get traffic predictions
            predictions = self._get_traffic_predictions(prediction_time)
            
            # Log the time and traffic pattern being used
            time_str = "current time" if prediction_time is None else prediction_time.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Updating traffic predictions for {time_str}")
            
            # Track statistics for reporting
            total_edges_updated = 0
            min_speed = float('inf')
            max_speed = 0
            sum_speed = 0
            
            # Update edge weights for each site
            for site_id, flow in predictions.items():
                if site_id not in self.graph.nodes:
                    continue
                    
                # Get all neighbors of this site
                neighbors = self.graph.get_neighbors(site_id)
                
                # Update edge weights for each neighbor
                for neighbor in neighbors:
                    # Get edge data
                    edge_data = self.graph.get_edge_data(site_id, neighbor)
                    if not edge_data:
                        continue
                        
                    distance = edge_data.get('distance', 0)
                    
                    # Convert flow to speed using our simplified method
                    # This ensures different times of day produce different speeds
                    speed = flow_to_speed(flow)
                    
                    # Track statistics
                    min_speed = min(min_speed, speed)
                    max_speed = max(max_speed, speed)
                    sum_speed += speed
                    total_edges_updated += 1
                    
                    # Calculate new travel time (seconds) plus 30 seconds intersection delay
                    # Formula: time = distance / speed * 3600 (to convert to seconds) + intersection delay
                    travel_time = (distance / speed) * 3600 + 30
                    
                    # Log the travel time calculation for debugging
                    if total_edges_updated <= 3:  # Only log a few samples to avoid flooding
                        logger.info(f"Edge {site_id}->{neighbor}: Distance={distance:.2f}km, Speed={speed:.1f}km/h, Travel Time={travel_time:.1f}s")

                    
                    # Update edge weight
                    self.graph.update_edge(site_id, neighbor, travel_time)
            
            # Report summary statistics
            if total_edges_updated > 0:
                avg_speed = sum_speed / total_edges_updated
                logger.info(f"Updated {total_edges_updated} edges based on traffic predictions")
                logger.info(f"Speed statistics - Min: {min_speed:.1f} km/h, Max: {max_speed:.1f} km/h, Avg: {avg_speed:.1f} km/h")
                
            logger.info(f"Updated edge weights based on traffic predictions for {time_str}")
            
        except Exception as e:
            logger.error(f"Error updating traffic predictions: {str(e)}")
            # Log but don't raise to make the demo more robust
    
    def get_routes(self, origin_scats: str, destination_scats: str,
                  prediction_time: Optional[datetime] = None,
                  max_routes: int = 5,
                  confidence_level: float = 0.95,
                  algorithms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get optimal routes between SCATS sites.
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time: Time for which to predict traffic
                If None, uses current time
            max_routes: Maximum number of routes to return
            confidence_level: Confidence level for prediction intervals (0.0-1.0)
            algorithms: List of routing algorithms to use
                If None, uses all available algorithms
                
        Returns:
            List[Dict[str, Any]]: List of route information dictionaries, sorted by travel time
        """
        if self.route_predictor is None:
            logger.error("Route Predictor not initialized")
            return []
        
        return self.route_predictor.get_routes(
            origin_scats=origin_scats,
            destination_scats=destination_scats,
            prediction_time=prediction_time,
            max_routes=max_routes,
            confidence_level=confidence_level,
            algorithms=algorithms
        )
        
    def set_model_type(self, model_type: str) -> bool:
        """
        Set the active ML model type.
        
        Args:
            model_type (str): Model type to set as active (LSTM, GRU, CNN-RNN)
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_type = model_type.upper()
        if model_type in model_integration.get_available_models():
            self.model_type = model_type
            model_integration.set_active_model(self.model_type)
            logger.info(f"Active model set to {self.model_type}")
            return True
        else:
            logger.error(f"Model type {model_type} not available")
            return False
    
    def enable_ensemble(self, enable: bool = True) -> None:
        """
        Enable or disable ensemble prediction.
        
        Args:
            enable: Whether to enable ensemble prediction
        """
        self.use_ensemble = enable
        model_integration.enable_ensemble(enable)
        logger.info(f"Ensemble prediction {'enabled' if enable else 'disabled'}")
    
    def set_ensemble_weights(self, weights: Dict[str, float]) -> bool:
        """
        Set weights for ensemble prediction.
        
        Args:
            weights: Dictionary mapping model types to weights
            
        Returns:
            bool: True if successful, False otherwise
        """
        return model_integration.set_ensemble_weights(weights)

def main():
    """
    Main function to demonstrate the SCATS Router using real SCATS data.
    """
    try:
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="TBRGS SCATS Router")
        parser.add_argument("--model", type=str, choices=["lstm", "gru", "cnnrnn", "ensemble"], default="gru",
                            help="Model type to use (lstm, gru, cnnrnn, ensemble)")
        parser.add_argument("--origin", type=str, default="970",
                            help="SCATS ID of the origin site")
        parser.add_argument("--destination", type=str, default="1066",
                            help="SCATS ID of the destination site")
        parser.add_argument("--time", type=str, default=None,
                            help="Prediction time (YYYY-MM-DD HH:MM:SS)")
        parser.add_argument("--max-routes", type=int, default=5,
                            help="Maximum number of routes to return")
        parser.add_argument("--algorithms", type=str, default=None,
                            help="Comma-separated list of routing algorithms to use")
        
        args = parser.parse_args()
        
        # Parse prediction time if provided
        prediction_time = None
        if args.time:
            try:
                prediction_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Invalid time format: {args.time}. Using current time.")
        
        # Parse algorithms if provided
        algorithms = args.algorithms.split(",") if args.algorithms else None
        
        # Create a SCATS Router with the specified model
        router = SCATSRouter(
            model_type=args.model,
            use_ensemble=args.model == "ensemble"
        )
        
        # Get routes
        routes = router.get_routes(
            origin_scats=args.origin,
            destination_scats=args.destination,
            prediction_time=prediction_time,
            max_routes=args.max_routes,
            algorithms=algorithms
        )
        
        # Print routes
        print(f"\nRoutes from {args.origin} to {args.destination}:")
        print(f"Time: {args.time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {args.model.upper()}")
        print(f"Number of routes: {len(routes)}")
        
        for i, route in enumerate(routes):
            print(f"\nRoute {i+1}:")
            print(f"  Algorithm: {route['algorithm']}")
            print(f"  Travel time: {route['travel_time']:.1f} seconds")
            print(f"  Distance: {route['distance']:.2f} km")
            print(f"  Average speed: {route['average_speed']:.1f} km/h")
            print(f"  Path: {' -> '.join(route['scats_path'])}")
            # Evening peak (5 PM)
            evening_peak = datetime.now().replace(hour=17, minute=0, second=0)
            print(f"\nEVENING PEAK TRAFFIC ({evening_peak.strftime('%H:%M')})")
            print("-" * 30)
            
            # Get routes for evening peak
            print(f"Finding optimal route during evening peak traffic...")
            peak_routes = router.get_routes(origin_scats, destination_scats, evening_peak.strftime('%Y-%m-%d %H:%M:%S'), max_routes=1)
            
            if peak_routes:
                route = peak_routes[0]
                print(f"  Travel Time: {route.get('travel_time', 0):.1f} seconds ({route.get('travel_time', 0)/60:.1f} minutes)")
                print(f"  Distance: {route.get('distance', 0):.2f} km")
                print(f"  Average Speed: {route.get('average_speed', 0):.1f} km/h")
                
                # Print the path
                path = route.get('path', [])
                if path:
                    path_str = " → ".join(path)
                    print(f"  Path: {path_str}")
            else:
                print("  No route found for evening peak")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TBRGS SCATS Router Demo Completed")
    print("=" * 80)

if __name__ == "__main__":
    main()
