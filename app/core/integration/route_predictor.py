#!/usr/bin/env python3
"""
TBRGS Route Predictor Module

This module implements the integration of ML traffic predictions with Part A routing algorithms
for the Traffic-Based Route Guidance System. It provides functionality to calculate
up to 5 optimal routes between SCATS sites based on predicted traffic conditions.

It provides functionality to:
1. Integrate ML predictions with Part A routing algorithms
2. Calculate routes between SCATS sites using site numbers
3. Generate top-5 routes as required by the assignment
4. Estimate travel times with ML predictions
5. Provide prediction confidence intervals
6. Implement fallback routing for missing predictions
7. Rank routes by total estimated travel time
"""

import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import lru_cache
from datetime import datetime, timedelta
import statistics
# Use relative imports
import sys
import os

# Add the parent directory to the path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Use relative imports that will work when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.config.config import config
from app.core.logging import TBRGSLogger
from app.core.integration.site_mapper import get_node_for_site, get_site_for_node, get_site_coordinate
from app.core.integration.geo_calculator import haversine_distance
from app.core.integration.flow_speed_converter import flow_to_speed
from app.core.integration.travel_time_calculator import TravelTimeCalculator
from app.core.integration.edge_weight_updater import EdgeWeightUpdater
from app.core.ml.traffic_predictor import TrafficPredictor
from app.core.ml.model_integration import model_integration, predict_traffic_flows

# Import the routing adapter for Part A algorithms
from app.core.integration.routing_adapter import find_route, get_available_algorithms
from app.core.methods.graph import Graph

# Import the routing algorithms directly for testing
from app.core.methods.astar_search import astar
from app.core.methods.bfs_search import bfs
from app.core.methods.dfs_search import dfs
from app.core.methods.gbfs_search import gbfs
from app.core.methods.iddfs_search import iddfs
from app.core.methods.bdwa_search import bdwa

# Initialize logger
logger = TBRGSLogger.get_logger("integration.route_predictor")

# Type aliases for clarity
NodeID = str  # Graph node ID
SCATSID = str  # SCATS site ID
EdgeWeight = float  # Edge weight in seconds
FlowValue = float  # Traffic flow in vehicles per 15 minutes
Coordinate = Tuple[float, float]  # (latitude, longitude)
Route = List[NodeID]  # List of node IDs representing a route
RouteInfo = Dict[str, Any]  # Dictionary with route information

class RoutePredictor:
    """
    Route Predictor for the Traffic-Based Route Guidance System.
    
    This class integrates ML traffic predictions with routing algorithms to calculate
    optimal routes between SCATS sites based on predicted traffic conditions.
    """
    
    def __init__(self, graph: Graph, traffic_predictor: Optional[TrafficPredictor] = None, 
                 model_type: str = "GRU", use_ensemble: bool = False,
                 travel_time_calculator=None, edge_weight_updater=None, cache_timeout=300):
        """
        Initialize the Route Predictor.
        
        Args:
            graph: Graph object with nodes and edges
            traffic_predictor: Optional TrafficPredictor for ML-based predictions
            model_type: Type of ML model to use (LSTM, GRU, CNN-RNN)
            use_ensemble: Whether to use ensemble prediction with all models
            travel_time_calculator: Optional calculator for travel times
            edge_weight_updater: Optional updater for edge weights
            cache_timeout: Cache timeout in seconds (default: 5 minutes)
        """
        self.graph = graph
        self.travel_time_calculator = travel_time_calculator or TravelTimeCalculator()
        
        # ML model configuration
        self.traffic_predictor = traffic_predictor
        self.model_type = model_type.upper()
        self.use_ensemble = use_ensemble
        self.prediction_metrics = {
            'rmse': [],
            'mae': [],
            'prediction_count': 0,
            'fallback_count': 0
        }
        
        # Create the edge weight updater if not provided
        if edge_weight_updater is None:
            # For testing purposes, we'll skip the EdgeWeightUpdater initialization
            # since it requires a graph parameter
            self.edge_weight_updater = None
        else:
            self.edge_weight_updater = edge_weight_updater
        
        # Cache for route predictions
        self.route_cache = {}
        self.last_update_time = time.time()
        self.cache_timeout = cache_timeout
        
        # Get available routing algorithms
        self.routing_algorithms = ['astar', 'bfs', 'dfs', 'gbfs', 'iddfs', 'bdwa']
        
        logger.info(f"Route Predictor initialized with {len(self.routing_algorithms)} algorithms: {self.routing_algorithms}")
        if self.traffic_predictor:
            logger.info(f"Using ML model type: {self.model_type}")
            if self.use_ensemble:
                logger.info("Ensemble prediction enabled")
    
    def get_routes(self, origin_scats: SCATSID, destination_scats: SCATSID,
                   prediction_time: Optional[Union[datetime, str]] = None,
                   max_routes: int = 5,
                   confidence_level: float = 0.95,
                   algorithms: Optional[List[str]] = None) -> List[RouteInfo]:
        """
        Get up to max_routes optimal routes from origin to destination using the specified algorithms.
        
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
            List[RouteInfo]: List of route information dictionaries, sorted by travel time
        """
        # Get traffic flow predictions for the specified time
        if prediction_time is None:
            prediction_time = datetime.now()
        elif isinstance(prediction_time, str):
            # Parse string to datetime
            try:
                prediction_time = datetime.strptime(prediction_time, '%Y-%m-%d %H:%M:%S')
                logger.info(f"Parsed prediction time: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except ValueError as e:
                logger.error(f"Error parsing prediction time: {e}")
                logger.warning(f"Using current time instead of: {prediction_time}")
                prediction_time = datetime.now()
        
        # Default to all available algorithms if not specified
        if algorithms is None:
            algorithms = self.routing_algorithms
        
        # Log the request
        logger.info(f"Getting routes from {origin_scats} to {destination_scats} at {prediction_time}")
        
        # Map SCATS IDs to node IDs
        origin_node = get_node_for_site(origin_scats)
        destination_node = get_node_for_site(destination_scats)
        
        # Check if the mapping exists
        if not origin_node:
            logger.warning(f"SCATS ID {origin_scats} not mapped to any node. Using SCATS ID as node ID.")
            origin_node = origin_scats
        
        if not destination_node:
            logger.warning(f"SCATS ID {destination_scats} not mapped to any node. Using SCATS ID as node ID.")
            destination_node = destination_scats
        
        # Verify the nodes exist in the graph
        if origin_node not in self.graph.nodes:
            logger.error(f"Origin node {origin_node} not found in graph.")
            return []
        
        if destination_node not in self.graph.nodes:
            logger.error(f"Destination node {destination_node} not found in graph.")
            return []
        
        # Get traffic flow predictions for all SCATS sites
        flow_predictions = self._get_traffic_predictions(prediction_time)
        
        # Update edge weights based on predicted traffic flows
        if self.edge_weight_updater and flow_predictions:
            self.edge_weight_updater.update_edge_weights(self.graph, flow_predictions)
        
        # Calculate routes using each algorithm
        all_routes = []
        
        for algorithm in algorithms:
            try:
                # Find a route using the specified algorithm
                logger.info(f"Finding route using {algorithm}")
                
                # Call the appropriate routing algorithm directly
                if algorithm == 'astar':
                    goal_reached, nodes_generated, path = astar(origin_node, [destination_node], self.graph.edges, self.graph.nodes)
                elif algorithm == 'bfs':
                    goal_reached, nodes_generated, path = bfs(origin_node, [destination_node], self.graph.edges)
                elif algorithm == 'dfs':
                    goal_reached, nodes_generated, path = dfs(origin_node, [destination_node], self.graph.edges)
                elif algorithm == 'gbfs':
                    goal_reached, nodes_generated, path = gbfs(origin_node, [destination_node], self.graph.edges, self.graph.nodes)
                elif algorithm == 'iddfs':
                    goal_reached, nodes_generated, path = iddfs(origin_node, [destination_node], self.graph.edges)
                elif algorithm == 'bdwa':
                    goal_reached, nodes_generated, path = bdwa(origin_node, [destination_node], self.graph.edges, self.graph.nodes)
                else:
                    logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                # Create a route result dictionary
                route_result = {
                    'path': path if goal_reached else [],
                    'nodes_generated': nodes_generated,
                    'goal_reached': goal_reached
                }
                
                if goal_reached and route_result['path']:
                    path = route_result['path']
                    nodes_generated = route_result.get('nodes_generated', 0)
                    
                    # Map node IDs back to SCATS IDs
                    scats_path = []
                    for node_id in path:
                        scats_id = get_site_for_node(node_id)
                        if not scats_id:
                            scats_id = node_id  # Use node ID if no mapping exists
                        scats_path.append(scats_id)
                    
                    # Calculate travel time and other metrics
                    travel_time_info = self._calculate_route_travel_time(
                        path, flow_predictions, confidence_level)
                    
                    # Create route information dictionary
                    route_info = {
                        'algorithm': algorithm,
                        'path': path,
                        'scats_path': scats_path,
                        'travel_time': travel_time_info['total_time'],
                        'distance': travel_time_info['total_distance'],
                        'average_speed': travel_time_info['average_speed'],
                        'nodes_generated': nodes_generated,
                        'segments': travel_time_info['segments']
                    }
                    
                    # Add confidence interval if available
                    if 'confidence_interval' in travel_time_info and travel_time_info['confidence_interval']:
                        route_info['confidence_interval'] = travel_time_info['confidence_interval']
                    
                    all_routes.append(route_info)
                    
                    logger.info(f"Found route using {algorithm}: {len(path)} nodes, "
                              f"{travel_time_info['total_time']:.1f} seconds")
                else:
                    logger.warning(f"No route found using {algorithm}")
            except Exception as e:
                logger.error(f"Error finding route using {algorithm}: {e}")
        
        # Sort routes by travel time
        all_routes.sort(key=lambda x: x['travel_time'])
        
        # Return the top N routes
        return all_routes[:max_routes]
    
    def _get_traffic_predictions(self, prediction_time: datetime) -> Dict[SCATSID, FlowValue]:
        """
        Get traffic flow predictions for all SCATS sites at the specified time.
        
        Args:
            prediction_time: Time for which to predict traffic
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
        """
        # If a traffic predictor is provided, use it for ML-based predictions
        if self.traffic_predictor is not None:
            try:
                start_time = time.time()
                
                # Generate predictions using the traffic predictor
                if self.use_ensemble:
                    # Use model integration for ensemble prediction
                    model_integration.enable_ensemble(True)
                    predictions = predict_traffic_flows(prediction_time)
                else:
                    # Use specific model type
                    model_integration.enable_ensemble(False)
                    model_integration.set_active_model(self.model_type)
                    predictions = predict_traffic_flows(prediction_time, model_type=self.model_type)
                
                prediction_time = time.time() - start_time
                
                # Log prediction metrics
                self.prediction_metrics['prediction_count'] += 1
                logger.info(f"Generated ML traffic predictions for {len(predictions)} sites in {prediction_time:.2f} seconds")
                logger.info(f"Using model type: {self.model_type if not self.use_ensemble else 'ENSEMBLE'}")
                
                return predictions
            except Exception as e:
                # Log error and increment fallback counter
                self.prediction_metrics['fallback_count'] += 1
                logger.error(f"Error generating ML traffic predictions: {e}")
                logger.warning(f"Falling back to default predictions (fallback count: {self.prediction_metrics['fallback_count']})")
        else:
            logger.info("No traffic predictor provided, using default predictions")
        
        # Fall back to default predictions based on time of day
        return self._get_default_predictions(prediction_time)
    
    def _get_default_predictions(self, prediction_time: datetime) -> Dict[SCATSID, FlowValue]:
        """
        Get default traffic flow predictions based on time of day.
        
        This is a fallback when no prediction model is available or when predictions fail.
        
        Args:
            prediction_time: Time for which to predict traffic
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
        """
        # Simple time-of-day based default predictions
        hour = prediction_time.hour
        
        # Default flow values based on time of day (vehicles per hour)
        if 7 <= hour < 9:  # Morning peak
            default_flow = 500.0
            time_period = "Morning Peak"
        elif 16 <= hour < 19:  # Evening peak
            default_flow = 600.0
            time_period = "Evening Peak"
        elif 9 <= hour < 16:  # Midday
            default_flow = 300.0
            time_period = "Midday"
        else:  # Night
            default_flow = 150.0
            time_period = "Night/Off-Peak"
            
        logger.info(f"Using {time_period} traffic pattern for {prediction_time.strftime('%H:%M')}")
        
        # Create predictions for all SCATS sites
        predictions = {}
        
        # Get all SCATS sites from the site mapper
        # This is a placeholder - in a real implementation, we would get the actual SCATS sites
        # For now, just use a fixed set of 40 sites (typical for the SCATS dataset)
        for site_id in range(1, 41):
            predictions[str(site_id)] = default_flow
        
        logger.info(f"Generated default traffic predictions for {len(predictions)} sites")
        logger.info(f"Time: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}, Flow: {default_flow:.1f} veh/h")
        
        return predictions
    
    def _calculate_route_travel_time(self, path: Route, flow_predictions: Dict[SCATSID, FlowValue],
                                    confidence_level: float) -> Dict[str, Any]:
        """
        Calculate detailed travel time information for a route.
        
        Args:
            path: List of node IDs representing the route
            flow_predictions: Dictionary mapping SCATS IDs to predicted flows
            confidence_level: Confidence level for prediction intervals (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Dictionary with travel time information
        """
        if not path or len(path) < 2:
            return {
                'total_time': 0.0,
                'total_distance': 0.0,
                'average_speed': 0.0,
                'segments': []
            }
            
        # Track prediction quality metrics if ground truth is available
        # This is for logging and monitoring purposes only
        try:
            # In a real system, we would compare predictions to actual traffic data
            # For now, we'll just log the prediction metrics
            if hasattr(self, 'prediction_metrics') and self.prediction_metrics['prediction_count'] > 0:
                fallback_rate = self.prediction_metrics['fallback_count'] / self.prediction_metrics['prediction_count']
                if self.prediction_metrics['rmse']:
                    avg_rmse = sum(self.prediction_metrics['rmse']) / len(self.prediction_metrics['rmse'])
                    avg_mae = sum(self.prediction_metrics['mae']) / len(self.prediction_metrics['mae'])
                    logger.info(f"Prediction metrics - Fallback rate: {fallback_rate:.2f}, Avg RMSE: {avg_rmse:.2f}, Avg MAE: {avg_mae:.2f}")
        except Exception as e:
            # Don't let metrics tracking interfere with the main functionality
            logger.debug(f"Error tracking prediction metrics: {e}")
        
        # Calculate travel time for each segment
        segments = []
        total_time = 0.0
        total_distance = 0.0
        
        # Uncertainty values for confidence interval calculation
        uncertainties = []
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Get SCATS IDs for the nodes
            from_scats = get_site_for_node(from_node)
            to_scats = get_site_for_node(to_node)
            
            # If no mapping exists, use the node IDs directly
            if not from_scats:
                from_scats = str(from_node)
            if not to_scats:
                to_scats = str(to_node)
            
            # Get flow prediction for the from_node
            flow = flow_predictions.get(from_scats, None)
            
            # Calculate segment distance using node coordinates
            distance = 0.0
            if from_node in self.graph.nodes and to_node in self.graph.nodes:
                from_coords = self.graph.nodes[from_node]
                to_coords = self.graph.nodes[to_node]
                distance = haversine_distance(from_coords, to_coords)
            
            # Calculate travel time based on edge cost
            travel_time = 0.0
            speed = 0.0
            for edge_to, edge_cost in self.graph.edges.get(from_node, []):
                if edge_to == to_node:
                    # Edge cost is in seconds
                    travel_time = edge_cost
                    # Calculate speed in km/h
                    if distance > 0 and travel_time > 0:
                        speed = distance / (travel_time / 3600)
                    break
            
            # Determine traffic regime based on flow
            regime = "unknown"
            if flow is not None:
                if flow < 60:  # vehicles per 15 minutes
                    regime = "free_flow"
                elif flow < 120:
                    regime = "moderate"
                else:
                    regime = "congested"
            
            # Add segment information
            segment = {
                'from_node': from_node,
                'to_node': to_node,
                'from_scats': from_scats,
                'to_scats': to_scats,
                'distance': distance,
                'travel_time': travel_time,
                'speed': speed,
                'flow': flow,
                'regime': regime
            }
            
            segments.append(segment)
            
            # Update totals
            total_time += travel_time
            total_distance += distance
            
            # Store uncertainty for confidence interval calculation
            # Simple model: higher uncertainty for congested conditions
            uncertainty = 0.0
            if flow is not None:
                if regime == "free_flow":
                    uncertainty = travel_time * 0.05  # 5% uncertainty
                elif regime == "moderate":
                    uncertainty = travel_time * 0.15  # 15% uncertainty
                else:  # congested
                    uncertainty = travel_time * 0.30  # 30% uncertainty
            uncertainties.append(uncertainty)
        
        # Calculate average speed (km/h)
        if total_time > 0:
            # Convert travel_time from seconds to hours for speed calculation
            average_speed = total_distance / (total_time / 3600)
            
            # Cap the speed at a realistic maximum for urban environments
            max_speed = 60.0  # km/h - typical urban speed limit
            average_speed = min(average_speed, max_speed)
        else:
            average_speed = 0.0
        
        # Calculate confidence interval if uncertainties are available
        confidence_interval = None
        if any(u > 0 for u in uncertainties):
            # Calculate the z-score for the given confidence level
            # For 95% confidence, z = 1.96
            if confidence_level == 0.95:
                z = 1.96
            elif confidence_level == 0.99:
                z = 2.58
            elif confidence_level == 0.90:
                z = 1.645
            else:
                z = 1.96  # Default to 95% confidence
            
            # Calculate the combined uncertainty
            # This is a simplified approach - in reality, we would need to account for
            # correlations between segments
            combined_uncertainty = math.sqrt(sum(u**2 for u in uncertainties))
            
            # Calculate the confidence interval
            margin = z * combined_uncertainty
            confidence_interval = {
                'lower_bound': max(0, total_time - margin),
                'upper_bound': total_time + margin,
                'confidence_level': confidence_level
            }
        
        return {
            'total_time': total_time,
            'total_distance': total_distance,
            'average_speed': average_speed,
            'segments': segments,
            'confidence_interval': confidence_interval
        }

    @lru_cache(maxsize=100)
    def predict_route(self, origin_scats: SCATSID, destination_scats: SCATSID,
                      prediction_time_str: str, algorithm: str) -> RouteInfo:
        """
        Predict a route between SCATS sites with caching.
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time_str: Time string for which to predict traffic
            algorithm: Routing algorithm to use
            
        Returns:
            RouteInfo: Route information dictionary
        """
        return self._predict_route_uncached(origin_scats, destination_scats, prediction_time_str, algorithm)
    
    def _predict_route_uncached(self, origin_scats: SCATSID, destination_scats: SCATSID,
                              prediction_time_str: str, algorithm: str) -> RouteInfo:
        """
        Predict a route between SCATS sites (uncached version).
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time_str: Time string for which to predict traffic
            algorithm: Routing algorithm to use
            
        Returns:
            RouteInfo: Route information dictionary
        """
        # Parse the prediction time
        prediction_time = datetime.strptime(prediction_time_str, '%Y-%m-%d %H:%M:%S')
        
        # Get routes using the specified algorithm
        routes = self.get_routes(
            origin_scats, destination_scats, prediction_time, 
            max_routes=1, algorithms=[algorithm])
        
        # Return the first route or an empty dictionary if no routes found
        return routes[0] if routes else {}
    
    def is_cache_valid(self) -> bool:
        """
        Check if the cache is still valid.
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        current_time = time.time()
        return (current_time - self.last_update_time) < self.cache_timeout
    
    def invalidate_cache(self) -> None:
        """
        Invalidate the route cache.
        """
        self.route_cache.clear()
        self.predict_route.cache_clear()
        self.last_update_time = time.time()
        logger.debug("Route cache invalidated")
    
    def save_routes(self, routes: List[RouteInfo], output_path: str) -> None:
        """
        Save routes to a JSON file.
        
        Args:
            routes: List of route information dictionaries
            output_path: Path to save the routes
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(routes, f, indent=2)
            logger.info(f"Saved {len(routes)} routes to {output_path}")
        except Exception as e:
            logger.error(f"Error saving routes to {output_path}: {e}")
    
    def load_routes(self, input_path: str) -> List[RouteInfo]:
        """
        Load routes from a JSON file.
        
        Args:
            input_path: Path to the routes JSON file
            
        Returns:
            List[RouteInfo]: List of route information dictionaries
        """
        try:
            with open(input_path, 'r') as f:
                routes = json.load(f)
            logger.info(f"Loaded {len(routes)} routes from {input_path}")
            return routes
        except Exception as e:
            logger.error(f"Error loading routes from {input_path}: {e}")
            return []
    
    def set_model_type(self, model_type: str) -> bool:
        """
        Set the active ML model type.
        
        Args:
            model_type (str): Model type to set as active (LSTM, GRU, CNN-RNN)
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_type = model_type.upper()
        if model_type in ["LSTM", "GRU", "CNN-RNN"]:
            self.model_type = model_type
            if self.traffic_predictor:
                self.traffic_predictor.model_type = model_type
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
        logger.info(f"Ensemble prediction {'enabled' if enable else 'disabled'}")


def create_test_graph() -> Graph:
    """
    Create a graph using real SCATS site data from the reference CSV file.
    
    Returns:
        Graph: A graph with real SCATS sites as nodes
    """
    import os
    import pandas as pd
    from app.core.integration.geo_calculator import haversine_distance
    
    # Path to the SCATS site reference data
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           'dataset', 'processed', 'scats_site_reference.csv')
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        logger.error(f"SCATS site reference data not found at {csv_path}")
        # Fall back to a minimal test graph
        nodes = {
            '970': (-37.86703, 145.09159),  # WARRIGAL_RD N of HIGH STREET_RD
            '2000': (-37.8516827, 145.0943457),  # WARRIGAL_RD N of TOORAK_RD
            '3002': (-37.86723, 145.09103)   # HIGH STREET_RD W of WARRIGAL_RD
        }
        edges = {}
        for from_node in nodes:
            edges[from_node] = []
            for to_node in nodes:
                if from_node != to_node:
                    # Calculate distance and use it to estimate travel time
                    distance = haversine_distance(nodes[from_node], nodes[to_node])
                    # Assume average speed of 40 km/h for travel time in seconds
                    travel_time = (distance / 40) * 3600
                    edges[from_node].append((to_node, travel_time))
        return Graph(nodes, edges)
    
    # Load the SCATS site reference data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} SCATS site records from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading SCATS site reference data: {e}")
        return Graph({}, {})
    
    # Extract unique SCATS sites with their coordinates
    unique_sites = {}
    for _, row in df.iterrows():
        site_id = str(row['SCATS_ID']).zfill(4)  # Ensure 4-digit format
        if site_id not in unique_sites and not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
            unique_sites[site_id] = (row['Latitude'], row['Longitude'])
    
    logger.info(f"Extracted {len(unique_sites)} unique SCATS sites with coordinates")
    
    # Create edges between sites based on proximity
    edges = {}
    for from_site in unique_sites:
        edges[from_site] = []
        for to_site in unique_sites:
            if from_site != to_site:
                # Calculate distance between sites
                distance = haversine_distance(unique_sites[from_site], unique_sites[to_site])
                
                # Only connect sites that are within 5 km of each other
                if distance <= 5.0:
                    # Estimate travel time based on distance (assuming 40 km/h average speed)
                    travel_time = (distance / 40) * 3600  # Convert to seconds
                    edges[from_site].append((to_site, travel_time))
    
    # Create the graph with nodes and edges
    graph = Graph(unique_sites, edges)
    
    logger.info(f"Created graph with {len(unique_sites)} nodes and {sum(len(e) for e in edges.values())} edges")
    return graph


def get_routes(origin_scats: SCATSID, destination_scats: SCATSID,
              prediction_time: Optional[datetime] = None,
              max_routes: int = 5,
              confidence_level: float = 0.95,
              algorithms: Optional[List[str]] = None,
              model_type: str = "GRU",
              use_ensemble: bool = False) -> List[RouteInfo]:
    """
    Convenience function to get optimal routes between SCATS sites.
    
    Args:
        origin_scats: SCATS ID of the origin site
        destination_scats: SCATS ID of the destination site
        prediction_time: Time for which to predict traffic
            If None, uses current time
        max_routes: Maximum number of routes to return
        confidence_level: Confidence level for prediction intervals (0.0-1.0)
        algorithms: List of routing algorithms to use
            If None, uses all available algorithms
        model_type: Type of ML model to use (LSTM, GRU, CNN-RNN)
        use_ensemble: Whether to use ensemble prediction with all models
            
    Returns:
        List[RouteInfo]: List of route information dictionaries, sorted by travel time
    """
    # Create a test graph
    graph = create_test_graph()
    
    # Create a traffic predictor
    try:
        traffic_predictor = TrafficPredictor(model_type=model_type)
        logger.info(f"Created traffic predictor with model type: {model_type}")
    except Exception as e:
        logger.error(f"Error creating traffic predictor: {e}")
        traffic_predictor = None
    
    # Create a route predictor with the traffic predictor
    route_predictor = RoutePredictor(
        graph=graph,
        traffic_predictor=traffic_predictor,
        model_type=model_type,
        use_ensemble=use_ensemble
    )
    
    # Get routes
    return route_predictor.get_routes(
        origin_scats=origin_scats,
        destination_scats=destination_scats,
        prediction_time=prediction_time,
        max_routes=max_routes,
        confidence_level=confidence_level,
        algorithms=algorithms
    )


def main():
    """Run a simple test of the Route Predictor."""
    # Set up logging for debugging
    import logging
    # Suppress site mapper warnings for the test
    logging.getLogger('tbrgs.integration.site_mapper').setLevel(logging.ERROR)
    # Enable detailed logging for other components
    logging.getLogger('tbrgs').setLevel(logging.INFO)
    
    print("=" * 80)
    print("TBRGS Route Predictor Test")
    print("=" * 80)
    
    # Create a simple test graph directly
    # For this test, we'll use string node IDs that match the SCATS IDs directly
    # This simplifies testing by avoiding the need for a separate mapping
    nodes = {
        '2000': (-37.85192, 145.09432),  # WARRIGAL_RD/TOORAK_RD
        '3002': (-37.81514, 145.02655),  # DENMARK_ST/BARKERS_RD
        '970': (-37.86730, 145.09151),   # WARRIGAL_RD/HIGH STREET_RD
        '2200': (-37.816540, 145.098047), # Another intersection
        '3001': (-37.814655, 145.022137), # Another intersection
        '4035': (-37.81830, 145.05811)    # BARKERS_RD/BURKE_RD
    }
    
    # Create edges - fully connect all nodes with integer costs
    edges = {}
    for source in nodes:
        edges[source] = []
        for target in nodes:
            if source != target:
                # Add a direct connection with a reasonable cost (integer)
                edges[source].append((target, 60))
    
    # Create the graph
    test_graph = Graph(nodes, edges)
    print(f"Created test graph with {len(test_graph.nodes)} nodes and {sum(len(edges) for edges in test_graph.edges.values())} edges")
    
    # For testing, we'll use a simple identity mapping (node ID = SCATS ID)
    # This bypasses the site mapper which seems to be having issues
    def test_get_node_for_site(scats_id):
        return scats_id if scats_id in nodes else None
    
    def test_get_site_for_node(node_id):
        return node_id if node_id in nodes else None
    
    # Monkey patch the site mapper functions for testing
    # Do this BEFORE creating the RoutePredictor to ensure it uses our test functions
    import app.core.integration.site_mapper
    app.core.integration.site_mapper.get_node_for_site = test_get_node_for_site
    app.core.integration.site_mapper.get_site_for_node = test_get_site_for_node
    
    # Print the mapping for debugging
    print(f"\nTest mapping set up:")
    print(f"SCATS ID 2000 -> Node ID {test_get_node_for_site('2000')}")
    print(f"SCATS ID 3002 -> Node ID {test_get_node_for_site('3002')}")
    
    # Get the mapped node IDs
    origin_node = test_get_node_for_site('2000')
    destination_node = test_get_node_for_site('3002')
    
    print(f"\nCalculating routes from 2000 to 3002")
    print(f"Node mapping: 2000 -> {origin_node}, 3002 -> {destination_node}")
    
    # Verify the nodes exist in the graph
    print(f"\nVerifying nodes exist in graph:")
    print(f"Origin node '{origin_node}' in graph: {origin_node in test_graph.nodes}")
    print(f"Destination node '{destination_node}' in graph: {destination_node in test_graph.nodes}")
    print(f"Graph nodes: {list(test_graph.nodes.keys())}")
    
    # Print edge information for debugging
    print(f"\nEdge information:")
    print(f"Edges from origin node '{origin_node}': {test_graph.edges.get(origin_node, [])}")
    print(f"Total edges in graph: {sum(len(edges) for edges in test_graph.edges.values())}")
    
    # Create a route predictor
    predictor = RoutePredictor(test_graph)
    
    # Print algorithm information
    print(f"\nAvailable algorithms: {predictor.routing_algorithms}")
    
    # Define test origin and destination
    origin_scats = "2000"  # WARRIGAL_RD/TOORAK_RD
    destination_scats = "3002"  # DENMARK_ST/BARKERS_RD
    
    # Get routes for current time
    routes = predictor.get_routes(origin_scats, destination_scats)
    
    if not routes:
        print("No routes found")
    else:
        print(f"\nFound {len(routes)} routes:")
        
        for i, route in enumerate(routes):
            print(f"\nRoute {i+1} ({route['algorithm']}):")
            print(f"  Travel Time: {route['travel_time']:.1f} seconds ({route['travel_time']/60:.1f} minutes)")
            print(f"  Distance: {route['distance']:.2f} km")
            print(f"  Average Speed: {route['average_speed']:.1f} km/h")
            
            if 'confidence_interval' in route and route['confidence_interval'] is not None:
                ci = route['confidence_interval']
                if 'lower_bound' in ci and 'upper_bound' in ci:
                    print(f"  Confidence Interval: {ci['lower_bound']/60:.1f} - {ci['upper_bound']/60:.1f} minutes " +
                         f"({ci['confidence_level']*100:.0f}% confidence)")
            
            print(f"  Path: {' -> '.join(route['scats_path'])}")
    
    # Test with different prediction times
    print("\nTesting different prediction times:")
    
    # Morning peak (8 AM)
    morning_peak = datetime.now().replace(hour=8, minute=0, second=0)
    print(f"\nMorning Peak ({morning_peak.strftime('%H:%M')})")
    
    routes = predictor.get_routes(origin_scats, destination_scats, morning_peak, max_routes=1)
    
    if routes:
        route = routes[0]
        print(f"  Travel Time: {route['travel_time']:.1f} seconds ({route['travel_time']/60:.1f} minutes)")
        print(f"  Average Speed: {route['average_speed']:.1f} km/h")
    
    # Test caching
    print("\nTesting Route Caching:")
    
    import time
    start_time = time.time()
    
    # Call predict_route multiple times with the same parameters
    for _ in range(5):
        route = predictor.predict_route(
            origin_scats, destination_scats,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'astar')
    
    end_time = time.time()
    print(f"Retrieved route 5 times in {(end_time - start_time)*1000:.2f} ms")
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()