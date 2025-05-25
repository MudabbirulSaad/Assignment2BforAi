#!/usr/bin/env python3
"""
TBRGS ML-Route Integration Module

This module integrates the ML traffic prediction models with the routing system.
It provides a bridge between the ML models and the route predictor.
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import project-specific modules
from app.core.logging import TBRGSLogger
from app.core.ml.model_integration import model_integration, predict_traffic_flows
from app.core.integration.route_predictor import RoutePredictor

# Initialize logger
logger = TBRGSLogger.get_logger("integration.ml_route_integration")

# Disable annoying error logs from traffic predictor
import logging
logging.getLogger('tbrgs.ml.traffic_predictor').setLevel(logging.CRITICAL)

# Type aliases for clarity
SCATSID = str
FlowValue = float
NodeID = str
EdgeWeight = float
Route = List[NodeID]
RouteInfo = Dict[str, Any]


class MLRouteIntegration:
    """
    ML-Route Integration class for connecting ML models with the routing system.
    
    This class provides a bridge between the ML traffic prediction models and the
    route predictor, allowing the routing system to use ML predictions for edge weights.
    
    Attributes:
        route_predictor (RoutePredictor): The route predictor instance
        model_type (str): The active ML model type
        use_ensemble (bool): Whether to use ensemble prediction
    """
    
    def __init__(self, route_predictor: RoutePredictor, model_type: str = "GRU", use_ensemble: bool = False):
        """
        Initialize the ML-Route Integration.
        
        Args:
            route_predictor (RoutePredictor): The route predictor instance
            model_type (str): The active ML model type
            use_ensemble (bool): Whether to use ensemble prediction
        """
        self.route_predictor = route_predictor
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        
        # Configure the model integration
        if use_ensemble:
            model_integration.enable_ensemble(True)
        else:
            model_integration.enable_ensemble(False)
            model_integration.set_active_model(model_type)
        
        # Replace the route predictor's traffic prediction method
        self._patch_route_predictor()
        
        logger.info(f"ML-Route Integration initialized with {model_type} model")
        if use_ensemble:
            logger.info("Ensemble prediction enabled")
    
    def _patch_route_predictor(self):
        """
        Patch the route predictor to use ML predictions.
        """
        # Store the original method for fallback
        self.route_predictor._original_get_traffic_predictions = self.route_predictor._get_traffic_predictions
        
        # Replace with ML prediction method
        self.route_predictor._get_traffic_predictions = self._ml_get_traffic_predictions
        
        logger.info("Route predictor patched to use ML predictions")
    
    def _ml_get_traffic_predictions(self, prediction_time: datetime) -> Dict[SCATSID, FlowValue]:
        """
        Get traffic flow predictions using ML models.
        
        Args:
            prediction_time: Time for which to predict traffic
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
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
    
    def set_model_type(self, model_type: str) -> bool:
        """
        Set the active ML model type.
        
        Args:
            model_type (str): Model type to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_type.upper() in model_integration.get_available_models():
            self.model_type = model_type.upper()
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
    
    def get_routes(self, origin_scats: SCATSID, destination_scats: SCATSID,
                 prediction_time: Optional[Union[datetime, str]] = None,
                 max_routes: int = 5,
                 confidence_level: float = 0.95,
                 algorithms: Optional[List[str]] = None) -> List[RouteInfo]:
        """
        Get optimal routes between SCATS sites using ML predictions.
        
        Args:
            origin_scats: SCATS ID of the origin site
            destination_scats: SCATS ID of the destination site
            prediction_time: Time for which to predict traffic
            max_routes: Maximum number of routes to return
            confidence_level: Confidence level for prediction intervals
            algorithms: List of routing algorithms to use
            
        Returns:
            List[RouteInfo]: List of route information dictionaries
        """
        return self.route_predictor.get_routes(
            origin_scats=origin_scats,
            destination_scats=destination_scats,
            prediction_time=prediction_time,
            max_routes=max_routes,
            confidence_level=confidence_level,
            algorithms=algorithms
        )


def create_ml_route_integration(model_type: str = "GRU", use_ensemble: bool = False) -> MLRouteIntegration:
    """
    Create an ML-Route Integration instance.
    
    Args:
        model_type (str): The active ML model type
        use_ensemble (bool): Whether to use ensemble prediction
        
    Returns:
        MLRouteIntegration: The ML-Route Integration instance
    """
    # Normalize model type for consistency
    if model_type.lower() in ['cnnrnn', 'cnn-rnn', 'cnn_rnn']:
        model_type = 'CNN-RNN'
    elif model_type.lower() == 'lstm':
        model_type = 'LSTM'
    elif model_type.lower() == 'gru':
        model_type = 'GRU'
    
    # Create a test graph
    from app.core.integration.route_predictor import create_test_graph
    graph = create_test_graph()
    
    # Create a route predictor with the test graph
    route_predictor = RoutePredictor(graph)
    
    # Create an ML-Route Integration
    integration = MLRouteIntegration(route_predictor, model_type, use_ensemble)
    
    return integration


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TBRGS ML-Route Integration")
    parser.add_argument("--model", type=str, choices=["lstm", "gru", "cnnrnn", "ensemble"], default="gru",
                        help="Model type to use (lstm, gru, cnnrnn, ensemble)")
    parser.add_argument("--origin", type=str, required=True,
                        help="SCATS ID of the origin site")
    parser.add_argument("--destination", type=str, required=True,
                        help="SCATS ID of the destination site")
    parser.add_argument("--time", type=str, default=None,
                        help="Prediction time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--max-routes", type=int, default=5,
                        help="Maximum number of routes to return")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated list of routing algorithms to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Enable verbose output to stdout
    if args.verbose:
        # Set up a console handler with higher level
        import logging
        root_logger = logging.getLogger()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Enable all loggers
        logging.getLogger('tbrgs').setLevel(logging.INFO)
        logging.getLogger('tbrgs.integration').setLevel(logging.INFO)
        logging.getLogger('tbrgs.ml').setLevel(logging.INFO)
    
    # Parse algorithms if provided
    algorithms = args.algorithms.split(",") if args.algorithms else None
    
    # Check if SCATS site reference data exists
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           'dataset', 'processed', 'scats_site_reference.csv')
    if not os.path.exists(csv_path):
        print(f"WARNING: SCATS site reference data not found at {csv_path}")
        print("The system will use a minimal test graph instead.")
    else:
        print(f"Found SCATS site reference data at {csv_path}")
    
    # Create ML-Route Integration
    print("Creating ML-Route Integration...")
    try:
        integration = create_ml_route_integration(
            model_type=args.model,
            use_ensemble=args.model == "ensemble"
        )
        print(f"Integration created successfully with model type: {args.model}")
    except Exception as e:
        print(f"Error creating integration: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Get routes
    print(f"\nGetting routes from {args.origin} to {args.destination}...")
    try:
        # Check if the nodes exist in the graph
        graph = integration.route_predictor.graph
        origin_node = args.origin
        destination_node = args.destination
        
        # Print graph information
        print(f"Graph has {len(graph.nodes)} nodes and {sum(len(e) for e in graph.edges.values())} edges")
        
        # Check if origin and destination are in the graph
        if origin_node not in graph.nodes:
            print(f"WARNING: Origin node {origin_node} not found in graph")
        else:
            print(f"Origin node {origin_node} found at coordinates {graph.nodes[origin_node]}")
        
        if destination_node not in graph.nodes:
            print(f"WARNING: Destination node {destination_node} not found in graph")
        else:
            print(f"Destination node {destination_node} found at coordinates {graph.nodes[destination_node]}")
        
        # Get routes
        routes = integration.get_routes(
            origin_scats=args.origin,
            destination_scats=args.destination,
            prediction_time=args.time,
            max_routes=args.max_routes,
            algorithms=algorithms
        )
        print(f"Found {len(routes)} routes")
    except Exception as e:
        print(f"Error getting routes: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Print routes
    print(f"Routes from {args.origin} to {args.destination}:")
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
        
        # Print more details if verbose
        if args.verbose and 'segments' in route:
            print("  Segments:")
            for segment in route['segments']:
                print(f"    {segment['from_node']} -> {segment['to_node']}: {segment['travel_time']:.1f} seconds, {segment['distance']:.2f} km")
                if 'flow' in segment and segment['flow'] is not None:
                    print(f"      Flow: {segment['flow']:.1f}, Regime: {segment['regime']}")
        
        # Print confidence interval if available
        if args.verbose and 'confidence_interval' in route and route['confidence_interval']:
            ci = route['confidence_interval']
            print(f"  Confidence Interval ({ci['confidence_level']:.0%}):")
            print(f"    Lower bound: {ci['lower_bound']:.1f} seconds")
            print(f"    Upper bound: {ci['upper_bound']:.1f} seconds")
