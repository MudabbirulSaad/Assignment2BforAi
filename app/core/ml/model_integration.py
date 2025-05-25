#!/usr/bin/env python3
"""
TBRGS Model Integration Module

This module integrates the ML traffic prediction models with the routing system.
It provides a unified interface for using multiple ML models for traffic prediction.
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import project-specific modules
from app.core.logging import TBRGSLogger
from app.config.config import config
from app.core.ml.traffic_predictor import TrafficPredictor
from app.core.ml.lstm_model import LSTMModel
from app.core.ml.gru_model import GRUModel
from app.core.ml.cnnrnn_model import CNNRNNModel

# Initialize logger
logger = TBRGSLogger.get_logger("ml.model_integration")

# Type aliases for clarity
SCATSID = str
FlowValue = float
ModelType = str
Predictions = Dict[SCATSID, FlowValue]


class ModelIntegration:
    """
    Model Integration class for managing multiple ML models for traffic prediction.
    
    This class provides a unified interface for using LSTM, GRU, and CNN-RNN models
    for traffic prediction and integrating them with the routing system.
    
    Attributes:
        predictors (Dict[str, TrafficPredictor]): Dictionary of traffic predictors by model type
        active_model (str): Currently active model type
        ensemble_weights (Dict[str, float]): Weights for ensemble prediction
    """
    
    def __init__(self, use_ensemble: bool = False):
        """
        Initialize the Model Integration.
        
        Args:
            use_ensemble (bool): Whether to use ensemble prediction
        """
        self.predictors = {}
        self.active_model = "GRU"  # Default model
        self.use_ensemble = use_ensemble
        self.ensemble_weights = {
            "LSTM": 0.3,
            "GRU": 0.4,
            "CNN-RNN": 0.3
        }
        
        # Initialize predictors for all model types
        self._initialize_predictors()
        
        logger.info(f"Model Integration initialized with active model: {self.active_model}")
        if self.use_ensemble:
            logger.info(f"Ensemble prediction enabled with weights: {self.ensemble_weights}")
    
    def _initialize_predictors(self):
        """
        Initialize predictors for all model types.
        """
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize LSTM predictor
        try:
            self.predictors["LSTM"] = TrafficPredictor(model_type="LSTM")
            logger.info("LSTM predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing LSTM predictor: {e}")
        
        # Initialize GRU predictor
        try:
            self.predictors["GRU"] = TrafficPredictor(model_type="GRU")
            logger.info("GRU predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing GRU predictor: {e}")
        
        # Initialize CNN-RNN predictor
        try:
            self.predictors["CNN-RNN"] = TrafficPredictor(model_type="CNN-RNN")
            logger.info("CNN-RNN predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing CNN-RNN predictor: {e}")
    
    def set_active_model(self, model_type: str) -> bool:
        """
        Set the active model type.
        
        Args:
            model_type (str): Model type to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Normalize model type
        if model_type.lower() in ['cnnrnn', 'cnn-rnn', 'cnn_rnn']:
            model_type = 'CNN-RNN'
        else:
            model_type = model_type.upper()
            
        if model_type in self.predictors:
            self.active_model = model_type
            logger.info(f"Active model set to {model_type}")
            return True
        else:
            logger.error(f"Model type {model_type} not available")
            return False
    
    def predict(self, prediction_time: Optional[Union[datetime, str]] = None,
               site_ids: Optional[List[SCATSID]] = None) -> Predictions:
        """
        Generate traffic predictions using the active model or ensemble.
        
        Args:
            prediction_time: Time for which to predict traffic
            site_ids: List of SCATS site IDs to predict
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
        """
        # Parse prediction time
        if prediction_time is None:
            prediction_time = datetime.now()
        elif isinstance(prediction_time, str):
            try:
                prediction_time = datetime.strptime(prediction_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.warning(f"Invalid time format: {prediction_time}. Using current time.")
                prediction_time = datetime.now()
        
        logger.info(f"Generating predictions for {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Use ensemble prediction if enabled
        if self.use_ensemble:
            return self._predict_ensemble(prediction_time, site_ids)
        
        # Use active model for prediction
        if self.active_model in self.predictors:
            predictor = self.predictors[self.active_model]
            predictions = predictor.predict(prediction_time, site_ids)
            logger.info(f"Generated {len(predictions)} predictions using {self.active_model} model")
            return predictions
        else:
            logger.error(f"Active model {self.active_model} not available")
            return self._get_time_based_predictions(prediction_time, site_ids)
    
    def _predict_ensemble(self, prediction_time: datetime,
                        site_ids: Optional[List[SCATSID]] = None) -> Predictions:
        """
        Generate ensemble predictions using all available models.
        
        Args:
            prediction_time: Time for which to predict traffic
            site_ids: List of SCATS site IDs to predict
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
        """
        # Get predictions from all available models
        model_predictions = {}
        for model_type, predictor in self.predictors.items():
            try:
                predictions = predictor.predict(prediction_time, site_ids)
                model_predictions[model_type] = predictions
                logger.info(f"Generated {len(predictions)} predictions using {model_type} model")
            except Exception as e:
                logger.error(f"Error generating predictions with {model_type} model: {e}")
        
        # If no models generated predictions, fall back to time-based predictions
        if not model_predictions:
            logger.warning("No models generated predictions. Falling back to time-based predictions.")
            return self._get_time_based_predictions(prediction_time, site_ids)
        
        # Combine predictions using ensemble weights
        ensemble_predictions = {}
        
        # Determine the set of all site IDs across all model predictions
        all_site_ids = set()
        for predictions in model_predictions.values():
            all_site_ids.update(predictions.keys())
        
        # If site_ids is provided, filter to only those sites
        if site_ids is not None:
            all_site_ids = all_site_ids.intersection(site_ids)
        
        # Calculate weighted average for each site
        for site_id in all_site_ids:
            # Get predictions from each model for this site
            site_predictions = {}
            for model_type, predictions in model_predictions.items():
                if site_id in predictions:
                    site_predictions[model_type] = predictions[site_id]
            
            # Calculate weighted average
            if site_predictions:
                total_weight = 0
                weighted_sum = 0
                
                for model_type, prediction in site_predictions.items():
                    weight = self.ensemble_weights.get(model_type, 0)
                    weighted_sum += prediction * weight
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    ensemble_predictions[site_id] = weighted_sum / total_weight
                else:
                    # If no weights, use simple average
                    ensemble_predictions[site_id] = sum(site_predictions.values()) / len(site_predictions)
        
        logger.info(f"Generated {len(ensemble_predictions)} ensemble predictions")
        return ensemble_predictions
    
    def _get_time_based_predictions(self, prediction_time: datetime,
                                  site_ids: Optional[List[SCATSID]] = None) -> Predictions:
        """
        Get time-based predictions as fallback.
        
        Args:
            prediction_time: Time for which to predict traffic
            site_ids: List of SCATS site IDs to predict
            
        Returns:
            Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
        """
        hour = prediction_time.hour
        
        # Simple time-of-day based default predictions
        if 7 <= hour < 9:  # Morning peak
            base_flow = 500.0
        elif 16 <= hour < 19:  # Evening peak
            base_flow = 600.0
        elif 9 <= hour < 16:  # Midday
            base_flow = 300.0
        else:  # Night
            base_flow = 150.0
        
        # If no site IDs provided, use a default set
        if site_ids is None:
            # Default to 40 SCATS sites (typical for the dataset)
            site_ids = [str(i) for i in range(970, 4200) if i % 100 < 40]
        
        # Generate predictions for each site
        predictions = {}
        for site_id in site_ids:
            # Add some randomness (±10%)
            flow = base_flow * np.random.uniform(0.9, 1.1)
            predictions[site_id] = flow
        
        logger.info(f"Generated {len(predictions)} time-based predictions")
        return predictions
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List[str]: List of available model types
        """
        return list(self.predictors.keys())
    
    def set_ensemble_weights(self, weights: Dict[str, float]) -> bool:
        """
        Set weights for ensemble prediction.
        
        Args:
            weights: Dictionary mapping model types to weights
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate weights
        for model_type in weights:
            if model_type not in self.predictors:
                logger.error(f"Model type {model_type} not available")
                return False
        
        # Update weights
        self.ensemble_weights.update(weights)
        
        # Normalize weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for model_type in self.ensemble_weights:
                self.ensemble_weights[model_type] /= total_weight
        
        logger.info(f"Ensemble weights updated: {self.ensemble_weights}")
        return True
    
    def enable_ensemble(self, enable: bool = True) -> None:
        """
        Enable or disable ensemble prediction.
        
        Args:
            enable: Whether to enable ensemble prediction
        """
        self.use_ensemble = enable
        logger.info(f"Ensemble prediction {'enabled' if enable else 'disabled'}")


# Create a singleton instance for easy import
model_integration = ModelIntegration()


def predict_traffic_flows(prediction_time: Optional[Union[datetime, str]] = None,
                         site_ids: Optional[List[SCATSID]] = None,
                         model_type: Optional[str] = None) -> Predictions:
    """
    Convenience function to predict traffic flows.
    
    Args:
        prediction_time: Time for which to predict traffic
        site_ids: List of SCATS site IDs to predict
        model_type: Type of model to use (LSTM, GRU, CNN-RNN)
        
    Returns:
        Dict[SCATSID, FlowValue]: Dictionary mapping SCATS IDs to predicted flows
    """
    global model_integration
    
    # Set active model if specified
    if model_type is not None:
        model_integration.set_active_model(model_type)
    
    # Generate predictions
    return model_integration.predict(prediction_time, site_ids)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TBRGS Model Integration")
    parser.add_argument("--model", type=str, choices=["lstm", "gru", "cnnrnn", "ensemble"], default="gru",
                        help="Model type to use (lstm, gru, cnnrnn, ensemble)")
    parser.add_argument("--time", type=str, default=None,
                        help="Prediction time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--sites", type=str, default=None,
                        help="Comma-separated list of SCATS site IDs")
    
    args = parser.parse_args()
    
    # Parse site IDs if provided
    site_ids = args.sites.split(",") if args.sites else None
    
    # Set up model integration
    if args.model == "ensemble":
        model_integration.enable_ensemble(True)
    else:
        model_integration.enable_ensemble(False)
        model_integration.set_active_model(args.model)
    
    # Generate predictions
    predictions = model_integration.predict(args.time, site_ids)
    
    # Print predictions
    print(f"Traffic Predictions ({args.model.upper()} model):")
    print(f"Time: {args.time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of sites: {len(predictions)}")
    print("\nSample predictions:")
    for site_id, flow in list(predictions.items())[:10]:
        print(f"Site {site_id}: {flow:.2f} veh/h")
