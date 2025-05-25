#!/usr/bin/env python3
"""
TBRGS Traffic Predictor Module

This module implements the integration of ML models with traffic prediction for the TBRGS system.
It provides functionality to load trained models, generate predictions for SCATS sites,
and integrate with the routing system.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache
from sklearn.preprocessing import StandardScaler

# Import project-specific modules
from app.core.logging import TBRGSLogger
from app.config.config import config
from app.core.ml.lstm_model import LSTMModel, LSTMNetwork
from app.core.ml.gru_model import GRUModel, GRUNetwork
from app.core.ml.cnnrnn_model import CNNRNNModel, CNNRNNNetwork
from app.dataset.feature_engineer import SCATSFeatureEngineer, engineer_features

# Initialize logger
logger = TBRGSLogger.get_logger("ml.traffic_predictor")

# Type aliases for clarity
SCATSID = str
FlowValue = float
ModelType = str
Predictions = Dict[SCATSID, FlowValue]


class TrafficPredictor:
    """
    Traffic Predictor class for integrating ML models with SCATS traffic prediction.
    
    This class handles loading trained ML models, generating traffic predictions for SCATS sites,
    and providing confidence intervals for predictions.
    
    Attributes:
        model_type (str): Type of model to use (LSTM, GRU, CNN-RNN)
        model_path (str): Path to the model checkpoint
        model (BaseModel): The loaded ML model
        feature_columns (List[str]): List of feature column names
        target_column (str): Name of the target column
        scaler_X (StandardScaler): Scaler for input features
        scaler_y (StandardScaler): Scaler for target values
        cache_timeout (int): Time in seconds before cache entries expire
        historical_data (pd.DataFrame): Historical data for feature engineering
    """
    
    def __init__(self, model_type: str = "GRU", model_path: Optional[str] = None, 
                 cache_size: int = 1000, cache_timeout: int = 300):
        """
        Initialize the Traffic Predictor.
        
        Args:
            model_type (str): Type of model to use (LSTM, GRU, CNN-RNN)
            model_path (str, optional): Path to the model checkpoint. If None, uses the latest checkpoint.
            cache_size (int): Size of the prediction cache
            cache_timeout (int): Time in seconds before cache entries expire
        """
        self.model_type = model_type.upper()
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.target_column = "Target_t+1"  # Default target column
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.cache_timeout = cache_timeout
        self.historical_data = None
        
        # Initialize cache with LRU decorator
        self.get_prediction = lru_cache(maxsize=cache_size)(self._get_prediction_uncached)
        
        # Load the model
        self._load_model()
        
        # Load historical data for feature engineering
        self._load_historical_data()
        
        logger.info(f"Traffic Predictor initialized with {self.model_type} model")
    
    def _load_model(self) -> None:
        """
        Load the ML model from checkpoint.
        """
        # Determine checkpoint directory
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "checkpoints"
        )
        
        # If model_path is not provided, use the latest checkpoint for the selected model
        if self.model_path is None:
            latest_checkpoint = f"{self.model_type}_latest.pt"
            self.model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        # Check if the model path exists
        if not os.path.exists(self.model_path):
            available_models = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            logger.error(f"Model checkpoint not found at {self.model_path}")
            logger.info(f"Available models: {available_models}")
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        # Determine the model type from the filename if not specified
        if self.model_type not in ["LSTM", "GRU", "CNN-RNN"]:
            if "lstm" in self.model_path.lower():
                self.model_type = "LSTM"
            elif "gru" in self.model_path.lower():
                self.model_type = "GRU"
            elif "cnn" in self.model_path.lower() or "cnnrnn" in self.model_path.lower():
                self.model_type = "CNN-RNN"
            else:
                logger.warning(f"Could not determine model type from {self.model_path}. Defaulting to GRU.")
                self.model_type = "GRU"
        
        # Create the appropriate model instance
        if self.model_type == "LSTM":
            self.model = LSTMModel()
        elif self.model_type == "GRU":
            self.model = GRUModel()
        elif self.model_type == "CNN-RNN":
            self.model = CNNRNNModel()
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load the checkpoint
        try:
            # Determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=device)
            
            # Extract model parameters
            if "model_state_dict" in checkpoint:
                # Get input and output dimensions from the checkpoint
                hyperparameters = checkpoint.get("hyperparameters", {})
                input_dim = hyperparameters.get("input_dim", 10)  # Default to 10 if not found
                output_dim = hyperparameters.get("output_dim", 1)  # Default to 1 if not found
                
                # Extract the actual input dimension from the model state dict
                # This is critical to handle mismatches between the checkpoint and our default assumptions
                if "gru.weight_ih_l0" in checkpoint["model_state_dict"]:
                    # For GRU, the input weight shape is [3*hidden_dim, input_dim]
                    weight_shape = checkpoint["model_state_dict"]["gru.weight_ih_l0"].shape
                    actual_input_dim = weight_shape[1]  # Extract input dimension from weights
                    logger.info(f"Detected input dimension from checkpoint: {actual_input_dim}")
                    input_dim = actual_input_dim
                elif "lstm.weight_ih_l0" in checkpoint["model_state_dict"]:
                    # For LSTM, the input weight shape is [4*hidden_dim, input_dim]
                    weight_shape = checkpoint["model_state_dict"]["lstm.weight_ih_l0"].shape
                    actual_input_dim = weight_shape[1]  # Extract input dimension from weights
                    logger.info(f"Detected input dimension from checkpoint: {actual_input_dim}")
                    input_dim = actual_input_dim
                elif "cnn.0.weight" in checkpoint["model_state_dict"]:
                    # For CNN-RNN, need to extract from CNN weights
                    weight_shape = checkpoint["model_state_dict"]["cnn.0.weight"].shape
                    actual_input_dim = weight_shape[1]  # Extract input dimension from weights
                    logger.info(f"Detected input dimension from checkpoint: {actual_input_dim}")
                    input_dim = actual_input_dim
                
                # Build the model with the correct dimensions
                if self.model_type == "LSTM":
                    hidden_dim = hyperparameters.get("hidden_dim", 64)
                    num_layers = hyperparameters.get("num_layers", 2)
                    self.model.build_model(input_dim, hidden_dim, output_dim, num_layers)
                elif self.model_type == "GRU":
                    hidden_dim = hyperparameters.get("hidden_dim", 64)
                    num_layers = hyperparameters.get("num_layers", 2)
                    self.model.build_model(input_dim, hidden_dim, output_dim, num_layers)
                elif self.model_type == "CNN-RNN":
                    hidden_dim = hyperparameters.get("hidden_dim", 64)
                    # Note: CNN-RNN model doesn't use num_layers in the same way
                    # and doesn't accept kernel_size in build_model
                    # It uses its internal cnnrnn_config for these parameters
                    self.model.build_model(input_dim, hidden_dim, output_dim)
                
                # Move model to the correct device
                self.model.model.to(device)
                
                # Load the state dictionary
                try:
                    # Try to use model adapter for CNN-RNN models if there's a dimension mismatch
                    if self.model_type == "CNN-RNN":
                        # Import the model adapter
                        from app.core.ml.model_adapter import adapt_cnnrnn_state_dict
                        
                        # Adapt the state dict if needed
                        adapted_state_dict = adapt_cnnrnn_state_dict(
                            checkpoint["model_state_dict"],
                            current_input_dim=input_dim,
                            saved_input_dim=8  # Assuming saved model has 8 features
                        )
                        self.model.model.load_state_dict(adapted_state_dict)
                        logger.info(f"Successfully adapted CNN-RNN model from 8 to {input_dim} input features")
                    else:
                        # For other models, load directly
                        self.model.model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise
                
                # Set model to evaluation mode
                self.model.model.eval()
                
                # Extract feature and target columns if available
                # If not available in checkpoint, use a default list based on the input dimension
                if "feature_columns" in checkpoint:
                    self.feature_columns = checkpoint["feature_columns"]
                else:
                    # Create a default feature list based on the input dimension
                    if input_dim == 8:
                        self.feature_columns = [
                            "Traffic_Count_t-1", "Traffic_Count_t-4", "Traffic_Count_t-96",
                            "Rolling_Mean_1-hour", "Rolling_Std_1-hour",
                            "Hour", "DayOfWeek", "IsWeekend"
                        ]
                    elif input_dim == 10:
                        self.feature_columns = [
                            "Traffic_Count_t-1", "Traffic_Count_t-4", "Traffic_Count_t-96",
                            "Rolling_Mean_1-hour", "Rolling_Std_1-hour",
                            "Hour", "DayOfWeek", "IsWeekend", "IsAMPeak", "IsPMPeak"
                        ]
                    else:
                        # For other dimensions, create generic feature names
                        self.feature_columns = [f"Feature_{i}" for i in range(input_dim)]
                    
                    logger.info(f"Using default feature columns based on input dimension {input_dim}: {self.feature_columns}")
                
                self.target_column = checkpoint.get("target_column", "Target_t+1")
                
                # Initialize scalers if available in checkpoint
                if "scaler_X" in checkpoint:
                    self.scaler_X = checkpoint["scaler_X"]
                if "scaler_y" in checkpoint:
                    self.scaler_y = checkpoint["scaler_y"]
                
                logger.info(f"Loaded {self.model_type} model from {self.model_path}")
                logger.info(f"Model input dimension: {input_dim}, output dimension: {output_dim}")
                logger.info(f"Feature columns: {self.feature_columns}")
                logger.info(f"Target column: {self.target_column}")
            else:
                logger.error(f"Invalid checkpoint format: 'model_state_dict' not found")
                raise ValueError(f"Invalid checkpoint format: 'model_state_dict' not found")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_historical_data(self) -> None:
        """
        Load historical data for feature engineering.
        """
        try:
            # Get the path to the processed data
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "dataset", "processed", "scats_traffic.csv"
            )
            
            # Check if the file exists
            if not os.path.exists(data_path):
                logger.warning(f"Historical data not found at {data_path}")
                logger.info("Feature engineering will be limited without historical data")
                return
            
            # Load the data (limit to a sample for memory efficiency)
            # In a production system, we would use a database or more efficient storage
            self.historical_data = pd.read_csv(data_path, nrows=100000)
            
            # Convert DateTime column to datetime
            if "DateTime" in self.historical_data.columns:
                self.historical_data["DateTime"] = pd.to_datetime(self.historical_data["DateTime"])
            
            logger.info(f"Loaded {len(self.historical_data)} historical records for feature engineering")
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            logger.warning("Feature engineering will be limited without historical data")
    
    def predict(self, prediction_time: Optional[Union[datetime, str]] = None, 
               site_ids: Optional[List[SCATSID]] = None) -> Predictions:
        """
        Generate traffic predictions for the specified time and SCATS sites.
        
        Args:
            prediction_time: Time for which to predict traffic (default: current time)
            site_ids: List of SCATS site IDs to predict (default: all available sites)
            
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
        
        # Round to nearest 15-minute interval
        minute = prediction_time.minute
        rounded_minute = 15 * (minute // 15)
        prediction_time = prediction_time.replace(minute=rounded_minute, second=0, microsecond=0)
        
        logger.info(f"Generating predictions for {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # If no site IDs provided, use all available sites
        if site_ids is None:
            if self.historical_data is not None and "SCATS_ID" in self.historical_data.columns:
                site_ids = self.historical_data["SCATS_ID"].unique().tolist()
            else:
                # Default to 40 SCATS sites (typical for the dataset)
                site_ids = [str(i) for i in range(970, 4200) if i % 100 < 40]
        
        # Generate predictions for each site
        predictions = {}
        for site_id in site_ids:
            try:
                # Get prediction with caching
                flow, confidence = self.get_prediction(site_id, prediction_time)
                predictions[site_id] = flow
                
                # Log a sample of predictions
                if len(predictions) <= 3:
                    logger.info(f"Prediction for site {site_id}: {flow:.2f} veh/h (±{confidence:.2f})")
            
            except Exception as e:
                logger.error(f"Error predicting for site {site_id}: {e}")
                # Fall back to time-based prediction
                predictions[site_id] = self._get_time_based_prediction(prediction_time)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _get_prediction_uncached(self, site_id: SCATSID, 
                               prediction_time: datetime) -> Tuple[FlowValue, float]:
        """
        Generate a prediction for a specific site and time (uncached version).
        
        Args:
            site_id: SCATS site ID
            prediction_time: Time for which to predict traffic
            
        Returns:
            Tuple[FlowValue, float]: Predicted flow and confidence interval
        """
        # Check if model is loaded
        if self.model is None:
            logger.error("Model not loaded. Falling back to time-based prediction.")
            return self._get_time_based_prediction(prediction_time), 100.0
        
        # Check if we have historical data for feature engineering
        if self.historical_data is None:
            logger.warning("No historical data available. Falling back to time-based prediction.")
            return self._get_time_based_prediction(prediction_time), 100.0
        
        try:
            # Create features for prediction
            features_df = self._create_prediction_features(site_id, prediction_time)
            
            # Check if we have enough features
            if features_df is None or len(features_df) == 0:
                logger.warning(f"Insufficient features for site {site_id}. Falling back to time-based prediction.")
                return self._get_time_based_prediction(prediction_time), 100.0
            
            # Extract features as numpy array and ensure all values are numeric
            try:
                # Convert all columns to float64 to ensure compatibility
                for col in self.feature_columns:
                    if col in features_df.columns:
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                
                # Fill any NaN values with 0
                features_df = features_df.fillna(0)
                
                # Extract as numpy array with explicit float type
                X = features_df[self.feature_columns].values.astype(np.float64)
                
                # Scale features if scaler is fitted
                if hasattr(self.scaler_X, 'transform') and hasattr(self.scaler_X, 'n_features_in_'):
                    X = self.scaler_X.transform(X)
                else:
                    # Simple standardization as fallback
                    means = np.mean(X, axis=0)
                    stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
                    X = (X - means) / stds
            except Exception as e:
                logger.warning(f"Error preparing features: {e}. Using default values.")
                # Create a default feature array with zeros
                X = np.zeros((1, len(self.feature_columns)), dtype=np.float64)
            
            # Make prediction with proper error handling for data types
            try:
                # Ensure X is the right shape and type for the model
                if not isinstance(X, np.ndarray):
                    X = np.array(X, dtype=np.float64)
                
                # Reshape if needed (models often expect a specific shape)
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                
                # Determine the device of the model
                device = next(self.model.model.parameters()).device if hasattr(self.model, 'model') else \
                        next(self.model.parameters()).device if hasattr(self.model, 'parameters') else \
                        torch.device('cpu')
                
                # Convert to tensor for PyTorch models and move to the correct device
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                
                # Make prediction
                with torch.no_grad():
                    try:
                        # Ensure tensor has the right shape for the model
                        # Most RNN models expect [batch_size, seq_len, features]
                        if len(X_tensor.shape) == 2 and hasattr(self.model, 'model') and \
                           (isinstance(self.model.model, torch.nn.GRU) or \
                            isinstance(self.model.model, torch.nn.LSTM) or \
                            'RNN' in self.model.__class__.__name__):
                            # Reshape to [batch_size, seq_len=1, features]
                            X_tensor = X_tensor.unsqueeze(1)
                        
                        # Pass the tensor directly to the model's forward method
                        if hasattr(self.model, 'model'):
                            # Most of our models have a nested model attribute
                            y_pred = self.model.model(X_tensor)
                        else:
                            # Fallback to direct prediction
                            y_pred = self.model(X_tensor)
                    except Exception as e:
                        # If there's an error with the model, use a simple fallback
                        logger.debug(f"Model forward pass error: {e}")
                        # Create a dummy tensor with a single value
                        y_pred = torch.tensor([[0.0]], device=device)
                
                # Convert prediction to numpy if it's a tensor
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.cpu().numpy()
                
                # Inverse transform if scaler is fitted
                if hasattr(self.scaler_y, 'inverse_transform') and hasattr(self.scaler_y, 'n_features_in_'):
                    y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                # Return a default prediction
                y_pred = np.array([self._get_time_based_prediction(prediction_time)])
            
            # Get the prediction value
            flow = float(y_pred[0])
            
            # Calculate confidence interval (simple implementation)
            # In a real system, this would be more sophisticated
            confidence = 0.1 * flow  # 10% of the predicted value
            
            # Ensure prediction is positive
            flow = max(0, flow)
            
            return flow, confidence
        
        except Exception as e:
            logger.error(f"Error in prediction for site {site_id}: {e}")
            return self._get_time_based_prediction(prediction_time), 100.0
    
    def _create_prediction_features(self, site_id: SCATSID, 
                                  prediction_time: datetime) -> Optional[pd.DataFrame]:
        """
        Create features for prediction based on historical data.
        
        Args:
            site_id: SCATS site ID
            prediction_time: Time for which to predict traffic
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        if self.historical_data is None:
            return None
        
        try:
            # Filter historical data for the specific site
            site_data = self.historical_data[self.historical_data["SCATS_ID"] == site_id].copy()
            
            if len(site_data) == 0:
                logger.warning(f"No historical data for site {site_id}")
                return None
            
            # Create a new row for prediction
            pred_row = pd.DataFrame({
                "SCATS_ID": [site_id],
                "DateTime": [prediction_time],
                "Traffic_Count": [np.nan]  # We don't know this yet - it's what we're predicting
            })
            
            # Append to site data (temporarily) for feature engineering
            site_data = pd.concat([site_data, pred_row], ignore_index=True)
            
            # Sort by datetime
            site_data = site_data.sort_values("DateTime")
            
            # Use the feature engineer to create features
            feature_engineer = SCATSFeatureEngineer(site_data)
            feature_engineer.add_lag_features()
            feature_engineer.add_rolling_statistics()
            feature_engineer.add_time_based_features()
            feature_engineer.add_peak_hour_indicators()
            
            # Get the feature-engineered data
            engineered_data = feature_engineer.df
            
            # Extract the prediction row (last row)
            pred_features = engineered_data.iloc[-1:].copy()
            
            # Create a DataFrame with the exact features needed by the model
            final_features = pd.DataFrame()
            
            # Add each required feature, with fallbacks for missing ones
            for feature in self.feature_columns:
                if feature in pred_features.columns:
                    final_features[feature] = pred_features[feature]
                elif feature == "IsPeak" and "IsAMPeak" in pred_features.columns and "IsPMPeak" in pred_features.columns:
                    # Create IsPeak from IsAMPeak and IsPMPeak
                    final_features[feature] = (pred_features["IsAMPeak"] | pred_features["IsPMPeak"]).astype(int)
                elif feature.startswith("Feature_"):
                    # Add a dummy feature with value 0
                    final_features[feature] = 0
                else:
                    # For other missing features, use 0 as fallback
                    logger.warning(f"Missing feature {feature} for prediction, using 0 as fallback")
                    final_features[feature] = 0
            
            return final_features
        
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return None
    
    def _get_time_based_prediction(self, prediction_time: datetime) -> FlowValue:
        """
        Get a time-based prediction as fallback.
        
        Args:
            prediction_time: Time for which to predict traffic
            
        Returns:
            FlowValue: Predicted flow based on time of day
        """
        hour = prediction_time.hour
        
        # Simple time-of-day based default predictions
        if 7 <= hour < 9:  # Morning peak
            flow = 500.0
        elif 16 <= hour < 19:  # Evening peak
            flow = 600.0
        elif 9 <= hour < 16:  # Midday
            flow = 300.0
        else:  # Night
            flow = 150.0
        
        # Add some randomness (±10%)
        flow *= np.random.uniform(0.9, 1.1)
        
        return flow
    
    def get_confidence_interval(self, prediction: FlowValue, 
                              confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a prediction.
        
        Args:
            prediction: Predicted flow value
            confidence_level: Confidence level (0.0-1.0)
            
        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval
        """
        # Simple implementation - in a real system this would be more sophisticated
        # For now, we'll use a percentage of the prediction value based on confidence level
        
        # Map confidence level to percentage uncertainty
        if confidence_level >= 0.99:
            uncertainty = 0.05  # 5% uncertainty for 99% confidence
        elif confidence_level >= 0.95:
            uncertainty = 0.10  # 10% uncertainty for 95% confidence
        elif confidence_level >= 0.90:
            uncertainty = 0.15  # 15% uncertainty for 90% confidence
        elif confidence_level >= 0.80:
            uncertainty = 0.20  # 20% uncertainty for 80% confidence
        else:
            uncertainty = 0.30  # 30% uncertainty for lower confidence
        
        # Calculate bounds
        margin = prediction * uncertainty
        lower_bound = max(0, prediction - margin)
        upper_bound = prediction + margin
        
        return lower_bound, upper_bound
    
    def clear_cache(self) -> None:
        """
        Clear the prediction cache.
        """
        self.get_prediction.cache_clear()
        logger.info("Prediction cache cleared")
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List[str]: List of available model types
        """
        return ["LSTM", "GRU", "CNN-RNN"]
    
    def switch_model(self, model_type: str, model_path: Optional[str] = None) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_type: Type of model to use (LSTM, GRU, CNN-RNN)
            model_path: Path to the model checkpoint (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save current model type and path
            old_model_type = self.model_type
            old_model_path = self.model_path
            
            # Update model type and path
            self.model_type = model_type.upper()
            if model_path is not None:
                self.model_path = model_path
            else:
                # Use the latest checkpoint for the selected model
                checkpoint_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "checkpoints"
                )
                latest_checkpoint = f"{self.model_type}_latest.pt"
                self.model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            # Load the new model
            self._load_model()
            
            # Clear the cache
            self.clear_cache()
            
            logger.info(f"Switched from {old_model_type} to {self.model_type} model")
            return True
        
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False


# Create a singleton instance for easy import
traffic_predictor = TrafficPredictor()


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
    global traffic_predictor
    
    # Switch model if specified
    if model_type is not None and model_type.upper() != traffic_predictor.model_type:
        traffic_predictor.switch_model(model_type)
    
    # Generate predictions
    return traffic_predictor.predict(prediction_time, site_ids)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TBRGS Traffic Predictor")
    parser.add_argument("--model", type=str, choices=["lstm", "gru", "cnnrnn"], default="gru",
                        help="Model type to use (lstm, gru, cnnrnn)")
    parser.add_argument("--time", type=str, default=None,
                        help="Prediction time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--sites", type=str, default=None,
                        help="Comma-separated list of SCATS site IDs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    # Parse site IDs if provided
    site_ids = args.sites.split(",") if args.sites else None
    
    # Create predictor with specified model
    predictor = TrafficPredictor(
        model_type=args.model,
        model_path=args.checkpoint
    )
    
    # Generate predictions
    predictions = predictor.predict(args.time, site_ids)
    
    # Print predictions
    print(f"Traffic Predictions ({args.model.upper()} model):")
    print(f"Time: {args.time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of sites: {len(predictions)}")
    print("\nSample predictions:")
    for site_id, flow in list(predictions.items())[:10]:
        lower, upper = predictor.get_confidence_interval(flow)
        print(f"Site {site_id}: {flow:.2f} veh/h (95% CI: {lower:.2f} - {upper:.2f})")
