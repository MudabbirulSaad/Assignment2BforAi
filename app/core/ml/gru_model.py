#!/usr/bin/env python3
"""
GRU Model Implementation for Traffic Prediction

This module implements a GRU-based model for traffic prediction in the TBRGS project.
It inherits from the BaseModel class and implements the specific GRU architecture.
The GRU (Gated Recurrent Unit) is a variant of LSTM with fewer parameters and
comparable performance for many tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union

from app.core.logging import logger
from app.core.ml.base_model import BaseModel


class GRUNetwork(nn.Module):
    """
    GRU Neural Network for time series prediction.
    
    This class implements a PyTorch GRU model with optional dropout and batch normalization.
    GRUs are generally faster to train than LSTMs and may perform better on smaller datasets.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units
        output_dim (int): Number of output units
        num_layers (int): Number of GRU layers
        dropout (float): Dropout rate
        batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1,
                dropout: float = 0.0, batch_norm: bool = False, bidirectional: bool = False):
        """
        Initialize the GRU network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            output_dim (int): Number of output units
            num_layers (int, optional): Number of GRU layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            bidirectional (bool, optional): Whether to use bidirectional GRU. Defaults to False.
        """
        super(GRUNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output dimension for bidirectional GRU
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Batch normalization (optional)
        self.batch_norm = nn.BatchNorm1d(fc_input_dim) if batch_norm else None
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output layer
        self.fc = nn.Linear(fc_input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # GRU forward pass
        # Output shape: (batch_size, seq_len, hidden_dim * (2 if bidirectional else 1))
        gru_out, _ = self.gru(x)
        
        # Take only the last time step output
        # Shape: (batch_size, hidden_dim * (2 if bidirectional else 1))
        out = gru_out[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Apply output layer
        # Shape: (batch_size, output_dim)
        out = self.fc(out)
        
        return out


class GRUModel(BaseModel):
    """
    GRU Model for traffic prediction.
    
    This class implements a GRU-based model for traffic prediction,
    inheriting from the BaseModel class. GRU models are often more
    efficient than LSTM models while maintaining similar performance.
    """
    
    def __init__(self, model_name: str = "GRU_Traffic_Predictor", config: Dict = None):
        """
        Initialize the GRU model.
        
        Args:
            model_name (str, optional): Name of the model. Defaults to "GRU_Traffic_Predictor".
            config (Dict, optional): Model configuration. Defaults to None.
        """
        super(GRUModel, self).__init__(model_name, "GRU", config)
        
        # Simplified GRU configuration to reduce overfitting
        self.gru_config = {
            'hidden_dim': 64,         # Reduced from 128 to prevent overfitting
            'num_layers': 2,          # Reduced from 3 to simplify model
            'dropout': 0.5,           # Increased from 0.3 for stronger regularization
            'batch_norm': True,       # Keep batch normalization
            'bidirectional': False,   # Disable bidirectional to reduce complexity
            'learning_rate': 0.0001,  # Reduced for more stable training
            'weight_decay': 5e-4,     # Increased for stronger regularization
            'sequence_length': 24     # Reduced from 36 to focus on more relevant patterns
        }
        
        # Update with user config if provided
        if config and 'gru_config' in config:
            self.gru_config.update(config['gru_config'])
    
    def build_model(self, input_dim: int, hidden_dim: int = None, output_dim: int = 1, 
                   num_layers: int = None) -> None:
        """
        Build the GRU model architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int, optional): Number of hidden units. Defaults to None.
            output_dim (int, optional): Number of output units. Defaults to 1.
            num_layers (int, optional): Number of GRU layers. Defaults to None.
        """
        # Use provided parameters or defaults from config
        hidden_dim = hidden_dim or self.gru_config['hidden_dim']
        num_layers = num_layers or self.gru_config['num_layers']
        
        # Create the GRU network
        self.model = GRUNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=self.gru_config['dropout'],
            batch_norm=self.gru_config['batch_norm'],
            bidirectional=self.gru_config['bidirectional']
        )
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.gru_config['learning_rate'],
            weight_decay=self.gru_config['weight_decay']
        )
        
        self.criterion = nn.MSELoss()
        
        # Log model architecture
        bidirectional_str = "bidirectional " if self.gru_config['bidirectional'] else ""
        logger.info(f"Built {bidirectional_str}GRU model with {input_dim} input features, "
                   f"{hidden_dim} hidden units, {num_layers} layers, "
                   f"and {output_dim} output units")
        
        # If we have a temporary state dict from loading, apply it now
        if hasattr(self, '_temp_state_dict'):
            self.model.load_state_dict(self._temp_state_dict)
            delattr(self, '_temp_state_dict')
            logger.info("Loaded model weights from saved state")
    
    def prepare_data_for_training(self, data: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Prepare data for training the GRU model.
        
        This method extracts the input dimension from the data loaders and builds the model.
        
        Args:
            data (Dict[str, DataLoader]): Dictionary with train, val, and test data loaders
        """
        # Get a batch from the train loader to determine input shape
        for X_batch, _ in data['train']:
            input_shape = X_batch.shape
            break
        
        # Extract dimensions
        _, seq_len, n_features = input_shape
        
        # Build the model with the correct input dimension
        self.build_model(input_dim=n_features)
        
        logger.info(f"Model prepared for training with input shape: {input_shape}")
    
    def compare_with_lstm(self, lstm_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compare GRU performance with LSTM.
        
        This method calculates the relative performance difference between
        GRU and LSTM models.
        
        Args:
            lstm_metrics (Dict[str, float]): Performance metrics from an LSTM model
            
        Returns:
            Dict[str, float]: Relative performance differences (positive values indicate GRU is better)
        """
        if not self.metrics:
            raise ValueError("GRU model has not been evaluated yet. Call evaluate() first.")
        
        comparison = {}
        
        # Calculate relative differences for each metric
        for metric, value in self.metrics.items():
            if metric in lstm_metrics:
                # For metrics where lower is better (MSE, RMSE, MAE, MAPE)
                if metric in ['mse', 'rmse', 'mae', 'mape', 'loss']:
                    # Negative value means GRU is better (lower error)
                    rel_diff = (lstm_metrics[metric] - value) / lstm_metrics[metric] * 100
                # For metrics where higher is better (R2)
                elif metric in ['r2']:
                    # Positive value means GRU is better (higher score)
                    rel_diff = (value - lstm_metrics[metric]) / lstm_metrics[metric] * 100
                
                comparison[f"{metric}_diff_pct"] = rel_diff
        
        # Add a summary metric
        if 'rmse' in comparison:
            comparison['summary'] = "GRU outperforms LSTM" if comparison['rmse_diff_pct'] > 0 else "LSTM outperforms GRU"
        
        logger.info(f"GRU vs LSTM comparison: {comparison}")
        return comparison
