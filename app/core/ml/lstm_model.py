#!/usr/bin/env python3
"""
LSTM Model Implementation for Traffic Prediction

This module implements an LSTM-based model for traffic prediction in the TBRGS project.
It inherits from the BaseModel class and implements the specific LSTM architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union

from app.core.logging import logger
from app.core.ml.base_model import BaseModel


class LSTMNetwork(nn.Module):
    """
    LSTM Neural Network for time series prediction.
    
    This class implements a PyTorch LSTM model with optional dropout and batch normalization.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units
        output_dim (int): Number of output units
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1,
                dropout: float = 0.0, batch_norm: bool = False):
        """
        Initialize the LSTM network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            output_dim (int): Number of output units
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        """
        super(LSTMNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization (optional)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        # Output shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take only the last time step output
        # Shape: (batch_size, hidden_dim)
        out = lstm_out[:, -1, :]
        
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


class LSTMModel(BaseModel):
    """
    LSTM Model for traffic prediction.
    
    This class implements an LSTM-based model for traffic prediction,
    inheriting from the BaseModel class.
    """
    
    def __init__(self, model_name: str = "LSTM_Traffic_Predictor", config: Dict = None):
        """
        Initialize the LSTM model.
        
        Args:
            model_name (str, optional): Name of the model. Defaults to "LSTM_Traffic_Predictor".
            config (Dict, optional): Model configuration. Defaults to None.
        """
        super(LSTMModel, self).__init__(model_name, "LSTM", config)
        
        # Default LSTM configuration with improved settings
        self.lstm_config = {
            'hidden_dim': 64,           # Reduced from 128 to prevent overfitting
            'num_layers': 2,            # Reduced from 3 to simplify the model
            'dropout': 0.5,             # Increased from 0.2 for stronger regularization
            'batch_norm': True,         # Keep batch normalization for stable training
            'learning_rate': 0.0001,    # Reduced from 0.001 for more stable training
            'weight_decay': 5e-4        # Increased from 1e-5 for better regularization
        }
        
        # Update with user config if provided
        if config and 'lstm_config' in config:
            self.lstm_config.update(config['lstm_config'])
    
    def build_model(self, input_dim: int, hidden_dim: int = None, output_dim: int = 1, 
                   num_layers: int = None) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int, optional): Number of hidden units. Defaults to None.
            output_dim (int, optional): Number of output units. Defaults to 1.
            num_layers (int, optional): Number of LSTM layers. Defaults to None.
        """
        # Use provided parameters or defaults from config
        hidden_dim = hidden_dim or self.lstm_config['hidden_dim']
        num_layers = num_layers or self.lstm_config['num_layers']
        
        # Create the LSTM network
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=self.lstm_config['dropout'],
            batch_norm=self.lstm_config['batch_norm']
        )
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lstm_config['learning_rate'],
            weight_decay=self.lstm_config['weight_decay']
        )
        
        self.criterion = nn.MSELoss()
        
        # Log model architecture
        logger.info(f"Built LSTM model with {input_dim} input features, "
                   f"{hidden_dim} hidden units, {num_layers} layers, "
                   f"and {output_dim} output units")
        
        # If we have a temporary state dict from loading, apply it now
        if hasattr(self, '_temp_state_dict'):
            self.model.load_state_dict(self._temp_state_dict)
            delattr(self, '_temp_state_dict')
            logger.info("Loaded model weights from saved state")
    
    def prepare_data_for_training(self, data: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Prepare data for training the LSTM model.
        
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
