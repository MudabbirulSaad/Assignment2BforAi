#!/usr/bin/env python3
"""
TBRGS Model Adapter Module

This module provides adapters for handling model compatibility issues,
particularly when the input feature dimensions have changed between model versions.
"""

import torch
import logging
from typing import Dict, Any

# Initialize logger
logger = logging.getLogger("tbrgs.ml.model_adapter")

def adapt_cnnrnn_state_dict(state_dict: Dict[str, Any], current_input_dim: int, saved_input_dim: int = 8) -> Dict[str, Any]:
    """
    Adapt a CNN-RNN model state dictionary to handle input dimension mismatches.
    
    This function modifies the state dictionary of a saved CNN-RNN model to make it compatible
    with a model instance that has a different input dimension. It specifically handles:
    1. feature_weights tensor
    2. First convolutional layer weights
    
    Args:
        state_dict (Dict[str, Any]): The loaded state dictionary
        current_input_dim (int): The input dimension of the current model
        saved_input_dim (int): The input dimension of the saved model
        
    Returns:
        Dict[str, Any]: The adapted state dictionary
    """
    if current_input_dim == saved_input_dim:
        # No adaptation needed
        return state_dict
    
    logger.info(f"Adapting CNN-RNN model from {saved_input_dim} to {current_input_dim} input features")
    
    # Create a copy of the state dict to avoid modifying the original
    adapted_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    
    # Handle feature_weights tensor
    if 'feature_weights' in adapted_dict:
        old_weights = adapted_dict['feature_weights']
        new_weights = torch.zeros(current_input_dim, device=old_weights.device)
        
        # Copy existing weights
        min_dim = min(current_input_dim, saved_input_dim)
        new_weights[:min_dim] = old_weights[:min_dim]
        
        # Initialize new weights with mean of existing weights
        if current_input_dim > saved_input_dim:
            mean_weight = old_weights.mean().item()
            new_weights[saved_input_dim:] = mean_weight
        
        adapted_dict['feature_weights'] = new_weights
        logger.info(f"Adapted feature_weights from shape {old_weights.shape} to {new_weights.shape}")
    
    # Handle first convolutional layer weights
    conv_key = 'conv_layers.0.0.weight'
    if conv_key in adapted_dict:
        old_conv = adapted_dict[conv_key]
        kernel_size = old_conv.size(2)
        
        # Create new conv weights tensor [out_channels, in_channels, kernel_size]
        new_conv = torch.zeros(old_conv.size(0), current_input_dim, kernel_size, device=old_conv.device)
        
        # Copy existing weights
        min_dim = min(current_input_dim, saved_input_dim)
        new_conv[:, :min_dim, :] = old_conv[:, :min_dim, :]
        
        # Initialize new channels with mean of existing channels
        if current_input_dim > saved_input_dim:
            channel_means = old_conv.mean(dim=1, keepdim=True).expand(-1, current_input_dim - saved_input_dim, -1)
            new_conv[:, saved_input_dim:, :] = channel_means
        
        adapted_dict[conv_key] = new_conv
        logger.info(f"Adapted {conv_key} from shape {old_conv.shape} to {new_conv.shape}")
    
    return adapted_dict
