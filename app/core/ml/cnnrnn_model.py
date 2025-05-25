#!/usr/bin/env python3
"""
CNN-RNN Hybrid Model Implementation for Traffic Prediction

This module implements a CNN-RNN hybrid model for traffic prediction in the TBRGS project.
The model combines convolutional layers for feature extraction with recurrent layers
for temporal modeling, providing advantages over pure RNN approaches.

Key advantages of CNN-RNN hybrid models for traffic prediction:
1. CNNs extract spatial and local temporal patterns from the input sequence
2. RNNs capture long-term temporal dependencies
3. Reduced parameter count compared to pure RNN models
4. Better handling of noisy traffic data through CNN's feature extraction
5. Improved gradient flow during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union, Any
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import optuna

from app.core.logging import logger
from app.core.ml.base_model import BaseModel


class CNNRNNNetwork(nn.Module):
    """
    CNN-RNN Hybrid Neural Network for time series prediction.
    
    This architecture first applies 1D convolutions to extract features from the input sequence,
    then passes these features through a recurrent layer (GRU or LSTM), and finally
    uses a fully connected layer to produce the output.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in the RNN
        output_dim (int): Number of output units
        cnn_layers (int): Number of CNN layers
        rnn_layers (int): Number of RNN layers
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout rate
        rnn_type (str): Type of RNN ('lstm' or 'gru')
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 cnn_layers: int = 2, rnn_layers: int = 1, kernel_size: int = 3,
                 dropout: float = 0.2, rnn_type: str = 'gru', use_attention: bool = False):
        """
        Initialize the CNN-RNN hybrid network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in the RNN
            output_dim (int): Number of output units
            cnn_layers (int, optional): Number of CNN layers. Defaults to 2.
            rnn_layers (int, optional): Number of RNN layers. Defaults to 1.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            rnn_type (str, optional): Type of RNN ('lstm' or 'gru'). Defaults to 'gru'.
            use_attention (bool, optional): Whether to use attention mechanism. Defaults to False.
        """
        super(CNNRNNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cnn_layers = cnn_layers
        self.rnn_layers = rnn_layers
        self.kernel_size = kernel_size
        self.rnn_type = rnn_type.lower()
        self.use_attention = use_attention
        
        # Validate RNN type
        if self.rnn_type not in ['lstm', 'gru']:
            raise ValueError("RNN type must be 'lstm' or 'gru'")
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Additional convolutional layers
        for _ in range(1, cnn_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # RNN layer (LSTM or GRU)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0
            )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Feature importance layer (for analysis)
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply feature importance weights
        x = x * self.feature_weights
        
        # Transpose for CNN: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Transpose back for RNN: (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Apply RNN layer
        if self.rnn_type == 'lstm':
            rnn_out, (_, _) = self.rnn(x)
        else:  # GRU
            rnn_out, _ = self.rnn(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(rnn_out).squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(1)
            
            # Apply attention weights
            context = torch.bmm(attention_weights, rnn_out).squeeze(1)
        else:
            # Use the last output from the RNN
            context = rnn_out[:, -1, :]
        
        # Apply output layer
        out = self.fc(context)
        
        return out
    
    def get_feature_importance(self) -> torch.Tensor:
        """
        Get the learned feature importance weights.
        
        Returns:
            torch.Tensor: Feature importance weights
        """
        return F.softmax(self.feature_weights, dim=0).detach()


class CNNRNNModel(BaseModel):
    """
    CNN-RNN Hybrid Model for traffic prediction.
    
    This class implements a CNN-RNN hybrid model for traffic prediction,
    inheriting from the BaseModel class. The hybrid approach combines
    the strengths of CNNs for feature extraction and RNNs for temporal modeling.
    
    The CNN-RNN hybrid model offers several advantages for traffic prediction:
    1. Effective feature extraction from CNN layers to capture spatial patterns
    2. Temporal modeling from RNN layers to capture time dependencies
    3. Reduced parameter count compared to pure RNN models
    4. Optional attention mechanism to focus on important time steps
    5. Feature importance analysis for interpretability
    """
    
    def __init__(self, model_name: str = "CNNRNN_Traffic_Predictor", config: Dict = None):
        """
        Initialize the CNN-RNN hybrid model.
        
        Args:
            model_name (str, optional): Name of the model. Defaults to "CNNRNN_Traffic_Predictor".
            config (Dict, optional): Model configuration. Defaults to None.
        """
        super(CNNRNNModel, self).__init__(model_name, "CNN-RNN", config)
        
        # Default CNN-RNN configuration with improved settings
        self.cnnrnn_config = {
            'hidden_dim': 64,           # Reduced from 128 to prevent overfitting
            'cnn_layers': 2,            # Keep 2 CNN layers for feature extraction
            'rnn_layers': 1,            # Single RNN layer for simplicity
            'kernel_size': 3,           # Standard kernel size for temporal patterns
            'dropout': 0.5,             # Increased from 0.2 for stronger regularization
            'rnn_type': 'gru',          # GRU performs well for this task
            'use_attention': False,     # Keep attention disabled for simplicity
            'learning_rate': 0.0001,    # Reduced from 0.001 for more stable training
            'weight_decay': 5e-4        # Increased from 1e-5 for better regularization
        }
        
        # Update with user config if provided
        if config and 'cnnrnn_config' in config:
            self.cnnrnn_config.update(config['cnnrnn_config'])
        
        # Store feature importance
        self.feature_importance = None
    
    def build_model(self, input_dim: int, hidden_dim: int = None, output_dim: int = 1, 
                   num_layers: int = None) -> None:
        """
        Build the CNN-RNN hybrid model architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int, optional): Number of hidden units. Defaults to None.
            output_dim (int, optional): Number of output units. Defaults to 1.
            num_layers (int, optional): Not used, kept for compatibility. Defaults to None.
        """
        # Use provided parameters or defaults from config
        hidden_dim = hidden_dim or self.cnnrnn_config['hidden_dim']
        
        # Create the CNN-RNN network
        self.model = CNNRNNNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            cnn_layers=self.cnnrnn_config['cnn_layers'],
            rnn_layers=self.cnnrnn_config['rnn_layers'],
            kernel_size=self.cnnrnn_config['kernel_size'],
            dropout=self.cnnrnn_config['dropout'],
            rnn_type=self.cnnrnn_config['rnn_type'],
            use_attention=self.cnnrnn_config['use_attention']
        )
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cnnrnn_config['learning_rate'],
            weight_decay=self.cnnrnn_config['weight_decay']
        )
        
        self.criterion = nn.MSELoss()
        
        # Log model architecture
        attention_str = "with attention" if self.cnnrnn_config['use_attention'] else "without attention"
        logger.info(f"Built CNN-{self.cnnrnn_config['rnn_type'].upper()} hybrid model {attention_str} "
                   f"with {input_dim} input features, {hidden_dim} hidden units, "
                   f"{self.cnnrnn_config['cnn_layers']} CNN layers, "
                   f"{self.cnnrnn_config['rnn_layers']} RNN layers, "
                   f"and {output_dim} output units")
        
        # If we have a temporary state dict from loading, apply it now
        if hasattr(self, '_temp_state_dict'):
            self.model.load_state_dict(self._temp_state_dict)
            delattr(self, '_temp_state_dict')
            logger.info("Loaded model weights from saved state")
    
    def prepare_data_for_training(self, data: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Prepare data for training the CNN-RNN hybrid model.
        
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
    
    def analyze_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Analyze the importance of input features.
        
        This method extracts feature importance weights from the model
        and returns them as a dictionary.
        
        Args:
            feature_names (List[str], optional): Names of input features. 
                                               Defaults to None (uses indices).
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get feature importance weights
        importance = self.model.get_feature_importance().cpu().numpy()
        self.feature_importance = importance
        
        # Create dictionary with feature names
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        importance_dict = {name: float(score) for name, score in zip(feature_names, importance)}
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def plot_feature_importance(self, feature_names: List[str] = None, 
                               top_n: int = None, save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str], optional): Names of input features. 
                                               Defaults to None (uses indices).
            top_n (int, optional): Number of top features to plot. Defaults to None (all).
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if self.feature_importance is None:
            self.analyze_feature_importance(feature_names)
        
        importance_dict = self.analyze_feature_importance(feature_names)
        
        # Limit to top N features if specified
        if top_n is not None:
            importance_dict = dict(list(importance_dict.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(list(importance_dict.keys()), list(importance_dict.values()))
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def tune_hyperparameters(self, train_loader, val_loader, 
                            n_trials: int = 20, timeout: int = 3600) -> Dict:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_trials (int, optional): Number of trials. Defaults to 20.
            timeout (int, optional): Timeout in seconds. Defaults to 3600 (1 hour).
            
        Returns:
            Dict: Best hyperparameters
        """
        def objective(trial):
            # Define hyperparameters to tune
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                'cnn_layers': trial.suggest_int('cnn_layers', 1, 3),
                'rnn_layers': trial.suggest_int('rnn_layers', 1, 2),
                'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'rnn_type': trial.suggest_categorical('rnn_type', ['lstm', 'gru']),
                'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
            }
            
            # Update model configuration
            for key, value in params.items():
                self.cnnrnn_config[key] = value
            
            # Get input dimension from train_loader
            for X_batch, _ in train_loader:
                input_shape = X_batch.shape
                break
            
            _, _, n_features = input_shape
            
            # Rebuild model with new hyperparameters
            self.build_model(input_dim=n_features)
            
            # Train for a few epochs
            epochs = 10
            patience = 3
            
            # Train the model
            self.train(train_loader, val_loader, epochs=epochs, 
                      learning_rate=params['learning_rate'], 
                      patience=patience, verbose=False)
            
            # Evaluate on validation set
            val_loss, _ = self._validate(val_loader)
            
            return val_loss
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Update model configuration with best parameters
        for key, value in best_params.items():
            self.cnnrnn_config[key] = value
        
        # Rebuild model with best parameters
        for X_batch, _ in train_loader:
            input_shape = X_batch.shape
            break
        
        _, _, n_features = input_shape
        self.build_model(input_dim=n_features)
        
        return best_params
    
    def train_step(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single training step (forward and backward pass).
        
        Args:
            data (torch.Tensor): Input data batch
            target (torch.Tensor): Target data batch
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss and output tensors
        """
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, output
    
    def validation_step(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single validation step (forward pass only).
        
        Args:
            data (torch.Tensor): Input data batch
            target (torch.Tensor): Target data batch
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss and output tensors
        """
        with torch.no_grad():
            output = self.model(data)
            loss = self.criterion(output, target)
        
        return loss, output
    
    def predict_step(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform a single prediction step.
        
        Args:
            data (torch.Tensor): Input data batch
            
        Returns:
            torch.Tensor: Output tensor
        """
        with torch.no_grad():
            output = self.model(data)
        
        return output
    
    def compare_with_models(self, models_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Compare CNN-RNN performance with other models.
        
        Args:
            models_metrics (Dict[str, Dict[str, float]]): Metrics from other models
                                                        {model_name: {metric: value}}
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if not self.metrics:
            raise ValueError("CNN-RNN model has not been evaluated yet. Call evaluate() first.")
        
        comparison = {'model_ranking': {}, 'metric_comparison': {}}
        
        # Add CNN-RNN metrics to the comparison
        all_models_metrics = {**models_metrics, 'CNN-RNN': self.metrics}
        
        # For each metric, rank the models
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            if all(metric in model_metrics for model_metrics in all_models_metrics.values()):
                # For metrics where lower is better (RMSE, MAE, MAPE)
                if metric in ['rmse', 'mae', 'mape']:
                    ranked_models = sorted(all_models_metrics.keys(), 
                                         key=lambda model: all_models_metrics[model][metric])
                # For metrics where higher is better (R2)
                else:
                    ranked_models = sorted(all_models_metrics.keys(), 
                                         key=lambda model: all_models_metrics[model][metric],
                                         reverse=True)
                
                comparison['model_ranking'][metric] = ranked_models
                
                # Calculate percentage differences relative to the best model
                best_model = ranked_models[0]
                best_value = all_models_metrics[best_model][metric]
                
                comparison['metric_comparison'][metric] = {}
                for model, metrics in all_models_metrics.items():
                    if metric in ['rmse', 'mae', 'mape']:
                        # For metrics where lower is better
                        pct_diff = (metrics[metric] - best_value) / best_value * 100
                    else:
                        # For metrics where higher is better
                        pct_diff = (best_value - metrics[metric]) / best_value * 100
                    
                    comparison['metric_comparison'][metric][model] = {
                        'value': metrics[metric],
                        'pct_diff_from_best': pct_diff
                    }
        
        # Calculate overall ranking based on average rank across metrics
        avg_ranks = {}
        for model in all_models_metrics.keys():
            ranks = [comparison['model_ranking'][metric].index(model) + 1 
                    for metric in comparison['model_ranking'].keys()]
            avg_ranks[model] = sum(ranks) / len(ranks)
        
        # Sort models by average rank
        comparison['overall_ranking'] = sorted(avg_ranks.keys(), key=lambda model: avg_ranks[model])
        comparison['average_ranks'] = avg_ranks
        
        # Add summary
        cnn_rnn_overall_rank = comparison['overall_ranking'].index('CNN-RNN') + 1
        total_models = len(comparison['overall_ranking'])
        
        if cnn_rnn_overall_rank == 1:
            comparison['summary'] = "CNN-RNN hybrid model outperforms all other models overall."
        else:
            best_model = comparison['overall_ranking'][0]
            comparison['summary'] = f"CNN-RNN hybrid model ranks {cnn_rnn_overall_rank} out of {total_models} models. " \
                                  f"The best performing model is {best_model}."
        
        logger.info(f"Model comparison results: {comparison['summary']}")
        return comparison
        
    def predict_sequence(self, initial_sequence: np.ndarray, steps: int = 1, 
                         scaler_X: Any = None, scaler_y: Any = None) -> np.ndarray:
        """
        Generate a sequence of predictions using the model.
        
        This method uses an initial sequence to predict the next value,
        then incorporates that prediction into the sequence to predict
        the next value, and so on.
        
        Args:
            initial_sequence (np.ndarray): Initial input sequence
            steps (int, optional): Number of steps to predict. Defaults to 1.
            scaler_X (Any, optional): Feature scaler. Defaults to None.
            scaler_y (Any, optional): Target scaler. Defaults to None.
            
        Returns:
            np.ndarray: Sequence of predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Use provided scalers or instance scalers
        scaler_X = scaler_X or self.scaler_X
        scaler_y = scaler_y or self.scaler_y
        
        # Move model to evaluation mode
        self.model.eval()
        
        # Initialize sequence
        sequence = initial_sequence.copy()
        predictions = []
        
        # Generate predictions
        for _ in range(steps):
            # Prepare input
            X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                y_pred = self.model(X)
            
            # Convert to numpy
            y_pred_np = y_pred.cpu().numpy().reshape(1, -1)
            
            # Inverse transform if scaler provided
            if scaler_y is not None:
                y_pred_np = scaler_y.inverse_transform(y_pred_np)
            
            # Store prediction
            predictions.append(y_pred_np[0])
            
            # Update sequence for next prediction
            # Remove first element and add prediction at the end
            sequence = np.vstack([sequence[1:], y_pred_np])
        
        return np.array(predictions)
