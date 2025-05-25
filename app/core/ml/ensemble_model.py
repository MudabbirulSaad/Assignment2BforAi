#!/usr/bin/env python3
"""
Ensemble Model Implementation for Traffic Prediction

This module implements an ensemble model that combines predictions from multiple
base models (LSTM, GRU, CNN-RNN) to improve prediction accuracy and robustness.
The ensemble approach leverages the strengths of different model architectures
to achieve better overall performance.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from app.core.logging import logger
from app.config.config import config
from app.core.ml.base_model import BaseModel
from app.core.ml.lstm_model import LSTMModel
from app.core.ml.gru_model import GRUModel
from app.core.ml.cnnrnn_model import CNNRNNModel


class EnsembleModel(BaseModel):
    """
    Ensemble Model for traffic prediction.
    
    This class implements an ensemble model that combines predictions from
    multiple base models (LSTM, GRU, CNN-RNN) using various ensemble strategies
    such as averaging, weighted averaging, or stacking.
    
    Attributes:
        base_models (Dict[str, BaseModel]): Dictionary of base models
        ensemble_method (str): Method for combining predictions
        model_weights (Dict[str, float]): Weights for each base model
        stacking_model (nn.Module): Model for stacking ensemble
    """
    
    def __init__(self, model_name: str = "Ensemble_Traffic_Predictor", 
                 config: Dict = None, base_models: Dict[str, BaseModel] = None,
                 ensemble_method: str = "weighted_average"):
        """
        Initialize the ensemble model.
        
        Args:
            model_name (str, optional): Name of the model. Defaults to "Ensemble_Traffic_Predictor".
            config (Dict, optional): Model configuration. Defaults to None.
            base_models (Dict[str, BaseModel], optional): Dictionary of base models. Defaults to None.
            ensemble_method (str, optional): Method for combining predictions. 
                                           Defaults to "weighted_average".
        """
        super(EnsembleModel, self).__init__(model_name, "Ensemble", config)
        
        # Default ensemble configuration
        self.ensemble_config = {
            'ensemble_method': ensemble_method,  # 'average', 'weighted_average', 'stacking'
            'model_weights': {},  # For weighted_average
            'stacking_hidden_dim': 32,  # For stacking
            'learning_rate': 0.001,
            'weight_decay': 1e-5
        }
        
        # Update with user config if provided
        if config and 'ensemble_config' in config:
            self.ensemble_config.update(config['ensemble_config'])
        
        # Initialize base models
        self.base_models = base_models or {}
        
        # Initialize model weights (equal by default)
        if not self.ensemble_config['model_weights'] and self.base_models:
            weight = 1.0 / len(self.base_models)
            self.ensemble_config['model_weights'] = {name: weight for name in self.base_models.keys()}
        
        # Initialize stacking model
        self.stacking_model = None
        
        logger.info(f"Initialized Ensemble model with {len(self.base_models)} base models "
                   f"using {self.ensemble_config['ensemble_method']} method")
    
    def add_base_model(self, model_name: str, model: BaseModel, weight: float = None) -> None:
        """
        Add a base model to the ensemble.
        
        Args:
            model_name (str): Name of the model
            model (BaseModel): Model instance
            weight (float, optional): Weight for the model in weighted averaging. 
                                    Defaults to None (equal weighting).
        """
        self.base_models[model_name] = model
        
        # Update weights if provided
        if weight is not None:
            self.ensemble_config['model_weights'][model_name] = weight
        # Otherwise, use equal weighting
        elif model_name not in self.ensemble_config['model_weights']:
            # Recalculate equal weights
            weight = 1.0 / len(self.base_models)
            self.ensemble_config['model_weights'] = {name: weight for name in self.base_models.keys()}
        
        logger.info(f"Added base model: {model_name} with weight: "
                   f"{self.ensemble_config['model_weights'].get(model_name, 'N/A')}")
    
    def build_model(self, input_dim: int, hidden_dim: int = None, output_dim: int = 1, 
                   num_layers: int = None) -> None:
        """
        Build the ensemble model architecture.
        
        For 'average' and 'weighted_average' methods, this doesn't create a new model.
        For 'stacking', this creates a meta-model that combines base model predictions.
        
        Args:
            input_dim (int): Number of input features (number of base models for stacking)
            hidden_dim (int, optional): Number of hidden units for stacking. Defaults to None.
            output_dim (int, optional): Number of output units. Defaults to 1.
            num_layers (int, optional): Not used, kept for compatibility. Defaults to None.
        """
        ensemble_method = self.ensemble_config['ensemble_method']
        
        if ensemble_method == 'stacking':
            # For stacking, we need a meta-model
            hidden_dim = hidden_dim or self.ensemble_config['stacking_hidden_dim']
            
            # Create a simple neural network for stacking
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Set up optimizer and loss function
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.ensemble_config['learning_rate'],
                weight_decay=self.ensemble_config['weight_decay']
            )
            
            self.criterion = nn.MSELoss()
            
            logger.info(f"Built stacking ensemble model with {input_dim} base models, "
                       f"{hidden_dim} hidden units, and {output_dim} output units")
        else:
            # For average and weighted_average, we don't need a separate model
            logger.info(f"Using {ensemble_method} ensemble method (no additional model needed)")
    
    def prepare_data_for_training(self, data: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Prepare data for training the ensemble model.
        
        For 'stacking', this prepares the meta-model for training.
        For other methods, this ensures all base models are prepared.
        
        Args:
            data (Dict[str, DataLoader]): Dictionary with train, val, and test data loaders
        """
        ensemble_method = self.ensemble_config['ensemble_method']
        
        # Ensure all base models are prepared
        for model_name, model in self.base_models.items():
            if not hasattr(model, 'model') or model.model is None:
                logger.info(f"Preparing base model: {model_name}")
                model.prepare_data_for_training(data)
        
        if ensemble_method == 'stacking':
            # For stacking, we need to build a meta-model
            # The input dimension is the number of base models
            input_dim = len(self.base_models)
            
            # Get a batch from the train loader to determine output shape
            for _, y_batch in data['train']:
                output_shape = y_batch.shape
                break
            
            # Extract output dimension
            output_dim = output_shape[-1]
            
            # Build the stacking model
            self.build_model(input_dim=input_dim, output_dim=output_dim)
            
            logger.info(f"Prepared stacking ensemble model with {input_dim} base models "
                       f"and {output_dim} output dimensions")
        else:
            logger.info(f"Using {ensemble_method} ensemble method (no additional preparation needed)")
    
    def _get_base_model_predictions(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from all base models for a given dataloader.
        
        Args:
            dataloader (DataLoader): Data loader
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Base model predictions and targets
        """
        all_predictions = []
        all_targets = []
        
        # Set all base models to evaluation mode
        for model in self.base_models.values():
            model.model.eval()
        
        # Get predictions from each base model
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                batch_predictions = []
                for model in self.base_models.values():
                    pred = model.model(X_batch)
                    batch_predictions.append(pred)
                
                # Stack predictions along a new dimension
                batch_predictions = torch.stack(batch_predictions, dim=1)
                
                all_predictions.append(batch_predictions)
                all_targets.append(y_batch)
        
        # Concatenate batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return all_predictions, all_targets
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader = None,
             epochs: int = 100, learning_rate: float = 0.001,
             patience: int = 10, verbose: bool = True) -> Dict:
        """
        Train the ensemble model.
        
        For 'stacking', this trains the meta-model.
        For other methods, this trains all base models if they haven't been trained.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            patience (int, optional): Early stopping patience. Defaults to 10.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            
        Returns:
            Dict: Training history
        """
        ensemble_method = self.ensemble_config['ensemble_method']
        
        # Train base models if they haven't been trained
        for model_name, model in self.base_models.items():
            if not model.history['train_loss']:
                logger.info(f"Training base model: {model_name}")
                model.train(train_loader, val_loader, epochs, learning_rate, patience, verbose)
        
        if ensemble_method == 'stacking':
            # For stacking, we need to train the meta-model
            if self.model is None:
                raise ValueError("Stacking model not built. Call prepare_data_for_training() first.")
            
            # Get base model predictions for training data
            logger.info("Getting base model predictions for stacking...")
            train_predictions, train_targets = self._get_base_model_predictions(train_loader)
            
            if val_loader is not None:
                val_predictions, val_targets = self._get_base_model_predictions(val_loader)
            else:
                val_predictions, val_targets = None, None
            
            # Train the stacking model
            logger.info("Training stacking model...")
            
            # Move model to device (CPU/GPU)
            self.model = self.model.to(self.device)
            
            # Initialize training variables
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Initialize history
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': {'mae': []},
                'val_metrics': {'mae': []}
            }
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                
                # Reshape predictions for stacking model input
                # From [batch_size, n_models, output_dim] to [batch_size, n_models * output_dim]
                batch_size, n_models, output_dim = train_predictions.shape
                X_train = train_predictions.view(batch_size, n_models)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(X_train)
                loss = self.criterion(output, train_targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                train_loss = loss.item()
                train_mae = torch.mean(torch.abs(output - train_targets)).item()
                
                # Validation phase
                if val_loader is not None:
                    self.model.eval()
                    
                    # Reshape predictions for stacking model input
                    batch_size, n_models, output_dim = val_predictions.shape
                    X_val = val_predictions.view(batch_size, n_models)
                    
                    with torch.no_grad():
                        val_output = self.model(X_val)
                        val_loss = self.criterion(val_output, val_targets).item()
                        val_mae = torch.mean(torch.abs(val_output - val_targets)).item()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    val_loss = None
                    val_mae = None
                
                # Store history
                self.history['train_loss'].append(train_loss)
                self.history['train_metrics']['mae'].append(train_mae)
                
                if val_loss is not None:
                    self.history['val_loss'].append(val_loss)
                    self.history['val_metrics']['mae'].append(val_mae)
                
                # Print progress
                if verbose and (epoch + 1) % 10 == 0:
                    val_str = f", val_loss: {val_loss:.4f}, val_mae: {val_mae:.4f}" if val_loss is not None else ""
                    logger.info(f"Epoch {epoch+1}/{epochs}, train_loss: {train_loss:.4f}, "
                               f"train_mae: {train_mae:.4f}{val_str}")
            
            # Load best model if early stopping occurred
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info("Loaded best model from early stopping")
            
            logger.info(f"Stacking model training completed after {len(self.history['train_loss'])} epochs")
            
        else:
            # For average and weighted_average, we don't need to train a separate model
            # Just aggregate the training history from base models
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': {'mae': []},
                'val_metrics': {'mae': []}
            }
            
            # Calculate weighted average of base model losses
            for epoch in range(min(len(model.history['train_loss']) for model in self.base_models.values())):
                train_loss = 0
                val_loss = 0
                train_mae = 0
                val_mae = 0
                
                for model_name, model in self.base_models.items():
                    weight = self.ensemble_config['model_weights'].get(model_name, 1.0 / len(self.base_models))
                    train_loss += weight * model.history['train_loss'][epoch]
                    
                    if 'val_loss' in model.history and len(model.history['val_loss']) > epoch:
                        val_loss += weight * model.history['val_loss'][epoch]
                    
                    if 'mae' in model.history['train_metrics'] and len(model.history['train_metrics']['mae']) > epoch:
                        train_mae += weight * model.history['train_metrics']['mae'][epoch]
                    
                    if 'val_metrics' in model.history and 'mae' in model.history['val_metrics'] and len(model.history['val_metrics']['mae']) > epoch:
                        val_mae += weight * model.history['val_metrics']['mae'][epoch]
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics']['mae'].append(train_mae)
                self.history['val_metrics']['mae'].append(val_mae)
            
            logger.info(f"Aggregated training history from {len(self.base_models)} base models")
        
        return self.history
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the ensemble model.
        
        Args:
            X (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predictions
        """
        ensemble_method = self.ensemble_config['ensemble_method']
        
        # Get predictions from all base models
        base_predictions = []
        for model_name, model in self.base_models.items():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        base_predictions = np.stack(base_predictions, axis=1)
        
        if ensemble_method == 'average':
            # Simple average
            predictions = np.mean(base_predictions, axis=1)
        
        elif ensemble_method == 'weighted_average':
            # Weighted average
            weights = np.array([self.ensemble_config['model_weights'].get(name, 1.0 / len(self.base_models)) 
                              for name in self.base_models.keys()])
            predictions = np.sum(base_predictions * weights.reshape(1, -1, 1), axis=1)
        
        elif ensemble_method == 'stacking':
            # Stacking
            if self.model is None:
                raise ValueError("Stacking model not built. Call prepare_data_for_training() first.")
            
            # Reshape predictions for stacking model input
            batch_size, n_models, output_dim = base_predictions.shape
            X_stack = torch.FloatTensor(base_predictions.reshape(batch_size, n_models)).to(self.device)
            
            # Get stacking model predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_stack).cpu().numpy()
        
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        return predictions
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Evaluate the ensemble model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            Dict: Evaluation metrics
        """
        ensemble_method = self.ensemble_config['ensemble_method']
        
        # Evaluate all base models if they haven't been evaluated
        for model_name, model in self.base_models.items():
            if not model.metrics:
                logger.info(f"Evaluating base model: {model_name}")
                model.evaluate(test_loader)
        
        # Initialize predictions and targets
        all_predictions = []
        all_targets = []
        
        # Get ensemble predictions
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.cpu().numpy()
            
            # Get ensemble predictions
            if ensemble_method == 'stacking':
                # For stacking, we need to get base model predictions first
                base_preds = []
                for model in self.base_models.values():
                    model.model.eval()
                    with torch.no_grad():
                        pred = model.model(X_batch).cpu().numpy()
                    base_preds.append(pred)
                
                # Stack predictions
                base_preds = np.stack(base_preds, axis=1)
                
                # Reshape for stacking model
                batch_size, n_models, output_dim = base_preds.shape
                X_stack = torch.FloatTensor(base_preds.reshape(batch_size, n_models)).to(self.device)
                
                # Get stacking model predictions
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(X_stack).cpu().numpy()
            
            else:
                # For average and weighted_average, get predictions from all base models
                base_preds = []
                for model_name, model in self.base_models.items():
                    model.model.eval()
                    with torch.no_grad():
                        pred = model.model(X_batch).cpu().numpy()
                    base_preds.append(pred)
                
                # Stack predictions
                base_preds = np.stack(base_preds, axis=1)
                
                if ensemble_method == 'average':
                    # Simple average
                    y_pred = np.mean(base_preds, axis=1)
                
                elif ensemble_method == 'weighted_average':
                    # Weighted average
                    weights = np.array([self.ensemble_config['model_weights'].get(name, 1.0 / len(self.base_models)) 
                                      for name in self.base_models.keys()])
                    y_pred = np.sum(base_preds * weights.reshape(1, -1, 1), axis=1)
            
            # Store predictions and targets
            all_predictions.append(y_pred)
            all_targets.append(y_batch)
        
        # Concatenate batches
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # Calculate MAPE
        mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-10))) * 100
        
        # Store metrics
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Ensemble model evaluation results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.4f}%")
        
        return self.metrics
    
    def compare_with_base_models(self) -> Dict:
        """
        Compare ensemble model performance with base models.
        
        Returns:
            Dict: Comparison results
        """
        if not self.metrics:
            raise ValueError("Ensemble model has not been evaluated yet. Call evaluate() first.")
        
        # Ensure all base models have been evaluated
        for model_name, model in self.base_models.items():
            if not model.metrics:
                raise ValueError(f"Base model {model_name} has not been evaluated yet.")
        
        # Create comparison dictionary
        comparison = {
            'models': {},
            'best_model': {},
            'improvement': {}
        }
        
        # Add ensemble model metrics
        comparison['models']['Ensemble'] = self.metrics
        
        # Add base model metrics
        for model_name, model in self.base_models.items():
            comparison['models'][model_name] = model.metrics
        
        # Find best model for each metric
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            # For metrics where lower is better (RMSE, MAE, MAPE)
            if metric in ['rmse', 'mae', 'mape']:
                best_model = min(comparison['models'].items(), key=lambda x: x[1][metric])[0]
                best_value = min(model_metrics[metric] for model_metrics in comparison['models'].values())
            # For metrics where higher is better (R2)
            else:
                best_model = max(comparison['models'].items(), key=lambda x: x[1][metric])[0]
                best_value = max(model_metrics[metric] for model_metrics in comparison['models'].values())
            
            comparison['best_model'][metric] = {
                'model': best_model,
                'value': best_value
            }
            
            # Calculate improvement over base models
            ensemble_value = self.metrics[metric]
            base_values = [model.metrics[metric] for model in self.base_models.values()]
            
            if metric in ['rmse', 'mae', 'mape']:
                # For metrics where lower is better
                avg_base = np.mean(base_values)
                improvement = (avg_base - ensemble_value) / avg_base * 100
            else:
                # For metrics where higher is better
                avg_base = np.mean(base_values)
                improvement = (ensemble_value - avg_base) / avg_base * 100
            
            comparison['improvement'][metric] = improvement
        
        # Calculate overall best model
        model_scores = {model_name: 0 for model_name in comparison['models'].keys()}
        
        for metric, best in comparison['best_model'].items():
            model_scores[best['model']] += 1
        
        comparison['overall_best_model'] = max(model_scores.items(), key=lambda x: x[1])[0]
        
        # Create summary
        if comparison['overall_best_model'] == 'Ensemble':
            comparison['summary'] = "The ensemble model outperforms all base models overall."
        else:
            best_model = comparison['overall_best_model']
            comparison['summary'] = f"The {best_model} model outperforms the ensemble model overall."
        
        logger.info(f"Model comparison results: {comparison['summary']}")
        
        return comparison
    
    def plot_comparison(self, save_path: str = None) -> None:
        """
        Plot comparison of ensemble model with base models.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if not self.metrics:
            raise ValueError("Ensemble model has not been evaluated yet. Call evaluate() first.")
        
        # Ensure all base models have been evaluated
        for model_name, model in self.base_models.items():
            if not model.metrics:
                raise ValueError(f"Base model {model_name} has not been evaluated yet.")
        
        # Create comparison dictionary
        models_metrics = {
            'Ensemble': self.metrics
        }
        
        # Add base model metrics
        for model_name, model in self.base_models.items():
            models_metrics[model_name] = model.metrics
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['rmse', 'mae', 'r2', 'mape']
        titles = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            # Get values for each model
            values = [metrics[metric] for metrics in models_metrics.values()]
            
            # Create bar chart
            bars = axes[i].bar(models_metrics.keys(), values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', rotation=0)
            
            axes[i].set_title(title)
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Highlight the best model
            if metric in ['rmse', 'mae', 'mape']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_color('green')
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_ensemble_from_trainer(trainer, ensemble_method: str = "weighted_average") -> EnsembleModel:
    """
    Create an ensemble model from a ModelTrainer instance.
    
    Args:
        trainer: ModelTrainer instance
        ensemble_method (str, optional): Ensemble method. Defaults to "weighted_average".
        
    Returns:
        EnsembleModel: Ensemble model
    """
    # Create ensemble model
    ensemble = EnsembleModel(
        model_name=f"Ensemble_{ensemble_method.capitalize()}",
        ensemble_method=ensemble_method
    )
    
    # Add base models
    for model_name, model in trainer.models.items():
        # If model has been evaluated, use its performance for weighting
        if model.metrics and 'r2' in model.metrics:
            # Use R² as weight (higher is better)
            weight = max(0.1, model.metrics['r2'])  # Ensure positive weight
        else:
            weight = None  # Use default equal weighting
        
        ensemble.add_base_model(model_name, model, weight)
    
    return ensemble
