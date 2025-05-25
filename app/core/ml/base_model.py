#!/usr/bin/env python3
"""
Base Model Class for Traffic Prediction Models

This module defines an abstract base class for all machine learning models used in the TBRGS project.
It provides common functionality for training, evaluation, prediction, and model management.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn imports for metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import project-specific modules
from app.core.logging import logger
from app.config.config import config


class BaseModel(ABC):
    """
    Abstract base class for all traffic prediction models.
    
    This class defines the interface and common functionality that all models must implement.
    It handles data preprocessing, model training, evaluation, prediction, and model persistence.
    
    Attributes:
        model_name (str): Name of the model
        model_type (str): Type of model (LSTM, GRU, CNN-RNN, etc.)
        model_path (str): Path to save/load the model
        config (dict): Model configuration parameters
        model (nn.Module): The actual PyTorch model instance
        optimizer (optim.Optimizer): PyTorch optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to run the model on (CPU or CUDA)
        history (Dict): Training history
        scaler_X (Any): Feature scaler
        scaler_y (Any): Target scaler
        metrics (Dict): Performance metrics
        hyperparameters (Dict): Model hyperparameters
        feature_columns (List[str]): List of feature column names
        target_columns (List[str]): List of target column names
    """
    
    def __init__(self, model_name: str, model_type: str, config: Dict = None):
        """
        Initialize the base model with name, type and configuration.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model (LSTM, GRU, CNN-RNN, etc.)
            config (Dict, optional): Model configuration parameters. Defaults to None.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.config = config or {}
        
        # Set up model directory
        # Get the model directory from config or use a default
        default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models')
        
        # Try to get model_dir from config, with fallbacks
        if isinstance(config, dict):
            model_dir = self.config.get('model_dir', default_model_dir)
        elif config is not None and hasattr(config, 'model_dir'):
            model_dir = config.model_dir
        else:
            model_dir = default_model_dir
            
        self.model_path = os.path.join(
            model_dir,
            f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.model_path, exist_ok=True)
        
        # Set device (CPU or CUDA)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model attributes
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        self.hyperparameters = {}
        self.feature_columns = []
        self.target_columns = []
        
        logger.info(f"Initialized {self.model_type} model: {self.model_name} on {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"CUDA is available. Using GPU acceleration.")
        else:
            logger.info(f"CUDA is not available. Using CPU.")
    
    
    @abstractmethod
    def build_model(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        """
        Build the model architecture.
        
        This method must be implemented by all subclasses to define the specific model architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            output_dim (int): Number of output units
            num_layers (int, optional): Number of layers. Defaults to 1.
        """
        pass
    
    def preprocess_data(self, data: pd.DataFrame, 
                       feature_columns: List[str], 
                       target_columns: List[str],
                       sequence_length: int = 24,
                       batch_size: int = 32,
                       train_mode: bool = True,
                       scaler_type: str = 'standard') -> Tuple:
        """
        Preprocess data for model training or prediction.
        
        Args:
            data (pd.DataFrame): Input data
            feature_columns (List[str]): List of feature column names
            target_columns (List[str]): List of target column names
            sequence_length (int, optional): Length of input sequences. Defaults to 24.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
            train_mode (bool, optional): Whether in training mode. Defaults to True.
            scaler_type (str, optional): Type of scaler ('standard', 'minmax'). Defaults to 'standard'.
            
        Returns:
            Tuple: DataLoader for model training/prediction and data shapes
        """
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        # Extract features and targets
        X = data[feature_columns].values
        y = data[target_columns].values if target_columns else None
        
        # Scale features
        if train_mode:
            if scaler_type.lower() == 'standard':
                self.scaler_X = StandardScaler()
            else:
                self.scaler_X = MinMaxScaler()
            
            X_scaled = self.scaler_X.fit_transform(X)
            
            if y is not None:
                if scaler_type.lower() == 'standard':
                    self.scaler_y = StandardScaler()
                else:
                    self.scaler_y = MinMaxScaler()
                
                y_scaled = self.scaler_y.fit_transform(y)
            else:
                y_scaled = None
        else:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y) if y is not None else None
        
        # Create sequences for time series models
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, sequence_length)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq) if y_seq is not None else None
        
        # Create TensorDataset and DataLoader
        if y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train_mode,  # Shuffle only in training mode
            drop_last=False
        )
        
        logger.info(f"Preprocessed data: X shape={X_seq.shape}, " + 
                   (f"y shape={y_seq.shape}" if y_seq is not None else "no targets"))
        
        return dataloader, (X_seq.shape, y_seq.shape if y_seq is not None else None)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Target data
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            if y is not None:
                y_seq.append(y[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
             epochs: int = 100, learning_rate: float = 0.001,
             patience: int = 10, verbose: bool = True, clip_grad_value: float = None) -> Dict:
        """
        Train the model using PyTorch.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            patience (int, optional): Early stopping patience. Defaults to 10.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            clip_grad_value (float, optional): Value for gradient clipping. Defaults to None.
            
        Returns:
            Dict: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Move model to device (CPU/GPU)
        self.model = self.model.to(self.device)
        
        # Set up optimizer and loss function if not already set
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.criterion is None:
            self.criterion = nn.MSELoss()
            
        # Set up learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',           # Reduce LR when val_loss stops decreasing
            factor=0.5,          # Multiply LR by this factor
            patience=5,          # Number of epochs with no improvement
            min_lr=1e-6          # Lower bound on the learning rate
        )
        
        # Log scheduler creation
        logger.info(f"Created learning rate scheduler with patience={5}, factor={0.5}, min_lr={1e-6}")
        
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
        start_time = time.time()
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            train_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass and optimize
                loss.backward()
                
                # Apply gradient clipping if specified
                if clip_grad_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_value)
                    
                self.optimizer.step()
                
                # Calculate metrics
                train_loss += loss.item() * data.size(0)
                train_mae += torch.mean(torch.abs(output - target)).item() * data.size(0)
                train_samples += data.size(0)
            
            # Calculate average training metrics
            train_loss /= train_samples
            train_mae /= train_samples
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_mae = self._validate(val_loader)
                
                # Update learning rate scheduler based on validation loss
                scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Early stopping check
                # Update best validation loss and reset patience counter
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                    
                    # Save checkpoint for best model
                    self.save_checkpoint(epoch, train_loss, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Store metrics in history
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics']['mae'].append(train_mae)
            
            if val_loader is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics']['mae'].append(val_mae)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                val_str = f", Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}{val_str}")
        
        # Load best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save hyperparameters
        self.hyperparameters = {
            'epochs': epoch + 1,  # Actual epochs trained
            'learning_rate': learning_rate,
            'patience': patience,
            'training_time': training_time,
            'final_train_loss': train_loss
        }
        
        if val_loader is not None:
            self.hyperparameters['final_val_loss'] = val_loss
        
        # Save the final model checkpoint
        self.save_checkpoint(epoch, train_loss, val_loss if val_loader is not None else None)
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {train_loss:.4f}")
        if val_loader is not None:
            logger.info(f"Final validation loss: {val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float = None) -> None:
        """
        Save a model checkpoint.
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float, optional): Validation loss. Defaults to None.
        """
        # Use the fixed checkpoints directory
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint filename
        checkpoint_filename = f"{self.model_name}_epoch_{epoch+1}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'hyperparameters': self.hyperparameters
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save a copy as latest.pt
            latest_path = os.path.join(checkpoint_dir, f"{self.model_name}_latest.pt")
            torch.save(checkpoint, latest_path)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on a validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[float, float]: Validation loss and MAE
        """
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate metrics
                val_loss += loss.item() * data.size(0)
                val_mae += torch.mean(torch.abs(output - target)).item() * data.size(0)
                val_samples += data.size(0)
        
        # Calculate average validation metrics
        val_loss /= val_samples
        val_mae /= val_samples
        
        return val_loss, val_mae
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize variables for predictions and targets
        all_targets = []
        all_predictions = []
        test_loss = 0.0
        test_samples = 0
        
        # Make predictions without gradient calculation
        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Accumulate loss
                test_loss += loss.item() * data.size(0)
                test_samples += data.size(0)
                
                # Move predictions and targets to CPU and convert to numpy
                all_targets.append(target.cpu().numpy())
                all_predictions.append(output.cpu().numpy())
        
        # Calculate average loss
        test_loss /= test_samples
        
        # Concatenate all batches
        y_test = np.vstack(all_targets)
        y_pred = np.vstack(all_predictions)
        
        # Inverse transform if scalers are available
        if self.scaler_y is not None:
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred
        
        # Calculate metrics
        self.metrics = {
            'loss': test_loss,
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'r2': r2_score(y_test_orig, y_pred_orig),
            'mape': self._mean_absolute_percentage_error(y_test_orig, y_pred_orig)
        }
        
        logger.info(f"Model evaluation results:")
        for metric, value in self.metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Save metrics to file
        self._save_metrics()
        
        return self.metrics
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure model is on the correct device
        device = next(self.model.parameters()).device
        
        # Convert numpy array to PyTorch tensor and move to the same device as the model
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions without gradient calculation
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Convert predictions back to numpy array
        return predictions.cpu().numpy()
    
    def predict_site(self, site_data: pd.DataFrame, site_id: str, 
                   feature_columns: List[str], target_column: str,
                   sequence_length: int = 24, batch_size: int = 32) -> pd.DataFrame:
        """
        Make predictions for a specific site.
        
        Args:
            site_data (pd.DataFrame): Data for the site
            site_id (str): ID of the site
            feature_columns (List[str]): Feature columns
            target_column (str): Target column
            sequence_length (int, optional): Sequence length. Defaults to 24.
            batch_size (int, optional): Batch size. Defaults to 32.
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Filter data for the site
        site_df = site_data[site_data['SCATS_ID'] == site_id].copy()
        
        # Preprocess data
        dataloader, shapes = self.preprocess_data(
            site_df, feature_columns, [target_column], 
            sequence_length=sequence_length,
            batch_size=batch_size,
            train_mode=False
        )
        
        # Initialize arrays for predictions
        all_predictions = []
        all_targets = []
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            for data, target in dataloader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Move predictions to CPU and convert to numpy
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.numpy())
        
        # Concatenate all batches
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_targets)
        
        # Inverse transform predictions
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(y_true)
        
        # Get timestamps (skip sequence_length entries at the beginning)
        timestamps = site_df.iloc[sequence_length:]['DateTime'].values[:len(y_pred)]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'DateTime': timestamps,
            'SCATS_ID': site_id,
            'Actual': y_true.flatten(),
            'Predicted': y_pred.flatten()
        })
        
        return results
    
    def save(self, save_path: str = None) -> str:
        """
        Save the model and its components.
        
        Args:
            save_path (str, optional): Path to save the model. Defaults to None.
            
        Returns:
            str: Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        save_path = save_path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save PyTorch model
        model_file = os.path.join(save_path, 'model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'hyperparameters': self.hyperparameters
        }, model_file)
        
        # Save scalers
        if self.scaler_X is not None:
            with open(os.path.join(save_path, 'scaler_X.pkl'), 'wb') as f:
                pickle.dump(self.scaler_X, f)
        
        if self.scaler_y is not None:
            with open(os.path.join(save_path, 'scaler_y.pkl'), 'wb') as f:
                pickle.dump(self.scaler_y, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save training history
        if self.history:
            history_dict = {}
            
            # Convert lists to regular Python lists for JSON serialization
            for k, v in self.history.items():
                if isinstance(v, (list, np.ndarray)):
                    history_dict[k] = v.tolist() if isinstance(v, np.ndarray) else v
                elif isinstance(v, dict):
                    history_dict[k] = {}
                    for sub_k, sub_v in v.items():
                        history_dict[k][sub_k] = sub_v.tolist() if isinstance(sub_v, np.ndarray) else sub_v
            
            with open(os.path.join(save_path, 'history.json'), 'w') as f:
                json.dump(history_dict, f, indent=4)
        
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    @classmethod
    def load(cls, load_path: str) -> 'BaseModel':
        """
        Load a saved model.
        
        Args:
            load_path (str): Path to the saved model
            
        Returns:
            BaseModel: Loaded model instance
        """
        # Load metadata
        with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(metadata['model_name'], metadata['model_type'])
        instance.feature_columns = metadata['feature_columns']
        instance.target_columns = metadata['target_columns']
        instance.hyperparameters = metadata['hyperparameters']
        instance.metrics = metadata['metrics']
        
        # The model needs to be built before loading weights
        # This will be handled by the subclass implementation
        
        # Load PyTorch model
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'), map_location=instance.device)
        
        # The model state_dict will be loaded by the subclass after building the model
        # We'll store it temporarily
        instance._temp_state_dict = checkpoint['model_state_dict']
        
        # Load scalers
        scaler_X_path = os.path.join(load_path, 'scaler_X.pkl')
        if os.path.exists(scaler_X_path):
            with open(scaler_X_path, 'rb') as f:
                instance.scaler_X = pickle.load(f)
        
        scaler_y_path = os.path.join(load_path, 'scaler_y.pkl')
        if os.path.exists(scaler_y_path):
            with open(scaler_y_path, 'rb') as f:
                instance.scaler_y = pickle.load(f)
        
        # Load history
        history_path = os.path.join(load_path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                instance.history = json.load(f)
        
        logger.info(f"Model loaded from {load_path}")
        return instance
    
    def _save_metrics(self) -> None:
        """
        Save evaluation metrics to a file.
        """
        metrics_file = os.path.join(self.model_path, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_history(self, save_path: str = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if not self.history:
            logger.warning("No training history available.")
            return
        
        save_path = save_path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot other metrics if available
        if 'mae' in self.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history['mae'], label='Training MAE')
            if 'val_mae' in self.history:
                plt.plot(self.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")
    
    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, 
                       timestamps: np.ndarray = None, save_path: str = None) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            timestamps (np.ndarray, optional): Timestamps for x-axis. Defaults to None.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        save_path = save_path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if timestamps is not None:
            plt.plot(timestamps, actual, label='Actual', marker='o', markersize=3)
            plt.plot(timestamps, predicted, label='Predicted', marker='x', markersize=3)
            plt.xlabel('Time')
        else:
            plt.plot(actual, label='Actual', marker='o', markersize=3)
            plt.plot(predicted, label='Predicted', marker='x', markersize=3)
            plt.xlabel('Time Step')
        
        plt.title(f'{self.model_name} - Actual vs Predicted')
        plt.ylabel('Traffic Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text annotation
        if self.metrics:
            metrics_text = f"RMSE: {self.metrics['rmse']:.2f}\nMAE: {self.metrics['mae']:.2f}\nR²: {self.metrics['r2']:.2f}"
            plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'predictions.png'))
        plt.close()
        
        logger.info(f"Predictions plot saved to {save_path}")
    
    def summary(self) -> None:
        """
        Print a summary of the model.
        """
        print(f"\n{'='*50}")
        print(f"Model: {self.model_name} ({self.model_type})")
        print(f"{'='*50}")
        
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet.")
        
        if self.hyperparameters:
            print(f"\nHyperparameters:")
            for param, value in self.hyperparameters.items():
                print(f"  {param}: {value}")
        
        if self.metrics:
            print(f"\nPerformance Metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        print(f"{'='*50}\n")
