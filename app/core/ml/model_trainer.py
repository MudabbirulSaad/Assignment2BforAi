#!/usr/bin/env python3
"""
Model Training Manager for Traffic Prediction

This module provides a comprehensive framework for training, evaluating, and comparing
different machine learning models for traffic prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import pickle
from datetime import datetime
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold, TimeSeriesSplit

# Add the project root directory to the Python path
# This allows running the script directly with 'python model_trainer.py'
traefik_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if traefik_dir not in sys.path:
    sys.path.insert(0, traefik_dir)

# Import project-specific modules
from app.core.logging import logger
from app.config.config import config
from app.core.ml.base_model import BaseModel
from app.core.ml.lstm_model import LSTMModel
from app.core.ml.gru_model import GRUModel
from app.core.ml.cnnrnn_model import CNNRNNModel


class ModelTrainer:
    """
    Model Training Manager for Traffic Prediction.
    
    This class orchestrates the training, evaluation, and comparison of
    different machine learning models for traffic prediction.
    
    Attributes:
        models (Dict[str, BaseModel]): Dictionary of models to train
        data (Dict[str, Any]): Dictionary of data loaders
        results (Dict[str, Dict]): Dictionary of training and evaluation results
        config (Dict): Configuration parameters
        output_dir (str): Directory to save results
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the model trainer.
        
        Args:
            config (Dict, optional): Configuration parameters. Defaults to None.
        """
        self.config = config or {}
        
        # Set up output directory
        if 'output_dir' in self.config:
            self.output_dir = self.config['output_dir']
        else:
            # Create a default model directory if not specified in config
            if hasattr(config, 'model_dir'):
                model_dir = config.model_dir
            else:
                # Use a default path based on project structure
                model_dir = os.path.join(config.project_root, 'models')
                
            self.output_dir = os.path.join(model_dir, 'comparison')
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize data dictionary
        self.data = {}
        
        # Initialize results dictionary with all required keys
        self.results = {
            'training': {},
            'evaluation': {},
            'comparison': {},
            'cross_validation': {},
            'hyperparameter_optimization': {}
        }
        
        # Initialize training progress tracking
        self.training_progress = {}
        
        logger.info(f"Initialized Model Trainer with output directory: {self.output_dir}")
    
    def add_model(self, model_name: str, model: BaseModel) -> None:
        """
        Add a model to the trainer.
        
        Args:
            model_name (str): Name of the model
            model (BaseModel): Model instance
        """
        self.models[model_name] = model
        logger.info(f"Added model: {model_name}")
    
    def create_default_models(self) -> None:
        """
        Create default models for training.
        
        This method creates instances of LSTM, GRU, and CNN-RNN models
        with default configurations.
        """
        # Create LSTM model
        lstm_model = LSTMModel(model_name="LSTM_Default")
        self.add_model("LSTM", lstm_model)
        
        # Create GRU model
        gru_model = GRUModel(model_name="GRU_Default")
        self.add_model("GRU", gru_model)
        
        # Create CNN-RNN model
        cnnrnn_model = CNNRNNModel(model_name="CNNRNN_Default")
        self.add_model("CNN-RNN", cnnrnn_model)
        
        logger.info("Created default models: LSTM, GRU, CNN-RNN")
    
    def prepare_data(self, data: pd.DataFrame, 
                    feature_columns: List[str], 
                    target_columns: List[str],
                    sequence_length: int = 24,
                    batch_size: int = 32,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    shuffle: bool = False,
                    time_column: str = None) -> Dict[str, DataLoader]:
        """
        Prepare data for model training.
        
        Args:
            data (pd.DataFrame): Input data
            feature_columns (List[str]): List of feature column names
            target_columns (List[str]): List of target column names
            sequence_length (int, optional): Length of input sequences. Defaults to 24.
            batch_size (int, optional): Batch size. Defaults to 32.
            train_ratio (float, optional): Ratio of training data. Defaults to 0.7.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.15.
            test_ratio (float, optional): Ratio of test data. Defaults to 0.15.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            time_column (str, optional): Name of time column for chronological split. Defaults to None.
            
        Returns:
            Dict[str, DataLoader]: Dictionary of data loaders
        """
        # Store feature and target columns
        self.data['feature_columns'] = feature_columns
        self.data['target_columns'] = target_columns
        
        # Extract features and targets
        X = data[feature_columns].values
        y = data[target_columns].values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y, sequence_length)
        
        # Store sequence data
        self.data['X_seq'] = X_seq
        self.data['y_seq'] = y_seq
        self.data['sequence_length'] = sequence_length
        
        # Split data
        if time_column is not None and time_column in data.columns:
            # Chronological split
            logger.info("Using chronological split based on time column")
            
            # Get unique timestamps after sequence creation
            timestamps = data[time_column].values[sequence_length:]
            
            # Calculate split indices
            n_samples = len(timestamps)
            train_idx = int(n_samples * train_ratio)
            val_idx = train_idx + int(n_samples * val_ratio)
            
            # Create train/val/test indices
            train_indices = list(range(train_idx))
            val_indices = list(range(train_idx, val_idx))
            test_indices = list(range(val_idx, n_samples))
        else:
            # Random split
            logger.info("Using random split")
            
            # Calculate split indices
            n_samples = len(X_seq)
            indices = list(range(n_samples))
            
            if shuffle:
                np.random.shuffle(indices)
            
            train_idx = int(n_samples * train_ratio)
            val_idx = train_idx + int(n_samples * val_ratio)
            
            # Create train/val/test indices
            train_indices = indices[:train_idx]
            val_indices = indices[train_idx:val_idx]
            test_indices = indices[val_idx:]
        
        # Store split indices
        self.data['train_indices'] = train_indices
        self.data['val_indices'] = val_indices
        self.data['test_indices'] = test_indices
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Create subsets for train/val/test
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Store data loaders
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        self.data['loaders'] = data_loaders
        
        logger.info(f"Data prepared: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test samples")
        
        return data_loaders
    
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
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    def train_model(self, model_name: str, epochs: int = 100, patience: int = 10,
                   learning_rate: float = 0.001, verbose: bool = True, clip_grad_value: float = None) -> Dict:
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            epochs (int, optional): Number of training epochs. Defaults to 100.
            patience (int, optional): Early stopping patience. Defaults to 10.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            clip_grad_value (float, optional): Value for gradient clipping. Defaults to None.
            
        Returns:
            Dict: Training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Add it first with add_model().")
        
        if 'loaders' not in self.data:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        model = self.models[model_name]
        train_loader = self.data['loaders']['train']
        val_loader = self.data['loaders']['val']
        
        # Initialize progress tracking for this model
        self.training_progress[model_name] = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'training_time': 0
        }
        
        # Prepare model for training (extract input dimensions from data loader)
        model.prepare_data_for_training({'train': train_loader, 'val': val_loader})
        
        # Train the model
        start_time = time.time()
        
        # Define callback for tracking progress
        def progress_callback(epoch, train_loss, val_loss):
            # Initialize progress tracking if not already done
            if 'epochs' not in self.training_progress[model_name]:
                self.training_progress[model_name]['epochs'] = []
            if 'train_loss' not in self.training_progress[model_name]:
                self.training_progress[model_name]['train_loss'] = []
            if 'val_loss' not in self.training_progress[model_name]:
                self.training_progress[model_name]['val_loss'] = []
                
            # Add current epoch data
            self.training_progress[model_name]['epochs'].append(epoch)
            self.training_progress[model_name]['train_loss'].append(train_loss)
            self.training_progress[model_name]['val_loss'].append(val_loss)
            
            # Update best validation loss if needed
            if 'best_val_loss' not in self.training_progress[model_name] or val_loss < self.training_progress[model_name]['best_val_loss']:
                self.training_progress[model_name]['best_val_loss'] = val_loss
                self.training_progress[model_name]['best_epoch'] = epoch
                
            # Log progress
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Train the model
        history = model.train(train_loader, val_loader, epochs=epochs,
                            learning_rate=learning_rate, patience=patience,
                            verbose=verbose, clip_grad_value=clip_grad_value)
        
        # Update progress tracking
        training_time = time.time() - start_time
        self.training_progress[model_name]['training_time'] = training_time
        
        # Ensure history has the correct structure
        if not isinstance(history, dict) or 'train_loss' not in history:
            logger.warning(f"Model {model_name} returned invalid history format. Creating default structure.")
            # Create a default history structure based on the model's history attribute
            if hasattr(model, 'history') and isinstance(model.history, dict):
                history = model.history
                logger.info(f"Using model's history attribute with {len(history.get('train_loss', []))} epochs")
            else:
                logger.warning(f"No valid history found for {model_name}. Creating empty history.")
                history = {'train_loss': [], 'val_loss': [], 'train_metrics': {}, 'val_metrics': {}}
        
        # Log the history structure
        logger.info(f"History for {model_name}: {len(history.get('train_loss', []))} training loss points, "
                   f"{len(history.get('val_loss', []))} validation loss points")
        
        # Store training results with detailed logging
        self.results['training'][model_name] = {
            'history': history,
            'training_time': training_time,
            'epochs_completed': len(history.get('train_loss', [])),
            'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None
        }
        
        # Log the stored results
        logger.info(f"Stored training results for {model_name} with {self.results['training'][model_name]['epochs_completed']} epochs")
        if self.results['training'][model_name]['final_train_loss'] is not None:
            logger.info(f"Final training loss: {self.results['training'][model_name]['final_train_loss']:.6f}")
        if self.results['training'][model_name]['final_val_loss'] is not None:
            logger.info(f"Final validation loss: {self.results['training'][model_name]['final_val_loss']:.6f}")
        
        logger.info(f"Trained {model_name} for {len(history['train_loss'])} epochs in {training_time:.2f} seconds")
        
        return self.results['training'][model_name]
    
    def train_all_models(self, epochs: int = 100, patience: int = 10,
                        learning_rate: float = 0.001, verbose: bool = True, clip_grad_value: float = None) -> Dict:
        """
        Train all models.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 100.
            patience (int, optional): Early stopping patience. Defaults to 10.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            clip_grad_value (float, optional): Value for gradient clipping. Defaults to None.
            
        Returns:
            Dict: Training results for all models
        """
        logger.info(f"Training all models: {list(self.models.keys())}")
        
        for model_name in self.models.keys():
            logger.info(f"\nTraining {model_name}...")
            self.train_model(model_name, epochs, patience, learning_rate, verbose, clip_grad_value)
        
        return self.results['training']
    
    def evaluate_model(self, model_name: str) -> Dict:
        """
        Evaluate a specific model on the test set.
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            Dict: Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Add it first with add_model().")
        
        if 'loaders' not in self.data:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if model_name not in self.results['training']:
            raise ValueError(f"Model {model_name} not trained. Call train_model() first.")
        
        model = self.models[model_name]
        test_loader = self.data['loaders']['test']
        
        # Evaluate the model
        metrics = model.evaluate(test_loader)
        
        # Store evaluation results
        self.results['evaluation'][model_name] = metrics
        
        logger.info(f"Evaluated {model_name} on test set:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all trained models on the test set.
        
        Returns:
            Dict: Evaluation results for all models
        """
        logger.info(f"Evaluating all trained models: {list(self.results['training'].keys())}")
        
        for model_name in self.results['training'].keys():
            logger.info(f"\nEvaluating {model_name}...")
            self.evaluate_model(model_name)
        
        return self.results['evaluation']
    
    def cross_validate(self, model_name: str, n_splits: int = 5, epochs: int = 50,
                      patience: int = 5, time_series_split: bool = True) -> Dict:
        """
        Perform cross-validation for a specific model.
        
        Args:
            model_name (str): Name of the model to cross-validate
            n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
            epochs (int, optional): Number of training epochs per fold. Defaults to 50.
            patience (int, optional): Early stopping patience. Defaults to 5.
            time_series_split (bool, optional): Whether to use time series split. Defaults to True.
            
        Returns:
            Dict: Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Add it first with add_model().")
        
        if 'X_seq' not in self.data or 'y_seq' not in self.data:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        logger.info(f"Performing {n_splits}-fold cross-validation for {model_name}")
        
        # Get data
        X_seq = self.data['X_seq']
        y_seq = self.data['y_seq']
        
        # Choose cross-validation strategy
        if time_series_split:
            cv = TimeSeriesSplit(n_splits=n_splits)
            logger.info(f"Using TimeSeriesSplit with {n_splits} splits")
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            logger.info(f"Using KFold with {n_splits} splits")
        
        # Initialize results
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_seq)):
            logger.info(f"\nFold {fold+1}/{n_splits}")
            
            # Create PyTorch tensors
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # Create datasets and loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create a fresh model instance for this fold
            model_class = self.models[model_name].__class__
            fold_model = model_class(model_name=f"{model_name}_fold{fold+1}")
            
            # Prepare model for training
            fold_model.prepare_data_for_training({'train': train_loader})
            
            # Train the model
            fold_model.train(train_loader, None, epochs=epochs, patience=patience, verbose=False)
            
            # Evaluate the model
            metrics = fold_model.evaluate(test_loader)
            
            # Store fold results
            cv_results['fold_metrics'].append(metrics)
            
            logger.info(f"Fold {fold+1} metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Calculate mean and std of metrics across folds
        all_metrics = list(cv_results['fold_metrics'][0].keys())
        
        for metric in all_metrics:
            values = [fold_metrics[metric] for fold_metrics in cv_results['fold_metrics']]
            cv_results['mean_metrics'][metric] = float(np.mean(values))
            cv_results['std_metrics'][metric] = float(np.std(values))
        
        # Store cross-validation results
        self.results['cross_validation'][model_name] = cv_results
        
        logger.info(f"\nCross-validation results for {model_name}:")
        for metric in all_metrics:
            mean = cv_results['mean_metrics'][metric]
            std = cv_results['std_metrics'][metric]
            logger.info(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")
        
        return cv_results
    
    def compare_models(self) -> Dict:
        """
        Compare all evaluated models.
        
        Returns:
            Dict: Comparison results
        """
        if not self.results['evaluation']:
            raise ValueError("No models evaluated. Call evaluate_all_models() first.")
        
        logger.info("Comparing all evaluated models")
        
        # Initialize comparison results
        comparison = {
            'best_model': {},
            'ranking': {},
            'relative_performance': {}
        }
        
        # Get all metrics from the first model
        first_model = list(self.results['evaluation'].keys())[0]
        metrics = list(self.results['evaluation'][first_model].keys())
        
        # For each metric, find the best model and rank all models
        for metric in metrics:
            # Skip non-numeric metrics
            if not isinstance(self.results['evaluation'][first_model][metric], (int, float)):
                continue
            
            # Determine if higher is better for this metric
            higher_is_better = metric in ['r2']
            
            # Get values for all models
            values = {}
            for model_name, model_metrics in self.results['evaluation'].items():
                if metric in model_metrics:
                    values[model_name] = model_metrics[metric]
            
            # Sort models by metric value
            if higher_is_better:
                sorted_models = sorted(values.keys(), key=lambda m: values[m], reverse=True)
            else:
                sorted_models = sorted(values.keys(), key=lambda m: values[m])
            
            # Store ranking
            comparison['ranking'][metric] = sorted_models
            
            # Store best model for this metric
            best_model = sorted_models[0]
            best_value = values[best_model]
            comparison['best_model'][metric] = {
                'model': best_model,
                'value': best_value
            }
            
            # Calculate relative performance compared to the best model
            comparison['relative_performance'][metric] = {}
            for model_name, value in values.items():
                if higher_is_better:
                    # For metrics where higher is better (e.g., R2)
                    # Calculate percentage decrease from best
                    rel_perf = (value / best_value - 1) * 100
                else:
                    # For metrics where lower is better (e.g., RMSE, MAE)
                    # Calculate percentage increase from best
                    rel_perf = (value / best_value - 1) * 100
                
                comparison['relative_performance'][metric][model_name] = rel_perf
        
        # Calculate overall ranking based on average rank across metrics
        model_ranks = {model: [] for model in self.results['evaluation'].keys()}
        
        for metric in comparison['ranking'].keys():
            for rank, model in enumerate(comparison['ranking'][metric]):
                model_ranks[model].append(rank + 1)  # 1-based ranking
        
        # Calculate average rank for each model
        avg_ranks = {model: np.mean(ranks) for model, ranks in model_ranks.items()}
        
        # Sort models by average rank
        overall_ranking = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
        
        comparison['overall_ranking'] = overall_ranking
        comparison['average_ranks'] = avg_ranks
        
        # Store comparison results
        self.results['comparison'] = comparison
        
        # Log comparison results
        logger.info("\nModel Comparison Results:")
        logger.info(f"Overall Ranking: {overall_ranking}")
        logger.info("\nBest Model per Metric:")
        for metric, best in comparison['best_model'].items():
            logger.info(f"  {metric.upper()}: {best['model']} ({best['value']:.4f})")
        
        return comparison
    
    def optimize_hyperparameters(self, model_name: str, param_grid: Dict = None, n_trials: int = 20,
                               timeout: int = 3600) -> Dict:
        """
        Optimize hyperparameters for a specific model using Optuna.
        
        Args:
            model_name (str): Name of the model to optimize
            param_grid (Dict, optional): Parameter grid to search. Defaults to None.
            n_trials (int, optional): Number of trials. Defaults to 20.
            timeout (int, optional): Timeout in seconds. Defaults to 3600 (1 hour).
            
        Returns:
            Dict: Best hyperparameters
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Add it first with add_model().")
        
        if 'loaders' not in self.data:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        model = self.models[model_name]
        train_loader = self.data['loaders']['train']
        val_loader = self.data['loaders']['val']
        
        logger.info(f"Optimizing hyperparameters for {model_name} with {n_trials} trials")
        
        # Define default parameter grids for each model type
        default_param_grids = {
            'LSTM': {
                'hidden_dim': [32, 64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
                'batch_norm': [True, False],
                'learning_rate': [0.0001, 0.001, 0.01]
            },
            'GRU': {
                'hidden_dim': [32, 64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
                'batch_norm': [True, False],
                'bidirectional': [True, False],
                'learning_rate': [0.0001, 0.001, 0.01]
            },
            'CNN-RNN': {
                'hidden_dim': [32, 64, 128, 256],
                'cnn_layers': [1, 2, 3],
                'rnn_layers': [1, 2],
                'kernel_size': [3, 5, 7],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
                'rnn_type': ['lstm', 'gru'],
                'use_attention': [True, False],
                'learning_rate': [0.0001, 0.001, 0.01]
            }
        }
        
        # Use provided parameter grid or default
        if param_grid is None:
            model_type = model.model_type
            if model_type in default_param_grids:
                param_grid = default_param_grids[model_type]
            else:
                raise ValueError(f"No default parameter grid for model type {model_type}. Please provide a parameter grid.")
        
        # Use Optuna for hyperparameter optimization
        import optuna
        
        def objective(trial):
            # Define hyperparameters to tune based on param_grid
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, bool) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values), log=True)
                    elif all(isinstance(v, str) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], log=True)
            
            # Create a fresh model instance for this trial
            model_class = self.models[model_name].__class__
            trial_model = model_class(model_name=f"{model_name}_trial{trial.number}")
            
            # Update model configuration with trial parameters
            if model.model_type == 'LSTM':
                trial_model.lstm_config.update(params)
            elif model.model_type == 'GRU':
                trial_model.gru_config.update(params)
            elif model.model_type == 'CNN-RNN':
                trial_model.cnnrnn_config.update(params)
            
            # Prepare model for training
            trial_model.prepare_data_for_training({'train': train_loader, 'val': val_loader})
            
            # Train the model with early stopping
            history = trial_model.train(
                train_loader, val_loader,
                epochs=50,  # Reduced epochs for optimization
                patience=5,  # Early stopping
                learning_rate=params.get('learning_rate', 0.001),
                verbose=False
            )
            
            # Return validation loss as the objective to minimize
            return history['val_loss'][-1] if 'val_loss' in history else float('inf')
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Store optimization results
        if 'hyperparameter_optimization' not in self.results:
            self.results['hyperparameter_optimization'] = {}
        
        self.results['hyperparameter_optimization'][model_name] = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': n_trials,
            'param_grid': param_grid
        }
        
        logger.info(f"Best hyperparameters for {model_name}:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best validation loss: {best_value:.4f}")
        
        # Update model with best parameters
        if model.model_type == 'LSTM':
            model.lstm_config.update(best_params)
        elif model.model_type == 'GRU':
            model.gru_config.update(best_params)
        elif model.model_type == 'CNN-RNN':
            model.cnnrnn_config.update(best_params)
        
        return best_params
    
    def plot_training_progress(self, model_names: List[str] = None, save_path: str = None):
        """
        Plot training progress for one or more models.
        
        Args:
            model_names (List[str], optional): Names of models to plot. Defaults to None (all trained models).
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        # Check if we have any training results
        if not self.results['training']:
            logger.warning("No training results to plot.")
            return
        
        if model_names is None:
            model_names = list(self.results['training'].keys())
        
        # Create figure with two subplots (training and validation loss)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot training loss
        ax1.set_title('Training Loss')
        for model_name in model_names:
            if model_name in self.results['training']:
                history = self.results['training'][model_name]['history']
                if 'train_loss' in history and len(history['train_loss']) > 0:
                    epochs = list(range(1, len(history['train_loss']) + 1))
                    ax1.plot(epochs, history['train_loss'], label=f'{model_name} (Train)')
                    logger.info(f"Plotting {len(history['train_loss'])} epochs of training loss for {model_name}")
                else:
                    logger.warning(f"No training loss data found for {model_name}")
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot validation loss
        ax2.set_title('Validation Loss')
        for model_name in model_names:
            if model_name in self.results['training']:
                history = self.results['training'][model_name]['history']
                if 'val_loss' in history and len(history['val_loss']) > 0:
                    epochs = list(range(1, len(history['val_loss']) + 1))
                    ax2.plot(epochs, history['val_loss'], label=f'{model_name} (Val)')
                    logger.info(f"Plotting {len(history['val_loss'])} epochs of validation loss for {model_name}")
                else:
                    logger.warning(f"No validation loss data found for {model_name}")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training progress plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(self, metrics: List[str] = None, save_path: str = None) -> None:
        """
        Plot model comparison for selected metrics.
        
        Args:
            metrics (List[str], optional): Metrics to plot. Defaults to None (all metrics).
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if not self.results.get('evaluation') or not any(self.results['evaluation'].values()):
            logger.warning("No evaluation results available. Skipping model comparison plot.")
            return
        
        # Get all available metrics
        all_metrics = set()
        for model_metrics in self.results['evaluation'].values():
            all_metrics.update(model_metrics.keys())
        
        # Filter to numeric metrics only
        numeric_metrics = []
        for metric in all_metrics:
            for model_name, model_metrics in self.results['evaluation'].items():
                if metric in model_metrics and isinstance(model_metrics[metric], (int, float)):
                    numeric_metrics.append(metric)
                    break
        
        # Use specified metrics or all numeric metrics
        if metrics is None:
            metrics = numeric_metrics
        else:
            # Ensure all specified metrics are available
            for metric in metrics:
                if metric not in numeric_metrics:
                    logger.warning(f"Metric {metric} not available or not numeric. Skipping.")
            metrics = [m for m in metrics if m in numeric_metrics]
        
        # Create figure
        n_metrics = len(metrics)
        fig_height = max(4, n_metrics * 2)  # Adjust height based on number of metrics
        plt.figure(figsize=(10, fig_height))
        
        # Get model names
        model_names = list(self.results['evaluation'].keys())
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            plt.subplot(n_metrics, 1, i+1)
            
            # Get values for each model
            values = []
            labels = []
            for model_name in model_names:
                if metric in self.results['evaluation'][model_name]:
                    values.append(self.results['evaluation'][model_name][metric])
                    labels.append(model_name)
            
            # Create bar chart
            bars = plt.bar(labels, values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title(f'{metric.upper()}')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: str = None) -> Dict:
        """
        Generate a comprehensive report of model training and evaluation.
        
        Args:
            output_path (str, optional): Path to save the report. Defaults to None.
            
        Returns:
            Dict: Report data
        """
        if not self.results.get('training'):
            raise ValueError("No training results. Train models first.")
            
        # Initialize evaluation results if not present
        if 'evaluation' not in self.results:
            self.results['evaluation'] = {}
        
        # Create report data structure
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {},
            'comparison': self.results['comparison'] if 'comparison' in self.results else None,
            'cross_validation': self.results['cross_validation'],
            'hyperparameter_optimization': self.results['hyperparameter_optimization'] if 'hyperparameter_optimization' in self.results else {}
        }
        
        # Add data for each model
        for model_name in self.results['training'].keys():
            model_report = {
                'type': self.models[model_name].model_type,
                'training': self.results['training'][model_name],
                'evaluation': self.results['evaluation'].get(model_name, {}),
                'cross_validation': self.results['cross_validation'].get(model_name, {}),
                'hyperparameters': {}
            }
            
            # Add hyperparameters based on model type
            model = self.models[model_name]
            if model.model_type == 'LSTM':
                model_report['hyperparameters'] = model.lstm_config
            elif model.model_type == 'GRU':
                model_report['hyperparameters'] = model.gru_config
            elif model.model_type == 'CNN-RNN':
                model_report['hyperparameters'] = model.cnnrnn_config
            
            report['models'][model_name] = model_report
        
        # Save report to file if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                json.dump(convert_numpy(report), f, indent=4)
            
            logger.info(f"Report saved to {output_path}")
            
            # Generate plots
            plots_dir = os.path.join(os.path.dirname(output_path), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Training progress plot
            self.plot_training_progress(
                save_path=os.path.join(plots_dir, 'training_progress.png')
            )
            
            # Model comparison plot
            self.plot_model_comparison(
                save_path=os.path.join(plots_dir, 'model_comparison.png')
            )
            
            logger.info(f"Plots saved to {plots_dir}")
        
        return report
    
    def save(self, save_path: str = None) -> str:
        """
        Save the model trainer state.
        
        Args:
            save_path (str, optional): Path to save the state. Defaults to None.
            
        Returns:
            str: Path where the state was saved
        """
        save_path = save_path or os.path.join(self.output_dir, f"model_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save models separately
        models_dir = os.path.join(os.path.dirname(save_path), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_paths = {}
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{model_name}")
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
            model_paths[model_name] = model_path
        
        # Create state to save (excluding models)
        state = {
            'data': self.data,
            'results': self.results,
            'training_progress': self.training_progress,
            'config': self.config,
            'output_dir': self.output_dir,
            'model_paths': model_paths,
            'model_types': {name: model.model_type for name, model in self.models.items()}
        }
        
        # Save state
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model trainer state saved to {save_path}")
        
        return save_path
    
    @classmethod
    def load(cls, load_path: str) -> 'ModelTrainer':
        """
        Load a saved model trainer state.
        
        Args:
            load_path (str): Path to the saved state
            
        Returns:
            ModelTrainer: Loaded model trainer instance
        """
        # Load state
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance
        instance = cls(config=state['config'])
        
        # Restore state
        instance.data = state['data']
        instance.results = state['results']
        instance.training_progress = state['training_progress']
        instance.output_dir = state['output_dir']
        
        # Load models
        for model_name, model_path in state['model_paths'].items():
            model_type = state['model_types'][model_name]
            
            # Create appropriate model instance
            if model_type == 'LSTM':
                model = LSTMModel.load(model_path)
            elif model_type == 'GRU':
                model = GRUModel.load(model_path)
            elif model_type == 'CNN-RNN':
                model = CNNRNNModel.load(model_path)
            else:
                logger.warning(f"Unknown model type {model_type} for {model_name}. Skipping.")
                continue
            
            # Add model to instance
            instance.models[model_name] = model
        
        logger.info(f"Model trainer loaded from {load_path} with {len(instance.models)} models")
        
        return instance


def main():
    """
    Main function to demonstrate the usage of the ModelTrainer class.
    This allows the script to be run directly with 'python model_trainer.py'.
    """
    import argparse
    from app.core.ml.lstm_model import LSTMModel
    from app.core.ml.gru_model import GRUModel
    from app.core.ml.cnnrnn_model import CNNRNNModel
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train traffic prediction models')
    parser.add_argument('--train_data', type=str, default=None, help='Path to training data CSV file')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data CSV file')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--sequence_length', type=int, default=24, help='Sequence length for time series (default: 24)')
    parser.add_argument('--model', type=str, default='all', help='Model to train (lstm, gru, cnnrnn, or all)')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    args = parser.parse_args()
    
    # Use default paths from config if not provided
    train_data_path = args.train_data or config.processed_data['train_data']
    val_data_path = args.val_data or config.processed_data['val_data']
    test_data_path = args.test_data or config.processed_data['test_data']
    
    # Create a default output directory if not provided
    if args.output_dir:
        output_dir = args.output_dir
    elif hasattr(config, 'model_dir'):
        output_dir = config.model_dir
    else:
        # Use a default path based on project structure
        output_dir = os.path.join(config.project_root, 'models')
    
    logger.info(f"Starting model training with data from:")
    logger.info(f"  Train: {train_data_path}")
    logger.info(f"  Validation: {val_data_path}")
    logger.info(f"  Test: {test_data_path}")
    
    # Check if data files exist
    for path, name in [(train_data_path, 'Training'), (val_data_path, 'Validation'), (test_data_path, 'Test')]:
        if not os.path.exists(path):
            logger.error(f"{name} data file not found: {path}")
            return
    
    # Load data files
    try:
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Loaded training data with {len(train_data)} records")
        logger.info(f"Loaded validation data with {len(val_data)} records")
        logger.info(f"Loaded test data with {len(test_data)} records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Define feature and target columns with more focused selection
    feature_columns = [
        # Core traffic features (most important)
        'Traffic_Count_t-1',      # Previous interval (strongest predictor)
        'Traffic_Count_t-4',      # One hour ago
        'Traffic_Count_t-96',     # Same time yesterday
        'Rolling_Mean_1-hour',    # Recent trend
        
        # Time features (cyclical patterns)
        'Hour',                   # Time of day
        'DayOfWeek',              # Day of week
        'IsWeekend',              # Weekend flag
        
        # Current traffic (optional - can be removed if causing issues)
        'Traffic_Count'
    ]
    
    # Simpler target - just predict next interval
    target_columns = ['Target_t+1']
    time_column = 'DateTime'
    
    # Initialize trainer with configuration
    trainer_config = {
        'output_dir': output_dir,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length
    }
    trainer = ModelTrainer(config=trainer_config)
    
    # Create models based on command line argument
    if args.model.lower() == 'all':
        trainer.create_default_models()
    elif args.model.lower() == 'lstm':
        trainer.add_model('LSTM', LSTMModel(model_name='LSTM'))
    elif args.model.lower() == 'gru':
        trainer.add_model('GRU', GRUModel(model_name='GRU'))
    elif args.model.lower() == 'cnnrnn':
        trainer.add_model('CNN-RNN', CNNRNNModel(model_name='CNN-RNN'))
    else:
        logger.error(f"Unknown model type: {args.model}. Please use 'lstm', 'gru', 'cnnrnn', or 'all'.")
        return
    
    # Create a new method to prepare pre-split data
    def prepare_pre_split_data(trainer, train_data, val_data, test_data, feature_columns, target_columns, 
                             sequence_length=24, batch_size=32, time_column=None):
        """
        Prepare pre-split data for model training.
        
        Args:
            trainer (ModelTrainer): The model trainer instance
            train_data (pd.DataFrame): Training data
            val_data (pd.DataFrame): Validation data
            test_data (pd.DataFrame): Test data
            feature_columns (List[str]): List of feature column names
            target_columns (List[str]): List of target column names
            sequence_length (int, optional): Length of input sequences. Defaults to 24.
            batch_size (int, optional): Batch size. Defaults to 32.
            time_column (str, optional): Name of time column. Defaults to None.
            
        Returns:
            Dict[str, DataLoader]: Dictionary of data loaders
        """
        # Store feature and target columns
        trainer.data['feature_columns'] = feature_columns
        trainer.data['target_columns'] = target_columns
        
        # Process each dataset
        datasets = {'train': train_data, 'val': val_data, 'test': test_data}
        loaders = {}
        
        # Create scalers for features and targets
        from sklearn.preprocessing import RobustScaler, StandardScaler
        # Use RobustScaler for features to handle outliers better (uses median and IQR)
        feature_scaler = RobustScaler()
        # Keep StandardScaler for targets
        target_scaler = StandardScaler()
        
        # First pass: fit scalers on training data only
        train_data_copy = datasets['train'].copy()
        
        # Ensure all feature columns are numeric
        for col in feature_columns + target_columns:
            if col in train_data_copy.columns:
                # Convert any non-numeric columns to numeric
                if not pd.api.types.is_numeric_dtype(train_data_copy[col]):
                    logger.info(f"Converting column {col} to numeric")
                    train_data_copy[col] = pd.to_numeric(train_data_copy[col], errors='coerce')
                    # Fill any NaN values with 0
                    train_data_copy[col] = train_data_copy[col].fillna(0)
        
        # Fit scalers on training data
        feature_scaler.fit(train_data_copy[feature_columns].values)
        target_scaler.fit(train_data_copy[target_columns].values)
        
        # Log scaling parameters based on scaler type
        if isinstance(feature_scaler, RobustScaler):
            logger.info(f"Feature scaling: RobustScaler with center={feature_scaler.center_}, scale={feature_scaler.scale_}")
        else:
            logger.info(f"Feature scaling: StandardScaler with mean={feature_scaler.mean_}, std={feature_scaler.scale_}")
            
        logger.info(f"Target scaling: mean={target_scaler.mean_}, std={target_scaler.scale_}")
        
        # Store scalers for later use
        trainer.data['feature_scaler'] = feature_scaler
        trainer.data['target_scaler'] = target_scaler
        
        # Second pass: transform all datasets
        for split_name, split_data in datasets.items():
            # Ensure all feature columns are numeric and handle outliers
            for col in feature_columns + target_columns:
                if col in split_data.columns:
                    # Convert any non-numeric columns to numeric
                    if not pd.api.types.is_numeric_dtype(split_data[col]):
                        logger.info(f"Converting column {col} to numeric")
                        split_data[col] = pd.to_numeric(split_data[col], errors='coerce')
                    
                    # Handle outliers for traffic-related columns
                    if 'Traffic' in col and split_name == 'train':
                        # Calculate percentiles for outlier detection
                        q1 = split_data[col].quantile(0.05)
                        q3 = split_data[col].quantile(0.95)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Cap outliers instead of removing them
                        outliers = ((split_data[col] < lower_bound) | (split_data[col] > upper_bound)).sum()
                        if outliers > 0:
                            logger.info(f"Capping {outliers} outliers in {col} ({outliers/len(split_data)*100:.2f}%)")
                            split_data[col] = split_data[col].clip(lower_bound, upper_bound)
                    
                    # Replace infinities with NaN
                    split_data[col] = split_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill any NaN values with appropriate values
                    if split_data[col].isna().any():
                        nan_count = split_data[col].isna().sum()
                        logger.info(f"Filling {nan_count} NaN values in {col} ({nan_count/len(split_data)*100:.2f}%)")
                        
                        # For traffic columns, use median (more robust to outliers)
                        if 'Traffic' in col:
                            fill_value = split_data[col].median() if not pd.isna(split_data[col].median()) else 0
                        else:
                            # For other columns, use mean
                            fill_value = split_data[col].mean() if not pd.isna(split_data[col].mean()) else 0
                            
                        split_data[col] = split_data[col].fillna(fill_value)
            
            # Extract and normalize features and targets
            X_raw = split_data[feature_columns].values
            y_raw = split_data[target_columns].values
            
            # Apply scaling
            X = feature_scaler.transform(X_raw).astype(np.float32)
            y = target_scaler.transform(y_raw).astype(np.float32)
            
            # Add site-specific normalization if SCATS_ID column exists
            if 'SCATS_ID' in split_data.columns:
                logger.info(f"Applying site-specific normalization for {split_name} data")
                # Group by site and normalize traffic values within each site
                site_groups = split_data.groupby('SCATS_ID')
                
                # Get indices for each site
                site_indices = {}
                for site_id, group in site_groups:
                    site_indices[site_id] = group.index.tolist()
                
                # For each traffic feature, apply site-specific normalization
                traffic_feature_indices = [i for i, col in enumerate(feature_columns) if 'Traffic' in col]
                
                if traffic_feature_indices:
                    logger.info(f"Found {len(traffic_feature_indices)} traffic features for site normalization")
                    
                    # Create a copy of X to modify
                    X_site_norm = X.copy()
                    
                    # Apply site-specific normalization
                    for site_id, indices in site_indices.items():
                        if len(indices) > 10:  # Only normalize if we have enough data
                            for feat_idx in traffic_feature_indices:
                                # Get site-specific values for this feature
                                site_values = X[indices, feat_idx]
                                
                                # Calculate site-specific mean and std
                                site_mean = np.mean(site_values)
                                site_std = np.std(site_values)
                                
                                if site_std > 0:
                                    # Apply z-score normalization within site
                                    X_site_norm[indices, feat_idx] = (site_values - site_mean) / site_std
                    
                    # Replace X with site-normalized version
                    X = X_site_norm
            
            # Check for any remaining NaN or inf values
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning(f"Found NaN or inf values in {split_name} features after preprocessing")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if np.isnan(y).any() or np.isinf(y).any():
                logger.warning(f"Found NaN or inf values in {split_name} targets after preprocessing")
                y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Create sequences
            X_seq, y_seq = trainer._create_sequences(X, y, sequence_length)
            
            # Create tensor dataset and loader
            tensor_dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
            loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=(split_name == 'train'))
            
            # Store loader
            loaders[split_name] = loader
            
            logger.info(f"Prepared {split_name} data: {len(split_data)} records, {len(tensor_dataset)} sequences")
        
        # Store loaders
        trainer.data['loaders'] = loaders
        trainer.data['sequence_length'] = sequence_length
        
        return loaders
    
    # Prepare data using the pre-split datasets
    prepare_pre_split_data(
        trainer=trainer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        time_column=time_column
    )
    
    # Train models
    training_results = {}
    if args.model.lower() == 'all':
        # Train all models with enhanced parameters
        training_results = trainer.train_all_models(
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            clip_grad_value=args.clip_grad,
            verbose=True
        )
    else:
        # Train single model with enhanced parameters
        model_name = args.model.upper() if args.model.lower() != 'cnnrnn' else 'CNN-RNN'
        try:
            # Update model parameters if they support it
            model = trainer.models[model_name]
            if hasattr(model, 'dropout') and args.dropout is not None:
                model.dropout = args.dropout
                logger.info(f"Set dropout to {args.dropout} for model {model_name}")
            
            # Train with gradient clipping
            results = trainer.train_model(
                model_name=model_name,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                clip_grad_value=args.clip_grad,
                verbose=True
            )
            
            # Store results for report
            training_results[model_name] = results
            
            # Check if training was successful
            if results and not np.isnan(results.get('val_loss', [0])).any():
                logger.info(f"Successfully trained {model_name} model")
            else:
                logger.warning(f"Training for {model_name} may have had issues. Check the loss values.")
                
        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")
    
    # Plot training progress
    try:
        # Use the fixed checkpoints directory for plots
        checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        plots_dir = os.path.join(checkpoints_dir, 'plots')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a more detailed plot showing actual training data
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.title('Training Loss')
        
        for model_name, results in training_results.items():
            if 'history' in results and 'train_loss' in results['history']:
                train_loss = results['history']['train_loss']
                epochs = list(range(1, len(train_loss) + 1))
                plt.plot(epochs, train_loss, label=f'{model_name} Training Loss')
                logger.info(f"Plotting {len(train_loss)} epochs of training loss")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot validation loss
        plt.subplot(2, 1, 2)
        plt.title('Validation Loss')
        
        for model_name, results in training_results.items():
            if 'history' in results and 'val_loss' in results['history']:
                val_loss = results['history']['val_loss']
                epochs = list(range(1, len(val_loss) + 1))
                plt.plot(epochs, val_loss, label=f'{model_name} Validation Loss')
                logger.info(f"Plotting {len(val_loss)} epochs of validation loss")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, 'training_progress.png')
        plt.savefig(plot_path)
        logger.info(f"Training progress plot saved to {plot_path}")
        
        # Also try the trainer's plotting method
        trainer_plot_path = os.path.join(checkpoints_dir, 'training_progress.png')
        trainer.plot_training_progress(save_path=trainer_plot_path)
    except Exception as e:
        logger.error(f"Error plotting training progress: {e}")
    
    # Only generate report if we have valid results
    if training_results and any(results for results in training_results.values()):
        try:
            # Store results in trainer
            trainer.results['training'] = training_results
            
            # Use the fixed checkpoints directory for reports
            checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            report_path = os.path.join(checkpoints_dir, 'training_report.json')
            
            # Generate training report
            trainer.generate_report(output_path=report_path)
            logger.info(f"Training report generated at {report_path}")
        except Exception as e:
            logger.error(f"Error generating training report: {e}")
    else:
        logger.warning("No valid training results to generate report")
        
    logger.info("Model training completed")
    
    logger.info("Model training completed successfully.")
    logger.info(f"Trained models saved to {output_dir}")


if __name__ == "__main__":
    main()
