#!/usr/bin/env python3
"""
Ensemble Model Training Script

This script demonstrates how to train an ensemble model that combines predictions
from LSTM, GRU, and CNN-RNN models for improved traffic prediction performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import project-specific modules
from app.core.logging import logger
from app.config.config import config
from app.core.ml.model_trainer import ModelTrainer
from app.core.ml.lstm_model import LSTMModel
from app.core.ml.gru_model import GRUModel
from app.core.ml.cnnrnn_model import CNNRNNModel
from app.core.ml.ensemble_model import EnsembleModel, create_ensemble_from_trainer
from app.core.ml.train_traffic_models import load_traffic_data, prepare_features_and_targets


def train_base_models(data: pd.DataFrame, feature_columns: List[str], 
                     target_columns: List[str], sequence_length: int = 24,
                     batch_size: int = 32, epochs: int = 100, patience: int = 10,
                     learning_rate: float = 0.001, verbose: bool = True) -> ModelTrainer:
    """
    Train base models (LSTM, GRU, CNN-RNN) for ensemble creation.
    
    Args:
        data (pd.DataFrame): Traffic data
        feature_columns (List[str]): Feature columns
        target_columns (List[str]): Target columns
        sequence_length (int, optional): Sequence length. Defaults to 24.
        batch_size (int, optional): Batch size. Defaults to 32.
        epochs (int, optional): Number of epochs. Defaults to 100.
        patience (int, optional): Early stopping patience. Defaults to 10.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        
    Returns:
        ModelTrainer: Trained model trainer instance
    """
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Create models
    lstm_model = LSTMModel(model_name="LSTM_Traffic_Predictor")
    gru_model = GRUModel(model_name="GRU_Traffic_Predictor")
    cnnrnn_model = CNNRNNModel(model_name="CNNRNN_Traffic_Predictor")
    
    # Add models to trainer
    trainer.add_model("LSTM", lstm_model)
    trainer.add_model("GRU", gru_model)
    trainer.add_model("CNN-RNN", cnnrnn_model)
    
    # Prepare data for training
    logger.info("Preparing data for training...")
    trainer.prepare_data(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,
        time_column='datetime' if 'datetime' in data.columns else None
    )
    
    # Train all models
    logger.info("Training all base models...")
    trainer.train_all_models(
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    # Evaluate all models
    logger.info("Evaluating all base models...")
    trainer.evaluate_all_models()
    
    # Compare models
    logger.info("Comparing base models...")
    trainer.compare_models()
    
    return trainer


def create_and_train_ensembles(trainer: ModelTrainer, data_loaders: Dict) -> Dict[str, EnsembleModel]:
    """
    Create and train ensemble models using different ensemble methods.
    
    Args:
        trainer (ModelTrainer): Trained model trainer instance
        data_loaders (Dict): Data loaders for training and evaluation
        
    Returns:
        Dict[str, EnsembleModel]: Dictionary of trained ensemble models
    """
    # Create ensemble models with different methods
    ensemble_methods = ["average", "weighted_average", "stacking"]
    ensembles = {}
    
    for method in ensemble_methods:
        logger.info(f"Creating {method} ensemble model...")
        ensemble = create_ensemble_from_trainer(trainer, ensemble_method=method)
        
        # Prepare data for training (especially important for stacking)
        ensemble.prepare_data_for_training(data_loaders)
        
        # Train ensemble model
        if method == "stacking":
            logger.info(f"Training {method} ensemble model...")
            ensemble.train(
                data_loaders['train'],
                data_loaders['val'],
                epochs=50,  # Fewer epochs for stacking
                patience=5,
                verbose=True
            )
        
        # Evaluate ensemble model
        logger.info(f"Evaluating {method} ensemble model...")
        ensemble.evaluate(data_loaders['test'])
        
        # Store ensemble model
        ensembles[method] = ensemble
    
    return ensembles


def compare_all_models(trainer: ModelTrainer, ensembles: Dict[str, EnsembleModel]) -> Dict:
    """
    Compare all models including base models and ensembles.
    
    Args:
        trainer (ModelTrainer): Trained model trainer instance
        ensembles (Dict[str, EnsembleModel]): Dictionary of trained ensemble models
        
    Returns:
        Dict: Comparison results
    """
    # Create comparison dictionary
    comparison = {
        'models': {},
        'best_model': {},
        'ranking': {}
    }
    
    # Add base model metrics
    for model_name, model in trainer.models.items():
        comparison['models'][model_name] = model.metrics
    
    # Add ensemble model metrics
    for method, ensemble in ensembles.items():
        comparison['models'][f"Ensemble_{method}"] = ensemble.metrics
    
    # Find best model for each metric
    for metric in ['rmse', 'mae', 'r2', 'mape']:
        # For metrics where lower is better (RMSE, MAE, MAPE)
        if metric in ['rmse', 'mae', 'mape']:
            sorted_models = sorted(comparison['models'].keys(), 
                                 key=lambda m: comparison['models'][m][metric])
        # For metrics where higher is better (R2)
        else:
            sorted_models = sorted(comparison['models'].keys(), 
                                 key=lambda m: comparison['models'][m][metric],
                                 reverse=True)
        
        comparison['ranking'][metric] = sorted_models
        
        # Store best model for this metric
        best_model = sorted_models[0]
        best_value = comparison['models'][best_model][metric]
        comparison['best_model'][metric] = {
            'model': best_model,
            'value': best_value
        }
    
    # Calculate overall ranking based on average rank across metrics
    avg_ranks = {}
    for model_name in comparison['models'].keys():
        ranks = [comparison['ranking'][metric].index(model_name) + 1 
                for metric in comparison['ranking'].keys()]
        avg_ranks[model_name] = sum(ranks) / len(ranks)
    
    # Sort models by average rank
    comparison['overall_ranking'] = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
    comparison['average_ranks'] = avg_ranks
    
    # Create summary
    best_model = comparison['overall_ranking'][0]
    comparison['summary'] = f"The best performing model overall is {best_model}."
    
    logger.info(f"Model comparison results: {comparison['summary']}")
    
    return comparison


def plot_model_comparison(comparison: Dict, save_path: str = None) -> None:
    """
    Plot comparison of all models.
    
    Args:
        comparison (Dict): Comparison results
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    metrics = ['rmse', 'mae', 'r2', 'mape']
    titles = ['RMSE (lower is better)', 'MAE (lower is better)', 
             'R² (higher is better)', 'MAPE % (lower is better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Get values for each model
        models = comparison['ranking'][metric]
        values = [comparison['models'][model][metric] for model in models]
        
        # Create bar chart with custom colors
        colors = []
        for model in models:
            if model.startswith('Ensemble'):
                colors.append('green')
            elif model == 'LSTM':
                colors.append('blue')
            elif model == 'GRU':
                colors.append('orange')
            elif model == 'CNN-RNN':
                colors.append('purple')
            else:
                colors.append('gray')
        
        bars = axes[i].bar(models, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', rotation=0)
        
        axes[i].set_title(title)
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function to demonstrate ensemble model training and evaluation.
    """
    logger.info("Starting ensemble model training for traffic prediction...")
    
    # Load traffic data
    data = load_traffic_data()
    
    # Prepare features and targets
    columns = prepare_features_and_targets(data)
    feature_columns = columns['feature_columns']
    target_columns = columns['target_columns']
    
    # Train base models
    trainer = train_base_models(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sequence_length=24,  # 6 hours (assuming 15-minute intervals)
        batch_size=32,
        epochs=100,
        patience=10,
        learning_rate=0.001,
        verbose=True
    )
    
    # Create and train ensemble models
    ensembles = create_and_train_ensembles(
        trainer=trainer,
        data_loaders=trainer.data['loaders']
    )
    
    # Compare all models
    comparison = compare_all_models(
        trainer=trainer,
        ensembles=ensembles
    )
    
    # Plot model comparison
    plot_path = os.path.join(config.model_dir, 'comparison', 'all_models_comparison.png')
    plot_model_comparison(
        comparison=comparison,
        save_path=plot_path
    )
    
    # Save best ensemble model
    best_ensemble = None
    best_ensemble_name = None
    
    for method, ensemble in ensembles.items():
        ensemble_name = f"Ensemble_{method}"
        if ensemble_name in comparison['overall_ranking'][:3]:  # If in top 3
            if best_ensemble is None or comparison['average_ranks'][ensemble_name] < comparison['average_ranks'].get(best_ensemble_name, float('inf')):
                best_ensemble = ensemble
                best_ensemble_name = ensemble_name
    
    if best_ensemble is not None:
        save_path = os.path.join(config.model_dir, f"{best_ensemble_name}")
        best_ensemble.save(save_path)
        logger.info(f"Best ensemble model ({best_ensemble_name}) saved to {save_path}")
    
    logger.info("Ensemble model training and evaluation completed successfully.")
    
    # Return trainer and ensembles for interactive use
    return trainer, ensembles, comparison


if __name__ == "__main__":
    trainer, ensembles, comparison = main()
