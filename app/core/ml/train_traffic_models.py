#!/usr/bin/env python3
"""
Traffic Prediction Model Training Script

This script demonstrates how to use the model training manager to train, evaluate,
and compare different machine learning models for traffic prediction.
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


def load_traffic_data(data_path: str = None) -> pd.DataFrame:
    """
    Load and prepare traffic data for model training.
    
    Args:
        data_path (str, optional): Path to the traffic data. Defaults to None.
        
    Returns:
        pd.DataFrame: Prepared traffic data
    """
    # Use provided path or default from config
    data_path = data_path or os.path.join(config.processed_data_dir, 'scats_traffic_enhanced.csv')
    
    logger.info(f"Loading traffic data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Convert datetime column to datetime type if it exists
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
    
    logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
    
    return data


def prepare_features_and_targets(data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Prepare feature and target columns for model training.
    
    Args:
        data (pd.DataFrame): Traffic data
        
    Returns:
        Dict[str, List[str]]: Dictionary with feature and target columns
    """
    # Identify feature columns (exclude targets, IDs, and datetime)
    exclude_columns = ['SCATS_ID', 'datetime', 'traffic_volume_t+1', 'traffic_volume_t+4']
    
    # Get all numeric columns for features
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    # Define target column(s)
    target_columns = ['traffic_volume_t+1']  # Predict next time step
    
    logger.info(f"Selected {len(feature_columns)} feature columns and {len(target_columns)} target columns")
    
    return {
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }


def train_and_evaluate_models(data: pd.DataFrame, feature_columns: List[str], 
                             target_columns: List[str], sequence_length: int = 24,
                             batch_size: int = 32, epochs: int = 100, patience: int = 10,
                             learning_rate: float = 0.001, verbose: bool = True,
                             optimize_hyperparams: bool = False) -> ModelTrainer:
    """
    Train and evaluate multiple models for traffic prediction.
    
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
        optimize_hyperparams (bool, optional): Whether to optimize hyperparameters. Defaults to False.
        
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
    
    # Optimize hyperparameters if requested
    if optimize_hyperparams:
        logger.info("Optimizing hyperparameters...")
        for model_name in trainer.models.keys():
            logger.info(f"Optimizing {model_name} hyperparameters...")
            trainer.optimize_hyperparameters(
                model_name=model_name,
                n_trials=10,  # Reduced for demonstration
                timeout=1800  # 30 minutes timeout
            )
    
    # Train all models
    logger.info("Training all models...")
    trainer.train_all_models(
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    # Evaluate all models
    logger.info("Evaluating all models...")
    trainer.evaluate_all_models()
    
    # Compare models
    logger.info("Comparing models...")
    comparison = trainer.compare_models()
    
    # Generate report
    logger.info("Generating report...")
    report_path = os.path.join(config.model_dir, 'comparison', 'model_comparison_report.json')
    trainer.generate_report(output_path=report_path)
    
    # Plot training progress
    logger.info("Plotting training progress...")
    plot_path = os.path.join(config.model_dir, 'comparison', 'training_progress.png')
    trainer.plot_training_progress(save_path=plot_path)
    
    # Plot model comparison
    logger.info("Plotting model comparison...")
    comparison_plot_path = os.path.join(config.model_dir, 'comparison', 'model_comparison.png')
    trainer.plot_model_comparison(save_path=comparison_plot_path)
    
    # Save trainer state
    logger.info("Saving trainer state...")
    save_path = os.path.join(config.model_dir, 'comparison', 'model_trainer_state.pkl')
    trainer.save(save_path=save_path)
    
    return trainer


def perform_cross_validation(trainer: ModelTrainer, n_splits: int = 5, 
                            epochs: int = 50, patience: int = 5) -> Dict:
    """
    Perform cross-validation for all models.
    
    Args:
        trainer (ModelTrainer): Model trainer instance
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        epochs (int, optional): Number of epochs per fold. Defaults to 50.
        patience (int, optional): Early stopping patience. Defaults to 5.
        
    Returns:
        Dict: Cross-validation results
    """
    logger.info(f"Performing {n_splits}-fold cross-validation...")
    
    cv_results = {}
    for model_name in trainer.models.keys():
        logger.info(f"Cross-validating {model_name}...")
        cv_results[model_name] = trainer.cross_validate(
            model_name=model_name,
            n_splits=n_splits,
            epochs=epochs,
            patience=patience,
            time_series_split=True
        )
    
    return cv_results


def analyze_feature_importance(trainer: ModelTrainer, feature_names: List[str] = None) -> Dict:
    """
    Analyze feature importance for CNN-RNN model.
    
    Args:
        trainer (ModelTrainer): Model trainer instance
        feature_names (List[str], optional): Feature names. Defaults to None.
        
    Returns:
        Dict: Feature importance results
    """
    if "CNN-RNN" not in trainer.models:
        logger.warning("CNN-RNN model not found in trainer. Cannot analyze feature importance.")
        return {}
    
    # Get CNN-RNN model
    cnnrnn_model = trainer.models["CNN-RNN"]
    
    # Analyze feature importance
    logger.info("Analyzing feature importance...")
    importance = cnnrnn_model.analyze_feature_importance(feature_names=feature_names)
    
    # Plot feature importance
    logger.info("Plotting feature importance...")
    plot_path = os.path.join(config.model_dir, 'comparison', 'feature_importance.png')
    cnnrnn_model.plot_feature_importance(
        feature_names=feature_names,
        top_n=20,  # Show top 20 features
        save_path=plot_path
    )
    
    return importance


def main():
    """
    Main function to demonstrate model training and evaluation.
    """
    logger.info("Starting traffic prediction model training...")
    
    # Load traffic data
    data = load_traffic_data()
    
    # Prepare features and targets
    columns = prepare_features_and_targets(data)
    feature_columns = columns['feature_columns']
    target_columns = columns['target_columns']
    
    # Train and evaluate models
    trainer = train_and_evaluate_models(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sequence_length=24,  # 6 hours (assuming 15-minute intervals)
        batch_size=32,
        epochs=100,
        patience=10,
        learning_rate=0.001,
        verbose=True,
        optimize_hyperparams=False  # Set to True to optimize hyperparameters
    )
    
    # Perform cross-validation
    cv_results = perform_cross_validation(
        trainer=trainer,
        n_splits=5,
        epochs=50,
        patience=5
    )
    
    # Analyze feature importance for CNN-RNN model
    importance = analyze_feature_importance(
        trainer=trainer,
        feature_names=feature_columns
    )
    
    logger.info("Traffic prediction model training completed successfully.")
    
    # Return trainer for interactive use
    return trainer


if __name__ == "__main__":
    trainer = main()
