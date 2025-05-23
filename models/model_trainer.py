"""
Model Trainer for Traffic Flow Prediction
This module provides functionality to train and evaluate all three ML models:
LSTM, GRU, and the custom CNN-RNN model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime

# Add parent directory to path to import from utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative imports
from models.lstm_model import TrafficLSTMPredictor
from models.gru_model import TrafficGRUPredictor
from models.cnnrnn_model import TrafficCNNRNNPredictor

def load_or_train_models(data_path, config):
    """
    Loads existing models or trains new ones if they don't exist.
    
    Args:
        data_path: Path to the preprocessed data file (NPZ format)
        config: Configuration dictionary loaded from JSON
        
    Returns:
        Dictionary containing the trained models
    """
    # Get paths from config
    models_dir = config['paths']['models_dir']
    results_dir = config['paths']['results_dir']
    force_retrain = config['training']['force_retrain']
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define model paths
    lstm_path = os.path.join(models_dir, 'lstm_model.pth')
    gru_path = os.path.join(models_dir, 'gru_model.pth')
    custom_path = os.path.join(models_dir, 'custom_model.pth')
    
    # Initialize models using config parameters
    lstm_model = TrafficLSTMPredictor(
        sequence_length=config['data']['sequence_length'],
        hidden_size=config['models']['lstm']['hidden_size'],
        num_layers=config['models']['lstm']['num_layers'],
        learning_rate=config['models']['lstm']['learning_rate'],
        batch_size=config['models']['lstm']['batch_size'],
        dropout=config['models']['lstm']['dropout'],
        config=config
    )
    
    gru_model = TrafficGRUPredictor(
        sequence_length=config['data']['sequence_length'],
        hidden_size=config['models']['gru']['hidden_size'],
        num_layers=config['models']['gru']['num_layers'],
        learning_rate=config['models']['gru']['learning_rate'],
        batch_size=config['models']['gru']['batch_size'],
        dropout=config['models']['gru']['dropout'],
        config=config
    )
    
    custom_model = TrafficCNNRNNPredictor(
        sequence_length=config['data']['sequence_length'],
        hidden_size=config['models']['custom']['hidden_size'],
        num_layers=config['models']['custom']['num_layers'],
        learning_rate=config['models']['custom']['learning_rate'],
        batch_size=config['models']['custom']['batch_size'],
        kernel_size=config['models']['custom']['kernel_size'],
        cnn_channels=config['models']['custom']['cnn_channels'],
        dropout=config['models']['custom']['dropout'],
        rnn_type=config['models']['custom']['rnn_type'],
        config=config
    )
    
    # Train or load LSTM model
    if not os.path.exists(lstm_path) or force_retrain:
        print("Training LSTM model...")
        lstm_model.train(
            data_path=data_path,
            epochs=config['models']['lstm']['epochs'],
            patience=config['training']['patience'],
            model_save_path=lstm_path
        )
        lstm_model.plot_history(save_path=os.path.join(results_dir, 'lstm_training_history.png'))
    else:
        print("Loading existing LSTM model...")
        try:
            lstm_model.load_model(lstm_path)
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            print("Training a new LSTM model...")
            lstm_model.train(
                data_path=data_path,
                epochs=config['training']['fallback_epochs'],
                patience=config['training']['patience'],
                model_save_path=lstm_path
            )
            lstm_model.plot_history(save_path=os.path.join(results_dir, 'lstm_training_history.png'))
    
    # Train or load GRU model
    if not os.path.exists(gru_path) or force_retrain:
        print("Training GRU model...")
        gru_model.train(
            data_path=data_path,
            epochs=config['models']['gru']['epochs'],
            patience=config['training']['patience'],
            model_save_path=gru_path
        )
        gru_model.plot_history(save_path=os.path.join(results_dir, 'gru_training_history.png'))
    else:
        print("Loading existing GRU model...")
        try:
            gru_model.load_model(gru_path)
        except Exception as e:
            print(f"Error loading GRU model: {e}")
            print("Training a new GRU model...")
            gru_model.train(
                data_path=data_path,
                epochs=config['training']['fallback_epochs'],
                patience=config['training']['patience'],
                model_save_path=gru_path
            )
            gru_model.plot_history(save_path=os.path.join(results_dir, 'gru_training_history.png'))
    
    # Train or load custom model
    if not os.path.exists(custom_path) or force_retrain:
        print("Training custom CNN-RNN model...")
        custom_model.train(
            data_path=data_path,
            epochs=config['models']['custom']['epochs'],
            patience=config['training']['patience'],
            model_save_path=custom_path
        )
        custom_model.plot_history(save_path=os.path.join(results_dir, 'custom_training_history.png'))
    else:
        print("Loading existing custom CNN-RNN model...")
        try:
            custom_model.load_model(custom_path)
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Training a new custom CNN-RNN model...")
            custom_model.train(
                data_path=data_path,
                epochs=config['training']['fallback_epochs'],
                patience=config['training']['patience'],
                model_save_path=custom_path
            )
            custom_model.plot_history(save_path=os.path.join(results_dir, 'custom_training_history.png'))
    
    return {
        'lstm': lstm_model,
        'gru': gru_model,
        'custom': custom_model
    }

def evaluate_models(models, data_path, config):
    """
    Evaluates all models on the test set and compares their performance.
    
    Args:
        models: Dictionary containing the trained models
        data_path: Path to the preprocessed data file (NPZ format)
        config: Configuration dictionary loaded from JSON
    """
    # Get results directory from config
    results_dir = config['paths']['results_dir']
    # Load test data
    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    data_min, data_max = data['data_min'], data['data_max']
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        # Get predictions
        predictions = model.predict_batch(X_test)
        
        # Denormalize true values
        y_test_denorm = y_test * (data_max - data_min) + data_min
        
        # Calculate metrics
        mse = np.mean((predictions - y_test_denorm) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_denorm))
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'true_values': y_test_denorm
        }
        
        print(f"{name.upper()} Model Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print("-" * 40)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot MSE comparison
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    mse_values = [results[name]['mse'] for name in model_names]
    plt.bar(model_names, mse_values)
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Comparison')
    
    # Plot RMSE comparison
    plt.subplot(2, 2, 2)
    rmse_values = [results[name]['rmse'] for name in model_names]
    plt.bar(model_names, rmse_values)
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error Comparison')
    
    # Plot MAE comparison
    plt.subplot(2, 2, 3)
    mae_values = [results[name]['mae'] for name in model_names]
    plt.bar(model_names, mae_values)
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error Comparison')
    
    # Plot predictions for a sample
    plt.subplot(2, 2, 4)
    sample_idx = np.random.randint(0, len(y_test))
    for name in model_names:
        plt.scatter(name, results[name]['predictions'][sample_idx], label=f"{name} prediction")
    plt.scatter('true', results[model_names[0]]['true_values'][sample_idx], color='red', label='True value')
    plt.ylabel('Traffic Flow')
    plt.title(f'Sample Prediction Comparison (Sample {sample_idx})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    plt.close()
    
    # Plot time series predictions
    plt.figure(figsize=(15, 8))
    num_samples = min(100, len(y_test))  # Plot first 100 samples or less
    time_steps = np.arange(num_samples)
    
    for name in model_names:
        plt.plot(time_steps, results[name]['predictions'][:num_samples], label=f"{name} predictions")
    
    plt.plot(time_steps, results[model_names[0]]['true_values'][:num_samples], 'k--', label='True values')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic Flow')
    plt.title('Time Series Predictions Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'time_series_comparison.png'))
    plt.close()
    
    return results

def main(model_name=None, epochs=None, force_retrain=False):
    """
    Main function to train and evaluate models.
    
    Args:
        model_name (str, optional): Name of the specific model to train ('lstm', 'gru', or 'custom'). 
                                   If None, all models will be trained/evaluated.
        epochs (int, optional): Number of epochs for training. If None, uses the value from config.
        force_retrain (bool): Whether to force retraining even if a saved model exists.
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override config values if specified in arguments
    if force_retrain:
        config['training']['force_retrain'] = True
    
    # Update epochs in config if specified
    if epochs is not None:
        if model_name:
            if model_name.lower() in config['models']:
                config['models'][model_name.lower()]['epochs'] = epochs
                print(f"Setting {epochs} epochs for {model_name} model")
        else:
            # Update epochs for all models
            for model_key in config['models']:
                config['models'][model_key]['epochs'] = epochs
            print(f"Setting {epochs} epochs for all models")
    
    # Path to the preprocessed data from config
    data_path = config['paths']['sequence_data']
    
    # Ensure we're using only the real SCATS data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: {data_path} not found. Please run data_processing.py first to process the SCATS data.")
    
    # Ensure model checkpoint directory exists
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Train specific model or all models
    if model_name:
        model_name = model_name.lower()
        if model_name not in ['lstm', 'gru', 'custom']:
            print(f"Error: Unknown model '{model_name}'. Valid options are 'lstm', 'gru', or 'custom'.")
            return
        
        print(f"Training/loading only the {model_name.upper()} model...")
        
        # Create a modified config for single model training
        single_model_config = config.copy()
        
        # Set force_retrain for the specific model
        if model_name == 'lstm':
            # Create a dictionary with only the LSTM model
            models = {}
            lstm_model = TrafficLSTMPredictor(
                sequence_length=config['data']['sequence_length'],
                hidden_size=config['models']['lstm']['hidden_size'],
                num_layers=config['models']['lstm']['num_layers'],
                learning_rate=config['models']['lstm']['learning_rate'],
                batch_size=config['models']['lstm']['batch_size'],
                dropout=config['models']['lstm']['dropout'],
                config=config
            )
            
            # Train or load the model
            lstm_path = os.path.join(config['paths']['models_dir'], 'lstm_model.pth')
            if not os.path.exists(lstm_path) or force_retrain:
                print(f"Training {model_name.upper()} model...")
                lstm_model.train(
                    data_path=data_path,
                    epochs=config['models']['lstm']['epochs'],
                    patience=config['training']['patience'],
                    model_save_path=lstm_path
                )
                lstm_model.plot_history(save_path=os.path.join(config['paths']['results_dir'], 'lstm_training_history.png'))
            else:
                print(f"Loading existing {model_name.upper()} model...")
                lstm_model.load_model(lstm_path)
            
            models['lstm'] = lstm_model
            
        elif model_name == 'gru':
            # Create a dictionary with only the GRU model
            models = {}
            gru_model = TrafficGRUPredictor(
                sequence_length=config['data']['sequence_length'],
                hidden_size=config['models']['gru']['hidden_size'],
                num_layers=config['models']['gru']['num_layers'],
                learning_rate=config['models']['gru']['learning_rate'],
                batch_size=config['models']['gru']['batch_size'],
                dropout=config['models']['gru']['dropout'],
                config=config
            )
            
            # Train or load the model
            gru_path = os.path.join(config['paths']['models_dir'], 'gru_model.pth')
            if not os.path.exists(gru_path) or force_retrain:
                print(f"Training {model_name.upper()} model...")
                gru_model.train(
                    data_path=data_path,
                    epochs=config['models']['gru']['epochs'],
                    patience=config['training']['patience'],
                    model_save_path=gru_path
                )
                gru_model.plot_history(save_path=os.path.join(config['paths']['results_dir'], 'gru_training_history.png'))
            else:
                print(f"Loading existing {model_name.upper()} model...")
                gru_model.load_model(gru_path)
            
            models['gru'] = gru_model
            
        elif model_name == 'custom':
            # Create a dictionary with only the custom model
            models = {}
            custom_model = TrafficCNNRNNPredictor(
                sequence_length=config['data']['sequence_length'],
                hidden_size=config['models']['custom']['hidden_size'],
                num_layers=config['models']['custom']['num_layers'],
                learning_rate=config['models']['custom']['learning_rate'],
                batch_size=config['models']['custom']['batch_size'],
                kernel_size=config['models']['custom']['kernel_size'],
                cnn_channels=config['models']['custom']['cnn_channels'],
                dropout=config['models']['custom']['dropout'],
                rnn_type=config['models']['custom']['rnn_type'],
                config=config
            )
            
            # Train or load the model
            custom_path = os.path.join(config['paths']['models_dir'], 'custom_model.pth')
            if not os.path.exists(custom_path) or force_retrain:
                print(f"Training {model_name.upper()} model...")
                custom_model.train(
                    data_path=data_path,
                    epochs=config['models']['custom']['epochs'],
                    patience=config['training']['patience'],
                    model_save_path=custom_path
                )
                custom_model.plot_history(save_path=os.path.join(config['paths']['results_dir'], 'custom_training_history.png'))
            else:
                print(f"Loading existing {model_name.upper()} model...")
                custom_model.load_model(custom_path)
            
            models['custom'] = custom_model
        
        # Evaluate only the specified model
        results = evaluate_models(
            models={model_name: models[model_name]},
            data_path=data_path,
            config=config
        )
    else:
        # Train or load all models
        models = load_or_train_models(
            data_path=data_path,
            config=config
        )
        
        # Evaluate all models
        results = evaluate_models(
            models=models,
            data_path=data_path,
            config=config
        )
    
    print("Model training and evaluation completed.")

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate traffic prediction models')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru', 'custom'], 
                        help='Specific model to train (lstm, gru, or custom)')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--force-retrain', action='store_true', 
                        help='Force retraining even if saved model exists')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    main(model_name=args.model, epochs=args.epochs, force_retrain=args.force_retrain)
