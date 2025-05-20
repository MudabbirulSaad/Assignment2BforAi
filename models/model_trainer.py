"""
Model Trainer for Traffic Flow Prediction
This module provides functionality to train and evaluate all three ML models:
LSTM, GRU, and the custom CNN-RNN model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from lstm_model import TrafficLSTMPredictor
from gru_model import TrafficGRUPredictor
from cnnrnn_model import TrafficCNNRNNPredictor

def load_or_train_models(data_path, models_dir='models/saved', results_dir='results', force_retrain=False):
    """
    Loads existing models or trains new ones if they don't exist.
    
    Args:
        data_path: Path to the preprocessed data file (NPZ format)
        models_dir: Directory to save/load models
        results_dir: Directory to save results
        force_retrain: If True, retrain models even if they already exist
        
    Returns:
        Dictionary containing the trained models
    """
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define model paths
    lstm_path = os.path.join(models_dir, 'lstm_model.pth')
    gru_path = os.path.join(models_dir, 'gru_model.pth')
    custom_path = os.path.join(models_dir, 'custom_model.pth')
    
    # Initialize models
    lstm_model = TrafficLSTMPredictor(
        sequence_length=16,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        dropout=0.2
    )
    
    gru_model = TrafficGRUPredictor(
        sequence_length=16,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        dropout=0.2
    )
    
    custom_model = TrafficCNNRNNPredictor(
        sequence_length=16,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        kernel_size=3,
        cnn_channels=16,
        dropout=0.2,
        rnn_type='lstm'
    )
    
    # Train or load LSTM model
    if not os.path.exists(lstm_path) or force_retrain:
        print("Training LSTM model...")
        lstm_model.train(
            data_path=data_path,
            epochs=50,
            patience=10,
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
                epochs=50,
                patience=10,
                model_save_path=lstm_path
            )
            lstm_model.plot_history(save_path=os.path.join(results_dir, 'lstm_training_history.png'))
    
    # Train or load GRU model
    if not os.path.exists(gru_path) or force_retrain:
        print("Training GRU model...")
        gru_model.train(
            data_path=data_path,
            epochs=50,
            patience=10,
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
                epochs=50,
                patience=10,
                model_save_path=gru_path
            )
            gru_model.plot_history(save_path=os.path.join(results_dir, 'gru_training_history.png'))
    
    # Train or load custom model
    if not os.path.exists(custom_path) or force_retrain:
        print("Training custom CNN-RNN model...")
        custom_model.train(
            data_path=data_path,
            epochs=50,
            patience=10,
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
                epochs=50,
                patience=10,
                model_save_path=custom_path
            )
            custom_model.plot_history(save_path=os.path.join(results_dir, 'custom_training_history.png'))
    
    return {
        'lstm': lstm_model,
        'gru': gru_model,
        'custom': custom_model
    }

def evaluate_models(models, data_path, results_dir='results'):
    """
    Evaluates all models on the test set and compares their performance.
    
    Args:
        models: Dictionary containing the trained models
        data_path: Path to the preprocessed data file (NPZ format)
        results_dir: Directory to save evaluation results
    """
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

def main():
    """
    Main function to train and evaluate all models.
    """
    # Path to the preprocessed data
    data_path = 'data/processed/sequence_data.npz'
    
    # Check if data exists, if not create synthetic data for testing
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Creating synthetic data for testing.")
        data_path = 'data/processed/synthetic_data.npz'
        
        # Create synthetic data directly
        def create_synthetic_data(save_path, num_samples=1000, sequence_length=16):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Generate synthetic time series data
            np.random.seed(42)
            t = np.linspace(0, 100, num_samples)
            # Create a sine wave with noise
            raw_values = 0.5 + 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(len(t))
            
            # Calculate min and max for scaling parameters
            data_min = np.min(raw_values)
            data_max = np.max(raw_values)
            
            # Normalize values to [0, 1] range
            values = (raw_values - data_min) / (data_max - data_min)
            
            # Create sequences
            X, y = [], []
            for i in range(len(values) - sequence_length):
                X.append(values[i:i + sequence_length])
                y.append(values[i + sequence_length])
            
            # Convert to numpy arrays
            X = np.array(X).reshape(-1, sequence_length, 1)
            y = np.array(y)
            
            # Split into train and test sets (80/20)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            
            # Save to file with scaling parameters
            np.savez(save_path, 
                     X_train=X_train, y_train=y_train, 
                     X_test=X_test, y_test=y_test,
                     data_min=data_min, data_max=data_max)
            print(f"Synthetic data saved to {save_path}")
            return save_path
        
        create_synthetic_data(data_path, num_samples=1000, sequence_length=16)
    
    # Train or load models
    models = load_or_train_models(
        data_path=data_path,
        models_dir='models/saved',
        results_dir='results',
        force_retrain=False
    )
    
    # Evaluate models
    results = evaluate_models(
        models=models,
        data_path=data_path,
        results_dir='results'
    )
    
    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()
