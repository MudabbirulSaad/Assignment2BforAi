"""
LSTM Model for Traffic Flow Prediction
This module implements a Long Short-Term Memory (LSTM) neural network
for predicting traffic flow based on historical data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import time

class LSTMModel(nn.Module):
    """
    LSTM neural network for time series prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (typically 1 for univariate time series)
            hidden_size: Number of hidden units in the LSTM layer
            num_layers: Number of LSTM layers
            output_size: Number of output features (typically 1 for single-step prediction)
            dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out


class TrafficLSTMPredictor:
    """
    Class for training, evaluating, and using LSTM models for traffic prediction.
    """
    def __init__(self, sequence_length=16, hidden_size=64, num_layers=2, learning_rate=0.001, 
                 batch_size=32, dropout=0.2, device=None):
        """
        Initialize the LSTM predictor.
        
        Args:
            sequence_length: Number of time steps to use for prediction
            hidden_size: Number of hidden units in the LSTM layer
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            dropout: Dropout rate for regularization
            device: Device to use for training (CPU or GPU)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Model will be initialized during training
        self.model = None
        self.scaler = None
        self.history = None
        
    def _prepare_data(self, data_path):
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to the preprocessed data file (NPZ format)
            
        Returns:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            scaler_params: Parameters for denormalizing predictions
        """
        # Load the preprocessed data
        data = np.load(data_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Get scaler parameters for denormalization
        data_min, data_max = data['data_min'], data['data_max']
        scaler_params = {'min': data_min, 'max': data_max}
        
        # Split training data into training and validation sets (80/20)
        val_size = int(0.2 * len(X_train))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Create DataLoader objects
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader, scaler_params
    
    def train(self, data_path, epochs=100, patience=10, model_save_path=None):
        """
        Train the LSTM model.
        
        Args:
            data_path: Path to the preprocessed data file (NPZ format)
            epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            model_save_path: Path to save the trained model
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        train_loader, val_loader, test_loader, scaler_params = self._prepare_data(data_path)
        self.scaler = scaler_params
        
        # Get input size from data
        input_size = next(iter(train_loader))[0].shape[2]
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': None,
            'train_rmse': [],
            'val_rmse': [],
            'test_rmse': None
        }
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_loader.dataset)
            
            # Calculate RMSE for training and validation
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(val_loss)
            
            # Store losses and metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | '
                  f'Train RMSE: {train_rmse:.6f} | '
                  f'Val RMSE: {val_rmse:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the best model
                if model_save_path:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch,
                        'scaler': self.scaler,
                        'model_params': {
                            'input_size': input_size,
                            'hidden_size': self.hidden_size,
                            'num_layers': self.num_layers,
                            'dropout': self.dropout
                        }
                    }, model_save_path)
                    print(f'Model saved to {model_save_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Calculate total training time
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        # Evaluate on test set
        test_loss, test_rmse = self.evaluate(test_loader)
        history['test_loss'] = test_loss
        history['test_rmse'] = test_rmse
        print(f'Test Loss: {test_loss:.6f} | Test RMSE: {test_rmse:.6f}')
        
        self.history = history
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader containing the evaluation data
            
        Returns:
            loss: Mean squared error
            rmse: Root mean squared error
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(data_loader.dataset)
        rmse = np.sqrt(avg_loss)
        
        return avg_loss, rmse
    
    def predict(self, sequence):
        """
        Make a prediction for a single sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, input_size)
            
        Returns:
            Predicted value (denormalized)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(sequence_tensor).item()
        
        # Denormalize prediction
        if self.scaler:
            prediction = prediction * (self.scaler['max'] - self.scaler['min']) + self.scaler['min']
        
        return prediction
    
    def predict_batch(self, sequences):
        """
        Make predictions for a batch of sequences.
        
        Args:
            sequences: Input sequences of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predicted values (denormalized)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences_tensor).cpu().numpy()
        
        # Denormalize predictions
        if self.scaler:
            predictions = predictions * (self.scaler['max'] - self.scaler['min']) + self.scaler['min']
        
        return predictions
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try to load with weights_only=True to avoid serialization issues
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            weights_only = True
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(model_path, map_location=self.device)
            weights_only = False
        
        if weights_only:
            # When using weights_only, we need to manually recreate the model structure
            # We'll use default parameters since we can't load them from the checkpoint
            self.model = LSTMModel(
                input_size=1,  # Default for univariate time series
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout
            ).to(self.device)
            
            # Load just the model weights
            self.model.load_state_dict(checkpoint)
            
            print(f"Model loaded from {model_path} (weights only)")
        else:
            # Extract model parameters
            model_params = checkpoint['model_params']
            
            # Initialize model with the same parameters
            self.model = LSTMModel(
                input_size=model_params['input_size'],
                hidden_size=model_params['hidden_size'],
                num_layers=model_params['num_layers'],
                output_size=1,
                dropout=model_params['dropout']
            ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load scaler parameters
            self.scaler = checkpoint['scaler']
            
            print(f"Model loaded from {model_path} (full checkpoint)")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def plot_history(self, save_path=None):
        """
        Plot training and validation loss/RMSE.
        
        Args:
            save_path: Path to save the plot image
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        if self.history['test_loss'] is not None:
            plt.axhline(y=self.history['test_loss'], color='r', linestyle='-', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot RMSE
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_rmse'], label='Train RMSE')
        plt.plot(self.history['val_rmse'], label='Validation RMSE')
        if self.history['test_rmse'] is not None:
            plt.axhline(y=self.history['test_rmse'], color='r', linestyle='-', label='Test RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to demonstrate the usage of the LSTM model.
    """
    # Path to the preprocessed data
    data_path = '../data/processed/sequence_data.npz'
    
    # Path to save the trained model
    model_save_path = '../models/saved/lstm_model.pth'
    
    # Initialize the LSTM predictor
    predictor = TrafficLSTMPredictor(
        sequence_length=16,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        dropout=0.2
    )
    
    # Train the model
    history = predictor.train(
        data_path=data_path,
        epochs=50,
        patience=10,
        model_save_path=model_save_path
    )
    
    # Plot training history
    predictor.plot_history(save_path='../results/lstm_training_history.png')
    
    print("LSTM model training completed.")


if __name__ == "__main__":
    main()
