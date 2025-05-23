# Machine Learning Models

This directory contains the implementation of machine learning models for traffic prediction.

## Implemented Models

- **LSTM (Long Short-Term Memory)**: `lstm_model.py` - Implementation of LSTM neural network for time series prediction
- **GRU (Gated Recurrent Unit)**: `gru_model.py` - Implementation of GRU neural network for time series prediction
- **CNN-RNN Hybrid**: `cnnrnn_model.py` - Custom hybrid model combining convolutional and recurrent layers

## Training and Prediction

- `model_trainer.py` - Contains code for training the models with sequence data
- `traffic_predictor.py` - Provides functionality for making traffic predictions using trained models

## Usage

Models are trained using the processed data from `data/processed/sequence_data.npz`. Trained model checkpoints are stored in the `models/models` directory.
