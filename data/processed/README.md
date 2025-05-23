# Processed Data

This directory contains processed and transformed data ready for use by the Traffic-based Route Guidance System.

## Files

- `cleaned_data.csv`: Cleaned and normalized traffic flow data
- `nodes.csv`: Node information for the traffic network, including IDs and coordinates
- `edges.csv`: Edge information for the traffic network, including source, target, distance, and traffic flow
- `sequence_data.npz`: Time series data prepared for machine learning models
- `synthetic_data.npz`: Synthetic data generated for testing and validation
- `synthetic_data_custom.npz`: Custom synthetic data for the CNN-RNN hybrid model
- `synthetic_data_gru.npz`: Synthetic data specifically formatted for the GRU model

## Usage

These processed files are used by:
- `route_finder.py`: Uses nodes.csv and edges.csv to build the traffic network graph
- Machine learning models: Use sequence_data.npz for training and evaluation
- Testing scripts: Use synthetic data files for validation and testing

## Data Format

- CSV files: Standard comma-separated values format
- NPZ files: NumPy compressed archive format containing multiple arrays for time series data
