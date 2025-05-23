"""
Traffic Predictor for TBRGS
This module provides functionality to make traffic flow predictions using trained ML models
and integrate them with the route finding system.
"""

import os
import numpy as np
import torch
import pandas as pd
import json
from datetime import datetime, timedelta

from .lstm_model import TrafficLSTMPredictor
from .gru_model import TrafficGRUPredictor
from .cnnrnn_model import TrafficCNNRNNPredictor

# Add parent directory to path to import from utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_utils import TrafficGraph

class TrafficPredictor:
    """
    Class for making traffic flow predictions and updating the traffic graph.
    """
    def __init__(self, models_dir='models/checkpoints', model_type='ensemble', config=None):
        """
        Initialize the traffic predictor.
        
        Args:
            models_dir: Directory containing the trained models
            model_type: Type of model to use ('lstm', 'gru', 'custom', or 'ensemble')
        """
        self.models_dir = models_dir
        self.model_type = model_type
        self.models = {}
        
        # Load configuration if not provided
        if config is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default_config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
            
        self.sequence_length = self.config['data']['sequence_length']  # Number of time steps used for prediction
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """
        Load the trained ML models.
        """
        # Print debug information about the models directory
        print(f"Looking for models in directory: {self.models_dir}")
        print(f"Absolute path: {os.path.abspath(self.models_dir)}")
        print(f"Directory exists: {os.path.exists(self.models_dir)}")
        if os.path.exists(self.models_dir):
            print(f"Contents of models directory: {os.listdir(self.models_dir)}")
        
        # Define model paths
        lstm_path = os.path.join(self.models_dir, 'lstm_model.pth')
        gru_path = os.path.join(self.models_dir, 'gru_model.pth')
        custom_path = os.path.join(self.models_dir, 'cnnrnn_model.pth')
        
        # Check if models exist
        models_exist = {
            'lstm': os.path.exists(lstm_path),
            'gru': os.path.exists(gru_path),
            'custom': os.path.exists(custom_path)
        }
        
        print(f"Model paths and existence:")
        print(f"LSTM: {lstm_path} - Exists: {models_exist['lstm']}")
        print(f"GRU: {gru_path} - Exists: {models_exist['gru']}")
        print(f"Custom: {custom_path} - Exists: {models_exist['custom']}")
        
        
        # Load LSTM model if needed
        if self.model_type in ['lstm', 'ensemble'] and models_exist['lstm']:
            try:
                lstm_model = TrafficLSTMPredictor()
                lstm_model.load_model(lstm_path)
                self.models['lstm'] = lstm_model
                print(f"LSTM model loaded from {lstm_path}")
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
        
        # Load GRU model if needed
        if self.model_type in ['gru', 'ensemble'] and models_exist['gru']:
            try:
                gru_model = TrafficGRUPredictor()
                gru_model.load_model(gru_path)
                self.models['gru'] = gru_model
                print(f"GRU model loaded from {gru_path}")
            except Exception as e:
                print(f"Error loading GRU model: {e}")
        
        # Load custom model if needed
        if self.model_type in ['custom', 'ensemble'] and models_exist['custom']:
            try:
                custom_model = TrafficCNNRNNPredictor()
                custom_model.load_model(custom_path)
                self.models['custom'] = custom_model
                print(f"Custom model loaded from {custom_path}")
            except Exception as e:
                print(f"Error loading custom model: {e}")
        
        # Check if any models were loaded
        if not self.models:
            raise ValueError(f"No models were loaded. Please train models first or check the model paths.")
    
    def predict_traffic_flow(self, historical_data):
        """
        Predict traffic flow based on historical data.
        
        Args:
            historical_data: Sequence of historical traffic flow values (sequence_length time steps)
            
        Returns:
            Predicted traffic flow
        """
        # Ensure historical data has the correct shape
        if len(historical_data) != self.sequence_length:
            raise ValueError(f"Historical data must have {self.sequence_length} time steps.")
        
        # Reshape data for model input
        sequence = np.array(historical_data).reshape(self.sequence_length, 1)
        
        # Make predictions with each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(sequence)
        
        # Return prediction based on model type
        if self.model_type == 'ensemble':
            # Average predictions from all models
            return np.mean(list(predictions.values()))
        else:
            # Return prediction from the specified model
            return predictions[self.model_type]
    
    def predict_traffic_flows(self, historical_data_dict):
        """
        Predict traffic flows for multiple edges based on historical data.
        
        Args:
            historical_data_dict: Dictionary mapping (source, destination) tuples to sequences of historical traffic flow values
            
        Returns:
            Dictionary mapping (source, destination) tuples to predicted traffic flow values
        """
        predictions = {}
        
        for (source, destination), historical_data in historical_data_dict.items():
            try:
                predictions[(source, destination)] = self.predict_traffic_flow(historical_data)
            except Exception as e:
                print(f"Error predicting traffic flow for edge ({source}, {destination}): {e}")
                # Use the most recent traffic flow as a fallback
                predictions[(source, destination)] = historical_data[-1]
        
        return predictions
    
    def update_traffic_graph(self, graph, historical_data_dict):
        """
        Update the traffic graph with predicted traffic flows.
        
        Args:
            graph: TrafficGraph instance
            historical_data_dict: Dictionary mapping (source, destination) tuples to sequences of historical traffic flow values
            
        Returns:
            Updated TrafficGraph instance
        """
        # Predict traffic flows
        predicted_flows = self.predict_traffic_flows(historical_data_dict)
        
        # Update the graph with predicted flows
        for (source, destination), flow in predicted_flows.items():
            graph.set_traffic_flow(source, destination, flow)
        
        # Update edge costs with travel times
        graph.update_edge_costs_with_travel_times()
        
        return graph
    
    def get_historical_data_from_dataframe(self, df, current_time, sequence_length=16, time_interval='1H'):
        """
        Extract historical data from a DataFrame for traffic prediction.
        
        Args:
            df: DataFrame containing traffic flow data with columns 'timestamp', 'source', 'destination', 'flow'
            current_time: Current timestamp
            sequence_length: Number of time steps to use for prediction
            time_interval: Time interval between consecutive data points
            
        Returns:
            Dictionary mapping (source, destination) tuples to sequences of historical traffic flow values
        """
        # Convert time_interval to timedelta
        if time_interval == '1H':
            delta = timedelta(hours=1)
        elif time_interval == '30min':
            delta = timedelta(minutes=30)
        elif time_interval == '15min':
            delta = timedelta(minutes=15)
        else:
            raise ValueError(f"Unsupported time interval: {time_interval}")
        
        # Generate timestamps for historical data
        timestamps = [current_time - (i + 1) * delta for i in range(sequence_length)]
        timestamps.reverse()  # Oldest first
        
        # Extract unique source-destination pairs
        source_dest_pairs = df[['source', 'destination']].drop_duplicates().values
        
        # Initialize historical data dictionary
        historical_data_dict = {}
        
        # Extract historical data for each source-destination pair
        for source, destination in source_dest_pairs:
            # Filter data for this source-destination pair
            edge_data = df[(df['source'] == source) & (df['destination'] == destination)]
            
            # Extract flow values for the timestamps
            historical_flows = []
            for ts in timestamps:
                # Find the closest timestamp in the data
                closest_row = edge_data.iloc[(edge_data['timestamp'] - ts).abs().argsort()[:1]]
                if not closest_row.empty:
                    flow = closest_row['flow'].values[0]
                else:
                    # If no data is available, use a default value (e.g., average flow)
                    flow = edge_data['flow'].mean() if not edge_data.empty else 0
                
                historical_flows.append(flow)
            
            # Store in dictionary
            historical_data_dict[(source, destination)] = historical_flows
        
        return historical_data_dict

def main():
    """
    Main function to demonstrate the usage of the traffic predictor.
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Create a simple traffic graph for testing
    nodes = {
        1: (0, 0),
        2: (1, 1),
        3: (2, 0),
        4: (1, -1)
    }
    
    edges = {
        1: [(2, 1.4), (4, 1.4)],
        2: [(1, 1.4), (3, 1.4)],
        3: [(2, 1.4), (4, 1.4)],
        4: [(1, 1.4), (3, 1.4)]
    }
    
    graph = TrafficGraph(nodes, edges)
    
    # Create some sample historical data
    historical_data_dict = {
        (1, 2): [500] * 16,  # Moderate traffic
        (1, 4): [200] * 16,  # Light traffic
        (2, 1): [500] * 16,
        (2, 3): [800] * 16,  # Heavy traffic
        (3, 2): [800] * 16,
        (3, 4): [700] * 16,
        (4, 1): [200] * 16,
        (4, 3): [700] * 16
    }
    
    # Initialize traffic predictor
    predictor = TrafficPredictor(models_dir='models/checkpoints', model_type='lstm', config=config)
    
    # Update traffic graph with predicted flows
    updated_graph = predictor.update_traffic_graph(graph, historical_data_dict)
    
    # Print predicted traffic flows and travel times
    print("Predicted Traffic Flows and Travel Times:")
    print("-" * 50)
    print(f"{'Source':^10} | {'Destination':^12} | {'Traffic Flow':^15} | {'Travel Time (s)':^15}")
    print("-" * 50)
    
    for source in updated_graph.edges:
        for destination, _ in updated_graph.get_neighbors(source):
            flow = updated_graph.get_traffic_flow(source, destination)
            time = updated_graph.calculate_travel_time(source, destination)
            print(f"{source:^10} | {destination:^12} | {flow:^15.2f} | {time:^15.2f}")

if __name__ == "__main__":
    main()
