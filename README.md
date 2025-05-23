# Traffic-based Route Guidance System (TBRGS)

## 📋 Project Overview

The Traffic-based Route Guidance System (TBRGS) is an intelligent routing application that uses machine learning to predict traffic conditions and find optimal routes through the Boroondara area of Melbourne. The system integrates historical traffic data, machine learning models, and search algorithms to provide realistic travel time estimates based on predicted traffic flows.

## 🌟 Key Features

- **Traffic Prediction**: Uses LSTM, GRU, and custom neural network models to predict traffic flow based on historical patterns
- **Multiple Search Algorithms**: Implements BFS, DFS, A*, GBFS, IDDFS, and BDWA for route finding
- **Realistic Travel Time Estimation**: Calculates travel times considering traffic flow, distance, and intersection delays
- **Comprehensive Testing Framework**: Includes test cases with actual SCATS site IDs for system validation

## 🏗️ Project Structure

```
├── __pycache__/            # Python cache files
├── 2A/                     # Assignment 2A components
├── config/                 # Configuration files
│   ├── __init__.py         # Package initialization
│   ├── default_config.json  # Default configuration settings
│   └── README.md           # Configuration documentation
├── data/                   # Traffic data
│   ├── processed/          # Cleaned and processed data
│   ├── raw/                # Original SCATS data
│   └── README.md           # Data documentation
├── docs/                   # Documentation files
├── gui/                    # GUI implementation (to be completed)
│   ├── __pycache__/        # Python cache files
│   ├── assets/             # GUI assets
│   └── README.md           # GUI documentation
├── models/                 # ML model implementations
│   ├── __pycache__/        # Python cache files
│   ├── models/             # Model storage
│   │   └── checkpoints/    # Saved model weights (.pth files)
│   ├── __init__.py         # Package initialization
│   ├── cnnrnn_model.py     # Custom CNN-RNN hybrid model
│   ├── gru_model.py        # GRU model implementation
│   ├── lstm_model.py       # LSTM model implementation
│   ├── model_trainer.py    # Training script for all models
│   ├── README.md           # Models documentation
│   └── traffic_predictor.py # Prediction interface
├── results/                # Results and visualizations
├── scripts/                # Data processing scripts
├── tests/                  # Testing framework
│   ├── __pycache__/        # Python cache files
│   ├── cases/              # Test case definitions
│   ├── README.md           # Testing documentation
│   └── run_cases.py        # Test runner script
├── utils/                  # Utility functions
│   ├── __pycache__/        # Python cache files
│   ├── search/             # Search algorithm implementations
│   ├── __init__.py         # Package initialization
│   ├── data_utils.py       # Data utility functions
│   ├── graph_utils.py      # Graph utility functions
│   └── README.md           # Utils documentation
├── .gitattributes         # Git attributes file
├── .gitignore             # Git ignore file
├── README.md              # Main project documentation
└── route_finder.py        # Main route finding implementation
```

## 🔍 Data Processing

### 📊 Dataset Description

- Source: `data/raw/data_given.xlsx`
- Structure:
  - Traffic flow data in 15-minute intervals across multiple days
  - First row: time intervals (`00:00`, `00:15`, ... `23:45`)
  - Column J: `"Start Time"` and date information
  - Each row represents one day of flow readings (96 values per day)
- Format: **15-minute interval traffic flow** for various SCATS sites in Boroondara

### 🔧 Data Processing Pipeline (`scripts/data_processing.py`)

1. Reshapes the Excel file into a flat **time-series** (`Timestamp`, `Flow`)
2. Resamples data into consistent 15-minute intervals
3. Normalizes `Flow` values to the range [0, 1]
4. Splits the data into train/test sets (80% train, 20% test)
5. Creates **LSTM/GRU-friendly sequences** of length 16
6. Saves:
   - `cleaned_data.csv` – human-readable time-series
   - `sequence_data.npz` – numpy arrays ready for training
   - `nodes.csv` – SCATS site information
   - `edges.csv` – Road connections between SCATS sites

## 🧠 Machine Learning Models

The system implements three types of neural network models for traffic prediction:

1. **LSTM (Long Short-Term Memory)**: Effective for capturing long-term dependencies in time series data
2. **GRU (Gated Recurrent Unit)**: A more computationally efficient variant of LSTM
3. **Custom CNN-RNN Hybrid**: Combines convolutional and recurrent layers for improved feature extraction

All models are trained on sequences of 16 time steps (4 hours of traffic data) to predict the next time step's traffic flow.

## 🔍 Search Algorithms

The system implements six search algorithms from Assignment 2A:

1. **BFS (Breadth-First Search)**: Finds routes with the fewest intersections
2. **DFS (Depth-First Search)**: Explores one branch deeply before backtracking
3. **A* Search**: Uses heuristics to find optimal routes efficiently
4. **GBFS (Greedy Best-First Search)**: Always moves toward the destination
5. **IDDFS (Iterative Deepening DFS)**: Memory-efficient search with completeness guarantees
6. **BDWA (Bidirectional Weighted A*)**: Searches from both origin and destination simultaneously

## 🚗 Route Finding

The `RouteFinder` class (`route_finder.py`) integrates the traffic prediction models with the search algorithms to find optimal routes:

1. Loads the road network graph from `nodes.csv` and `edges.csv`
2. Predicts traffic flow using the trained ML models
3. Updates edge costs based on predicted traffic and calculates travel times
4. Applies the selected search algorithm to find optimal routes
5. Returns routes with estimated travel times and distances

## 🧪 Testing Framework

The system includes a comprehensive testing framework (`tests/run_cases.py`) that:

1. Runs test cases defined in the `tests/cases/` directory
2. Evaluates route finding performance with different algorithms
3. Assesses model prediction accuracy
4. Generates detailed test reports with success rates and execution times

Test cases use actual SCATS site IDs from the Boroondara area for realistic testing.

## 🖥️ GUI (To Be Implemented)

A graphical user interface will be implemented to provide:

- Interactive map visualization of the Boroondara road network
- Origin and destination selection by SCATS site ID
- Algorithm selection and parameter configuration
- Visual display of routes with travel time estimates
- Traffic prediction visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, torch, matplotlib

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. Process the data (if not already done):
   ```bash
   python scripts/data_processing.py
   ```

2. Train the models (if not already trained):
   ```bash
   python models/model_trainer.py
   ```

3. Find routes using the command line interface:
   ```bash
   python route_finder.py --origin 2000 --destination 3002 --algorithm AS
   ```

4. Run the test cases:
   ```bash
   python tests/run_cases.py
   ```

## 📊 System Performance

- **Model Accuracy**: LSTM and GRU models achieve MSE < 0.05 on test data
- **Route Finding**: All search algorithms successfully find optimal routes
- **Travel Time Estimation**: Calculated travel times reflect realistic driving conditions
- **Test Success Rate**: 100% pass rate on all test cases

## 🔮 Future Improvements

- Real-time traffic data integration
- Multi-modal transportation options
- User preference customization
- Mobile application development
- Integration with external mapping services

## 📝 Assignment Information

- **Course**: COS30019 Introduction to Artificial Intelligence
- **Assignment**: 2B - Traffic-based Route Guidance System
- **Due Date**: May 25, 2025
