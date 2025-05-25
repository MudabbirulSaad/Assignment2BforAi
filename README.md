# Traffic-Based Route Guidance System (TBRGS)

## 🚀 Project Overview

The Traffic-Based Route Guidance System (TBRGS) is a comprehensive application designed to predict traffic flow and suggest optimal routes using real-world SCATS (Sydney Coordinated Adaptive Traffic System) data. It leverages machine learning models for traffic prediction and various graph-based search algorithms for route optimization. The system features a web-based GUI for interactive route planning and visualization.

---

## 📁 Directory Structure

The project is organized as follows:

```
mudabbirulsaad/assignment2bforai/
├── app/
│   ├── config/
│   │   └── config.py             # Central configuration settings
│   ├── core/
│   │   ├── integration/          # Modules for system integration (data, ML, routing)
│   │   │   ├── data/
│   │   │   │   └── coordinate_corrections.json # Manual SCATS coordinate fixes
│   │   │   └── ... (edge_weight_updater.py, route_predictor.py, scats_router.py, etc.)
│   │   ├── logging/
│   │   │   └── logger.py           # Centralized logging
│   │   ├── methods/                # Implementations of routing algorithms (A*, BFS, etc.)
│   │   ├── ml/                     # Machine learning models, training, and evaluation
│   │   │   ├── checkpoints/        # Saved model checkpoints
│   │   │   ├── evaluation/         # Evaluation reports and plots
│   │   │   └── ... (lstm_model.py, gru_model.py, model_trainer.py, etc.)
│   │   └── utils/                  # Utility scripts (graph handling, data loading)
│   ├── dataset/                    # Scripts for SCATS data processing and feature engineering
│   ├── gui/                        # Gradio web interface and map visualization
│   └── tests/                      # Test suite and results
│       ├── results/                # Stored test results for different models
│       └── run_tests.py            # Main script to execute tests
├── requirements.txt                # Project dependencies
├── run_gradio.py                   # Script to launch the Gradio web UI
└── run_tests.py                    # Script to run the test suite
```

---

## ✨ Core Components & Functionalities

### 1. Configuration (`app/config/config.py`)
Provides centralized configuration for file paths, project parameters, traffic conversion formulas, data processing settings, and ML model hyperparameters.

### 2. Data Processing (`app/dataset/`)
This suite of scripts handles the end-to-end processing of SCATS data:
* **Loading**: Ingests raw SCATS data, typically from an Excel file (`Scats Data October 2006.xls`).
* **Reshaping**: Converts data from a wide format (96 columns for 15-minute intervals) to a long format.
* **Time Conversion**: Handles Excel date serial numbers and extracts detailed time features (hour, day of week, etc.).
* **Feature Engineering**: Creates advanced features for ML models, including lag features, rolling statistics, peak hour indicators, and site-specific normalization.
* **Data Splitting**: Chronologically splits data into training, validation, and test sets.
* **Validation**: Includes scripts to analyze data quality and validate processed datasets.
* The main script `app/dataset/process_data.py` orchestrates this entire pipeline.

### 3. Machine Learning Models (`app/core/ml/`)
The system implements several deep learning models for traffic flow prediction:
* **LSTM (Long Short-Term Memory)**: Captures long-term temporal dependencies.
* **GRU (Gated Recurrent Unit)**: A more efficient variant of LSTM.
* **CNN-RNN Hybrid**: Combines Convolutional Neural Networks for spatial/local feature extraction and Recurrent Neural Networks for temporal modeling.
* **Ensemble Model**: Combines predictions from the base models (LSTM, GRU, CNN-RNN) to improve accuracy and robustness.
* **Base Model (`base_model.py`)**: An abstract class providing common functionalities for all ML models, including training, evaluation, prediction, and model persistence.
* **Model Training & Management**:
    * `ModelTrainer`: Manages the training, evaluation, and comparison of different models.
    * `TrafficPredictor`: Integrates trained ML models to provide traffic flow predictions for SCATS sites.
    * `ModelIntegration`: Provides a unified interface for using multiple ML models and supports ensemble predictions.
    * `ModelAdapter`: Handles model compatibility issues, especially for input feature dimension changes.
* **Evaluation (`evaluator.py`, `generate_model_comparison_plots.py`)**: Tools for comprehensive model assessment, including metrics calculation, statistical testing, site-specific analysis, temporal pattern analysis, and generation of comparison reports and visualizations.
    * Evaluation results (JSON and plots) are stored in `app/core/ml/evaluation/`.

### 4. Routing Algorithms & Graph Integration (`app/core/methods/`, `app/core/integration/`)
The system uses a graph representation of the road network and various algorithms to find optimal routes:
* **Search Algorithms**: Implements A*, Breadth-First Search (BFS), Depth-First Search (DFS), Greedy Best-First Search (GBFS), Iterative Deepening DFS (IDDFS), and a custom Bidirectional Weighted A* (BDWA).
* **Graph Representation**:
    * `app/core/methods/graph.py`: Basic graph class.
    * `app/core/utils/graph.py`: An `EnhancedGraph` class tailored for SCATS data, supporting coordinates and dynamic edge weights.
* **Integration Components**:
    * `SiteMapper`: Maps SCATS site IDs to graph nodes, addressing coordinate discrepancies using `coordinate_corrections.json`.
    * `FlowSpeedConverter`: Converts predicted traffic flow values into travel speeds.
    * `TravelTimeCalculator`: Estimates travel times between SCATS sites considering distance, speed, and intersection delays.
    * `EdgeWeightUpdater`: Dynamically adjusts graph edge weights based on real-time or predicted traffic conditions.
    * `RoutePredictor` & `SCATSRouter`: Core classes that integrate traffic predictions with routing algorithms to find and rank optimal routes.
    * `MLRouteIntegration`: Serves as a bridge between the ML traffic prediction models and the route prediction system.
    * `GeoCalculator`: Provides methods for geographic distance calculations (Euclidean, Haversine).
    * `RoutingAdapter`: Adapts Part A routing algorithms for use within the TBRGS.

### 5. Logging (`app/core/logging/logger.py`)
A centralized logging system (`TBRGSLogger`) is implemented to provide consistent logging throughout the application, with outputs to both console and timestamped log files.

### 6. Graphical User Interface (`app/gui/gradio_app.py`)
An interactive web interface built using Gradio:
* Allows users to select origin and destination SCATS sites.
* Lets users choose a specific time for traffic prediction (e.g., current time, peak hours).
* Supports selection of different ML models (LSTM, GRU, CNN-RNN, Ensemble) for traffic prediction.
* Visualizes the top N optimal routes on a map using Plotly (via `EnhancedMapVisualizer`).

### 7. Testing (`app/tests/`)
A dedicated test suite evaluates the system's performance:
* Uses 10 predefined real-life scenarios based on Melbourne SCATS data.
* Compares the routes and travel times predicted by LSTM, GRU, CNN-RNN, and Ensemble models.
* Tests are conducted across different times of day (Morning Peak, Midday, Evening Peak, Night).
* Results are saved in JSON format and a comprehensive Markdown report with visualizations is generated.

---

## 💾 Data

* **Raw Data**: The primary input is SCATS traffic data, expected to be in an Excel file format (e.g., `Scats Data October 2006.xls` as referenced in `app/config/config.py`).
* **Coordinate Corrections**: `app/core/integration/data/coordinate_corrections.json` stores manual adjustments for SCATS site geographical coordinates to improve mapping accuracy.
* **Processed Data**: The data processing pipeline generates several CSV files in the `app/dataset/processed/` directory (configurable), including:
    * `scats_traffic.csv`: Reshaped and feature-engineered traffic data.
    * `scats_site_reference.csv`: Information about unique SCATS sites.
    * `train_data.csv`, `val_data.csv`, `test_data.csv`: Split datasets for ML model training.
    * `scats_graph.json`: A graph representation of the SCATS network.

---

## 🛠️ Setup and Installation

1.  **Clone the repository.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The project dependencies are listed in `requirements.txt` at the root of the project and `app/gui/requirements.txt`. Consolidate these into a single `requirements.txt` at the root and install:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    * `pandas`
    * `numpy`
    * `torch` (PyTorch for ML models)
    * `scikit-learn`
    * `matplotlib`
    * `seaborn`
    * `openpyxl` & `xlrd` (for Excel file reading)
    * `tabulate`
    * `gradio>=3.50.0` (for the web UI)
    * `folium>=0.14.0` (alternative map visualizer)
    * `plotly` (for enhanced map visualization)
    * `scipy` (for statistical tests)
    * `pytz` (for timezone handling)
    * `geopy` (for geodesic distance calculations)
    * `optuna` (for hyperparameter optimization)

---

## ▶️ How to Run

### 1. Data Processing
Before training models or running the application, process the raw SCATS data:
* Ensure the path to your raw SCATS Excel file is correctly set in `app/config/config.py` (or provide it as a command-line argument).
* Run the main data processing script:
    ```bash
    python -m app.dataset.process_data
    ```
    Or, to specify file paths:
    ```bash
    python -m app.dataset.process_data --excel-file path/to/your/ScatsData.xls --output-dir path/to/processed_data
    ```
    This will generate the necessary CSV files in the specified output directory (default: `app/dataset/processed/`).

### 2. Model Training
* **Train Base Models (LSTM, GRU, CNN-RNN):**
    ```bash
    python -m app.core.ml.train_traffic_models --output_dir app/core/ml/checkpoints
    ```
    Adjust `--epochs`, `--learning_rate`, etc., as needed. Trained models and checkpoints will be saved in `app/core/ml/checkpoints/`.
* **Train Ensemble Model (after base models are trained):**
    ```bash
    python -m app.core.ml.train_ensemble_model
    ```

### 3. Running Tests
To execute the test suite and evaluate model performance across different scenarios:
```bash
python run_tests.py
```
This script internally calls `app/tests/run_tests.py`. Test results (JSON files, Markdown report, visualizations) will be saved in `app/tests/results/`.

### 4. Launching the Gradio Web UI
To start the interactive web application:
```bash
python run_gradio.py
```
This will launch the Gradio interface, typically accessible at `http://127.0.0.1:7860`.

---

## ⚙️ Configuration

The primary configuration file for the project is `app/config/config.py`. It allows you to set:
* File paths for raw and processed data.
* Parameters for traffic flow to travel time conversion.
* Data processing parameters (e.g., SCATS column names, expected sites).
* Time splits for train/validation/test datasets.
* Peak hour definitions.
* Default ML model hyperparameters.

---

## 🧪 Testing Framework

The project includes a testing framework located in `app/tests/`.
* **Test Scenarios**: 10 real-life scenarios using Melbourne SCATS data are defined to test route predictions.
* **Models Tested**: LSTM, GRU, CNN-RNN, and an Ensemble model are evaluated.
* **Time Periods**: Tests are run for Morning Peak, Midday, Evening Peak, and Night conditions.
* **Output**: Test results, including travel times and distances for each scenario and model, are saved in `app/tests/results/`. A comparative report in Markdown and visualizations are also generated.

---
