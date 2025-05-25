# Traffic-Based Route Guidance System (TBRGS)
## COS30019 - Introduction to AI - Assignment 2, Part B

## 🚀 Project Overview

This project is the **Part B: Machine Learning and Software Integration** component of Assignment 2 for COS30019 - Introduction to AI. [cite: 1] It involves implementing Machine Learning (ML) algorithms to train models for traffic prediction and integrating these predictors with the routing algorithms developed in Part A to create a fully functional Traffic-Based Route Guidance System (TBRGS). [cite: 2] The system focuses on the Boroondara area, utilizing a provided VicRoads dataset containing traffic flow data (number of cars passing an intersection every 15 minutes). [cite: 4, 17]

The core objective is to predict traffic conditions and use these predictions to estimate travel times, ultimately finding optimal routes between specified SCATS (Sydney Coordinated Adaptive Traffic System) sites. [cite: 12, 16, 18] The system includes a Graphical User Interface (GUI) for user interaction, parameter settings, and route visualization. [cite: 8]

---

## 🎯 Assignment Goals

As per the assignment requirements, the main tasks for this project are:
1.  **Data Processing**: Implement programs to extract and process data from the provided Boroondara dataset, structuring it appropriately for training and testing ML models. [cite: 9]
2.  **Machine Learning Model Training**: Implement at least three ML algorithms for traffic flow prediction. This must include **LSTM** and **GRU** as basic deep learning techniques, plus at least one other ML technique identified and approved by the tutor. [cite: 10, 6, 7, 13] These models are to be trained and tested using the provided dataset. [cite: 5, 14] A comprehensive comparison between the different models is required. [cite: 7]
3.  **Travel Time Estimation**: Implement a method to estimate travel time for each edge (road segment) on the Boroondara map, considering predicted traffic conditions. [cite: 11]
4.  **Integration with Part A**: Integrate the traffic prediction models and travel time estimations with the search algorithms from Part A. This involves replacing static edge costs with dynamic, predicted travel times to find the top-k (up to five) optimal paths between an origin (O) and destination (D). [cite: 12, 15, 16, 19]

---

## 📋 System Requirements (Key Features)

* **ML-Powered Traffic Prediction**:
    * Train ML models (LSTM, GRU, and at least one other) using the Boroondara dataset. [cite: 6, 7, 13]
    * Provide meaningful traffic predictions based on these trained models.
    * Evaluate and compare the performance of the different ML models. [cite: 14]
* **Traffic-Based Routing**:
    * Focus on the Boroondara road network. [cite: 17]
    * Accept user input for origin and destination as SCATS site numbers (e.g., O=2000, D=3002). [cite: 18]
    * Calculate and return up to five (5) optimal routes from origin to destination. [cite: 19]
    * Dynamically update edge costs in the graph with predicted travel times. [cite: 16]
* **Travel Time Calculation Assumptions**:
    * Default speed limit: 60km/h on all links. [cite: 20]
    * Travel time approximation: Based on accumulated traffic volume at the destination SCATS site and distance, using a provided conversion formula. [cite: 21]
    * Intersection delay: An average delay of 30 seconds to pass each controlled intersection. [cite: 22]
* **Graphical User Interface (GUI)**:
    * Provide a GUI for user input (origin, destination), parameter settings (e.g., ML model selection, time of day), and visualization of results. [cite: 8]
    * A configuration file should manage default settings. [cite: 8]
* **Coordinate Mapping**:
    * Address potential discrepancies between SCATS site latitude/longitude and actual intersection locations on maps like Google Maps. [cite: 54]

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
│   │   └── ml/                     # Machine learning models, training, and evaluation
│   │       ├── checkpoints/        # Saved model checkpoints
│   │       ├── evaluation/         # Evaluation reports and plots
│   │       │   ├── reports/
│   │       │   ├── site_analysis/
│   │       │   └── temporal_analysis/
│   │       └── ... (lstm_model.py, gru_model.py, model_trainer.py, etc.)
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
Provides centralized configuration for file paths (including the Boroondara dataset), project parameters, traffic conversion formulas, data processing settings, and ML model hyperparameters.

### 2. Data Processing (`app/dataset/`)
This suite of scripts handles the end-to-end processing of SCATS data as required by the assignment[cite: 9]:
* **Loading**: Ingests raw SCATS data, from the provided VicRoads Boroondara dataset (Excel file).
* **Reshaping**: Converts data from a wide format (96 columns for 15-minute intervals) to a long format.
* **Time Conversion**: Handles Excel date serial numbers and extracts detailed time features (hour, day of week, etc.).
* **Feature Engineering**: Creates advanced features for ML models, including lag features, rolling statistics, peak hour indicators, and site-specific normalization.
* **Data Splitting**: Chronologically splits data into training, validation, and test sets.
* **Validation**: Includes scripts to analyze data quality and validate processed datasets.
* The main script `app/dataset/process_data.py` orchestrates this entire pipeline.

### 3. Machine Learning Models (`app/core/ml/`)
The system implements several deep learning models for traffic flow prediction, fulfilling the assignment requirement of using LSTM, GRU, and at least one other technique[cite: 6, 7, 10, 13]:
* **LSTM (Long Short-Term Memory)**: Captures long-term temporal dependencies.
* **GRU (Gated Recurrent Unit)**: A more efficient variant of LSTM.
* **CNN-RNN Hybrid**: Combines Convolutional Neural Networks for spatial/local feature extraction and Recurrent Neural Networks for temporal modeling (serves as the third required ML technique).
* **Ensemble Model**: Combines predictions from the base models (LSTM, GRU, CNN-RNN) to potentially improve accuracy and robustness.
* **Base Model (`base_model.py`)**: An abstract class providing common functionalities for all ML models, including training, evaluation, prediction, and model persistence.
* **Model Training & Management**:
    * `ModelTrainer`: Manages the training, evaluation, and comparison of different models. The Boroondara dataset is used for training these models. [cite: 5]
    * `TrafficPredictor`: Integrates trained ML models to provide traffic flow predictions for SCATS sites.
    * `ModelIntegration`: Provides a unified interface for using multiple ML models and supports ensemble predictions.
    * `ModelAdapter`: Handles model compatibility issues, especially for input feature dimension changes.
* **Evaluation (`evaluator.py`, `generate_model_comparison_plots.py`)**: Tools for comprehensive model assessment, including metrics calculation, statistical testing, site-specific analysis, and temporal pattern analysis. JSON reports and plots are generated, fulfilling the requirement for a comprehensive comparison. [cite: 7]
    * Evaluation results (JSON and plots) are stored in `app/core/ml/evaluation/`.

### 4. Routing Algorithms & Graph Integration (`app/core/methods/`, `app/core/integration/`)
The system integrates Part A search algorithms with the Boroondara map and ML predictions[cite: 15, 12]:
* **Search Algorithms**: Implements A*, Breadth-First Search (BFS), Depth-First Search (DFS), Greedy Best-First Search (GBFS), Iterative Deepening DFS (IDDFS), and a custom Bidirectional Weighted A* (BDWA).
* **Graph Representation**:
    * `app/core/methods/graph.py`: Basic graph class.
    * `app/core/utils/graph.py`: An `EnhancedGraph` class tailored for SCATS data, supporting coordinates and dynamic edge weights.
* **Integration Components**:
    * `SiteMapper`: Maps SCATS site IDs (intersections) to graph nodes, addressing coordinate discrepancies using `coordinate_corrections.json` as highlighted in the assignment.
    * `FlowSpeedConverter`: Converts predicted traffic flow values (vehicles per 15 minutes/hour) into travel speeds (km/h) based on the provided formula.
    * `TravelTimeCalculator`: Estimates travel times for road segments considering distance, speed, and a 30-second intersection delay.
    * `EdgeWeightUpdater`: Dynamically adjusts graph edge weights (costs) based on the predicted travel times.
    * `RoutePredictor` & `SCATSRouter`: Core classes that integrate traffic predictions with routing algorithms to find and rank (up to 5) optimal routes.
    * `MLRouteIntegration`: Serves as a bridge between the ML traffic prediction models and the route prediction system.
    * `GeoCalculator`: Provides methods for geographic distance calculations (Euclidean, Haversine).
    * `RoutingAdapter`: Adapts Part A routing algorithms for use within the TBRGS.

### 5. Logging (`app/core/logging/logger.py`)
A centralized logging system (`TBRGSLogger`) is implemented to provide consistent logging throughout the application, with outputs to both console and timestamped log files.

### 6. Graphical User Interface (`app/gui/gradio_app.py`)
An interactive web interface built using Gradio, fulfilling the assignment requirement[cite: 8]:
* Allows users to select origin and destination SCATS sites by their numbers. [cite: 18]
* Lets users choose a specific time for traffic prediction (e.g., current time, peak hours).
* Supports selection of different ML models (LSTM, GRU, CNN-RNN, Ensemble) for traffic prediction.
* Visualizes the top N optimal routes on a map using Plotly (via `EnhancedMapVisualizer`).

### 7. Testing (`app/tests/`)
A dedicated test suite evaluates the system's performance as required[cite: 36]:
* Uses 10 predefined real-life scenarios based on Melbourne SCATS data for the Boroondara area.
* Compares the routes and travel times predicted by LSTM, GRU, CNN-RNN, and Ensemble models.
* Tests are conducted across different times of day (Morning Peak, Midday, Evening Peak, Night).
* Results are saved in JSON format and a comprehensive Markdown report with visualizations is generated, facilitating model comparison.

---

## 💾 Data

* **Raw Data**: The primary input is the VicRoads SCATS traffic flow dataset for the city of Boroondara (traffic volume per 15 minutes). This dataset is expected to be in an Excel file format (e.g., `Scats Data October 2006.xls` as referenced in `app/config/config.py`).
* **Coordinate Corrections**: `app/core/integration/data/coordinate_corrections.json` stores manual adjustments for SCATS site geographical coordinates to address the known issue of SCATS lat/long not aligning with actual map intersections.
* **Processed Data**: The data processing pipeline (`app/dataset/process_data.py`) generates several CSV files in the `app/dataset/processed/` directory (configurable), including:
    * `scats_traffic.csv`: Reshaped and feature-engineered traffic data.
    * `scats_site_reference.csv`: Information about unique SCATS sites.
    * `train_data.csv`, `val_data.csv`, `test_data.csv`: Split datasets for ML model training and testing. [cite: 5]
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
    The project utilizes libraries such as PyTorch, pandas, numpy, scikit-learn, Gradio, etc. [cite: 4] Consolidate dependencies from `requirements.txt` and `app/gui/requirements.txt` into a single root `requirements.txt` and install:
    ```bash
    pip install -r requirements.txt
    ```
    Ensure the following key dependencies are met:
    * `pandas`
    * `numpy`
    * `torch` (PyTorch for ML models)
    * `scikit-learn`
    * `matplotlib`
    * `seaborn`
    * `openpyxl` & `xlrd` (for Excel file reading)
    * `tabulate`
    * `gradio>=3.50.0` (for the web UI)
    * `folium>=0.14.0` (if used as an alternative map visualizer)
    * `plotly` (for enhanced map visualization)
    * `scipy` (for statistical tests)
    * `pytz` (for timezone handling)
    * `geopy` (for geodesic distance calculations)
    * `optuna` (for hyperparameter optimization, if used)

---

## ▶️ How to Run

### 1. Data Processing
Before training models or running the application, process the raw SCATS data:
* Ensure the path to your raw SCATS Excel file (Boroondara dataset) is correctly set in `app/config/config.py` (or provide it as a command-line argument). [cite: 4]
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
* **Train Base Models (LSTM, GRU, CNN-RNN):** [cite: 6, 7]
    ```bash
    python -m app.core.ml.train_traffic_models --output_dir app/core/ml/checkpoints
    ```
    Adjust `--epochs`, `--learning_rate`, etc., as needed. Trained models and checkpoints will be saved in `app/core/ml/checkpoints/`.
* **Train Ensemble Model (after base models are trained):**
    ```bash
    python -m app.core.ml.train_ensemble_model
    ```

### 3. Running Tests
To execute the test suite and evaluate model performance across different scenarios[cite: 36]:
```bash
python run_tests.py
```
This script internally calls `app/tests/run_tests.py`. Test results (JSON files, Markdown report, visualizations) will be saved in `app/tests/results/`.

### 4. Launching the Gradio Web UI
To start the interactive web application[cite: 8]:
```bash
python run_gradio.py
```
This will launch the Gradio interface, typically accessible at `http://127.0.0.1:7860`.

---

## ⚙️ Configuration

The primary configuration file for the project is `app/config/config.py`. This file is crucial for setting:
* Paths to raw and processed datasets (including the Boroondara dataset).
* Parameters for traffic flow to travel time conversion, adhering to assignment guidelines. [cite: 21]
* Data processing settings like SCATS column names and expected site identifiers.
* Time splits for train/validation/test datasets.
* Definitions for peak traffic hours.
* Default hyperparameters for the ML models.

---

## 🧪 Testing Framework

The project includes a testing framework located in `app/tests/` designed to ensure software quality and evaluate ML model performance. [cite: 36]
* **Test Scenarios**: 10 real-life scenarios using Melbourne SCATS data from the Boroondara area are defined to test route predictions under various conditions.
* **Models Tested**: The test suite is designed to compare the performance of LSTM, GRU, CNN-RNN, and an Ensemble model, fulfilling the assignment's requirement for model comparison.
* **Time Periods**: Tests are conducted across different times of day (Morning Peak, Midday, Evening Peak, Night) to assess model behavior under varying traffic loads.
* **Output**: Test results, including predicted travel times and distances for each scenario and model, are saved in `app/tests/results/`. A comprehensive Markdown report (`model_comparison_report.md`) and visualizations are automatically generated to aid in analyzing model performance.

---

## 💡 Research Initiatives & Further Development

The assignment suggests several avenues for research and extension[cite: 39]:
* **Expanded Datasets**: Utilizing more comprehensive VicRoads datasets for Victoria across multiple years, or exploring other open data sources like PEMS or UK Highways England data. [cite: 47, 50, 51, 52] This would involve tackling challenges of large-scale data processing. [cite: 48]
* **Advanced Visualization**: Enhancing route and traffic visualization, potentially inspired by Google Maps Traffic, using tools like OpenStreetMap. [cite: 53]
* **Custom Optimizations**: Exploring novel algorithms or clever optimizations for traffic prediction or route finding.

This project aims to follow good programming practices, including well-structured code and clear comments. [cite: 58]

---
