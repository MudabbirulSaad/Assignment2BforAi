# Data Directory

This directory contains the data used by the Traffic-based Route Guidance System (TBRGS).

## Structure

- `raw/`: Contains the original, unprocessed data files
- `processed/`: Contains processed and transformed data ready for use by the system

## Data Flow

1. Raw data is sourced from VicRoads SCATS traffic data
2. Data processing scripts convert raw data into usable formats
3. Processed data is used by the route finder and machine learning models

## Usage

The data in this directory is used for:
- Building the traffic network graph
- Training machine learning models for traffic prediction
- Evaluating route finding algorithms
