# Scripts

This directory contains utility scripts for data processing and preparation for the Traffic-based Route Guidance System (TBRGS).

## Files

- `data_processing.py`: Script for processing raw traffic data from VicRoads SCATS system
  - Converts Excel data to CSV format
  - Extracts node and edge information for the traffic network
  - Generates sequence data for machine learning models
  - Creates synthetic data for testing

## Usage

The data processing script is used to prepare the raw data for use in the system:

```bash
python scripts/data_processing.py
```

This script reads from the raw data directory and outputs processed files to the processed data directory as specified in the configuration file.

## Dependencies

- pandas
- numpy
- scikit-learn
- openpyxl (for Excel file processing)

The script uses configuration settings from `config/default_config.json` to determine file paths.
