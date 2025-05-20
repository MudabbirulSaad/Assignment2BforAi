# Assignment2BforAi
A2B
# COS30019 A2B – Data Processing (Member 1)

## 📥 What Was Given
- Raw Excel file: `data_given.xlsx`  
- It had traffic flow data in 15-minute intervals across multiple days.
- The first row had time intervals (0:00 to 23:45), and the "Date" values went down column J.

## 🔧 What I Have Done
- Converted the raw Excel into a proper timestamped time-series.
- Cleaned the data and filled any missing values.
- Normalized the traffic flow values between 0 and 1.
- Created training and testing sequences (16 time steps each) for ML models like LSTM and GRU.
- Saved:
  - `data/processed/cleaned_data.csv` → cleaned time-series data  
  - `data/processed/sequence_data.npz` → NumPy arrays ready for training
 

---

## 📊 Dataset Description

- Source: `data/raw/data_given.xlsx`
- Structure:
  - First row: time intervals (`0:00`, `0:15`, ... `23:45`)
  - Column J: `"Start Time"` and `"Date"` headings
  - From row 3 onward: each row = 1 day of flow readings (96 values per day)
- Format: **15-minute interval traffic flow** for various SCATS sites in Boroondara

---

## 🔧 What This Script Does (`scripts/data_processing.py`)

1. Reshapes the Excel file into a flat **time-series** (`Timestamp`, `Flow`)
2. Resamples data into consistent 15-minute intervals
3. Normalizes `Flow` values to the range [0, 1]
4. Splits the data into train/test sets (80% train, 20% test)
5. Creates **LSTM/GRU-friendly sequences** of length 16
6. Saves:
   - `cleaned_data.csv` – human-readable time-series
   - `sequence_data.npz` – numpy arrays ready for training

---

## 📁 What’s Inside `sequence_data.npz`

You can load it like this in Python:

```python
import numpy as np

data = np.load('data/processed/sequence_data.npz')

X_train = data['X_train']  # Shape: (samples, 16, 1)
y_train = data['y_train']  # Shape: (samples,)
X_test  = data['X_test']   # Shape: (samples, 16, 1)
y_test  = data['y_test']   # Shape: (samples,)


## 📦 How You Can Use It

To load the model-ready data:

```python
import numpy as np

data = np.load('data/processed/sequence_data.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
