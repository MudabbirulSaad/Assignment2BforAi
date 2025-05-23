# Raw Data

This directory contains the original, unprocessed data files used by the Traffic-based Route Guidance System.

## Files

- `data_given.xlsx`: Original traffic data from VicRoads SCATS system containing traffic flow information, intersection details, and geographical coordinates

## Data Source

The raw data is sourced from the VicRoads SCATS (Sydney Coordinated Adaptive Traffic System) which provides traffic flow information at signalized intersections across Melbourne, Australia.

## Usage

This raw data is processed and transformed into more usable formats stored in the `processed/` directory. The processing includes:
- Cleaning and normalizing the data
- Extracting node and edge information for the traffic network
- Preparing sequence data for machine learning models
