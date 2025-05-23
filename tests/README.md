# Tests

This directory contains test cases and testing utilities for the Traffic-based Route Guidance System (TBRGS).

## Files and Directories

- `cases/`: Directory containing test case files with different origin-destination pairs and parameters
  - Contains 10 test case files (`test_case1.txt` through `test_case10.txt`) with various route scenarios
- `run_cases.py`: Script to run and evaluate the route finder on the test cases
- `__init__.py`: Python package initialization file

## Usage

To run the test cases, use the following command:

```bash
python tests/run_cases.py
```

This will execute all test cases and provide a summary of the results, including route quality, execution time, and algorithm performance.
