# TBRGS Test Suite

This directory contains test scripts for the Traffic-Based Route Guidance System (TBRGS), focusing on evaluating the performance of different ML models for traffic prediction and route guidance.

## Test Scenarios

The test suite includes 10 real-life test scenarios using actual SCATS site data from Melbourne:

1. **Scenario 1**: Warrigal Rd/Toorak Rd to Union Rd/Maroondah Hwy
2. **Scenario 2**: Union Rd/Maroondah Hwy to Canterbury Rd
3. **Scenario 3**: High St/Warrigal Rd to Warrigal Rd/Toorak Rd
4. **Scenario 4**: Bulleen Rd to Burke Rd
5. **Scenario 5**: Princess St to High St
6. **Scenario 6**: Burke Rd to Warrigal Rd/High St
7. **Scenario 7**: High St to Bulleen Rd
8. **Scenario 8**: Warrigal Rd/Toorak Rd to Burke Rd
9. **Scenario 9**: Princess St to Union Rd/Maroondah Hwy
10. **Scenario 10**: Bulleen Rd to Warrigal Rd/High St

Each scenario is tested with four different ML models:
- LSTM
- GRU
- CNN-RNN
- Ensemble (combination of the above)

And at four different times of day:
- Morning Peak (8:00 AM)
- Midday (12:00 PM)
- Evening Peak (5:00 PM)
- Night (10:00 PM)

## Running the Tests

To run the test suite, execute the following command from the project root:

```bash
python -m app.tests.run_tests
```

## Test Results

The test results are saved in the `app/tests/results` directory:
- Individual JSON files for each model's results
- A comprehensive Markdown report comparing all models
- Visualizations showing performance comparisons

## Visualizations

The test suite generates the following visualizations:
1. Average Travel Time by Model
2. Travel Time Comparison by Scenario
3. Travel Time by Time of Day

These visualizations help identify which ML model performs best under different conditions and scenarios.
