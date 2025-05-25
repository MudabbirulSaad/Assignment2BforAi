#!/usr/bin/env python3
"""
TBRGS Test Suite

This script runs a comprehensive set of tests for the Traffic-Based Route Guidance System,
comparing the routes predicted by different ML models (LSTM, GRU, CNN-RNN, and Ensemble)
using real-life SCATS site data.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Warning: tabulate not available. Some formatting will be simplified.")

# Set up the path to include the app directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app_dir = os.path.join(project_root, 'traefik')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tbrgs.tests")

# Import TBRGS components
from app.core.integration.scats_router import SCATSRouter
from app.core.integration.ml_route_integration import create_ml_route_integration

# Logger is already initialized above

class TBRGSTestSuite:
    """
    Test suite for the Traffic-Based Route Guidance System.
    
    This class runs a set of predefined test scenarios to compare the performance
    of different ML models for traffic prediction and route guidance.
    """
    
    def __init__(self):
        """Initialize the test suite."""
        # Load SCATS site data
        self.load_scats_data()
        
        # Define test scenarios
        self.define_test_scenarios()
        
        # Initialize results storage
        self.results = {
            "LSTM": [],
            "GRU": [],
            "CNN-RNN": [],
            "Ensemble": []
        }
        
        # Test times
        self.test_times = [
            ("Morning Peak", datetime.now().replace(hour=8, minute=0, second=0)),
            ("Midday", datetime.now().replace(hour=12, minute=0, second=0)),
            ("Evening Peak", datetime.now().replace(hour=17, minute=0, second=0)),
            ("Night", datetime.now().replace(hour=22, minute=0, second=0))
        ]
        
        logger.info("TBRGS Test Suite initialized")
    
    def load_scats_data(self):
        """Load SCATS site data from the processed dataset."""
        try:
            # Try multiple possible paths for the SCATS site reference data
            possible_paths = [
                # Direct path from app_dir
                os.path.join(app_dir, 'dataset', 'processed', 'scats_site_reference.csv'),
                # Path from project_root
                os.path.join(project_root, 'traefik', 'app', 'dataset', 'processed', 'scats_site_reference.csv'),
                # Path without 'app' subdirectory
                os.path.join(app_dir, 'dataset', 'processed', 'scats_site_reference.csv'),
                # Path from current directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'processed', 'scats_site_reference.csv'),
                # Additional fallback path
                os.path.join(project_root, 'traefik', 'dataset', 'processed', 'scats_site_reference.csv')
            ]
            
            site_reference_path = None
            for path in possible_paths:
                logger.info(f"Trying SCATS data path: {path}")
                if os.path.exists(path):
                    site_reference_path = path
                    logger.info(f"Found SCATS data at: {site_reference_path}")
                    break
            
            if site_reference_path is None:
                raise FileNotFoundError("Could not find SCATS site reference data in any of the expected locations")
            
            # Load site reference data
            self.scats_data = pd.read_csv(site_reference_path)
            
            # Get unique SCATS IDs
            self.scats_ids = self.scats_data['SCATS_ID'].unique()
            self.scats_ids = [str(site_id) for site_id in self.scats_ids]
            
            logger.info(f"Loaded {len(self.scats_ids)} unique SCATS sites")
            
        except Exception as e:
            logger.error(f"Error loading SCATS data: {e}")
            raise
    
    def define_test_scenarios(self):
        """Define test scenarios using real-life SCATS sites."""
        # Define 10 real-life test scenarios
        self.test_scenarios = [
            {
                "name": "Scenario 1: Warrigal Rd to Maroondah Hwy",
                "description": "Route from Warrigal Rd/Toorak Rd to Union Rd/Maroondah Hwy",
                "origin": "2000",  # Warrigal Rd/Toorak Rd
                "destination": "2200"  # Union Rd/Maroondah Hwy
            },
            {
                "name": "Scenario 2: Maroondah Hwy to Canterbury Rd",
                "description": "Route from Union Rd/Maroondah Hwy to Canterbury Rd",
                "origin": "2200",  # Union Rd/Maroondah Hwy
                "destination": "3122"  # Canterbury Rd
            },
            {
                "name": "Scenario 3: High St to Warrigal Rd",
                "description": "Route from High St to Warrigal Rd/Toorak Rd",
                "origin": "0970",  # High St/Warrigal Rd
                "destination": "2000"  # Warrigal Rd/Toorak Rd
            },
            {
                "name": "Scenario 4: Bulleen Rd to Burke Rd",
                "description": "Route from Bulleen Rd to Burke Rd",
                "origin": "2827",  # Bulleen Rd
                "destination": "2825"  # Burke Rd
            },
            {
                "name": "Scenario 5: Princess St to High St",
                "description": "Route from Princess St to High St",
                "origin": "2820",  # Princess St
                "destination": "2846"  # High St
            },
            {
                "name": "Scenario 6: Burke Rd to Warrigal Rd",
                "description": "Route from Burke Rd to Warrigal Rd/High St",
                "origin": "2825",  # Burke Rd
                "destination": "0970"  # Warrigal Rd/High St
            },
            {
                "name": "Scenario 7: High St to Bulleen Rd",
                "description": "Route from High St to Bulleen Rd",
                "origin": "2846",  # High St
                "destination": "2827"  # Bulleen Rd
            },
            {
                "name": "Scenario 8: Warrigal Rd to Burke Rd",
                "description": "Route from Warrigal Rd/Toorak Rd to Burke Rd",
                "origin": "2000",  # Warrigal Rd/Toorak Rd
                "destination": "2825"  # Burke Rd
            },
            {
                "name": "Scenario 9: Princess St to Maroondah Hwy",
                "description": "Route from Princess St to Union Rd/Maroondah Hwy",
                "origin": "2820",  # Princess St
                "destination": "2200"  # Union Rd/Maroondah Hwy
            },
            {
                "name": "Scenario 10: Bulleen Rd to Warrigal Rd",
                "description": "Route from Bulleen Rd to Warrigal Rd/High St",
                "origin": "2827",  # Bulleen Rd
                "destination": "0970"  # Warrigal Rd/High St
            }
        ]
        
        logger.info(f"Defined {len(self.test_scenarios)} test scenarios")
    
    def run_tests(self):
        """Run all test scenarios with different ML models."""
        # Models to test
        models = ["LSTM", "GRU", "CNN-RNN", "Ensemble"]
        
        # Create output directory for results
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run tests for each model
        for model_type in models:
            logger.info(f"Testing with {model_type} model")
            
            # Create router with the specified model
            use_ensemble = model_type == "Ensemble"
            if use_ensemble:
                router = SCATSRouter(use_ensemble=True)
            else:
                router = SCATSRouter(model_type=model_type)
            
            # Run all test scenarios
            model_results = []
            
            for scenario in self.test_scenarios:
                scenario_results = {
                    "scenario": scenario["name"],
                    "origin": scenario["origin"],
                    "destination": scenario["destination"],
                    "time_periods": []
                }
                
                # Test at different times of day
                for time_label, prediction_time in self.test_times:
                    logger.info(f"Running {scenario['name']} at {time_label} with {model_type} model")
                    
                    try:
                        # Get routes
                        routes = router.get_routes(
                            origin_scats=scenario["origin"],
                            destination_scats=scenario["destination"],
                            prediction_time=prediction_time,
                            max_routes=3
                        )
                        
                        # Extract results
                        if routes:
                            # Get the best route (first one)
                            best_route = routes[0]
                            travel_time = best_route.get('travel_time', 0) / 60.0  # Convert to minutes
                            distance = best_route.get('distance', 0)
                            algorithm = best_route.get('algorithm', 'Unknown')
                            
                            # Store results
                            time_result = {
                                "time_period": time_label,
                                "travel_time": travel_time,
                                "distance": distance,
                                "algorithm": algorithm,
                                "route_count": len(routes)
                            }
                            
                            scenario_results["time_periods"].append(time_result)
                            logger.info(f"Result: {travel_time:.2f} minutes, {distance:.2f} km, {algorithm}")
                        else:
                            logger.warning(f"No routes found for {scenario['name']} at {time_label}")
                            scenario_results["time_periods"].append({
                                "time_period": time_label,
                                "travel_time": None,
                                "distance": None,
                                "algorithm": None,
                                "route_count": 0
                            })
                    
                    except Exception as e:
                        logger.error(f"Error running test: {e}")
                        scenario_results["time_periods"].append({
                            "time_period": time_label,
                            "travel_time": None,
                            "distance": None,
                            "algorithm": None,
                            "route_count": 0,
                            "error": str(e)
                        })
                
                model_results.append(scenario_results)
            
            # Store results for this model
            self.results[model_type] = model_results
            
            # Save results to JSON file
            result_file = os.path.join(output_dir, f"{model_type.lower()}_results.json")
            with open(result_file, 'w') as f:
                json.dump(model_results, f, indent=4)
            
            logger.info(f"Saved {model_type} results to {result_file}")
        
        # Generate comparison report
        self.generate_comparison_report(output_dir)
    
    def generate_comparison_report(self, output_dir):
        """Generate a comparison report of all models."""
        # Create report file
        report_file = os.path.join(output_dir, "model_comparison_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# TBRGS Model Comparison Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            
            # Create summary table
            summary_data = []
            for model_type in self.results.keys():
                model_results = self.results[model_type]
                
                # Calculate average travel time across all scenarios and time periods
                total_travel_time = 0
                count = 0
                
                for scenario in model_results:
                    for time_period in scenario["time_periods"]:
                        if time_period.get("travel_time") is not None:
                            total_travel_time += time_period["travel_time"]
                            count += 1
                
                avg_travel_time = total_travel_time / count if count > 0 else None
                
                # Add to summary data
                summary_data.append([
                    model_type,
                    f"{avg_travel_time:.2f} min" if avg_travel_time is not None else "N/A",
                    count
                ])
            
            # Write summary table
            f.write("| Model | Average Travel Time | Successful Tests |\n")
            f.write("|-------|---------------------|------------------|\n")
            for row in summary_data:
                f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
            
            f.write("\n")
            
            # Detailed comparison by scenario
            f.write("## Detailed Comparison by Scenario\n\n")
            
            for i, scenario in enumerate(self.test_scenarios):
                f.write(f"### {scenario['name']}\n\n")
                f.write(f"Origin: {scenario['origin']}, Destination: {scenario['destination']}\n\n")
                
                # Create comparison table for this scenario
                f.write("| Time Period | Model | Travel Time | Distance | Algorithm |\n")
                f.write("|-------------|-------|-------------|----------|----------|\n")
                
                for time_label, _ in self.test_times:
                    for model_type in self.results.keys():
                        # Get results for this model, scenario, and time period
                        scenario_results = self.results[model_type][i]
                        time_results = next((t for t in scenario_results["time_periods"] if t["time_period"] == time_label), None)
                        
                        if time_results:
                            travel_time = time_results.get("travel_time")
                            distance = time_results.get("distance")
                            algorithm = time_results.get("algorithm")
                            
                            travel_time_str = f"{travel_time:.2f} min" if travel_time is not None else "N/A"
                            distance_str = f"{distance:.2f} km" if distance is not None else "N/A"
                            algorithm_str = algorithm if algorithm else "N/A"
                            
                            f.write(f"| {time_label} | {model_type} | {travel_time_str} | {distance_str} | {algorithm_str} |\n")
                
                f.write("\n")
            
            # Generate visualizations
            self.generate_visualizations(output_dir)
            
            # Include visualization references in the report if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                f.write("## Visualizations\n\n")
                f.write("### Average Travel Time by Model\n\n")
                f.write("![Average Travel Time by Model](average_travel_time_by_model.png)\n\n")
                
                f.write("### Travel Time Comparison by Scenario\n\n")
                f.write("![Travel Time Comparison by Scenario](travel_time_by_scenario.png)\n\n")
                
                f.write("### Travel Time by Time of Day\n\n")
                f.write("![Travel Time by Time of Day](travel_time_by_time_of_day.png)\n\n")
            else:
                f.write("## Visualizations\n\n")
                f.write("Visualizations are not available because matplotlib is not installed.\n\n")
                f.write("To enable visualizations, install matplotlib: `pip install matplotlib`\n\n")
        
        logger.info(f"Generated comparison report at {report_file}")
        
        # Print report location
        print(f"\n=== Model Comparison Report saved to: {report_file} ===\n")
    
    def generate_visualizations(self, output_dir):
        """Generate visualizations for the test results."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualizations.")
            return
            
        try:
            # Set up matplotlib
            plt.style.use('ggplot')
            
            # 1. Average Travel Time by Model
            self.plot_average_travel_time_by_model(output_dir)
            
            # 2. Travel Time Comparison by Scenario
            self.plot_travel_time_by_scenario(output_dir)
            
            # 3. Travel Time by Time of Day
            self.plot_travel_time_by_time_of_day(output_dir)
            
            logger.info("Generated all visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            logger.info("Continuing without visualizations.")
            return
    
    def plot_average_travel_time_by_model(self, output_dir):
        """Plot average travel time by model."""
        # Calculate average travel time for each model
        avg_times = []
        model_names = []
        
        for model_type in self.results.keys():
            model_results = self.results[model_type]
            
            # Calculate average travel time
            total_travel_time = 0
            count = 0
            
            for scenario in model_results:
                for time_period in scenario["time_periods"]:
                    if time_period.get("travel_time") is not None:
                        total_travel_time += time_period["travel_time"]
                        count += 1
            
            if count > 0:
                avg_times.append(total_travel_time / count)
                model_names.append(model_type)
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, avg_times, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Average Travel Time (minutes)')
        plt.title('Average Travel Time by Model')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'average_travel_time_by_model.png'), dpi=300)
        plt.close()
    
    def plot_travel_time_by_scenario(self, output_dir):
        """Plot travel time comparison by scenario."""
        # Get data for morning peak time (as a representative time period)
        time_label = "Morning Peak"
        
        # Prepare data
        scenarios = []
        model_data = {model: [] for model in self.results.keys()}
        
        for i, scenario in enumerate(self.test_scenarios):
            scenarios.append(f"S{i+1}")  # Short scenario name
            
            for model_type in self.results.keys():
                # Get results for this model and scenario
                scenario_results = self.results[model_type][i]
                time_results = next((t for t in scenario_results["time_periods"] if t["time_period"] == time_label), None)
                
                if time_results and time_results.get("travel_time") is not None:
                    model_data[model_type].append(time_results["travel_time"])
                else:
                    model_data[model_type].append(0)  # Use 0 for missing data
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(scenarios))
        
        # Plot bars for each model
        for i, (model, data) in enumerate(model_data.items()):
            plt.bar(index + i * bar_width, data, bar_width, label=model, 
                   alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add labels and title
        plt.xlabel('Scenario')
        plt.ylabel('Travel Time (minutes)')
        plt.title(f'Travel Time Comparison by Scenario ({time_label})')
        plt.xticks(index + bar_width * (len(model_data) - 1) / 2, scenarios)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'travel_time_by_scenario.png'), dpi=300)
        plt.close()
    
    def plot_travel_time_by_time_of_day(self, output_dir):
        """Plot travel time by time of day."""
        # Use the first scenario as an example
        scenario_index = 0
        scenario = self.test_scenarios[scenario_index]
        
        # Prepare data
        time_periods = [t[0] for t in self.test_times]
        model_data = {model: [] for model in self.results.keys()}
        
        for model_type in self.results.keys():
            # Get results for this model and scenario
            scenario_results = self.results[model_type][scenario_index]
            
            for time_label, _ in self.test_times:
                time_results = next((t for t in scenario_results["time_periods"] if t["time_period"] == time_label), None)
                
                if time_results and time_results.get("travel_time") is not None:
                    model_data[model_type].append(time_results["travel_time"])
                else:
                    model_data[model_type].append(0)  # Use 0 for missing data
        
        # Create line chart
        plt.figure(figsize=(10, 6))
        
        # Plot lines for each model
        for model, data in model_data.items():
            plt.plot(time_periods, data, marker='o', linewidth=2, label=model)
        
        # Add labels and title
        plt.xlabel('Time of Day')
        plt.ylabel('Travel Time (minutes)')
        plt.title(f'Travel Time by Time of Day (Scenario: {scenario["name"]})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'travel_time_by_time_of_day.png'), dpi=300)
        plt.close()


def main():
    """Main function to run the TBRGS test suite."""
    try:
        # Create and run the test suite
        test_suite = TBRGSTestSuite()
        test_suite.run_tests()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running test suite: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
