"""
Test Runner for Traffic-based Route Guidance System (TBRGS)

This script runs test cases from the 'cases' directory and generates a report.
"""

import os
import sys
import time
import json
from datetime import datetime
import glob
import re

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Rich for better console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED
    rich_available = True
except ImportError:
    rich_available = False
    print("Rich library not available. Installing with 'pip install rich' for better console output...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        print("Rich installed successfully! Please run the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"Could not install Rich: {e}")
        print("Continuing with basic console output.")

# Initialize Rich console if available
if rich_available:
    console = Console()
else:
    console = None

# Import project modules
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import TrafficGraph
# Import RouteFinder from the main module instead of utils
from route_finder import RouteFinder
# Only import TrafficPredictor if it's needed
try:
    from models.traffic_predictor import TrafficPredictor
except ImportError:
    TrafficPredictor = None

def create_test_graph():
    """
    Create a test graph for testing purposes using actual SCATS site IDs.
    
    Returns:
        tuple: (nodes, edges) where nodes is a dictionary of node IDs to node data
              and edges is a dictionary mapping source nodes to lists of (target, distance) tuples
    """
    # Create a test graph with actual SCATS site IDs and coordinates
    nodes = {
        2000: {
            'id': 2000,
            'latitude': -37.82,
            'longitude': 145.09,
            'name': 'TOORAK_RD W of WARRIGAL_RD/060 H06'
        },
        3812: {
            'id': 3812,
            'latitude': -37.832,
            'longitude': 145.126,
            'name': 'CAMBERWELL_RD NW of TRAFALGAR_RD/059 K02'
        },
        4035: {
            'id': 4035,
            'latitude': -37.835,
            'longitude': 145.13,
            'name': 'TOORAK_RD W of BURKE_RD/045 H08'
        },
        4043: {
            'id': 4043,
            'latitude': -37.84,
            'longitude': 145.14,
            'name': 'TOORAK_RD W of GLENFERRIE_RD/045 H06'
        },
        4321: {
            'id': 4321,
            'latitude': -37.845,
            'longitude': 145.15,
            'name': 'TOORAK_RD W of AUBURN_RD/045 H05'
        },
        4030: {
            'id': 4030,
            'latitude': -37.85,
            'longitude': 145.13,
            'name': 'KILBY_RD W of BURKE_RD/045 K02'
        },
        3126: {
            'id': 3126,
            'latitude': -37.846,
            'longitude': 145.112,
            'name': 'CANTERBURY_RD W of WARRIGAL_RD/046 H11'
        },
        3001: {
            'id': 3001,
            'latitude': -37.821,
            'longitude': 145.11,
            'name': 'BARKERS_RD W of CHURCH_ST/045 A08'
        },
        3002: {
            'id': 3002,
            'latitude': -37.822,
            'longitude': 145.11,
            'name': 'BARKERS_RD W of DENMARK_ST/045 B08'
        }
    }
    
    # Create edges with distances between SCATS sites
    edges = {
        2000: [(3812, 1200), (4035, 1500), (3126, 1800)],
        3812: [(2000, 1200), (4035, 900), (4030, 1500)],
        4035: [(2000, 1500), (3812, 900), (4043, 800), (4321, 1100)],
        4043: [(4035, 800), (4321, 600), (3001, 1400)],
        4321: [(4035, 1100), (4043, 600), (4030, 750)],
        4030: [(3812, 1500), (4321, 750), (3126, 1200)],
        3126: [(2000, 1800), (4030, 1200)],
        3001: [(4043, 1400), (3002, 300)],
        3002: [(3001, 300)]
    }
    
    return nodes, edges

def parse_test_case(file_path):
    """
    Parse a test case file and extract parameters.
    
    Args:
        file_path: Path to the test case file
    
    Returns:
        Dictionary with test case parameters
    """
    params = {}
    description = ""
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        
        # Store raw content for later processing
        params['raw_content'] = content
        
        # Extract description from comments
        for line in lines:
            if line.strip().startswith('#'):
                description += line.strip()[1:].strip() + " "
            else:
                # Parse parameter lines (key=value)
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    
                    # Handle comments after parameter values
                    if '#' in value:
                        value = value.split('#', 1)[0].strip()
                    
                    params[key.strip()] = value.strip()
    
    # Set description
    if 'description' not in params and description:
        params['description'] = description.strip()
    
    # Set test case name based on file name
    file_name = os.path.basename(file_path)
    match = re.match(r'test_case(\d+)', file_name)
    if match:
        params['test_number'] = int(match.group(1))
        params['test_name'] = f"Test Case {params['test_number']}"
    else:
        params['test_name'] = os.path.splitext(file_name)[0]
    
    return params

def run_route_test(test_params):
    """
    Run a route finding test with the specified parameters.
    
    Args:
        test_params: Dictionary with test parameters
    
    Returns:
        Dictionary with test results
    """
    # Extract parameters
    origin = int(test_params.get('origin', 1))
    destination = int(test_params.get('destination', 9))
    algorithm = test_params.get('algorithm', 'AS')
    
    # Extract node descriptions from test case
    node_descriptions = {}
    for line in test_params.get('raw_content', '').split('\n'):
        if line.strip().startswith('# ') and ':' in line:
            parts = line.strip('# ').split(':', 1)
            try:
                node_id = int(parts[0].strip())
                description = parts[1].strip()
                if '#' in description:
                    description = description.split('#', 1)[0].strip()
                node_descriptions[node_id] = description
            except ValueError:
                pass
    
    if rich_available:
        # Create a panel with test information
        test_title = f"[bold cyan]{test_params['test_name']}[/bold cyan]: [yellow]{test_params.get('description', '')}[/yellow]"
        console.print(Panel(test_title, border_style="cyan"))
        
        # Show test parameters
        params_table = Table(show_header=False, box=ROUNDED, border_style="blue")
        params_table.add_column("Parameter", style="bold green")
        params_table.add_column("Value", style="yellow")
        
        # Add origin with description if available
        origin_desc = node_descriptions.get(origin, f"Node {origin}")
        params_table.add_row("Origin", f"{origin} ({origin_desc})")
        
        # Add destination with description if available
        dest_desc = node_descriptions.get(destination, f"Node {destination}")
        params_table.add_row("Destination", f"{destination} ({dest_desc})")
        
        params_table.add_row("Algorithm", f"{algorithm}")
        console.print(params_table)
        
        console.print(f"[bold green]Finding route from {origin} to {destination} using {algorithm}...[/bold green]")
    else:
        # Plain text output if Rich is not available
        print(f"\nRunning {test_params['test_name']}: {test_params.get('description', '')}")
        origin_desc = node_descriptions.get(origin, f"Node {origin}")
        dest_desc = node_descriptions.get(destination, f"Node {destination}")
        print(f"Finding route from {origin} ({origin_desc}) to {destination} ({dest_desc}) using {algorithm}")
    
    # Create test graph
    nodes, edges = create_test_graph()
    graph = TrafficGraph(nodes, edges)
    
    # Initialize RouteFinder with the default config path
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default_config.json')
    route_finder = RouteFinder(config_path=config_path)
    
    # Set the graph manually for testing
    route_finder.graph = graph
    
    # Map algorithm abbreviations to full names
    algorithm_mapping = {
        'AS': 'astar',
        'BFS': 'bfs',
        'DFS': 'dfs',
        'GBFS': 'gbfs',
        'IDDFS': 'iddfs',
        'BDWA': 'bdwa',
        'CUS1': 'astar',  # Map custom algorithms to standard ones for testing
        'CUS2': 'bdwa'
    }
    
    # Convert algorithm abbreviation to full name
    algo_full_name = algorithm_mapping.get(algorithm, algorithm)
    
    # Find routes
    start_time = time.time()
    try:
        routes = route_finder.find_routes(origin, destination, algorithm=algo_full_name)
    except Exception as e:
        if rich_available:
            console.print(f"[bold red]Error finding route: {str(e)}[/bold red]")
        else:
            print(f"Error finding route: {str(e)}")
        routes = []
    execution_time = time.time() - start_time
    
    if rich_available:
        console.print(f"[bold green]Route finding completed in {execution_time:.4f}s[/bold green]")
    else:
        print(f"Route finding completed in {execution_time:.4f}s")
    
    # Process results
    if routes:
        if rich_available:
            console.print(f"[bold green]Found {len(routes)} route(s) in {execution_time:.4f} seconds[/bold green]")
            
            # Create a table for routes
            routes_table = Table(title="Routes", box=ROUNDED, border_style="green")
            routes_table.add_column("Route", style="cyan")
            routes_table.add_column("Path", style="yellow")
            routes_table.add_column("Travel Time", style="green")
            routes_table.add_column("Distance", style="blue")
            
            for i, route in enumerate(routes):
                # Check if route is a dictionary (from RouteFinder) or a tuple (older format)
                if isinstance(route, dict):
                    path = route.get('path', [])
                    travel_time = route.get('travel_time', 0)
                    # If distance is already calculated, use it
                    if 'distance' in route:
                        distance = route['distance']
                    else:
                        # Calculate distance and travel time properly
                        distance = 0
                        travel_time = 0
                        intersection_delay = 30  # Same as in route_finder.py
                        
                        for j in range(len(path) - 1):
                            source = path[j]
                            target = path[j + 1]
                            
                            # Get distance for this segment
                            segment_distance = graph.calculate_distance(source, target)
                            distance += segment_distance
                            
                            # Get traffic flow and calculate speed
                            flow = graph.get_traffic_flow(source, target)
                            speed = graph.calculate_speed(flow)
                            
                            # Calculate driving time (in seconds)
                            distance_km = segment_distance / 1000.0
                            pure_driving_time = (distance_km / speed) * 3600
                            
                            # Add intersection delay
                            segment_travel_time = pure_driving_time + intersection_delay
                            travel_time += segment_travel_time
                else:
                    # Assume it's a tuple (path, travel_time)
                    try:
                        path, travel_time = route
                        # Calculate distance and travel time properly
                        distance = 0
                        travel_time = 0
                        intersection_delay = 30  # Same as in route_finder.py
                        
                        for j in range(len(path) - 1):
                            source = path[j]
                            target = path[j + 1]
                            
                            # Get distance for this segment
                            segment_distance = graph.calculate_distance(source, target)
                            distance += segment_distance
                            
                            # Get traffic flow and calculate speed
                            flow = graph.get_traffic_flow(source, target)
                            speed = graph.calculate_speed(flow)
                            
                            # Calculate driving time (in seconds)
                            distance_km = segment_distance / 1000.0
                            pure_driving_time = (distance_km / speed) * 3600
                            
                            # Add intersection delay
                            segment_travel_time = pure_driving_time + intersection_delay
                            travel_time += segment_travel_time
                    except ValueError:
                        # If unpacking fails, skip this route
                        continue
                
                # Format path with node names
                path_with_names = []
                for node in path:
                    node_desc = node_descriptions.get(node, f"Node {node}")
                    path_with_names.append(f"{node} ({node_desc})")
                
                formatted_path = " → ".join([str(node) for node in path])
                if len(path) > 1:
                    formatted_path += "\n" + " → ".join([node_descriptions.get(node, f"Node {node}") for node in path])
                
                routes_table.add_row(
                    f"Route {i+1}",
                    formatted_path,
                    f"{travel_time:.2f} seconds",
                    f"{distance:.2f} meters"
                )
            
            console.print(routes_table)
        else:
            print(f"Found {len(routes)} route(s) in {execution_time:.4f} seconds")
            for i, (path, travel_time) in enumerate(routes):
                # Format path with node IDs
                print(f"  Route {i+1}: {path}")
                
                # Format path with node names
                if node_descriptions:
                    path_with_names = []
                    for node in path:
                        node_desc = node_descriptions.get(node, f"Node {node}")
                        path_with_names.append(node_desc)
                    print(f"  Path names: {' → '.join(path_with_names)}")
                
                print(f"  Travel Time: {travel_time:.2f} seconds")
                
                # Calculate distance
                distance = 0
                for j in range(len(path) - 1):
                    source = path[j]
                    target = path[j + 1]
                    distance += graph.calculate_distance(source, target)
                print(f"  Distance: {distance:.2f} meters")
    else:
        if rich_available:
            console.print("[bold red]No routes found[/bold red]")
        else:
            print("No routes found")
    
    # Return results
    # Extract paths and travel times from routes
    paths = []
    travel_times = []
    
    if routes:
        for route in routes:
            # Check if route is a dictionary or a tuple
            if isinstance(route, dict):
                paths.append(route.get('path', []))
                travel_times.append(route.get('travel_time', 0))
            else:
                # Assume it's a tuple (path, travel_time)
                try:
                    path, travel_time = route
                    paths.append(path)
                    travel_times.append(travel_time)
                except ValueError:
                    # If unpacking fails, skip this route
                    continue
    
    return {
        "test_name": test_params["test_name"],
        "description": test_params.get("description", ""),
        "origin": origin,
        "destination": destination,
        "algorithm": algorithm,
        "routes_found": len(routes) if routes else 0,
        "execution_time": execution_time,
        "paths": paths,
        "travel_times": travel_times,
        "status": "PASSED" if routes else "FAILED"
    }

def run_model_test(test_params):
    """
    Run a model evaluation test with the specified parameters.
    
    Args:
        test_params: Dictionary with test parameters
    
    Returns:
        Dictionary with test results
    """
    # Extract parameters
    model_name = test_params.get('model', 'lstm')
    num_samples = int(test_params.get('num_samples', 10))
    
    if rich_available:
        # Create a panel with test information
        test_title = f"[bold magenta]{test_params['test_name']}[/bold magenta]: [yellow]{test_params.get('description', '')}[/yellow]"
        console.print(Panel(test_title, border_style="magenta"))
        
        # Show test parameters
        params_table = Table(show_header=False, box=ROUNDED, border_style="blue")
        params_table.add_column("Parameter", style="bold green")
        params_table.add_column("Value", style="yellow")
        params_table.add_row("Model", f"{model_name.upper()}")
        params_table.add_row("Samples", f"{num_samples}")
        console.print(params_table)
    else:
        print(f"\nRunning {test_params['test_name']}: {test_params.get('description', '')}")
        print(f"Evaluating {model_name.upper()} model with {num_samples} samples")
    
    # Initialize predictor
    predictor = TrafficPredictor(models_dir='models/models/checkpoints')
    
    # Load test data
    try:
        data = np.load('data/processed/sequence_data.npz')
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Limit to specified number of samples
        num_samples = min(num_samples, len(X_test))
        X_test_sample = X_test[:num_samples]
        y_test_sample = y_test[:num_samples]
        
        if rich_available:
            console.print(f"Using [cyan]{num_samples}[/cyan] samples from test data")
        else:
            print(f"Using {num_samples} samples from test data")
        
        # Get model
        model = predictor.models.get(model_name)
        if model is None:
            if rich_available:
                console.print(f"[bold red]Model {model_name} not found[/bold red]")
            else:
                print(f"Model {model_name} not found")
            return {
                "test_name": test_params['test_name'],
                "description": test_params.get('description', ''),
                "model": model_name,
                "status": "FAILED",
                "error": f"Model {model_name} not found"
            }
        
        # Make predictions
        if rich_available:
            console.print(f"[bold green]Making predictions with {model_name.upper()} model...[/bold green]")
            
            start_time = time.time()
            predictions = []
            
            for i in range(num_samples):
                try:
                    # Get a single sample
                    sample = X_test_sample[i]
                    
                    # Make prediction
                    pred = model.predict(sample)
                    predictions.append(pred)
                    
                    # Show progress periodically
                    if (i+1) % 5 == 0 or i+1 == num_samples:
                        console.print(f"[cyan]Processed {i+1}/{num_samples} samples[/cyan]")
                except Exception as e:
                    console.print(f"[red]Error predicting sample {i}: {e}[/red]")
                    predictions.append(None)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            console.print(f"[bold green]Predictions completed in {execution_time:.4f}s[/bold green]")
        
        else:
            start_time = time.time()
            predictions = []
            
            for i in range(num_samples):
                try:
                    # Get a single sample
                    sample = X_test_sample[i]
                    
                    # Make prediction
                    pred = model.predict(sample)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error predicting sample {i}: {e}")
                    predictions.append(None)
            
            # Calculate execution time
            execution_time = time.time() - start_time
        
        # Calculate metrics
        valid_predictions = [p for p in predictions if p is not None]
        valid_actuals = y_test_sample[:len(valid_predictions)]
        
        if valid_predictions:
            # Mean Squared Error
            mse = np.mean((valid_actuals - np.array(valid_predictions)) ** 2)
            
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(valid_actuals - np.array(valid_predictions)))
            
            if rich_available:
                # Create a table for metrics
                metrics_table = Table(title=f"Metrics for {model_name.upper()} model", box=ROUNDED, border_style="green")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="yellow")
                
                metrics_table.add_row("MSE", f"{mse:.4f}")
                metrics_table.add_row("RMSE", f"{rmse:.4f}")
                metrics_table.add_row("MAE", f"{mae:.4f}")
                metrics_table.add_row("Execution Time", f"{execution_time:.4f} seconds")
                metrics_table.add_row("Valid Predictions", f"{len(valid_predictions)}/{num_samples}")
                
                console.print(metrics_table)
            else:
                print(f"Metrics for {model_name.upper()} model:")
                print(f"  MSE: {mse:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
            
            return {
                "test_name": test_params['test_name'],
                "description": test_params.get('description', ''),
                "model": model_name,
                "num_samples": num_samples,
                "valid_predictions": len(valid_predictions),
                "execution_time": execution_time,
                "metrics": {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae
                },
                "status": "PASSED"
            }
        else:
            if rich_available:
                console.print("[bold red]No valid predictions[/bold red]")
            else:
                print("No valid predictions")
            return {
                "test_name": test_params['test_name'],
                "description": test_params.get('description', ''),
                "model": model_name,
                "status": "FAILED",
                "error": "No valid predictions"
            }
    
    except Exception as e:
        if rich_available:
            console.print(f"[bold red]Error in model test: {e}[/bold red]")
        else:
            print(f"Error in model test: {e}")
        return {
            "test_name": test_params['test_name'],
            "description": test_params.get('description', ''),
            "model": model_name,
            "status": "FAILED",
            "error": str(e)
        }

def run_test_case(test_params):
    """
    Run a test case based on its parameters.
    
    Args:
        test_params: Dictionary with test parameters
    
    Returns:
        Dictionary with test results
    """
    # Determine test type based on parameters
    if 'model' in test_params:
        # This is a model test
        return run_model_test(test_params)
    else:
        # This is a route test
        return run_route_test(test_params)

def save_test_results(results, output_dir='../results'):
    """
    Save test results to a JSON file.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save results
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if rich_available:
        console.print(f"Test results saved to [bold blue]{output_file}[/bold blue]")
    else:
        print(f"Test results saved to {output_file}")
    return output_file

def main():
    """Run all test cases and generate a report."""
    if rich_available:
        console.print("[bold blue]Running TBRGS Test Cases[/bold blue]")
        console.print("[blue]=======================[/blue]")
    else:
        print("Running TBRGS Test Cases")
        print("=======================")
    
    # Find all test case files
    cases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases')
    test_files = sorted(glob.glob(os.path.join(cases_dir, 'test_case*.txt')))
    
    if not test_files:
        if rich_available:
            console.print(f"[bold red]No test case files found in {cases_dir}[/bold red]")
        else:
            print(f"No test case files found in {cases_dir}")
        return
    
    if rich_available:
        console.print(f"Found [bold green]{len(test_files)}[/bold green] test case files")
    else:
        print(f"Found {len(test_files)} test case files")
    
    # Import numpy for model tests
    global np
    import numpy as np
    
    # Run tests
    results = []
    start_time = time.time()
    
    if rich_available:
        # Create a progress bar for running tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Running tests..."),
            BarColumn(),
            TextColumn("[bold cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running test cases", total=len(test_files))
            
            for test_file in test_files:
                # Parse test case
                test_params = parse_test_case(test_file)
                
                # Run test
                result = run_test_case(test_params)
                results.append(result)
                
                progress.update(task, advance=1)
    else:
        for test_file in test_files:
            # Parse test case
            test_params = parse_test_case(test_file)
            
            # Run test
            result = run_test_case(test_params)
            results.append(result)
    
    # Calculate total execution time
    total_execution_time = time.time() - start_time
    
    # Count passed/failed tests
    passed_tests = sum(1 for r in results if r.get('status') == 'PASSED')
    failed_tests = sum(1 for r in results if r.get('status') == 'FAILED')
    
    # Prepare summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_execution_time": total_execution_time,
        "results": results
    }
    
    # Save results
    output_file = save_test_results(summary)
    
    # Print summary
    if rich_available:
        # Create a summary table
        summary_table = Table(title="Test Summary", box=ROUNDED, border_style="green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Total Tests", f"{summary['total_tests']}")
        summary_table.add_row("Passed", f"[bold green]{summary['passed_tests']}[/bold green]")
        summary_table.add_row("Failed", f"[bold red]{summary['failed_tests']}[/bold red]")
        
        success_color = "green" if summary['passed_tests'] == summary['total_tests'] else "yellow"
        success_rate = f"[bold {success_color}]{summary['passed_tests'] / summary['total_tests'] * 100:.2f}%[/bold {success_color}]"
        summary_table.add_row("Success Rate", success_rate)
        
        summary_table.add_row("Total Execution Time", f"{total_execution_time:.2f} seconds")
        summary_table.add_row("Results File", f"{output_file}")
        
        console.print(summary_table)
    else:
        print("\nTest Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success rate: {summary['passed_tests'] / summary['total_tests'] * 100:.2f}%")
        print(f"  Total execution time: {total_execution_time:.2f} seconds")
        print(f"  Results saved to: {output_file}")

if __name__ == "__main__":
    main()
