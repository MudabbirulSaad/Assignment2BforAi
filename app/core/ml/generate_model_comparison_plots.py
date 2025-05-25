#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate model comparison visualizations for TBRGS project.
This script creates visualizations comparing the performance of LSTM, GRU, and CNN-RNN models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})


def load_comparison_report(file_path: str) -> Dict:
    """
    Load the model comparison report from a JSON file.
    
    Args:
        file_path (str): Path to the model comparison report JSON file
        
    Returns:
        Dict: The loaded comparison report
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def create_metric_comparison_plot(report: Dict, output_dir: str) -> None:
    """
    Create a bar plot comparing the main metrics across all models.
    
    Args:
        report (Dict): The model comparison report
        output_dir (str): Directory to save the plot
    """
    # Extract metrics for each model
    metrics = report['comparison']['metrics']
    model_names = list(report['models'].keys())
    
    # Metrics to include in the plot
    metric_names = ['rmse', 'mae', 'r2', 'nrmse', 'theil_u']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 15), sharex=True)
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        # Extract values for this metric
        values = [metrics[metric][model] for model in model_names]
        
        # Create bar plot
        bars = ax.bar(model_names, values, color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Set title and labels
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        
        # Highlight the best model for this metric
        best_idx = np.argmin(values) if metric in ['rmse', 'mae', 'mape', 'nrmse', 'theil_u'] else np.argmax(values)
        bars[best_idx].set_color('#f39c12')
        bars[best_idx].set_hatch('///')
    
    # Add overall title
    plt.suptitle('Model Performance Comparison', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot
    output_path = os.path.join(output_dir, 'model_metric_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metric comparison plot saved to {output_path}")


def create_site_performance_plot(report: Dict, output_dir: str) -> None:
    """
    Create a plot comparing model performance across different sites.
    
    Args:
        report (Dict): The model comparison report
        output_dir (str): Directory to save the plot
    """
    # Extract site results for each model
    model_names = list(report['models'].keys())
    
    # Get list of sites (assuming all models have the same sites)
    sites = list(report['models'][model_names[0]]['site_results'].keys())
    
    # Extract RMSE values for each model and site
    site_rmse = {site: [] for site in sites}
    for model in model_names:
        for site in sites:
            site_rmse[site].append(report['models'][model]['site_results'][site]['rmse'])
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of the bars on X axis
    r = np.arange(len(sites))
    
    # Create bars
    for i, model in enumerate(model_names):
        values = [site_rmse[site][i] for site in sites]
        ax.bar(r + i*bar_width, values, width=bar_width, label=model, 
               alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add labels and title
    ax.set_xlabel('Site ID', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Model Performance by Site (RMSE)', fontweight='bold')
    ax.set_xticks(r + bar_width)
    ax.set_xticklabels(sites, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'model_site_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Site performance comparison plot saved to {output_path}")


def create_radar_chart(report: Dict, output_dir: str) -> None:
    """
    Create a radar chart comparing models across different metrics.
    
    Args:
        report (Dict): The model comparison report
        output_dir (str): Directory to save the plot
    """
    # Extract metrics for each model
    metrics = report['comparison']['metrics']
    model_names = list(report['models'].keys())
    
    # Metrics to include in the radar chart
    metric_names = ['rmse', 'mae', 'r2', 'nrmse', 'theil_u']
    
    # Normalize metrics to 0-1 scale for comparison
    normalized_metrics = {}
    for metric in metric_names:
        values = [metrics[metric][model] for model in model_names]
        
        # For metrics where lower is better, invert the normalization
        if metric in ['rmse', 'mae', 'mape', 'nrmse', 'theil_u']:
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val > 0:
                normalized_metrics[metric] = [1 - (val - min_val) / (max_val - min_val) for val in values]
            else:
                normalized_metrics[metric] = [0.5 for _ in values]
        else:
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val > 0:
                normalized_metrics[metric] = [(val - min_val) / (max_val - min_val) for val in values]
            else:
                normalized_metrics[metric] = [0.5 for _ in values]
    
    # Number of variables
    N = len(metric_names)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metric labels
    plt.xticks(angles[:-1], [m.upper() for m in metric_names], fontsize=12)
    
    # Draw one axis per variable and add labels
    for angle, metric in zip(angles[:-1], metric_names):
        ax.text(angle, 1.15, metric.upper(), 
                horizontalalignment='center', size=12, 
                verticalalignment='center', fontweight='bold')
    
    # Set y-ticks
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, model in enumerate(model_names):
        values = [normalized_metrics[metric][i] for metric in metric_names]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=model)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Model Performance Comparison (Normalized Metrics)', size=15, y=1.1)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'model_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to {output_path}")


def create_temporal_performance_plot(report: Dict, output_dir: str) -> None:
    """
    Create a plot comparing model performance across different hours of the day.
    
    Args:
        report (Dict): The model comparison report
        output_dir (str): Directory to save the plot
    """
    # Extract hourly results for each model
    model_names = list(report['models'].keys())
    
    # Extract RMSE values for each hour
    hours = range(24)
    hourly_rmse = {model: [] for model in model_names}
    
    for model in model_names:
        for hour in hours:
            hour_str = str(hour)
            hourly_rmse[model].append(report['models'][model]['temporal_results']['hourly'][hour_str]['rmse'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot line for each model
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, model in enumerate(model_names):
        ax.plot(hours, hourly_rmse[model], marker='o', linewidth=2, 
                label=model, color=colors[i])
    
    # Add labels and title
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Model Performance by Hour of Day', fontweight='bold')
    ax.set_xticks(hours)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'model_hourly_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Temporal performance plot saved to {output_path}")


def main():
    """Main function to generate all comparison plots."""
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, 'evaluation', 'reports', 'model_comparison_report.json')
    output_dir = os.path.join(base_dir, 'evaluation', 'plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the comparison report
    report = load_comparison_report(report_path)
    
    # Generate plots
    create_metric_comparison_plot(report, output_dir)
    create_site_performance_plot(report, output_dir)
    create_radar_chart(report, output_dir)
    create_temporal_performance_plot(report, output_dir)
    
    print("All model comparison plots generated successfully.")


if __name__ == "__main__":
    main()
