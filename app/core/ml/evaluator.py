#!/usr/bin/env python3
"""
Model Evaluation Framework for Traffic Prediction

This module provides comprehensive evaluation tools for assessing and comparing
the performance of different traffic prediction models. It includes metrics calculation,
statistical testing, visualization, and reporting capabilities.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy import stats
import json
import calendar
from datetime import datetime

# Add the project root directory to the Python path
# This allows running the script directly with 'python evaluator.py'
traefik_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if traefik_dir not in sys.path:
    sys.path.insert(0, traefik_dir)

# For model evaluation
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For statistical testing
from scipy.stats import ttest_ind, wilcoxon, friedmanchisquare

# Import project-specific modules
from app.core.logging import logger
from app.config.config import config
from app.core.ml.base_model import BaseModel


class ModelEvaluator:
    """
    Comprehensive evaluation framework for traffic prediction models.
    
    This class provides tools for evaluating and comparing different models,
    including metrics calculation, statistical testing, visualization, and reporting.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str, optional): Directory to save evaluation results. Defaults to None.
        """
        # Create a default model directory if not specified in config
        if not output_dir:
            if hasattr(config, 'model_dir'):
                model_dir = config.model_dir
            else:
                # Use a default path based on project structure
                model_dir = os.path.join(config.project_root, 'models')
                
            self.output_dir = os.path.join(model_dir, 'evaluation')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different types of results
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.reports_dir = os.path.join(self.output_dir, 'reports')
        self.site_analysis_dir = os.path.join(self.output_dir, 'site_analysis')
        self.temporal_analysis_dir = os.path.join(self.output_dir, 'temporal_analysis')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.site_analysis_dir, exist_ok=True)
        os.makedirs(self.temporal_analysis_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.site_results = {}
        self.temporal_results = {}
        self.statistical_tests = {}
        
        logger.info(f"Initialized Model Evaluator with output directory: {self.output_dir}")
    
    def evaluate_model(self, model: BaseModel, test_data: Union[pd.DataFrame, DataLoader], 
                     feature_columns: List[str] = None, target_columns: List[str] = None,
                     site_column: str = None, time_column: str = None) -> Dict:
        """
        Evaluate a model using comprehensive metrics.
        
        Args:
            model (BaseModel): Model to evaluate
            test_data (Union[pd.DataFrame, DataLoader]): Test data
            feature_columns (List[str], optional): Feature column names if test_data is DataFrame. Defaults to None.
            target_columns (List[str], optional): Target column names if test_data is DataFrame. Defaults to None.
            site_column (str, optional): Column name for site IDs. Defaults to None.
            time_column (str, optional): Column name for timestamps. Defaults to None.
            
        Returns:
            Dict: Evaluation results
        """
        model_name = model.model_name
        logger.info(f"Evaluating model: {model_name}")
        
        # Prepare data if it's a DataFrame
        if isinstance(test_data, pd.DataFrame):
            if feature_columns is None or target_columns is None:
                raise ValueError("feature_columns and target_columns must be provided when test_data is a DataFrame")
            
            # Preprocess data using the model's preprocessing method
            test_loader, _ = model.preprocess_data(
                data=test_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
        else:
            # Assume it's already a DataLoader
            test_loader = test_data
        
        # Get predictions and actual values
        all_predictions = []
        all_actuals = []
        site_ids = []
        timestamps = []
        
        # If site_column and time_column are provided, extract them for site-specific and temporal analysis
        has_site_info = site_column is not None and site_column in test_data.columns if isinstance(test_data, pd.DataFrame) else False
        has_time_info = time_column is not None and time_column in test_data.columns if isinstance(test_data, pd.DataFrame) else False
        
        # Collect predictions and actual values
        for batch in test_loader:
            X_batch, y_batch = batch
            
            # Convert to numpy for consistent processing
            X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
            y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
            
            # Get predictions
            y_pred = model.predict(X_np)
            
            # Store predictions and actuals
            all_predictions.append(y_pred)
            all_actuals.append(y_np)
        
        # Concatenate batches
        all_predictions = np.vstack(all_predictions)
        all_actuals = np.vstack(all_actuals)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_actuals, all_predictions)
        
        # Store results
        self.results[model_name] = metrics
        
        # Perform site-specific evaluation if site information is available
        if has_site_info and isinstance(test_data, pd.DataFrame):
            self.site_results[model_name] = self._evaluate_by_site(
                test_data, model, feature_columns, target_columns, site_column
            )
        
        # Perform temporal pattern analysis if time information is available
        if has_time_info and isinstance(test_data, pd.DataFrame):
            self.temporal_results[model_name] = self._analyze_temporal_patterns(
                test_data, model, feature_columns, target_columns, time_column
            )
        
        logger.info(f"Evaluation results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def evaluate_multiple_models(self, models: Dict[str, BaseModel], test_data: Union[pd.DataFrame, DataLoader],
                               feature_columns: List[str] = None, target_columns: List[str] = None,
                               site_column: str = None, time_column: str = None) -> Dict:
        """
        Evaluate multiple models and compare their performance.
        
        Args:
            models (Dict[str, BaseModel]): Dictionary of models to evaluate
            test_data (Union[pd.DataFrame, DataLoader]): Test data
            feature_columns (List[str], optional): Feature column names if test_data is DataFrame. Defaults to None.
            target_columns (List[str], optional): Target column names if test_data is DataFrame. Defaults to None.
            site_column (str, optional): Column name for site IDs. Defaults to None.
            time_column (str, optional): Column name for timestamps. Defaults to None.
            
        Returns:
            Dict: Evaluation results for all models
        """
        logger.info(f"Evaluating {len(models)} models: {list(models.keys())}")
        
        # Evaluate each model
        for model_name, model in models.items():
            self.evaluate_model(
                model=model,
                test_data=test_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                site_column=site_column,
                time_column=time_column
            )
        
        # Perform statistical significance testing between models
        if len(models) > 1:
            self._perform_statistical_tests_for_models(models, test_data, feature_columns, target_columns)
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
        
    def _perform_statistical_tests_for_models(self, models: Dict[str, BaseModel], test_data: Union[pd.DataFrame, DataLoader],
                                    feature_columns: List[str] = None, target_columns: List[str] = None) -> Dict:
        """
        Perform statistical significance tests to compare multiple models.
        
        Args:
            models (Dict[str, BaseModel]): Dictionary of models to compare
            test_data (Union[pd.DataFrame, DataLoader]): Test data
            feature_columns (List[str], optional): Feature column names if test_data is DataFrame. Defaults to None.
            target_columns (List[str], optional): Target column names if test_data is DataFrame. Defaults to None.
            
        Returns:
            Dict: Results of statistical tests
        """
        logger.info(f"Performing statistical significance tests for {len(models)} models")
        
        # Initialize statistical tests dictionary
        self.statistical_tests = {}
        
        # Get model names
        model_names = list(models.keys())
        
        # Prepare data if it's a DataFrame
        if isinstance(test_data, pd.DataFrame):
            if feature_columns is None or target_columns is None:
                raise ValueError("feature_columns and target_columns must be provided when test_data is a DataFrame")
            
            # Get actual values (ground truth)
            y_true = test_data[target_columns].values
            
            # Get predictions for each model
            model_predictions = {}
            model_errors = {}
            
            for model_name, model in models.items():
                # Preprocess data using the model's preprocessing method
                test_loader, _ = model.preprocess_data(
                    data=test_data,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    train_mode=False
                )
                
                # Get predictions
                all_predictions = []
                all_actuals = []
                
                for batch in test_loader:
                    X_batch, y_batch = batch
                    
                    # Convert to numpy for consistent processing
                    X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                    y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                    
                    # Get predictions
                    y_pred = model.predict(X_np)
                    
                    # Store predictions and actuals
                    all_predictions.append(y_pred)
                    all_actuals.append(y_np)
                
                # Concatenate batches
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Store predictions and calculate errors
                model_predictions[model_name] = all_predictions
                model_errors[model_name] = np.abs(all_actuals - all_predictions)
        else:
            # If test_data is a DataLoader, we need to process it for each model
            # This is more complex and would require multiple passes through the data
            # For simplicity, we'll raise an error for now
            raise ValueError("Statistical testing with DataLoader input is not yet implemented. Please provide a DataFrame.")
        
        # Perform pairwise statistical tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Only test each pair once
                    # Perform statistical tests between model1 and model2
                    test_results = self._perform_statistical_tests(
                        model_errors[model1], model_errors[model2]
                    )
                    
                    # Store test results
                    self.statistical_tests[f"{model1}_vs_{model2}"] = test_results
                    
                    # Log summary
                    summary = test_results['summary']
                    logger.info(f"Statistical comparison {model1} vs {model2}:")
                    logger.info(f"  Overall better model: {summary['overall_better_model']} (confidence: {summary['confidence']})")
                    logger.info(f"  Significant tests: {summary['significant_tests']}, Model1 wins: {summary['model1_wins']}, Model2 wins: {summary['model2_wins']}, Ties: {summary['ties']}")
        
        # Save statistical test results
        stats_path = os.path.join(self.output_dir, 'statistical_tests.json')
        
        # Convert the statistical tests to serializable format
        serializable_tests = self._convert_to_serializable(self.statistical_tests)
        
        with open(stats_path, 'w') as f:
            json.dump(serializable_tests, f, indent=4)
        
        logger.info(f"Statistical test results saved to {stats_path}")
        
        return self.statistical_tests
    
    def _perform_statistical_tests(self, model1_errors: np.ndarray, model2_errors: np.ndarray) -> Dict:
        """
        Perform statistical significance tests to compare model errors.
        
        Args:
            model1_errors (np.ndarray): Absolute errors from model 1
            model2_errors (np.ndarray): Absolute errors from model 2
            
        Returns:
            Dict: Results of statistical tests
        """
        # Ensure inputs are numpy arrays
        model1_errors = np.asarray(model1_errors).flatten()
        model2_errors = np.asarray(model2_errors).flatten()
        
        # Initialize results dictionary
        test_results = {}
        
        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(model1_errors, model2_errors)
            test_results['paired_t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better_model': 'model1' if t_stat < 0 else 'model2' if t_stat > 0 else 'tie'
            }
        except Exception as e:
            logger.error(f"Error performing paired t-test: {e}")
            test_results['paired_t_test'] = {'error': str(e)}
        
        # Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        try:
            w_stat, p_value = stats.wilcoxon(model1_errors, model2_errors)
            test_results['wilcoxon_test'] = {
                'statistic': float(w_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better_model': 'model1' if np.median(model1_errors) < np.median(model2_errors) else 'model2' if np.median(model1_errors) > np.median(model2_errors) else 'tie'
            }
        except Exception as e:
            logger.error(f"Error performing Wilcoxon test: {e}")
            test_results['wilcoxon_test'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test (tests if two samples are drawn from the same distribution)
        try:
            ks_stat, p_value = stats.ks_2samp(model1_errors, model2_errors)
            test_results['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'different_distributions': p_value < 0.05
            }
        except Exception as e:
            logger.error(f"Error performing KS test: {e}")
            test_results['ks_test'] = {'error': str(e)}
        
        # Mann-Whitney U test (tests if two samples are from the same distribution)
        try:
            u_stat, p_value = stats.mannwhitneyu(model1_errors, model2_errors, alternative='two-sided')
            test_results['mann_whitney_test'] = {
                'statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'better_model': 'model1' if np.median(model1_errors) < np.median(model2_errors) else 'model2' if np.median(model1_errors) > np.median(model2_errors) else 'tie'
            }
        except Exception as e:
            logger.error(f"Error performing Mann-Whitney U test: {e}")
            test_results['mann_whitney_test'] = {'error': str(e)}
        
        # F-test for equality of variances
        try:
            f_stat = np.var(model1_errors, ddof=1) / np.var(model2_errors, ddof=1)
            dfn = len(model1_errors) - 1  # degrees of freedom numerator
            dfd = len(model2_errors) - 1  # degrees of freedom denominator
            p_value = 1 - stats.f.cdf(f_stat, dfn, dfd) if f_stat > 1 else stats.f.cdf(f_stat, dfn, dfd)
            p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed test
            
            test_results['f_test'] = {
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'more_consistent_model': 'model1' if f_stat < 1 and p_value < 0.05 else 'model2' if f_stat > 1 and p_value < 0.05 else 'tie'
            }
        except Exception as e:
            logger.error(f"Error performing F-test: {e}")
            test_results['f_test'] = {'error': str(e)}
        
        # Effect size (Cohen's d)
        try:
            mean_diff = np.mean(model1_errors) - np.mean(model2_errors)
            pooled_std = np.sqrt((np.var(model1_errors, ddof=1) + np.var(model2_errors, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            # Interpret Cohen's d
            if abs(cohens_d) < 0.2:
                interpretation = 'negligible'
            elif abs(cohens_d) < 0.5:
                interpretation = 'small'
            elif abs(cohens_d) < 0.8:
                interpretation = 'medium'
            else:
                interpretation = 'large'
            
            test_results['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': interpretation,
                'better_model': 'model1' if cohens_d < 0 else 'model2' if cohens_d > 0 else 'tie'
            }
        except Exception as e:
            logger.error(f"Error calculating effect size: {e}")
            test_results['effect_size'] = {'error': str(e)}
        
        # Summarize test results
        test_results['summary'] = self._summarize_statistical_tests(test_results)
        
        return test_results
    
    def _summarize_statistical_tests(self, test_results: Dict) -> Dict:
        """
        Summarize the results of statistical tests.
        
        Args:
            test_results (Dict): Results of statistical tests
            
        Returns:
            Dict: Summary of test results
        """
        # Count how many tests indicate each model is better
        model1_count = 0
        model2_count = 0
        tie_count = 0
        significant_tests = 0
        
        for test_name, result in test_results.items():
            # Skip summary and tests with errors
            if test_name == 'summary' or 'error' in result:
                continue
            
            # Count significant tests
            if result.get('significant', False):
                significant_tests += 1
                
                # Count which model is better according to this test
                if 'better_model' in result:
                    if result['better_model'] == 'model1':
                        model1_count += 1
                    elif result['better_model'] == 'model2':
                        model2_count += 1
                    else:  # tie
                        tie_count += 1
        
        # Determine overall better model
        if model1_count > model2_count:
            overall_better = 'model1'
        elif model2_count > model1_count:
            overall_better = 'model2'
        else:
            overall_better = 'tie'
        
        # Determine confidence level
        total_tests = model1_count + model2_count + tie_count
        if total_tests == 0:
            confidence = 'none'
        else:
            max_count = max(model1_count, model2_count)
            confidence_ratio = max_count / total_tests
            
            if confidence_ratio >= 0.8:
                confidence = 'high'
            elif confidence_ratio >= 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'
        
        return {
            'significant_tests': significant_tests,
            'model1_wins': model1_count,
            'model2_wins': model2_count,
            'ties': tie_count,
            'overall_better_model': overall_better,
            'confidence': confidence
        }
        
    def generate_comparison_report(self) -> Dict:
        """
        Generate a comprehensive comparison report for all evaluated models.
        
        Returns:
            Dict: Comparison report data
        """
        
    def save_comparison_report(self, report_path: str) -> None:
        """
        Save the comparison report to a JSON file.
        
        Args:
            report_path (str): Path to save the report
        """
        report = self.generate_comparison_report()
        serializable_report = self._convert_to_serializable(report)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=4)
        
        logger.info(f"Model comparison report saved to {report_path}")
        
    def generate_comparison_report(self) -> Dict:
        """
        Generate a comprehensive comparison report for all evaluated models.
        
        Returns:
            Dict: Comparison report data
        """
        if not self.results:
            raise ValueError("No evaluation results available. Evaluate models first.")
        
        logger.info("Generating model comparison report...")
        
        # Create report data structure
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {},
            'comparison': {
                'metrics': {},
                'rankings': {},
                'statistical_tests': {},
                'best_model': {}
            }
        }
        
        # Add model results
        for model_name, metrics in self.results.items():
            report['models'][model_name] = {
                'metrics': metrics
            }
            
            # Add site-specific results if available
            if model_name in self.site_results:
                report['models'][model_name]['site_results'] = self.site_results[model_name]
            
            # Add temporal pattern results if available
            if model_name in self.temporal_results:
                report['models'][model_name]['temporal_results'] = self.temporal_results[model_name]
        
        # Add statistical test results if available
        if self.statistical_tests:
            report['comparison']['statistical_tests'] = self.statistical_tests
        
        # Compare metrics across models
        metrics = list(next(iter(self.results.values())).keys())  # Get metrics from first model
        
        for metric in metrics:
            # Get metric values for all models
            metric_values = {model_name: result[metric] for model_name, result in self.results.items()}
            
            # Store metric values for comparison
            report['comparison']['metrics'][metric] = metric_values
            
            # Rank models by metric (lower is better for error metrics, higher is better for R²)
            if metric in ['rmse', 'mae', 'mape', 'mbe', 'nrmse', 'theil_u']:
                # Lower is better
                ranked_models = sorted(metric_values.keys(), key=lambda m: metric_values[m])
            else:
                # Higher is better (R²)
                ranked_models = sorted(metric_values.keys(), key=lambda m: metric_values[m], reverse=True)
            
            # Store rankings
            report['comparison']['rankings'][metric] = ranked_models
        
        # Determine best model overall
        # Calculate average rank across all metrics
        avg_ranks = {}
        for model_name in self.results.keys():
            ranks = []
            for metric, ranked_models in report['comparison']['rankings'].items():
                # Get rank (add 1 because ranks are 1-based)
                rank = ranked_models.index(model_name) + 1
                ranks.append(rank)
            
            # Calculate average rank
            avg_ranks[model_name] = sum(ranks) / len(ranks)
        
        # Sort models by average rank
        overall_ranking = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
        report['comparison']['best_model']['overall_ranking'] = overall_ranking
        report['comparison']['best_model']['average_ranks'] = avg_ranks
        report['comparison']['best_model']['best_model'] = overall_ranking[0]
        
        # Generate summary
        report['summary'] = self._generate_report_summary(report)
        
        # Save report to file
        report_path = os.path.join(self.reports_dir, 'model_comparison_report.json')
        
        # Convert report to serializable format
        serializable_report = self._convert_to_serializable(report)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=4)
        
        # Generate plots
        self._plot_model_comparison(report)
        
        logger.info(f"Model comparison report saved to {report_path}")
        
        return report
    
    def _generate_report_summary(self, report: Dict) -> str:
        """
        Generate a text summary of the comparison report.
        
        Args:
            report (Dict): Comparison report data
            
        Returns:
            str: Report summary
        """
        # Get best model
        best_model = report['comparison']['best_model']['best_model']
        
        # Get metrics for best model
        best_model_metrics = report['models'][best_model]['metrics']
        
        # Generate summary
        summary = [f"Model Comparison Report ({report['timestamp']})"]
        summary.append("\n1. Overall Results:")
        summary.append(f"   - Best performing model: {best_model}")
        summary.append(f"   - Key metrics for {best_model}:")
        for metric, value in best_model_metrics.items():
            summary.append(f"     * {metric.upper()}: {value:.4f}")
        
        # Add statistical test results if available
        if 'statistical_tests' in report['comparison'] and report['comparison']['statistical_tests']:
            summary.append("\n2. Statistical Significance:")
            for comparison, results in report['comparison']['statistical_tests'].items():
                model1, model2 = comparison.split('_vs_')
                overall_better = results['summary']['overall_better_model']
                confidence = results['summary']['confidence']
                better_model = model1 if overall_better == 'model1' else model2 if overall_better == 'model2' else 'Neither'
                
                if better_model != 'Neither':
                    summary.append(f"   - {better_model} is statistically better than {model1 if better_model == model2 else model2} (confidence: {confidence})")
                else:
                    summary.append(f"   - No statistically significant difference between {model1} and {model2}")
        
        # Add model rankings
        summary.append("\n3. Model Rankings by Metric:")
        for metric, ranked_models in report['comparison']['rankings'].items():
            summary.append(f"   - {metric.upper()}: {', '.join(ranked_models)}")
        
        # Add recommendations
        summary.append("\n4. Recommendations:")
        summary.append(f"   - Primary model: {best_model}")
        
        # If there are multiple models, suggest ensemble approach
        if len(report['models']) > 1:
            summary.append("   - Consider using an ensemble approach combining the top models for potentially better performance")
        
        return '\n'.join(summary)
    
    def _plot_model_comparison(self, report: Dict) -> None:
        """
        Create comparison plots for model evaluation metrics.
        
        Args:
            report (Dict): Comparison report data
        """
        # Create directory for plots
        plots_dir = os.path.join(self.plots_dir, 'comparison')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get metrics and models
        metrics = list(report['comparison']['metrics'].keys())
        models = list(report['models'].keys())
        
        # Plot comparison for key metrics (RMSE, MAE, MAPE, R²)
        key_metrics = ['rmse', 'mae', 'mape', 'r2']
        key_metrics = [m for m in key_metrics if m in metrics]  # Filter to available metrics
        
        # Create subplots for key metrics
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 4 * len(key_metrics)))
        if len(key_metrics) == 1:
            axes = [axes]  # Make axes iterable if only one subplot
        
        # Plot each metric
        for i, metric in enumerate(key_metrics):
            # Get values for this metric
            values = [report['comparison']['metrics'][metric][model] for model in models]
            
            # Create bar chart
            bars = axes[i].bar(models, values, color='skyblue', alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            # Set title and labels
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel('Value')
            axes[i].set_xlabel('Model')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Highlight the best model
            best_idx = np.argmin(values) if metric in ['rmse', 'mae', 'mape', 'mbe', 'nrmse', 'theil_u'] else np.argmax(values)
            bars[best_idx].set_color('green')
            
            # Rotate x-axis labels for better readability
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'key_metrics_comparison.png'), dpi=300)
        plt.close()
        
        # Create radar chart for overall comparison
        if len(models) > 1 and len(metrics) > 2:
            # Normalize metrics for radar chart
            normalized_metrics = {}
            for metric in metrics:
                values = np.array([report['comparison']['metrics'][metric][model] for model in models])
                
                # Normalize based on whether higher or lower is better
                if metric in ['rmse', 'mae', 'mape', 'mbe', 'nrmse', 'theil_u']:
                    # Lower is better, so invert the normalization
                    if np.max(values) - np.min(values) > 0:
                        normalized_metrics[metric] = 1 - (values - np.min(values)) / (np.max(values) - np.min(values))
                    else:
                        normalized_metrics[metric] = np.ones_like(values)
                else:
                    # Higher is better
                    if np.max(values) - np.min(values) > 0:
                        normalized_metrics[metric] = (values - np.min(values)) / (np.max(values) - np.min(values))
                    else:
                        normalized_metrics[metric] = np.ones_like(values)
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set number of metrics and angles
            N = len(metrics)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Plot each model
            for i, model in enumerate(models):
                values = [normalized_metrics[metric][i] for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            
            # Add legend
            plt.legend(loc='upper right')
            
            plt.title('Model Performance Comparison (Normalized)')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'radar_chart_comparison.png'), dpi=300)
            plt.close()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Dictionary of metrics
        """
        # Handle NaN values
        # Find indices where neither array has NaN
        valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        # Filter out NaN values
        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]
        
        # Log the number of valid samples
        logger.info(f"Calculating metrics on {len(y_true_valid)} valid samples out of {len(y_true)} total")
        
        if len(y_true_valid) == 0:
            logger.error("No valid samples for metric calculation")
            return {
                'rmse': float('nan'),
                'mae': float('nan'),
                'mape': float('nan'),
                'r2': float('nan'),
                'mbe': float('nan'),
                'nrmse': float('nan'),
                'theil_u': float('nan')
            }
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        
        # Mean Absolute Percentage Error (MAPE)
        # Add small epsilon to avoid division by zero
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (y_true_valid + 1e-10))) * 100
        
        # R-squared (coefficient of determination)
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        # Additional metrics
        # Mean Bias Error (MBE)
        mbe = np.mean(y_pred_valid - y_true_valid)
        
        # Normalized RMSE (NRMSE)
        y_range = np.max(y_true_valid) - np.min(y_true_valid)
        nrmse = rmse / y_range if y_range > 0 else np.nan
        
        # Theil's U statistic (forecast accuracy)
        # Add small epsilon to avoid division by zero
        numerator = np.sqrt(np.mean(np.square(y_pred_valid - y_true_valid)))
        denominator = np.sqrt(np.mean(np.square(y_true_valid))) + np.sqrt(np.mean(np.square(y_pred_valid)))
        theil_u = numerator / denominator if denominator > 0 else np.nan
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'mbe': float(mbe),
            'nrmse': float(nrmse),
            'theil_u': float(theil_u)
        }
    
    def _evaluate_by_site(self, test_data: pd.DataFrame, model: BaseModel,
                         feature_columns: List[str], target_columns: List[str],
                         site_column: str) -> Dict:
        """
        Evaluate model performance for each site.
        
        Args:
            test_data (pd.DataFrame): Test data
            model (BaseModel): Model to evaluate
            feature_columns (List[str]): Feature column names
            target_columns (List[str]): Target column names
            site_column (str): Column name for site IDs
            
        Returns:
            Dict: Site-specific evaluation results
        """
        logger.info(f"Performing site-specific evaluation for {model.model_name}")
        
        # Get unique sites
        unique_sites = test_data[site_column].unique()
        
        # Initialize results dictionary
        site_results = {}
        
        # Evaluate for each site
        for site in unique_sites:
            # Filter data for this site
            site_data = test_data[test_data[site_column] == site]
            
            # Skip if not enough data
            if len(site_data) < 10:  # Arbitrary threshold
                logger.warning(f"Skipping site {site} due to insufficient data ({len(site_data)} samples)")
                continue
            
            # Preprocess site data
            site_loader, _ = model.preprocess_data(
                data=site_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
            
            # Get predictions and actual values
            all_predictions = []
            all_actuals = []
            
            for batch in site_loader:
                X_batch, y_batch = batch
                
                # Convert to numpy for consistent processing
                X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                
                # Get predictions
                y_pred = model.predict(X_np)
                
                # Store predictions and actuals
                all_predictions.append(y_pred)
                all_actuals.append(y_np)
            
            # Concatenate batches
            if all_predictions and all_actuals:
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Calculate metrics
                metrics = self._calculate_metrics(all_actuals, all_predictions)
                
                # Store results
                site_results[site] = metrics
                
                logger.info(f"  Site {site} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        
        # Convert NumPy types to Python native types for JSON serialization
        serializable_results = self._convert_to_serializable(site_results)
        
        # Save site-specific results
        site_results_path = os.path.join(self.site_analysis_dir, f"{model.model_name}_site_results.json")
        with open(site_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # Create site comparison visualization
        self._plot_site_comparison(model.model_name, site_results)
        
        return site_results
    
    def _analyze_temporal_patterns(self, test_data: pd.DataFrame, model: BaseModel,
                                 feature_columns: List[str], target_columns: List[str],
                                 time_column: str) -> Dict:
        """
        Analyze model performance across different temporal patterns.
        
        Args:
            test_data (pd.DataFrame): Test data
            model (BaseModel): Model to evaluate
            feature_columns (List[str]): Feature column names
            target_columns (List[str]): Target column names
            time_column (str): Column name for timestamps
            
        Returns:
            Dict: Temporal pattern analysis results
        """
        logger.info(f"Performing temporal pattern analysis for {model.model_name}")
        
        # Ensure time column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(test_data[time_column]):
            test_data[time_column] = pd.to_datetime(test_data[time_column])
        
        # Extract temporal features
        test_data['hour'] = test_data[time_column].dt.hour
        test_data['day_of_week'] = test_data[time_column].dt.dayofweek  # 0=Monday, 6=Sunday
        test_data['is_weekend'] = test_data['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
        test_data['part_of_day'] = pd.cut(
            test_data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Initialize results dictionary
        temporal_results = {
            'hourly': {},
            'daily': {},
            'part_of_day': {},
            'weekend_vs_weekday': {}
        }
        
        # Analyze by hour of day
        for hour in range(24):
            hour_data = test_data[test_data['hour'] == hour]
            
            # Skip if not enough data
            if len(hour_data) < 10:  # Arbitrary threshold
                logger.warning(f"Skipping hour {hour} due to insufficient data ({len(hour_data)} samples)")
                continue
            
            # Preprocess hour data
            hour_loader, _ = model.preprocess_data(
                data=hour_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
            
            # Get predictions and actual values
            all_predictions = []
            all_actuals = []
            
            for batch in hour_loader:
                X_batch, y_batch = batch
                
                # Convert to numpy for consistent processing
                X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                
                # Get predictions
                y_pred = model.predict(X_np)
                
                # Store predictions and actuals
                all_predictions.append(y_pred)
                all_actuals.append(y_np)
            
            # Concatenate batches
            if all_predictions and all_actuals:
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Calculate metrics
                metrics = self._calculate_metrics(all_actuals, all_predictions)
                
                # Store results
                temporal_results['hourly'][str(hour)] = metrics
        
        # Analyze by day of week
        for day in range(7):
            day_name = calendar.day_name[day]
            day_data = test_data[test_data['day_of_week'] == day]
            
            # Skip if not enough data
            if len(day_data) < 10:  # Arbitrary threshold
                logger.warning(f"Skipping {day_name} due to insufficient data ({len(day_data)} samples)")
                continue
            
            # Preprocess day data
            day_loader, _ = model.preprocess_data(
                data=day_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
            
            # Get predictions and actual values
            all_predictions = []
            all_actuals = []
            
            for batch in day_loader:
                X_batch, y_batch = batch
                
                # Convert to numpy for consistent processing
                X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                
                # Get predictions
                y_pred = model.predict(X_np)
                
                # Store predictions and actuals
                all_predictions.append(y_pred)
                all_actuals.append(y_np)
            
            # Concatenate batches
            if all_predictions and all_actuals:
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Calculate metrics
                metrics = self._calculate_metrics(all_actuals, all_predictions)
                
                # Store results
                temporal_results['daily'][day_name] = metrics
        
        # Analyze by part of day
        for part in ['night', 'morning', 'afternoon', 'evening']:
            part_data = test_data[test_data['part_of_day'] == part]
            
            # Skip if not enough data
            if len(part_data) < 10:  # Arbitrary threshold
                logger.warning(f"Skipping {part} due to insufficient data ({len(part_data)} samples)")
                continue
            
            # Preprocess part of day data
            part_loader, _ = model.preprocess_data(
                data=part_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
            
            # Get predictions and actual values
            all_predictions = []
            all_actuals = []
            
            for batch in part_loader:
                X_batch, y_batch = batch
                
                # Convert to numpy for consistent processing
                X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                
                # Get predictions
                y_pred = model.predict(X_np)
                
                # Store predictions and actuals
                all_predictions.append(y_pred)
                all_actuals.append(y_np)
            
            # Concatenate batches
            if all_predictions and all_actuals:
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Calculate metrics
                metrics = self._calculate_metrics(all_actuals, all_predictions)
                
                # Store results
                temporal_results['part_of_day'][part] = metrics
        
        # Analyze weekend vs weekday
        for is_weekend, label in [(0, 'weekday'), (1, 'weekend')]:
            weekend_data = test_data[test_data['is_weekend'] == is_weekend]
            
            # Skip if not enough data
            if len(weekend_data) < 10:  # Arbitrary threshold
                logger.warning(f"Skipping {label} due to insufficient data ({len(weekend_data)} samples)")
                continue
            
            # Preprocess weekend/weekday data
            weekend_loader, _ = model.preprocess_data(
                data=weekend_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                train_mode=False
            )
            
            # Get predictions and actual values
            all_predictions = []
            all_actuals = []
            
            for batch in weekend_loader:
                X_batch, y_batch = batch
                
                # Convert to numpy for consistent processing
                X_np = X_batch.numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                y_np = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
                
                # Get predictions
                y_pred = model.predict(X_np)
                
                # Store predictions and actuals
                all_predictions.append(y_pred)
                all_actuals.append(y_np)
            
            # Concatenate batches
            if all_predictions and all_actuals:
                all_predictions = np.vstack(all_predictions)
                all_actuals = np.vstack(all_actuals)
                
                # Calculate metrics
                metrics = self._calculate_metrics(all_actuals, all_predictions)
                
                # Store results
                temporal_results['weekend_vs_weekday'][label] = metrics
        
        # Save temporal analysis results
        temporal_results_path = os.path.join(self.temporal_analysis_dir, f"{model.model_name}_temporal_results.json")
        with open(temporal_results_path, 'w') as f:
            json.dump(temporal_results, f, indent=4)
        
        # Create temporal pattern visualizations
        self._plot_temporal_patterns(model.model_name, temporal_results)
        
        return temporal_results
        
    def _plot_temporal_patterns(self, model_name: str, temporal_results: Dict):
        """
        Plot temporal pattern analysis results.
        
        Args:
            model_name (str): Name of the model
            temporal_results (Dict): Temporal pattern analysis results
        """
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(self.plots_dir, model_name, 'temporal_patterns')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot hourly patterns
        if temporal_results['hourly']:
            plt.figure(figsize=(12, 8))
            hours = sorted([int(h) for h in temporal_results['hourly'].keys()])
            rmse_values = [temporal_results['hourly'][str(h)]['rmse'] for h in hours]
            mae_values = [temporal_results['hourly'][str(h)]['mae'] for h in hours]
            r2_values = [temporal_results['hourly'][str(h)]['r2'] for h in hours]
            
            # Plot RMSE and MAE on primary y-axis
            ax1 = plt.gca()
            ax1.plot(hours, rmse_values, 'b-', marker='o', label='RMSE')
            ax1.plot(hours, mae_values, 'g-', marker='s', label='MAE')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Error (RMSE/MAE)')
            ax1.set_xticks(hours)
            ax1.set_xticklabels([f'{h:02d}:00' for h in hours])
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot R² on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(hours, r2_values, 'r-', marker='^', label='R²')
            ax2.set_ylabel('R² Score')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.title(f'Hourly Performance Metrics for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'hourly_performance.png'), dpi=300)
            plt.close()
        
        # Plot daily patterns
        if temporal_results['daily']:
            plt.figure(figsize=(12, 8))
            days = list(temporal_results['daily'].keys())
            rmse_values = [temporal_results['daily'][d]['rmse'] for d in days]
            mae_values = [temporal_results['daily'][d]['mae'] for d in days]
            r2_values = [temporal_results['daily'][d]['r2'] for d in days]
            
            # Plot RMSE and MAE on primary y-axis
            ax1 = plt.gca()
            x = np.arange(len(days))
            width = 0.35
            ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='blue', alpha=0.7)
            ax1.bar(x + width/2, mae_values, width, label='MAE', color='green', alpha=0.7)
            ax1.set_xlabel('Day of Week')
            ax1.set_ylabel('Error (RMSE/MAE)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(days)
            
            # Plot R² on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(x, r2_values, 'r-', marker='^', linewidth=2, label='R²')
            ax2.set_ylabel('R² Score')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.title(f'Daily Performance Metrics for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'daily_performance.png'), dpi=300)
            plt.close()
        
        # Plot part of day patterns
        if temporal_results['part_of_day']:
            plt.figure(figsize=(10, 6))
            parts = list(temporal_results['part_of_day'].keys())
            rmse_values = [temporal_results['part_of_day'][p]['rmse'] for p in parts]
            mae_values = [temporal_results['part_of_day'][p]['mae'] for p in parts]
            r2_values = [temporal_results['part_of_day'][p]['r2'] for p in parts]
            
            # Create a radar chart
            angles = np.linspace(0, 2*np.pi, len(parts), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Add the first value at the end to close the loop
            rmse_values += [rmse_values[0]]
            mae_values += [mae_values[0]]
            r2_values += [r2_values[0]]
            parts += [parts[0]]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            
            # Plot each metric
            ax.plot(angles, rmse_values, 'b-', linewidth=2, label='RMSE')
            ax.plot(angles, mae_values, 'g-', linewidth=2, label='MAE')
            
            # Set the labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(parts[:-1])
            
            # Add legend
            plt.legend(loc='upper right')
            
            plt.title(f'Part of Day Performance for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'part_of_day_performance.png'), dpi=300)
            plt.close()
            
            # Create a separate plot for R² values
            plt.figure(figsize=(10, 6))
            plt.bar(parts[:-1], r2_values[:-1], color='r', alpha=0.7)
            plt.xlabel('Part of Day')
            plt.ylabel('R² Score')
            plt.title(f'R² Score by Part of Day for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'part_of_day_r2.png'), dpi=300)
            plt.close()
        
        # Plot weekend vs weekday comparison
        if temporal_results['weekend_vs_weekday']:
            plt.figure(figsize=(10, 6))
            categories = list(temporal_results['weekend_vs_weekday'].keys())
            
            # Extract metrics
            metrics = {}
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                metrics[metric] = [temporal_results['weekend_vs_weekday'][c][metric] for c in categories]
            
            # Create grouped bar chart
            x = np.arange(len(categories))
            width = 0.2
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot error metrics on primary y-axis
            ax1.bar(x - width, metrics['rmse'], width, label='RMSE', color='blue', alpha=0.7)
            ax1.bar(x, metrics['mae'], width, label='MAE', color='green', alpha=0.7)
            ax1.bar(x + width, metrics['mape'], width, label='MAPE (%)', color='purple', alpha=0.7)
            
            ax1.set_xlabel('Time Period')
            ax1.set_ylabel('Error Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            
            # Plot R² on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(x, metrics['r2'], 'r-', marker='o', linewidth=2, label='R²')
            ax2.set_ylabel('R² Score')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.title(f'Weekend vs Weekday Performance for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'weekend_weekday_comparison.png'), dpi=300)
            plt.close()
            
    def _convert_to_serializable(self, obj):
        """
        Convert NumPy types to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert (can be dict, list, NumPy array, or scalar)
            
        Returns:
            Object with NumPy types converted to Python native types
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        elif isinstance(obj, bool):
            return obj
        else:
            return str(obj)
    
    def _plot_site_comparison(self, model_name: str, site_results: Dict):
        """
        Plot site-specific performance comparison.
        
        Args:
            model_name (str): Name of the model
            site_results (Dict): Site-specific evaluation results
        """
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(self.plots_dir, model_name, 'site_comparison')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract site IDs and metrics
        sites = list(site_results.keys())
        
        # Skip if no sites
        if not sites:
            logger.warning(f"No site results available for {model_name}")
            return
        
        # Extract metrics for each site
        rmse_values = [site_results[site]['rmse'] for site in sites]
        mae_values = [site_results[site]['mae'] for site in sites]
        r2_values = [site_results[site]['r2'] for site in sites]
        
        # Sort sites by RMSE (ascending)
        sorted_indices = np.argsort(rmse_values)
        sorted_sites = [sites[i] for i in sorted_indices]
        sorted_rmse = [rmse_values[i] for i in sorted_indices]
        sorted_mae = [mae_values[i] for i in sorted_indices]
        sorted_r2 = [r2_values[i] for i in sorted_indices]
        
        # Create horizontal bar chart for RMSE and MAE
        plt.figure(figsize=(12, max(8, len(sites) * 0.3)))
        y_pos = np.arange(len(sorted_sites))
        
        # Plot RMSE and MAE
        plt.barh(y_pos - 0.2, sorted_rmse, 0.4, label='RMSE', color='blue', alpha=0.7)
        plt.barh(y_pos + 0.2, sorted_mae, 0.4, label='MAE', color='green', alpha=0.7)
        
        plt.yticks(y_pos, sorted_sites)
        plt.xlabel('Error Value')
        plt.ylabel('SCATS Site ID')
        plt.title(f'Site-Specific Error Metrics for {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'site_error_metrics.png'), dpi=300)
        plt.close()
        
        # Create horizontal bar chart for R²
        plt.figure(figsize=(12, max(8, len(sites) * 0.3)))
        
        # Sort sites by R² (descending)
        r2_sorted_indices = np.argsort(r2_values)[::-1]
        r2_sorted_sites = [sites[i] for i in r2_sorted_indices]
        r2_sorted_values = [r2_values[i] for i in r2_sorted_indices]
        
        plt.barh(np.arange(len(r2_sorted_sites)), r2_sorted_values, 0.6, color='red', alpha=0.7)
        plt.yticks(np.arange(len(r2_sorted_sites)), r2_sorted_sites)
        plt.xlabel('R² Score')
        plt.ylabel('SCATS Site ID')
        plt.title(f'Site-Specific R² Scores for {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'site_r2_scores.png'), dpi=300)
        plt.close()
        
        # Create heatmap of site performance
        if len(sites) > 1:
            # Create a dataframe for the heatmap
            heatmap_data = pd.DataFrame({
                'Site': sites,
                'RMSE': rmse_values,
                'MAE': mae_values,
                'R²': r2_values
            })
            
            # Set Site as index
            heatmap_data = heatmap_data.set_index('Site')
            
            # Create heatmap
            plt.figure(figsize=(10, max(8, len(sites) * 0.4)))
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', linewidths=0.5)
            plt.title(f'Performance Metrics Heatmap for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'site_metrics_heatmap.png'), dpi=300)
            plt.close()


def create_model_comparison_visualization(report_path=None, output_dir=None):
    """Create a comprehensive comparison visualization of all models.
        
    Args:
        report_path (str): Path to the model comparison report JSON file.
        output_dir (str): Directory to save the comparison visualizations.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Set default paths if not provided
        if not report_path:
            report_path = os.path.join(os.path.dirname(__file__), 'evaluation', 'model_comparison_report.json')
            if not os.path.exists(report_path):
                report_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model_comparison_report.json')
        
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(__file__), 'evaluation', 'plots', 'comparison')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the comparison report
        if not os.path.exists(report_path):
            logger.error(f"Model comparison report not found at {report_path}")
            return False
        
        with open(report_path, 'r') as f:
            comparison_data = json.load(f)
        
        # Extract model names and metrics
        model_names = list(comparison_data.keys())
        if not model_names:
            logger.error("No models found in the comparison report.")
            return False
        
        # Check if we have metrics for each model
        models_with_metrics = [model for model in model_names if 'metrics' in comparison_data[model]]
        if not models_with_metrics:
            logger.error("No models with metrics found in the comparison report.")
            return False
        
        # Extract common metrics across all models
        common_metrics = set()
        for model in models_with_metrics:
            if 'metrics' in comparison_data[model]:
                metrics = comparison_data[model]['metrics']
                if not common_metrics:
                    common_metrics = set(metrics.keys())
                else:
                    common_metrics = common_metrics.intersection(set(metrics.keys()))
        
        if not common_metrics:
            logger.error("No common metrics found across models.")
            return False
        
        # Prepare data for visualization
        metrics_data = {}
        for metric in common_metrics:
            metrics_data[metric] = []
            for model in models_with_metrics:
                if 'metrics' in comparison_data[model] and metric in comparison_data[model]['metrics']:
                    metrics_data[metric].append(comparison_data[model]['metrics'][metric])
                else:
                    metrics_data[metric].append(np.nan)
        
        # Create bar charts for each metric
        for metric in common_metrics:
            plt.figure(figsize=(10, 6))
            
            # Format metric name for display
            metric_display = metric.upper() if metric in ['rmse', 'mae', 'mape', 'mbe', 'nrmse'] else \
                            'R²' if metric == 'r2' else \
                            "Theil's U" if metric == 'theil_u' else metric
            
            # Create bar chart
            bars = plt.bar(models_with_metrics, metrics_data[metric], color=['#3498db', '#e74c3c', '#2ecc71'][:len(models_with_metrics)])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Set title and labels
            plt.title(f'Comparison of {metric_display} Across Models', fontsize=14, fontweight='bold')
            plt.ylabel(metric_display, fontsize=12)
            plt.xlabel('Model', fontsize=12)
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300)
            plt.close()
        
        # Create a radar chart for overall comparison
        # Normalize metrics for radar chart (all metrics should be on a 0-1 scale where 1 is best)
        radar_metrics = ['rmse', 'mae', 'mape', 'r2', 'nrmse', 'theil_u']
        radar_metrics = [m for m in radar_metrics if m in common_metrics]
        
        if len(radar_metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
            # Prepare data for radar chart
            radar_data = {}
            for model in models_with_metrics:
                radar_data[model] = []
            
            # Normalize each metric (lower is better for all except r2)
            for metric in radar_metrics:
                values = [comparison_data[model]['metrics'][metric] for model in models_with_metrics]
                
                # For R², higher is better (1 is perfect), so we invert the normalization
                if metric == 'r2':
                    # Handle the case where all values are the same
                    if max(values) == min(values):
                        normalized = [0.5] * len(values)  # Assign a neutral value
                    else:
                        # Normalize to 0-1 scale where 1 is best (highest R²)
                        normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
                else:
                    # For other metrics, lower is better
                    # Handle the case where all values are the same
                    if max(values) == min(values):
                        normalized = [0.5] * len(values)  # Assign a neutral value
                    else:
                        # Normalize to 0-1 scale and invert (1 - norm) so 1 is best (lowest error)
                        normalized = [1 - ((v - min(values)) / (max(values) - min(values))) for v in values]
                
                # Add normalized values to radar data
                for i, model in enumerate(models_with_metrics):
                    radar_data[model].append(normalized[i])
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set number of angles based on number of metrics
            angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
            
            # Close the plot
            angles += angles[:1]
            
            # Set labels for radar chart
            metric_labels = []
            for metric in radar_metrics:
                if metric == 'rmse':
                    metric_labels.append('RMSE')
                elif metric == 'mae':
                    metric_labels.append('MAE')
                elif metric == 'mape':
                    metric_labels.append('MAPE')
                elif metric == 'r2':
                    metric_labels.append('R²')
                elif metric == 'nrmse':
                    metric_labels.append('NRMSE')
                elif metric == 'theil_u':
                    metric_labels.append("Theil's U")
                else:
                    metric_labels.append(metric.upper())
            
            # Set radar chart labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=12)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Plot each model
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']  # Add more colors if needed
            for i, model in enumerate(models_with_metrics):
                values = radar_data[model]
                values += values[:1]  # Close the polygon
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            plt.title('Model Performance Comparison (Higher is Better)', fontsize=15, fontweight='bold', y=1.1)
            
            # Save radar chart
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined bar chart with all metrics
        plt.figure(figsize=(12, 8))
        
        # Number of metrics and models
        n_metrics = len(common_metrics)
        n_models = len(models_with_metrics)
        
        # Set width of bars
        bar_width = 0.8 / n_models
        
        # Set positions of bars on x-axis
        r = np.arange(n_metrics)
        
        # Create bars for each model
        for i, model in enumerate(models_with_metrics):
            # Extract values for this model
            values = []
            for metric in common_metrics:
                if 'metrics' in comparison_data[model] and metric in comparison_data[model]['metrics']:
                    values.append(comparison_data[model]['metrics'][metric])
                else:
                    values.append(np.nan)
            
            # Plot bars
            position = [x + bar_width * i for x in r]
            bars = plt.bar(position, values, width=bar_width, label=model, 
                          color=colors[i % len(colors)])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Add labels and title
        metric_labels = [m.upper() if m not in ['r2', 'theil_u'] else 'R²' if m == 'r2' else "Theil's U" for m in common_metrics]
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Comparison of All Metrics Across Models', fontsize=14, fontweight='bold')
        plt.xticks([r + bar_width * (n_models - 1) / 2 for r in range(n_metrics)], metric_labels)
        
        # Add legend
        plt.legend()
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300)
        plt.close()
        
        # Create a table visualization of all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Hide axes
        ax.axis('off')
        ax.axis('tight')
        
        # Create data for table
        table_data = []
        for metric in common_metrics:
            row = [metric.upper() if metric not in ['r2', 'theil_u'] else 'R²' if metric == 'r2' else "Theil's U"]
            for model in models_with_metrics:
                if 'metrics' in comparison_data[model] and metric in comparison_data[model]['metrics']:
                    row.append(f"{comparison_data[model]['metrics'][metric]:.4f}")
                else:
                    row.append('N/A')
            table_data.append(row)
        
        # Create column labels
        columns = ['Metric'] + models_with_metrics
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
        
        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the header
        for i, key in enumerate(columns):
            cell = table[(0, i)]
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f0f0f0')
        
        # Set title
        plt.title('Detailed Metrics Comparison Table', fontsize=14, fontweight='bold', pad=20)
        
        # Save table
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary image with key insights
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Hide axes
        ax.axis('off')
        
        # Determine best model for each metric
        best_models = {}
        for metric in common_metrics:
            values = []
            for model in models_with_metrics:
                if 'metrics' in comparison_data[model] and metric in comparison_data[model]['metrics']:
                    values.append((model, comparison_data[model]['metrics'][metric]))
            
            if values:
                # For R², higher is better
                if metric == 'r2':
                    best_model = max(values, key=lambda x: x[1])
                else:  # For other metrics, lower is better
                    best_model = min(values, key=lambda x: x[1])
                
                best_models[metric] = best_model
        
        # Create text for summary
        summary_text = "MODEL COMPARISON SUMMARY\n\n"
        
        # Add best model for each metric
        summary_text += "Best Model by Metric:\n"
        for metric, (model, value) in best_models.items():
            metric_display = metric.upper() if metric not in ['r2', 'theil_u'] else 'R²' if metric == 'r2' else "Theil's U"
            summary_text += f"  {metric_display}: {model} ({value:.4f})\n"
        
        # Calculate overall best model (most wins)
        model_wins = {}
        for model, _ in best_models.values():
            if model not in model_wins:
                model_wins[model] = 0
            model_wins[model] += 1
        
        if model_wins:
            overall_best = max(model_wins.items(), key=lambda x: x[1])
            summary_text += f"\nOverall Best Model: {overall_best[0]} (best in {overall_best[1]} metrics)\n"
        
        # Add average rank for each model
        summary_text += "\nAverage Rank by Model:\n"
        model_ranks = {model: [] for model in models_with_metrics}
        
        for metric in common_metrics:
            values = []
            for model in models_with_metrics:
                if 'metrics' in comparison_data[model] and metric in comparison_data[model]['metrics']:
                    values.append((model, comparison_data[model]['metrics'][metric]))
            
            if values:
                # Sort models by performance (ascending for error metrics, descending for R²)
                if metric == 'r2':
                    sorted_models = sorted(values, key=lambda x: x[1], reverse=True)
                else:
                    sorted_models = sorted(values, key=lambda x: x[1])
                
                # Assign ranks
                for rank, (model, _) in enumerate(sorted_models, 1):
                    model_ranks[model].append(rank)
        
        # Calculate average rank
        avg_ranks = {}
        for model, ranks in model_ranks.items():
            if ranks:
                avg_ranks[model] = sum(ranks) / len(ranks)
        
        # Sort models by average rank
        sorted_avg_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        # Add to summary text
        for model, avg_rank in sorted_avg_ranks:
            summary_text += f"  {model}: {avg_rank:.2f}\n"
        
        # Add the summary text to the plot
        ax.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # Save summary
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created {len(common_metrics) + 3} comparison visualizations in {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating model comparison visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_model_comparison_visualization(report_path=None, output_dir=None):
    """
    Create comprehensive comparison visualizations for all models in the report.
    
    Args:
        report_path (str, optional): Path to the model comparison report JSON file.
        output_dir (str, optional): Directory to save the visualizations.
    """
    import json
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Set default paths if not provided
    if report_path is None:
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation', 'reports', 'model_comparison_report.json')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation', 'plots', 'comparison')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the report
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model comparison report: {e}")
        return
    
    # Check if we have models to compare
    if 'models' not in report or len(report['models']) < 1:
        logger.warning("No models found in the report for comparison")
        return
    
    # Extract model names and metrics
    model_names = list(report['models'].keys())
    
    # If we have a comparison section, use it
    if 'comparison' in report and 'metrics' in report['comparison']:
        metrics_data = report['comparison']['metrics']
        # Check if metrics data is not empty
        if not metrics_data:
            logger.warning("No metrics found in the comparison section")
            return
        
        # Get the first metric to determine available models
        first_metric = list(metrics_data.keys())[0]
        model_names = list(metrics_data[first_metric].keys())
    else:
        # If no comparison section, create one from the models section
        metrics_data = {}
        for metric in ['rmse', 'mae', 'mape', 'r2', 'mbe', 'nrmse', 'theil_u']:
            metrics_data[metric] = {}
            for model in model_names:
                if metric in report['models'][model]['metrics']:
                    metrics_data[metric][model] = report['models'][model]['metrics'][metric]
    
    # Create a DataFrame for easier plotting
    import pandas as pd
    metrics_df = pd.DataFrame(index=model_names)
    
    # Fill the DataFrame with metrics
    for metric in metrics_data:
        for model in model_names:
            if model in metrics_data[metric]:
                metrics_df.loc[model, metric] = metrics_data[metric][model]
    
    # Create a more readable version of the DataFrame for display
    display_df = metrics_df.copy()
    display_df.columns = [
        'RMSE', 'MAE', 'MAPE', 'R²', 'MBE', 'NRMSE', "Theil's U"
    ]
    
    # Create a bar chart for each metric
    metrics = ['rmse', 'mae', 'r2', 'mape', 'theil_u']
    metric_titles = ['RMSE (lower is better)', 'MAE (lower is better)', 'R² (higher is better)', 
                     'MAPE (lower is better)', "Theil's U (lower is better)"]
    
    # Set up the figure with 5 subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Create bar charts for each metric
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        
        # Sort models by metric value
        if metric in ['rmse', 'mae', 'mape', 'theil_u']:
            # Lower is better, sort ascending
            sorted_df = metrics_df.sort_values(by=metric)
        else:
            # Higher is better, sort descending
            sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        
        # Get values for the current metric
        values = sorted_df[metric].values
        models = sorted_df.index
        
        # Create bar chart
        bars = ax.bar(models, values, color=sns.color_palette("muted", len(models)))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Set title and labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper())
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Create a radar chart for overall comparison
    ax_radar = fig.add_subplot(gs[2, :], polar=True)
    
    # Normalize metrics for radar chart
    normalized_metrics = {}
    for metric in metrics:
        values = np.array([metrics_df.loc[model, metric] for model in model_names])
        
        # Normalize based on whether higher or lower is better
        if metric in ['rmse', 'mae', 'mape', 'theil_u']:
            # Lower is better, so invert the normalization
            if np.max(values) - np.min(values) > 0:
                normalized_metrics[metric] = 1 - (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized_metrics[metric] = np.ones_like(values)
        else:
            # Higher is better
            if np.max(values) - np.min(values) > 0:
                normalized_metrics[metric] = (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized_metrics[metric] = np.ones_like(values)
    
    # Set number of metrics and angles
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set labels for radar chart
    metric_labels = ['RMSE', 'MAE', 'R²', 'MAPE', "Theil's U"]
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels)
    
    # Plot each model on the radar chart
    for i, model in enumerate(model_names):
        values = [normalized_metrics[metric][i] for metric in metrics]
        values += values[:1]  # Close the loop
        
        ax_radar.plot(angles, values, linewidth=2, label=model)
        ax_radar.fill(angles, values, alpha=0.1)
    
    # Add legend
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.set_title('Model Performance Comparison (Normalized)', fontsize=12, fontweight='bold')
    
    # Add a table with the actual metric values
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis('off')
    
    # Create a formatted table with the metrics
    cell_text = []
    for model in model_names:
        row = [f"{metrics_df.loc[model, metric]:.4f}" for metric in metrics]
        cell_text.append(row)
    
    table = table_ax.table(
        cellText=cell_text,
        rowLabels=model_names,
        colLabels=metric_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.15] * len(metrics)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title to the table
    table_ax.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comprehensive model comparison visualization saved to {os.path.join(output_dir, 'comprehensive_model_comparison.png')}")
    
    # Create a performance summary table image
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    ax.axis('off')
    
    # Format the display DataFrame for better readability
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # Add a title
    ax.set_title('Traffic Prediction Model Performance Comparison', fontsize=14, fontweight='bold')
    
    # Create the table
    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.12] * len(display_df.columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight the best value in each column
    for j, col in enumerate(display_df.columns):
        # Determine if higher or lower is better for this metric
        if col in ['R²']:
            # Higher is better
            best_idx = display_df[col].astype(float).idxmax()
        else:
            # Lower is better
            best_idx = display_df[col].astype(float).idxmin()
        
        # Get the row index for the best value
        row_idx = display_df.index.get_loc(best_idx)
        
        # Highlight the cell
        cell = table[(row_idx+1, j)]  # +1 for the header row
        cell.set_facecolor('lightgreen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model performance table saved to {os.path.join(output_dir, 'model_performance_table.png')}")
    
    return True


def main():
    """
    Main function to demonstrate the usage of the ModelEvaluator class.
    This allows the script to be run directly with 'python evaluator.py'.
    """
    import argparse
    import sys
    from app.core.ml.lstm_model import LSTMModel
    from app.core.ml.gru_model import GRUModel
    from app.core.ml.cnnrnn_model import CNNRNNModel
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate traffic prediction models')
    parser.add_argument('--data_path', type=str, default=None, help='Path to test data CSV file')
    parser.add_argument('--models_dir', type=str, default=None, help='Directory containing trained models')
    parser.add_argument('--model_path', type=str, nargs='*', default=None, help='Path to one or more model checkpoint files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--metrics', type=str, default='all', help='Metrics to evaluate (comma-separated, or "all")')
    parser.add_argument('--site_analysis', action='store_true', help='Perform site-specific analysis')
    parser.add_argument('--temporal_analysis', action='store_true', help='Perform temporal pattern analysis')
    parser.add_argument('--statistical_tests', action='store_true', help='Perform statistical significance tests')
    parser.add_argument('--generate_report', action='store_true', help='Generate comprehensive evaluation report')
    parser.add_argument('--create_comparison', action='store_true', help='Create comprehensive model comparison visualization')
    parser.add_argument('--report_path', type=str, default=None, help='Path to the model comparison report JSON file for visualization')
    parser.add_argument('--comparison_dir', type=str, default=None, help='Directory to save the comparison visualizations')
    
    # Print help if no arguments are provided or --help is specified
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        return
        
    args = parser.parse_args()
    
    # Use default paths from config if not provided
    data_path = args.data_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'processed', 'test_data.csv')
    
    # Set up default models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    
    model_path = args.model_path
    
    # Set up default output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
    
    logger.info(f"Starting model evaluation with data from {data_path}")
    
    # Check if we should create a comparison visualization
    if args.create_comparison:
        logger.info("Creating comprehensive model comparison visualization...")
        success = create_model_comparison_visualization(
            report_path=args.report_path,
            output_dir=args.comparison_dir
        )
        if success:
            logger.info("Model comparison visualization created successfully.")
            print("\nModel Comparison Visualization created successfully.\n")
            print("Visualizations saved to: " + 
                  (args.comparison_dir or os.path.join(os.path.dirname(__file__), 'evaluation', 'plots', 'comparison')))
        else:
            logger.error("Failed to create model comparison visualization.")
            print("\nFailed to create model comparison visualization.\n")
        return
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Test data file not found: {data_path}")
        return
    
    # Load test data
    try:
        test_data = pd.read_csv(data_path)
        logger.info(f"Loaded test data with {len(test_data)} records")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Define feature and target columns with the improved selection
    # These must match what was used during training
    feature_columns = [
        # Core traffic features (most important)
        'Traffic_Count_t-1',      # Previous interval (strongest predictor)
        'Traffic_Count_t-4',      # One hour ago
        'Traffic_Count_t-96',     # Same time yesterday
        'Rolling_Mean_1-hour',    # Recent trend
        
        # Time features (cyclical patterns)
        'Hour',                   # Time of day
        'DayOfWeek',              # Day of week
        'IsWeekend',              # Weekend flag
        
        # Current traffic
        'Traffic_Count'
    ]
    target_columns = ['Target_t+1']
    site_column = 'SCATS_ID'
    time_column = 'DateTime'
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Load and evaluate models
    models = {}
    
    # If model paths are provided, load those models
    if model_path:
        # Convert single path to list if needed
        if isinstance(model_path, str):
            model_paths = [model_path]
        else:
            model_paths = model_path
            
        for path in model_paths:
            try:
                # Determine model type from filename
                model_filename = os.path.basename(path)
                if 'lstm' in model_filename.lower():
                    model_name = 'LSTM'
                    model = LSTMModel(model_name=model_name)
                elif 'gru' in model_filename.lower():
                    model_name = 'GRU'
                    model = GRUModel(model_name=model_name)
                elif 'cnn' in model_filename.lower() or 'rnn' in model_filename.lower():
                    model_name = 'CNN-RNN'
                    model = CNNRNNModel(model_name=model_name)
                else:
                    # Default to GRU if model type can't be determined
                    model_name = 'GRU'
                    model = GRUModel(model_name=model_name)
                    
                # Load model from checkpoint
                logger.info(f"Loading model from checkpoint: {path}")
                
                # Determine device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load checkpoint to the appropriate device
                checkpoint = torch.load(path, map_location=device)
                
                # Initialize model architecture
                input_dim = len(feature_columns)
                output_dim = len(target_columns)
                
                # Build the model architecture first
                if model_name == 'LSTM':
                    model.build_model(input_dim=input_dim, output_dim=output_dim)
                elif model_name == 'GRU':
                    model.build_model(input_dim=input_dim, output_dim=output_dim)
                elif model_name == 'CNN-RNN':
                    model.build_model(input_dim=input_dim, output_dim=output_dim)
                    
                # Move model to the appropriate device
                model.model.to(device)
                
                # Now load the state dict
                model.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Initialize scalers for the model with improved scaling approach
                from sklearn.preprocessing import StandardScaler, RobustScaler
                
                # Use RobustScaler for features to handle outliers better
                model.scaler_X = RobustScaler()
                # Keep StandardScaler for targets
                model.scaler_y = StandardScaler()
                
                # Handle outliers in the test data
                test_data_processed = test_data.copy()
                
                # Process each feature column
                for col in feature_columns:
                    if 'Traffic' in col:
                        # Calculate percentiles for outlier detection
                        q1 = test_data[col].quantile(0.05)
                        q3 = test_data[col].quantile(0.95)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Cap outliers
                        outliers = ((test_data[col] < lower_bound) | (test_data[col] > upper_bound)).sum()
                        if outliers > 0:
                            logger.info(f"Capping {outliers} outliers in {col} ({outliers/len(test_data)*100:.2f}%)")
                            test_data_processed[col] = test_data[col].clip(lower_bound, upper_bound)
                    
                    # Replace infinities with NaN
                    test_data_processed[col] = test_data_processed[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values
                    if test_data_processed[col].isna().any():
                        nan_count = test_data_processed[col].isna().sum()
                        logger.info(f"Filling {nan_count} NaN values in {col} ({nan_count/len(test_data_processed)*100:.2f}%)")
                        
                        # For traffic columns, use median (more robust to outliers)
                        if 'Traffic' in col:
                            fill_value = test_data_processed[col].median() if not pd.isna(test_data_processed[col].median()) else 0
                        else:
                            # For other columns, use mean
                            fill_value = test_data_processed[col].mean() if not pd.isna(test_data_processed[col].mean()) else 0
                            
                        test_data_processed[col] = test_data_processed[col].fillna(fill_value)
                
                # Fit the scalers on the processed test data
                X_sample = test_data_processed[feature_columns].values
                y_sample = test_data_processed[target_columns].values
                
                # Fit the scalers
                model.scaler_X.fit(X_sample)
                model.scaler_y.fit(y_sample)
                
                # Apply site-specific normalization if SCATS_ID column exists
                if site_column in test_data_processed.columns:
                    logger.info("Applying site-specific normalization for evaluation data")
                    # Store site information for later use
                    model.site_info = test_data_processed[site_column].values
                    model.use_site_normalization = True
                
                # If optimizer state dict is in the checkpoint, load it
                if 'optimizer_state_dict' in checkpoint and model.optimizer is not None:
                    try:
                        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except:
                        logger.warning(f"Could not load optimizer state for {model_name}")
                
                models[model_name] = model
                logger.info(f"Loaded {model_name} model from checkpoint: {path}")
            except Exception as e:
                logger.error(f"Error loading model from checkpoint {path}: {e}")
    else:
        # Load models from models_dir
        model_paths = {
            'LSTM': os.path.join(models_dir, 'lstm_model'),
            'GRU': os.path.join(models_dir, 'gru_model'),
            'CNN-RNN': os.path.join(models_dir, 'cnnrnn_model')
        }
        
        # Try to load each model
        for model_name, model_dir in model_paths.items():
            if os.path.exists(model_dir):
                try:
                    if model_name == 'LSTM':
                        model = LSTMModel(model_name=model_name)
                    elif model_name == 'GRU':
                        model = GRUModel(model_name=model_name)
                    elif model_name == 'CNN-RNN':
                        model = CNNRNNModel(model_name=model_name)
                    
                    # Load model weights
                    model.load(model_dir)
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_dir}")
                except Exception as e:
                    logger.error(f"Error loading {model_name} model: {e}")
    
    if not models:
        logger.error("No models could be loaded. Please train models first.")
        return
    
    # Evaluate all models
    evaluation_results = evaluator.evaluate_multiple_models(
        models=models,
        test_data=test_data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        site_column=site_column,
        time_column=time_column
    )
    
    logger.info("Model evaluation completed successfully.")
    logger.info(f"Evaluation results saved to {output_dir}")
    
    # Print a summary of the evaluation results
    if len(models) > 0:
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        for model_name, results in evaluator.results.items():
            if model_name in models and 'metrics' in results:
                print(f"\n{model_name} Model:")
                print("-" * 30)
                
                metrics = results['metrics']
                if 'rmse' in metrics:
                    print(f"RMSE: {metrics['rmse']:.4f}")
                if 'mae' in metrics:
                    print(f"MAE: {metrics['mae']:.4f}")
                if 'r2' in metrics:
                    print(f"R²: {metrics['r2']:.4f}")
                if 'mape' in metrics:
                    print(f"MAPE: {metrics['mape']:.4f}%")
                if 'mbe' in metrics:
                    print(f"MBE: {metrics['mbe']:.4f}")
                if 'nrmse' in metrics:
                    print(f"NRMSE: {metrics['nrmse']:.4f}")
                if 'theil_u' in metrics:
                    print(f"Theil's U: {metrics['theil_u']:.4f}")
        
        print("\n" + "=" * 50)
        
    # Generate comprehensive evaluation report if requested
    if args.generate_report or len(models) > 1:
        logger.info("Generating comprehensive evaluation report...")
        evaluator.generate_comparison_report()
        report_path = os.path.join(output_dir, 'model_comparison_report.json')
        evaluator.save_comparison_report(report_path)
        logger.info(f"Saved model comparison report to {report_path}")
        
        # Create comparison visualization automatically if multiple models were evaluated
        if len(models) > 1:
            logger.info("Creating model comparison visualization...")
            success = create_model_comparison_visualization(
                report_path=report_path,
                output_dir=os.path.join(output_dir, 'plots', 'comparison')
            )
            if success:
                logger.info("Model comparison visualization created successfully.")
                print("\nModel Comparison Visualization created successfully.\n")
                print(f"Visualizations saved to: {os.path.join(output_dir, 'plots', 'comparison')}")
            else:
                logger.error("Failed to create model comparison visualization.")
        
    logger.info("Model evaluation completed successfully.")
    logger.info(f"Evaluation results saved to {output_dir}")
    
    # Print summary of results
    print("\nModel Evaluation Summary:")
    print("=======================\n")
    for model_name, results in evaluator.results.items():
        if 'metrics' in results:
            print(f"{model_name} Model:")
            metrics = results['metrics']
            # Helper function to format metric values
            def format_metric(metric_value):
                if isinstance(metric_value, (int, float)):
                    return f"{metric_value:.4f}"
                return "N/A"
                
            print(f"  RMSE: {format_metric(metrics.get('rmse', 'N/A'))}")
            print(f"  MAE: {format_metric(metrics.get('mae', 'N/A'))}")
            print(f"  R²: {format_metric(metrics.get('r2', 'N/A'))}")
            print(f"  MAPE: {format_metric(metrics.get('mape', 'N/A'))}%" if isinstance(metrics.get('mape', 'N/A'), (int, float)) else "  MAPE: N/A")
            print(f"  NRMSE: {format_metric(metrics.get('nrmse', 'N/A'))}")
            print(f"  Theil's U: {format_metric(metrics.get('theil_u', 'N/A'))}")
            print()
    
    print("=" * 50)
    if args.generate_report or len(models) > 1:
        print(f"\nFull comparison report saved to: {os.path.join(output_dir, 'model_comparison_report.json')}")
        print(f"Visualizations saved to: {os.path.join(output_dir, 'plots')}")
    print("=" * 50 + "\n")
    
    logger.info("Model evaluation completed successfully")
    
    # This is the entry point when running the script directly
if __name__ == "__main__":
    # Only show running messages when actually running the evaluation, not when displaying help
    if len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
        print("Running Model Evaluator...")
    
    try:
        main()
        
        # Only show completion message when actually running the evaluation, not when displaying help
        if len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
            print("Model Evaluator completed.")
    except Exception as e:
        print(f"Error running Model Evaluator: {e}")
        import traceback
        traceback.print_exc()