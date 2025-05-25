#!/usr/bin/env python3
"""
SCATS Traffic Data Analyzer

This module provides functions for analyzing the SCATS traffic dataset structure,
validating data quality, and generating statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import datetime

# Add the parent directory to sys.path to allow importing app modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the logger and config
from app.core.logging import logger
from app.config.config import config

class SCATSDataAnalyzer:
    """
    Class for analyzing SCATS traffic data structure and quality.
    
    This class provides methods to:
    - Identify unique SCATS sites
    - Validate date ranges
    - Check data completeness
    - Display basic statistics
    - Assess data quality
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the SCATSDataAnalyzer with the path to the data file.
        
        Args:
            data_path (str, optional): Path to the SCATS traffic data file.
                If None, uses the default processed data path from config.
        """
        self.data_path = data_path or config.processed_data['scats_complete']
        self.site_ref_path = config.processed_data['site_reference']
        self.data_df = None
        self.site_ref_df = None
        self.load_data()
    
    def load_data(self):
        """
        Load the SCATS traffic data from CSV file.
        
        Returns:
            bool: True if data loaded successfully, False otherwise.
        """
        try:
            if os.path.exists(self.data_path):
                logger.info(f"Loading data from {self.data_path}")
                self.data_df = pd.read_csv(self.data_path)
                
                # Convert DateTime to datetime type if it exists
                if 'DateTime' in self.data_df.columns:
                    self.data_df['DateTime'] = pd.to_datetime(self.data_df['DateTime'])
                    
                # Load site reference data if available
                if os.path.exists(self.site_ref_path):
                    self.site_ref_df = pd.read_csv(self.site_ref_path)
                    logger.info(f"Loaded site reference data from {self.site_ref_path}")
                
                logger.info(f"Successfully loaded {len(self.data_df)} records")
                return True
            else:
                logger.error(f"Data file not found: {self.data_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def identify_unique_sites(self):
        """
        Identify unique SCATS sites in the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with unique SCATS sites and their counts.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Get unique SCATS IDs and their counts
            site_counts = self.data_df['SCATS_ID'].value_counts().reset_index()
            site_counts.columns = ['SCATS_ID', 'Record_Count']
            
            # Add site information if available
            if self.site_ref_df is not None:
                site_counts = pd.merge(
                    site_counts, 
                    self.site_ref_df[['SCATS_ID', 'Location', 'Latitude', 'Longitude']], 
                    on='SCATS_ID', 
                    how='left'
                )
            
            logger.info(f"Identified {len(site_counts)} unique SCATS sites")
            return site_counts
        except Exception as e:
            logger.error(f"Error identifying unique sites: {str(e)}")
            return None
    
    def validate_date_range(self):
        """
        Validate that the dataset covers the expected date range (Oct 1-31, 2006).
        
        Returns:
            dict: Dictionary with date range validation results.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Expected date range
            expected_start = pd.Timestamp('2006-10-01')
            expected_end = pd.Timestamp('2006-10-31 23:59:59')
            
            # Actual date range
            actual_start = self.data_df['DateTime'].min()
            actual_end = self.data_df['DateTime'].max()
            
            # Check if date range matches expectations
            starts_on_time = actual_start.date() == expected_start.date()
            ends_on_time = actual_end.date() == expected_end.date()
            
            # Create validation results
            results = {
                'expected_start': expected_start,
                'expected_end': expected_end,
                'actual_start': actual_start,
                'actual_end': actual_end,
                'starts_on_time': starts_on_time,
                'ends_on_time': ends_on_time,
                'complete_range': starts_on_time and ends_on_time,
                'days_covered': (actual_end.date() - actual_start.date()).days + 1
            }
            
            # Log results
            logger.info(f"Date range validation: {results['complete_range']}")
            logger.info(f"Dataset covers {results['days_covered']} days from {actual_start} to {actual_end}")
            
            return results
        except Exception as e:
            logger.error(f"Error validating date range: {str(e)}")
            return None
    
    def check_data_completeness(self):
        """
        Check the completeness of the dataset by identifying missing time intervals.
        
        Returns:
            dict: Dictionary with completeness check results.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Get unique SCATS IDs
            scats_ids = self.data_df['SCATS_ID'].unique()
            
            # Expected number of records per day (96 intervals per day)
            expected_intervals_per_day = 96
            
            # Get date range
            date_range = self.validate_date_range()
            if not date_range:
                return None
                
            expected_days = date_range['days_covered']
            expected_records_per_site = expected_intervals_per_day * expected_days
            
            # Check completeness for each site
            completeness_results = []
            for site_id in scats_ids:
                site_data = self.data_df[self.data_df['SCATS_ID'] == site_id]
                site_days = site_data['DateTime'].dt.date.nunique()
                site_records = len(site_data)
                
                # Calculate completeness
                day_completeness = site_days / expected_days
                record_completeness = site_records / expected_records_per_site
                
                # Check for missing intervals
                unique_dates = site_data['DateTime'].dt.date.unique()
                missing_intervals_by_date = {}
                
                for date in unique_dates:
                    date_data = site_data[site_data['DateTime'].dt.date == date]
                    if len(date_data) < expected_intervals_per_day:
                        missing_count = expected_intervals_per_day - len(date_data)
                        missing_intervals_by_date[str(date)] = missing_count
                
                completeness_results.append({
                    'SCATS_ID': site_id,
                    'Days_Present': site_days,
                    'Expected_Days': expected_days,
                    'Day_Completeness': day_completeness,
                    'Records_Present': site_records,
                    'Expected_Records': expected_records_per_site,
                    'Record_Completeness': record_completeness,
                    'Missing_Intervals': sum(missing_intervals_by_date.values()) if missing_intervals_by_date else 0
                })
            
            completeness_df = pd.DataFrame(completeness_results)
            
            # Overall completeness
            overall_completeness = {
                'total_sites': len(scats_ids),
                'expected_total_records': len(scats_ids) * expected_records_per_site,
                'actual_total_records': len(self.data_df),
                'overall_completeness': len(self.data_df) / (len(scats_ids) * expected_records_per_site),
                'site_completeness': completeness_df
            }
            
            logger.info(f"Overall data completeness: {overall_completeness['overall_completeness']:.2%}")
            logger.info(f"Sites with 100% completeness: {len(completeness_df[completeness_df['Record_Completeness'] == 1])}")
            
            return overall_completeness
        except Exception as e:
            logger.error(f"Error checking data completeness: {str(e)}")
            return None
    
    def display_basic_statistics(self):
        """
        Display basic statistics for the SCATS traffic data.
        
        Returns:
            dict: Dictionary with basic statistics.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Basic statistics
            traffic_stats = self.data_df['Traffic_Count'].describe()
            
            # Time-based statistics
            time_stats = {
                'by_hour': self.data_df.groupby(self.data_df['DateTime'].dt.hour)['Traffic_Count'].mean(),
                'by_day_of_week': self.data_df.groupby(self.data_df['DateTime'].dt.dayofweek)['Traffic_Count'].mean(),
                'weekday_vs_weekend': {
                    'weekday': self.data_df[self.data_df['DateTime'].dt.dayofweek < 5]['Traffic_Count'].mean(),
                    'weekend': self.data_df[self.data_df['DateTime'].dt.dayofweek >= 5]['Traffic_Count'].mean()
                }
            }
            
            # Site-based statistics
            site_stats = self.data_df.groupby('SCATS_ID')['Traffic_Count'].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            # Combine all statistics
            all_stats = {
                'traffic_stats': traffic_stats.to_dict(),
                'time_stats': {
                    'by_hour': time_stats['by_hour'].to_dict(),
                    'by_day_of_week': time_stats['by_day_of_week'].to_dict(),
                    'weekday_vs_weekend': time_stats['weekday_vs_weekend']
                },
                'site_stats': site_stats.to_dict('records')
            }
            
            # Log summary statistics
            logger.info(f"Traffic count statistics: min={traffic_stats['min']:.2f}, max={traffic_stats['max']:.2f}, mean={traffic_stats['mean']:.2f}")
            logger.info(f"Weekday vs Weekend average: Weekday={time_stats['weekday_vs_weekend']['weekday']:.2f}, Weekend={time_stats['weekday_vs_weekend']['weekend']:.2f}")
            
            return all_stats
        except Exception as e:
            logger.error(f"Error displaying basic statistics: {str(e)}")
            return None
    
    def assess_data_quality(self):
        """
        Assess the quality of the SCATS traffic data.
        
        Returns:
            dict: Dictionary with data quality assessment results.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Check for missing values
            missing_values = self.data_df.isnull().sum().to_dict()
            missing_percentage = (self.data_df.isnull().sum() / len(self.data_df) * 100).to_dict()
            
            # Check for outliers (using IQR method)
            Q1 = self.data_df['Traffic_Count'].quantile(0.25)
            Q3 = self.data_df['Traffic_Count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data_df[(self.data_df['Traffic_Count'] < lower_bound) | (self.data_df['Traffic_Count'] > upper_bound)]
            
            # Check for duplicate records
            duplicates = self.data_df.duplicated(['SCATS_ID', 'DateTime']).sum()
            
            # Check for zero values
            zero_values = (self.data_df['Traffic_Count'] == 0).sum()
            
            # Combine all quality checks
            quality_assessment = {
                'missing_values': missing_values,
                'missing_percentage': missing_percentage,
                'outliers': {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.data_df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                },
                'duplicates': {
                    'count': duplicates,
                    'percentage': duplicates / len(self.data_df) * 100
                },
                'zero_values': {
                    'count': zero_values,
                    'percentage': zero_values / len(self.data_df) * 100
                }
            }
            
            # Log quality assessment summary
            logger.info(f"Data quality assessment:")
            logger.info(f"- Missing values: {sum(missing_values.values())} ({sum(missing_values.values()) / (len(self.data_df) * len(self.data_df.columns)) * 100:.2f}%)")
            logger.info(f"- Outliers: {quality_assessment['outliers']['count']} ({quality_assessment['outliers']['percentage']:.2f}%)")
            logger.info(f"- Duplicates: {quality_assessment['duplicates']['count']} ({quality_assessment['duplicates']['percentage']:.2f}%)")
            logger.info(f"- Zero values: {quality_assessment['zero_values']['count']} ({quality_assessment['zero_values']['percentage']:.2f}%)")
            
            return quality_assessment
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return None
    
    def generate_summary_report(self, output_path=None):
        """
        Generate a comprehensive summary report of the SCATS traffic data.
        
        Args:
            output_path (str, optional): Path to save the report. If None, uses default location.
                
        Returns:
            dict: Dictionary with all analysis results.
        """
        if self.data_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        try:
            # Run all analyses
            unique_sites = self.identify_unique_sites()
            date_validation = self.validate_date_range()
            completeness = self.check_data_completeness()
            statistics = self.display_basic_statistics()
            quality = self.assess_data_quality()
            
            # Combine all results
            report = {
                'dataset_info': {
                    'file_path': self.data_path,
                    'record_count': len(self.data_df),
                    'column_count': len(self.data_df.columns),
                    'columns': list(self.data_df.columns)
                },
                'unique_sites': unique_sites.to_dict('records') if unique_sites is not None else None,
                'date_validation': date_validation,
                'completeness': completeness,
                'statistics': statistics,
                'quality': quality
            }
            
            # Save report if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save as JSON
                import json
                with open(output_path, 'w') as f:
                    # Convert datetime objects to strings
                    json_report = {k: v for k, v in report.items()}
                    if 'date_validation' in json_report and json_report['date_validation']:
                        for key in ['expected_start', 'expected_end', 'actual_start', 'actual_end']:
                            if key in json_report['date_validation']:
                                json_report['date_validation'][key] = str(json_report['date_validation'][key])
                    
                    json.dump(json_report, f, indent=4)
                logger.info(f"Summary report saved to {output_path}")
            
            return report
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return None


def analyze_scats_data(data_path=None, output_path=None):
    """
    Convenience function to analyze SCATS traffic data.
    
    Args:
        data_path (str, optional): Path to the SCATS traffic data file.
            If None, uses the default processed data path from config.
        output_path (str, optional): Path to save the report.
            If None, uses a default location in the processed directory.
            
    Returns:
        dict: Dictionary with all analysis results.
    """
    # Create analyzer
    analyzer = SCATSDataAnalyzer(data_path)
    
    # Set default output path if not provided
    if output_path is None and analyzer.data_df is not None:
        output_dir = os.path.dirname(config.processed_data['scats_complete'])
        output_path = os.path.join(output_dir, 'scats_data_analysis_report.json')
    
    # Generate report
    report = analyzer.generate_summary_report(output_path)
    
    return report


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze SCATS traffic data structure and quality')
    parser.add_argument('--data-path', type=str, help='Path to SCATS traffic data file')
    parser.add_argument('--output-path', type=str, help='Path to save the analysis report')
    args = parser.parse_args()
    
    # Run analysis
    logger.info("=== SCATS TRAFFIC DATA ANALYSIS ===")
    report = analyze_scats_data(args.data_path, args.output_path)
    
    if report:
        logger.info("Data analysis completed successfully")
    else:
        logger.error("Data analysis failed")
