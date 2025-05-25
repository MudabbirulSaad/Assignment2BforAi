#!/usr/bin/env python3
"""
SCATS Traffic Feature Engineering Module

This module provides functionality to create advanced features for SCATS traffic data.
It includes lag features, rolling statistics, time-based features, peak hour indicators,
site-specific normalization, and prediction targets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Tuple, Optional

# Import the logger and config
from app.core.logging import logger
from app.config.config import config

class SCATSFeatureEngineer:
    """
    Class for creating advanced features for SCATS traffic data.
    
    This class provides methods to:
    - Create lag features (t-1, t-4, t-96)
    - Add rolling statistics (1-hour, 3-hour averages)
    - Create time-based features (hour, day, weekend flags)
    - Add peak hour indicators
    - Implement site-specific normalization
    - Create prediction targets (t+1, t+4)
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the feature engineer with an optional DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame containing SCATS traffic data.
        """
        self.df = df
        
        # Get peak hours from config
        self.am_peak_start = config.scats_data['peak_hours']['am_start']
        self.am_peak_end = config.scats_data['peak_hours']['am_end']
        self.pm_peak_start = config.scats_data['peak_hours']['pm_start']
        self.pm_peak_end = config.scats_data['peak_hours']['pm_end']
        
        # Define lag periods (in 15-minute intervals)
        self.lag_periods = {
            't-1': 1,      # Previous 15-minute interval
            't-4': 4,      # 1 hour ago
            't-96': 96     # 24 hours ago (same time yesterday)
        }
        
        # Define rolling window sizes (in 15-minute intervals)
        self.rolling_windows = {
            '1-hour': 4,   # 1 hour (4 intervals)
            '3-hour': 12   # 3 hours (12 intervals)
        }
        
        # Define prediction targets (in 15-minute intervals)
        self.prediction_targets = {
            't+1': 1,      # Next 15-minute interval
            't+4': 4       # 1 hour ahead
        }
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame to use for feature engineering.
        
        Args:
            df (pd.DataFrame): DataFrame containing SCATS traffic data.
        """
        self.df = df
    
    def add_lag_features(self) -> bool:
        """
        Add lag features to the DataFrame.
        
        Creates features for previous time intervals (t-1, t-4, t-96).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding lag features...")
        
        # Ensure DataFrame is sorted by SCATS_ID and DateTime
        self.df = self.df.sort_values(['SCATS_ID', 'DateTime'])
        
        # Create lag features for each site separately
        for lag_name, lag_period in self.lag_periods.items():
            logger.info(f"Creating {lag_name} lag feature ({lag_period} intervals)")
            
            # Create lag feature name
            lag_col_name = f'Traffic_Count_{lag_name}'
            
            # Group by SCATS_ID and shift to create lag
            self.df[lag_col_name] = self.df.groupby('SCATS_ID')['Traffic_Count'].shift(lag_period)
            
            # Log the number of NaN values created
            nan_count = self.df[lag_col_name].isna().sum()
            nan_percentage = (nan_count / len(self.df)) * 100
            logger.info(f"Created {lag_col_name} with {nan_count} NaN values ({nan_percentage:.2f}%)")
        
        logger.info("Lag features added successfully")
        return True
    
    def add_rolling_statistics(self) -> bool:
        """
        Add rolling statistics to the DataFrame.
        
        Creates rolling mean, min, max, and standard deviation for different window sizes.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding rolling statistics...")
        
        # Ensure DataFrame is sorted by SCATS_ID and DateTime
        self.df = self.df.sort_values(['SCATS_ID', 'DateTime'])
        
        # Create rolling statistics for each site separately
        for window_name, window_size in self.rolling_windows.items():
            logger.info(f"Creating {window_name} rolling statistics ({window_size} intervals)")
            
            # Group by SCATS_ID and calculate rolling statistics
            grouped = self.df.groupby('SCATS_ID')['Traffic_Count']
            
            # Rolling mean
            mean_col_name = f'Rolling_Mean_{window_name}'
            self.df[mean_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
            
            # Rolling min
            min_col_name = f'Rolling_Min_{window_name}'
            self.df[min_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).min())
            
            # Rolling max
            max_col_name = f'Rolling_Max_{window_name}'
            self.df[max_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).max())
            
            # Rolling standard deviation
            std_col_name = f'Rolling_Std_{window_name}'
            self.df[std_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
            
            logger.info(f"Created rolling statistics for {window_name}")
        
        logger.info("Rolling statistics added successfully")
        return True
    
    def add_time_based_features(self) -> bool:
        """
        Add time-based features to the DataFrame.
        
        Creates features for hour, day of week, day of month, month, year, weekend flag.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding time-based features...")
        
        # Check if DateTime column exists and is datetime type
        if 'DateTime' not in self.df.columns:
            logger.error("DateTime column not found in DataFrame")
            return False
        
        # Ensure DateTime is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df['DateTime']):
            logger.info("Converting DateTime column to datetime type")
            self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        
        # Extract time components
        logger.info("Extracting time components...")
        
        # Hour of day (0-23)
        if 'Hour' not in self.df.columns:
            self.df['Hour'] = self.df['DateTime'].dt.hour
        
        # Minute of hour (0-59)
        if 'Minute' not in self.df.columns:
            self.df['Minute'] = self.df['DateTime'].dt.minute
        
        # Day of week (0=Monday, 6=Sunday)
        self.df['DayOfWeek'] = self.df['DateTime'].dt.weekday
        
        # Day name (Monday, Tuesday, etc.)
        self.df['DayName'] = self.df['DateTime'].dt.day_name()
        
        # Day of month (1-31)
        self.df['DayOfMonth'] = self.df['DateTime'].dt.day
        
        # Month (1-12)
        self.df['Month'] = self.df['DateTime'].dt.month
        
        # Year
        self.df['Year'] = self.df['DateTime'].dt.year
        
        # Weekend flag (True if Saturday or Sunday)
        self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: x >= 5)
        
        # Time of day category (Morning, Afternoon, Evening, Night)
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        
        self.df['TimeOfDay'] = self.df['Hour'].apply(get_time_of_day)
        
        logger.info("Time-based features added successfully")
        return True
    
    def add_peak_hour_indicators(self) -> bool:
        """
        Add peak hour indicators to the DataFrame.
        
        Creates features for AM peak, PM peak, and general peak hours.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding peak hour indicators...")
        
        # Check if Hour column exists
        if 'Hour' not in self.df.columns:
            logger.error("Hour column not found in DataFrame")
            return False
        
        # AM peak (typically 7-9 AM)
        self.df['IsAMPeak'] = (self.df['Hour'] >= self.am_peak_start) & (self.df['Hour'] <= self.am_peak_end)
        
        # PM peak (typically 4-6 PM)
        self.df['IsPMPeak'] = (self.df['Hour'] >= self.pm_peak_start) & (self.df['Hour'] <= self.pm_peak_end)
        
        # General peak (either AM or PM peak)
        self.df['IsPeakHour'] = self.df['IsAMPeak'] | self.df['IsPMPeak']
        
        logger.info("Peak hour indicators added successfully")
        return True
    
    def add_site_specific_normalization(self) -> bool:
        """
        Add site-specific normalized features to the DataFrame.
        
        Creates normalized traffic counts relative to each site's statistics.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding site-specific normalization...")
        
        # Calculate site-specific statistics
        site_stats = self.df.groupby('SCATS_ID')['Traffic_Count'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Merge statistics back to the main DataFrame
        self.df = self.df.merge(site_stats, on='SCATS_ID', how='left')
        
        # Create normalized features
        
        # Z-score normalization: (x - mean) / std
        self.df['Traffic_Count_ZScore'] = (self.df['Traffic_Count'] - self.df['mean']) / self.df['std'].replace(0, 1)
        
        # Min-max normalization: (x - min) / (max - min)
        min_max_diff = (self.df['max'] - self.df['min']).replace(0, 1)  # Avoid division by zero
        self.df['Traffic_Count_MinMax'] = (self.df['Traffic_Count'] - self.df['min']) / min_max_diff
        
        # Percentage of max: x / max
        self.df['Traffic_Count_PctOfMax'] = self.df['Traffic_Count'] / self.df['max'].replace(0, 1)
        
        # Remove the temporary statistics columns
        self.df = self.df.drop(columns=['mean', 'std', 'min', 'max'])
        
        logger.info("Site-specific normalization added successfully")
        return True
    
    def add_prediction_targets(self) -> bool:
        """
        Add prediction target features to the DataFrame.
        
        Creates target features for future time intervals (t+1, t+4).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding prediction targets...")
        
        # Ensure DataFrame is sorted by SCATS_ID and DateTime
        self.df = self.df.sort_values(['SCATS_ID', 'DateTime'])
        
        # Create prediction targets for each site separately
        for target_name, target_period in self.prediction_targets.items():
            logger.info(f"Creating {target_name} prediction target ({target_period} intervals ahead)")
            
            # Create target feature name
            target_col_name = f'Target_{target_name}'
            
            # Group by SCATS_ID and shift to create future target
            # Note: We use negative shift to get future values
            self.df[target_col_name] = self.df.groupby('SCATS_ID')['Traffic_Count'].shift(-target_period)
            
            # Log the number of NaN values created
            nan_count = self.df[target_col_name].isna().sum()
            nan_percentage = (nan_count / len(self.df)) * 100
            logger.info(f"Created {target_col_name} with {nan_count} NaN values ({nan_percentage:.2f}%)")
        
        logger.info("Prediction targets added successfully")
        return True
    
    def add_all_features(self) -> bool:
        """
        Add all features to the DataFrame.
        
        This is a convenience method that calls all feature engineering methods.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        logger.info("Adding all features...")
        
        # Add lag features
        success = self.add_lag_features()
        if not success:
            return False
        
        # Add rolling statistics
        success = self.add_rolling_statistics()
        if not success:
            return False
        
        # Add time-based features
        success = self.add_time_based_features()
        if not success:
            return False
        
        # Add peak hour indicators
        success = self.add_peak_hour_indicators()
        if not success:
            return False
        
        # Add site-specific normalization
        success = self.add_site_specific_normalization()
        if not success:
            return False
        
        # Add prediction targets
        success = self.add_prediction_targets()
        if not success:
            return False
        
        logger.info("All features added successfully")
        return True
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Get the DataFrame with all added features.
        
        Returns:
            pd.DataFrame: DataFrame with all added features.
        """
        return self.df


def engineer_features(df: pd.DataFrame, add_all: bool = True, 
                     add_lag: bool = False, add_rolling: bool = False,
                     add_time: bool = False, add_peak: bool = False,
                     add_norm: bool = False, add_targets: bool = False) -> pd.DataFrame:
    """
    Convenience function to add features to a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing SCATS traffic data.
        add_all (bool, optional): Add all features. Default is True.
        add_lag (bool, optional): Add lag features. Default is False.
        add_rolling (bool, optional): Add rolling statistics. Default is False.
        add_time (bool, optional): Add time-based features. Default is False.
        add_peak (bool, optional): Add peak hour indicators. Default is False.
        add_norm (bool, optional): Add site-specific normalization. Default is False.
        add_targets (bool, optional): Add prediction targets. Default is False.
        
    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    # Create feature engineer
    feature_engineer = SCATSFeatureEngineer(df)
    
    # Add features based on parameters
    if add_all:
        feature_engineer.add_all_features()
    else:
        if add_lag:
            feature_engineer.add_lag_features()
        if add_rolling:
            feature_engineer.add_rolling_statistics()
        if add_time:
            feature_engineer.add_time_based_features()
        if add_peak:
            feature_engineer.add_peak_hour_indicators()
        if add_norm:
            feature_engineer.add_site_specific_normalization()
        if add_targets:
            feature_engineer.add_prediction_targets()
    
    # Return DataFrame with added features
    return feature_engineer.get_feature_dataframe()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SCATS Traffic Feature Engineering')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--no-all', action='store_true', help='Do not add all features')
    parser.add_argument('--lag', action='store_true', help='Add lag features')
    parser.add_argument('--rolling', action='store_true', help='Add rolling statistics')
    parser.add_argument('--time', action='store_true', help='Add time-based features')
    parser.add_argument('--peak', action='store_true', help='Add peak hour indicators')
    parser.add_argument('--norm', action='store_true', help='Add site-specific normalization')
    parser.add_argument('--targets', action='store_true', help='Add prediction targets')
    
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Convert DateTime to datetime if it exists
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Add features
    add_all = not args.no_all
    logger.info(f"Adding features (add_all={add_all})...")
    
    df = engineer_features(
        df, 
        add_all=add_all,
        add_lag=args.lag,
        add_rolling=args.rolling,
        add_time=args.time,
        add_peak=args.peak,
        add_norm=args.norm,
        add_targets=args.targets
    )
    
    # Save output data
    logger.info(f"Saving data to {args.output}...")
    df.to_csv(args.output, index=False)
    
    logger.info("Feature engineering completed successfully")
