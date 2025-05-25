#!/usr/bin/env python3
"""
SCATS Traffic Data Splitter Module

This module provides functionality to split SCATS traffic data into train, validation, and test sets.
It implements chronological splitting for time series data, ensures no data leakage between splits,
and provides site-wise validation to ensure all SCATS sites are represented in each split.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
import logging

# Import the logger and config
from app.core.logging import logger
from app.config.config import config

class SCATSDataSplitter:
    """
    Class for splitting SCATS traffic data into train, validation, and test sets.
    
    This class provides methods to:
    - Implement chronological train/validation/test split
    - Ensure no data leakage between splits
    - Create site-wise validation
    - Save split datasets
    - Validate split proportions and date ranges
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the data splitter with an optional DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame containing SCATS traffic data.
        """
        self.df = df
        
        # Default split proportions
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Initialize split dataframes
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Tracking split information
        self.split_info = {
            'train': {'start_date': None, 'end_date': None, 'records': 0, 'sites': 0},
            'val': {'start_date': None, 'end_date': None, 'records': 0, 'sites': 0},
            'test': {'start_date': None, 'end_date': None, 'records': 0, 'sites': 0}
        }
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame to use for splitting.
        
        Args:
            df (pd.DataFrame): DataFrame containing SCATS traffic data.
        """
        self.df = df
    
    def set_split_ratios(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> bool:
        """
        Set the split ratios for train, validation, and test sets.
        
        Args:
            train_ratio (float, optional): Proportion of data for training. Default is 0.7.
            val_ratio (float, optional): Proportion of data for validation. Default is 0.15.
            test_ratio (float, optional): Proportion of data for testing. Default is 0.15.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Validate ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-10):
            logger.error(f"Split ratios must sum to 1.0, got {total_ratio}")
            return False
        
        # Set ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        logger.info(f"Split ratios set: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
        return True
    
    def chronological_split(self) -> bool:
        """
        Split data chronologically into train, validation, and test sets.
        
        This method respects the time series nature of the data by splitting
        it chronologically, ensuring that training data comes before validation,
        which comes before test data.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df is None:
            logger.error("DataFrame not set. Call set_dataframe() first.")
            return False
        
        # Ensure DataFrame is sorted by DateTime
        logger.info("Sorting data chronologically...")
        if 'DateTime' not in self.df.columns:
            logger.error("DateTime column not found in DataFrame")
            return False
        
        # Ensure DateTime is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df['DateTime']):
            logger.info("Converting DateTime column to datetime type")
            self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        
        # Sort by DateTime
        self.df = self.df.sort_values('DateTime')
        
        # Get unique dates
        unique_dates = self.df['DateTime'].dt.date.unique()
        total_days = len(unique_dates)
        
        logger.info(f"Data spans {total_days} days from {unique_dates[0]} to {unique_dates[-1]}")
        
        # Calculate split indices based on dates
        train_end_idx = int(total_days * self.train_ratio)
        val_end_idx = train_end_idx + int(total_days * self.val_ratio)
        
        # Get split dates
        train_end_date = unique_dates[train_end_idx]
        val_end_date = unique_dates[val_end_idx]
        
        # Convert to datetime for comparison
        train_end_datetime = pd.Timestamp(train_end_date).replace(hour=23, minute=59, second=59)
        val_end_datetime = pd.Timestamp(val_end_date).replace(hour=23, minute=59, second=59)
        
        # Split data
        logger.info(f"Splitting data chronologically...")
        logger.info(f"Train data: up to {train_end_date}")
        logger.info(f"Validation data: {train_end_date} to {val_end_date}")
        logger.info(f"Test data: from {val_end_date} onwards")
        
        self.train_df = self.df[self.df['DateTime'] <= train_end_datetime].copy()
        self.val_df = self.df[(self.df['DateTime'] > train_end_datetime) & 
                              (self.df['DateTime'] <= val_end_datetime)].copy()
        self.val_df.reset_index(drop=True, inplace=True)
        self.test_df = self.df[self.df['DateTime'] > val_end_datetime].copy()
        self.test_df.reset_index(drop=True, inplace=True)
        
        # Update split info
        self._update_split_info()
        
        # Log split results
        logger.info(f"Train set: {len(self.train_df)} records, " +
                   f"{self.train_df['SCATS_ID'].nunique()} sites, " +
                   f"{self.train_df['DateTime'].min()} to {self.train_df['DateTime'].max()}")
        logger.info(f"Validation set: {len(self.val_df)} records, " +
                   f"{self.val_df['SCATS_ID'].nunique()} sites, " +
                   f"{self.val_df['DateTime'].min()} to {self.val_df['DateTime'].max()}")
        logger.info(f"Test set: {len(self.test_df)} records, " +
                   f"{self.test_df['SCATS_ID'].nunique()} sites, " +
                   f"{self.test_df['DateTime'].min()} to {self.test_df['DateTime'].max()}")
        
        return True
    
    def site_wise_validation(self) -> bool:
        """
        Ensure all SCATS sites are represented in each split.
        
        This method checks if all sites are present in each split and logs warnings
        if any sites are missing from a split.
        
        Returns:
            bool: True if all sites are in all splits, False otherwise.
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.error("Data not split yet. Call chronological_split() first.")
            return False
        
        # Get unique sites in each split
        train_sites = set(self.train_df['SCATS_ID'].unique())
        val_sites = set(self.val_df['SCATS_ID'].unique())
        test_sites = set(self.test_df['SCATS_ID'].unique())
        
        # Get all unique sites
        all_sites = train_sites.union(val_sites).union(test_sites)
        
        # Check if all sites are in each split
        missing_in_train = all_sites - train_sites
        missing_in_val = all_sites - val_sites
        missing_in_test = all_sites - test_sites
        
        # Log results
        logger.info(f"Site-wise validation:")
        logger.info(f"Total unique sites: {len(all_sites)}")
        logger.info(f"Sites in train set: {len(train_sites)} ({len(train_sites)/len(all_sites)*100:.1f}%)")
        logger.info(f"Sites in validation set: {len(val_sites)} ({len(val_sites)/len(all_sites)*100:.1f}%)")
        logger.info(f"Sites in test set: {len(test_sites)} ({len(test_sites)/len(all_sites)*100:.1f}%)")
        
        if missing_in_train:
            logger.warning(f"{len(missing_in_train)} sites missing from train set: {sorted(list(missing_in_train))[:5]}...")
        if missing_in_val:
            logger.warning(f"{len(missing_in_val)} sites missing from validation set: {sorted(list(missing_in_val))[:5]}...")
        if missing_in_test:
            logger.warning(f"{len(missing_in_test)} sites missing from test set: {sorted(list(missing_in_test))[:5]}...")
        
        # Return True if all sites are in all splits
        return not (missing_in_train or missing_in_val or missing_in_test)
    
    def validate_split_proportions(self) -> Dict[str, float]:
        """
        Validate the actual split proportions against the target ratios.
        
        Returns:
            Dict[str, float]: Dictionary with actual split proportions.
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.error("Data not split yet. Call chronological_split() first.")
            return {}
        
        # Calculate actual proportions
        total_records = len(self.train_df) + len(self.val_df) + len(self.test_df)
        actual_train_ratio = len(self.train_df) / total_records
        actual_val_ratio = len(self.val_df) / total_records
        actual_test_ratio = len(self.test_df) / total_records
        
        # Calculate differences from target
        train_diff = actual_train_ratio - self.train_ratio
        val_diff = actual_val_ratio - self.val_ratio
        test_diff = actual_test_ratio - self.test_ratio
        
        # Log results
        logger.info(f"Split proportion validation:")
        logger.info(f"Target ratios: train={self.train_ratio:.2f}, val={self.val_ratio:.2f}, test={self.test_ratio:.2f}")
        logger.info(f"Actual ratios: train={actual_train_ratio:.2f}, val={actual_val_ratio:.2f}, test={actual_test_ratio:.2f}")
        logger.info(f"Differences: train={train_diff:.2f}, val={val_diff:.2f}, test={test_diff:.2f}")
        
        # Check if differences are acceptable (within 5%)
        if abs(train_diff) > 0.05 or abs(val_diff) > 0.05 or abs(test_diff) > 0.05:
            logger.warning("Split proportions differ from target by more than 5%")
        
        return {
            'train': actual_train_ratio,
            'val': actual_val_ratio,
            'test': actual_test_ratio
        }
    
    def validate_date_ranges(self) -> Dict[str, Dict[str, datetime]]:
        """
        Validate the date ranges of each split to ensure chronological integrity.
        
        Returns:
            Dict[str, Dict[str, datetime]]: Dictionary with date ranges for each split.
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.error("Data not split yet. Call chronological_split() first.")
            return {}
        
        # Get date ranges
        train_start = self.train_df['DateTime'].min()
        train_end = self.train_df['DateTime'].max()
        val_start = self.val_df['DateTime'].min()
        val_end = self.val_df['DateTime'].max()
        test_start = self.test_df['DateTime'].min()
        test_end = self.test_df['DateTime'].max()
        
        # Log results
        logger.info(f"Date range validation:")
        logger.info(f"Train: {train_start} to {train_end}")
        logger.info(f"Validation: {val_start} to {val_end}")
        logger.info(f"Test: {test_start} to {test_end}")
        
        # Check chronological order
        if train_end >= val_start:
            logger.error(f"Data leakage: Train end ({train_end}) >= Validation start ({val_start})")
        if val_end >= test_start:
            logger.error(f"Data leakage: Validation end ({val_end}) >= Test start ({test_start})")
        
        # Check for gaps
        train_val_gap = (val_start - train_end).total_seconds() / 3600  # Gap in hours
        val_test_gap = (test_start - val_end).total_seconds() / 3600  # Gap in hours
        
        if train_val_gap > 24:
            logger.warning(f"Gap between train and validation: {train_val_gap:.1f} hours")
        if val_test_gap > 24:
            logger.warning(f"Gap between validation and test: {val_test_gap:.1f} hours")
        
        return {
            'train': {'start': train_start, 'end': train_end},
            'val': {'start': val_start, 'end': val_end},
            'test': {'start': test_start, 'end': test_end}
        }
    
    def save_split_datasets(self, output_dir: Optional[str] = None) -> bool:
        """
        Save the split datasets to CSV files.
        
        Args:
            output_dir (str, optional): Directory to save the split datasets.
                If None, uses the processed_data directory from config.
                
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.error("Data not split yet. Call chronological_split() first.")
            return False
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(config.processed_data['train_data'])
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths
        train_path = os.path.join(output_dir, 'train_data.csv')
        val_path = os.path.join(output_dir, 'val_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        
        # Save datasets
        logger.info(f"Saving split datasets to {output_dir}...")
        self.train_df.to_csv(train_path, index=False)
        logger.info(f"Saved train dataset to {train_path} ({len(self.train_df)} records)")
        
        self.val_df.to_csv(val_path, index=False)
        logger.info(f"Saved validation dataset to {val_path} ({len(self.val_df)} records)")
        
        self.test_df.to_csv(test_path, index=False)
        logger.info(f"Saved test dataset to {test_path} ({len(self.test_df)} records)")
        
        return True
    
    def get_split_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the split datasets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
        """
        return self.train_df, self.val_df, self.test_df
    
    def get_split_info(self) -> Dict[str, Dict[str, Union[datetime, int]]]:
        """
        Get information about the splits.
        
        Returns:
            Dict[str, Dict[str, Union[datetime, int]]]: Dictionary with split information.
        """
        return self.split_info
    
    def _update_split_info(self) -> None:
        """
        Update the split information dictionary.
        """
        if self.train_df is not None:
            self.split_info['train']['start_date'] = self.train_df['DateTime'].min()
            self.split_info['train']['end_date'] = self.train_df['DateTime'].max()
            self.split_info['train']['records'] = len(self.train_df)
            self.split_info['train']['sites'] = self.train_df['SCATS_ID'].nunique()
        
        if self.val_df is not None:
            self.split_info['val']['start_date'] = self.val_df['DateTime'].min()
            self.split_info['val']['end_date'] = self.val_df['DateTime'].max()
            self.split_info['val']['records'] = len(self.val_df)
            self.split_info['val']['sites'] = self.val_df['SCATS_ID'].nunique()
        
        if self.test_df is not None:
            self.split_info['test']['start_date'] = self.test_df['DateTime'].min()
            self.split_info['test']['end_date'] = self.test_df['DateTime'].max()
            self.split_info['test']['records'] = len(self.test_df)
            self.split_info['test']['sites'] = self.test_df['SCATS_ID'].nunique()


def split_scats_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                    test_ratio: float = 0.15, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience function to split SCATS traffic data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing SCATS traffic data.
        train_ratio (float, optional): Proportion of data for training. Default is 0.7.
        val_ratio (float, optional): Proportion of data for validation. Default is 0.15.
        test_ratio (float, optional): Proportion of data for testing. Default is 0.15.
        output_dir (str, optional): Directory to save the split datasets.
            
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]: Train, validation, test datasets, and split info.
    """
    # Create data splitter
    splitter = SCATSDataSplitter(df)
    
    # Set split ratios
    splitter.set_split_ratios(train_ratio, val_ratio, test_ratio)
    
    # Split data chronologically
    success = splitter.chronological_split()
    if not success:
        logger.error("Failed to split data chronologically")
        return None, None, None, {}
    
    # Validate site representation
    splitter.site_wise_validation()
    
    # Validate split proportions
    splitter.validate_split_proportions()
    
    # Validate date ranges
    splitter.validate_date_ranges()
    
    # Save split datasets if output_dir is provided
    if output_dir is not None:
        splitter.save_split_datasets(output_dir)
    
    # Get split datasets and info
    train_df, val_df, test_df = splitter.get_split_datasets()
    split_info = splitter.get_split_info()
    
    return train_df, val_df, test_df, split_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SCATS Traffic Data Splitter')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save split datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Proportion of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Convert DateTime to datetime if it exists
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Split data
    logger.info(f"Splitting data with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}...")
    
    train_df, val_df, test_df, split_info = split_scats_data(
        df, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir
    )
    
    logger.info("Data splitting completed successfully")
