#!/usr/bin/env python3
"""
SCATS Traffic Data Preparation Module

This module extends the data loader to prepare the SCATS traffic dataset for machine learning.
It handles feature engineering, train/validation/test splits, and other ML-specific data preparation tasks.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import the logger and config
from app.core.logging import logger
from app.config.config import config
from app.core.utils.data_loader import SCATSDataLoader, load_scats_data

class SCATSDataPreparation:
    """
    Class for preparing SCATS traffic data for machine learning.
    
    This class extends the SCATSDataLoader functionality to create features
    and prepare datasets specifically for machine learning training.
    """
    
    def __init__(self, data_df=None, site_reference=None):
        """
        Initialize the data preparation with optional dataframes.
        
        Args:
            data_df (pd.DataFrame, optional): Processed SCATS data in long format.
            site_reference (pd.DataFrame, optional): Site reference data.
        """
        self.data_df = data_df
        self.site_reference = site_reference
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.prepared = False
        
        # Get time split configuration
        self.train_end = pd.to_datetime(config.scats_data['time_splits']['train_end'])
        self.val_end = pd.to_datetime(config.scats_data['time_splits']['val_end'])
        
        # Get peak hours configuration
        self.am_peak_start = config.scats_data['peak_hours']['am_start']
        self.am_peak_end = config.scats_data['peak_hours']['am_end']
        self.pm_peak_start = config.scats_data['peak_hours']['pm_start']
        self.pm_peak_end = config.scats_data['peak_hours']['pm_end']
    
    def load_data(self, excel_file=None, reload=False):
        """
        Load the SCATS data using the SCATSDataLoader.
        
        Args:
            excel_file (str, optional): Path to the SCATS Excel file.
            reload (bool, optional): Whether to reload data even if already loaded.
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if self.data_df is not None and self.site_reference is not None and not reload:
            logger.info("Data already loaded, skipping load_data step")
            return True
        
        try:
            logger.info("Loading SCATS data...")
            self.data_df, self.site_reference = load_scats_data(excel_file)
            
            if self.data_df is None or self.site_reference is None:
                logger.error("Failed to load SCATS data")
                return False
            
            logger.info(f"Successfully loaded {len(self.data_df)} records from SCATS data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SCATS data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_from_csv(self, csv_path=None, site_ref_path=None):
        """
        Load the processed SCATS data from CSV files.
        
        Args:
            csv_path (str, optional): Path to the processed SCATS data CSV file.
            site_ref_path (str, optional): Path to the site reference CSV file.
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Use default paths if not provided
            if csv_path is None:
                csv_path = config.processed_data['scats_complete']
            
            if site_ref_path is None:
                site_ref_path = config.processed_data['site_reference']
            
            logger.info(f"Loading processed data from {csv_path}")
            
            # Check if files exist
            if not os.path.exists(csv_path):
                logger.error(f"Processed data file not found: {csv_path}")
                return False
            
            if not os.path.exists(site_ref_path):
                logger.error(f"Site reference file not found: {site_ref_path}")
                return False
            
            # Load the data
            self.data_df = pd.read_csv(csv_path)
            self.site_reference = pd.read_csv(site_ref_path)
            
            # Convert DateTime column to datetime
            if 'DateTime' in self.data_df.columns:
                self.data_df['DateTime'] = pd.to_datetime(self.data_df['DateTime'])
            
            # Convert Date column to datetime.date
            if 'Date' in self.data_df.columns:
                self.data_df['Date'] = pd.to_datetime(self.data_df['Date']).dt.date
            
            logger.info(f"Successfully loaded {len(self.data_df)} records from CSV")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def add_feature_engineering(self):
        """
        Add feature engineering to the dataset.
        
        This includes:
        - Peak hour indicators
        - Lag features
        - Rolling statistics
        - Site-specific features
        - Prediction targets
        
        Returns:
            bool: True if feature engineering was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_data() or load_from_csv() first.")
            return False
        
        try:
            logger.info("Adding feature engineering...")
            
            # Make a copy to avoid modifying the original
            df = self.data_df.copy()
            
            # Sort by SCATS_ID and DateTime for proper lag calculation
            df = df.sort_values(['SCATS_ID', 'DateTime']).reset_index(drop=True)
            
            # Add peak hour indicator
            df['IsPeakHour'] = (
                ((df['Hour'] >= self.am_peak_start) & (df['Hour'] <= self.am_peak_end)) | 
                ((df['Hour'] >= self.pm_peak_start) & (df['Hour'] <= self.pm_peak_end))
            )
            logger.info("Added peak hour indicator")
            
            # Add lag features (within each site)
            df['Traffic_Lag1'] = df.groupby('SCATS_ID')['Traffic_Count'].shift(1)  # t-1 (15 min ago)
            df['Traffic_Lag4'] = df.groupby('SCATS_ID')['Traffic_Count'].shift(4)  # t-4 (1 hour ago)
            df['Traffic_Lag96'] = df.groupby('SCATS_ID')['Traffic_Count'].shift(96)  # t-96 (same time yesterday)
            logger.info("Added lag features")
            
            # Add rolling statistics (within each site)
            df['Traffic_Roll1H'] = df.groupby('SCATS_ID')['Traffic_Count'].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            )  # 1-hour rolling mean
            
            df['Traffic_Roll3H'] = df.groupby('SCATS_ID')['Traffic_Count'].transform(
                lambda x: x.rolling(12, min_periods=1).mean()
            )  # 3-hour rolling mean
            logger.info("Added rolling statistics")
            
            # Add site-specific features
            site_daily_avg = df.groupby('SCATS_ID')['Traffic_Count'].mean().reset_index()
            site_daily_avg.columns = ['SCATS_ID', 'Site_DailyAvg']
            df = pd.merge(df, site_daily_avg, on='SCATS_ID', how='left')
            
            # Add normalized traffic
            df['Traffic_Normalized'] = df['Traffic_Count'] / df['Site_DailyAvg']
            logger.info("Added site-specific features")
            
            # Add prediction targets
            df['Traffic_Next1'] = df.groupby('SCATS_ID')['Traffic_Count'].shift(-1)  # t+1 (next 15 min)
            df['Traffic_Next4'] = df.groupby('SCATS_ID')['Traffic_Count'].shift(-4)  # t+4 (next 1 hour)
            logger.info("Added prediction targets")
            
            # Update the dataframe
            self.data_df = df
            logger.info("Feature engineering completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feature engineering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def create_train_val_test_split(self):
        """
        Create train/validation/test splits based on time.
        
        Returns:
            bool: True if splitting was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_data() or load_from_csv() first.")
            return False
        
        try:
            logger.info("Creating train/validation/test splits...")
            
            # Make sure DateTime is datetime type
            if not pd.api.types.is_datetime64_any_dtype(self.data_df['DateTime']):
                self.data_df['DateTime'] = pd.to_datetime(self.data_df['DateTime'])
            
            # Create splits based on date
            self.train_df = self.data_df[self.data_df['DateTime'] <= self.train_end]
            self.val_df = self.data_df[(self.data_df['DateTime'] > self.train_end) & 
                                      (self.data_df['DateTime'] <= self.val_end)]
            self.test_df = self.data_df[self.data_df['DateTime'] > self.val_end]
            
            # Log split information
            logger.info(f"Train: {len(self.train_df)} records ({self.train_df['DateTime'].min()} to {self.train_df['DateTime'].max()})")
            logger.info(f"Validation: {len(self.val_df)} records ({self.val_df['DateTime'].min()} to {self.val_df['DateTime'].max()})")
            logger.info(f"Test: {len(self.test_df)} records ({self.test_df['DateTime'].min()} to {self.test_df['DateTime'].max()})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating train/validation/test splits: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def save_prepared_data(self, output_dir=None):
        """
        Save the prepared data to CSV files.
        
        Args:
            output_dir (str, optional): Directory to save the output files. If None, uses the path from config.
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.error("Data not prepared. Call create_train_val_test_split() first.")
            return False
        
        try:
            # Determine output directory
            if output_dir is None:
                # Get the directory part of the processed data path
                train_path = config.processed_data['train_data']
                output_dir = os.path.dirname(train_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save datasets
            train_path = os.path.join(output_dir, 'train_data.csv')
            val_path = os.path.join(output_dir, 'val_data.csv')
            test_path = os.path.join(output_dir, 'test_data.csv')
            
            self.train_df.to_csv(train_path, index=False)
            self.val_df.to_csv(val_path, index=False)
            self.test_df.to_csv(test_path, index=False)
            
            logger.info("=== DATASET PREPARATION FOR TRAINING COMPLETE ===")
            logger.info("Files created:")
            logger.info(f"- train_data.csv ({len(self.train_df)} records)")
            logger.info(f"- val_data.csv ({len(self.val_df)} records)")
            logger.info(f"- test_data.csv ({len(self.test_df)} records)")
            
            self.prepared = True
            return True
            
        except Exception as e:
            logger.error(f"Error saving prepared data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def prepare_data(self, excel_file=None, from_csv=True):
        """
        Prepare the data from loading to saving in one method.
        
        Args:
            excel_file (str, optional): Path to the SCATS Excel file.
            from_csv (bool, optional): Whether to load from processed CSV files.
            
        Returns:
            bool: True if the entire process was successful, False otherwise
        """
        try:
            # Step 1: Load data
            if from_csv:
                if not self.load_from_csv():
                    return False
            else:
                if not self.load_data(excel_file):
                    return False
            
            # Step 2: Add feature engineering
            if not self.add_feature_engineering():
                return False
            
            # Step 3: Create train/validation/test splits
            if not self.create_train_val_test_split():
                return False
            
            # Step 4: Save prepared data
            if not self.save_prepared_data():
                return False
            
            logger.info("Data preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_prepared_data(self):
        """
        Get the prepared data.
        
        Returns:
            tuple: (train_df, val_df, test_df, site_reference) if data is prepared,
                  (None, None, None, None) otherwise
        """
        if not self.prepared:
            logger.warning("Data not prepared. Call prepare_data() first.")
            return None, None, None, None
        
        return self.train_df, self.val_df, self.test_df, self.site_reference


def prepare_scats_data(excel_file=None, from_csv=True):
    """
    Prepare SCATS traffic data for machine learning.
    
    Args:
        excel_file (str, optional): Path to the SCATS Excel file.
        from_csv (bool, optional): Whether to load from processed CSV files.
        
    Returns:
        tuple: (train_df, val_df, test_df, site_reference) if successful,
              (None, None, None, None) otherwise
    """
    data_prep = SCATSDataPreparation()
    success = data_prep.prepare_data(excel_file, from_csv)
    
    if success:
        return data_prep.get_prepared_data()
    else:
        return None, None, None, None


# Example usage
if __name__ == "__main__":
    # Prepare data from processed CSV files
    train_df, val_df, test_df, site_reference = prepare_scats_data(from_csv=True)
    
    if train_df is not None:
        print(f"Train data: {len(train_df)} records")
        print(f"Validation data: {len(val_df)} records")
        print(f"Test data: {len(test_df)} records")
        print(f"Site reference: {len(site_reference)} sites")
        
        # Print feature list
        print("\nFeatures available:")
        for col in train_df.columns:
            print(f"- {col}")
    else:
        print("Failed to prepare data")
