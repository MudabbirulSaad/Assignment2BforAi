#!/usr/bin/env python3
"""
SCATS Traffic Data Processing Pipeline

This comprehensive script integrates all data processing steps for the TBRGS project:
1. Data Analysis - Analyze SCATS traffic data quality and structure
2. Data Reshaping - Convert from wide to long format
3. Time Conversion - Handle Excel dates and add time features
4. Feature Engineering - Create advanced features for ML models
5. Data Splitting - Split data chronologically for train/validation/test sets

Features:
- Progress tracking with detailed logging
- Data validation checkpoints
- Summary statistics output
- Error handling and recovery
- Comprehensive logging
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# Add the parent directory to sys.path to allow importing app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all necessary modules
from app.core.logging import logger, TBRGSLogger
from app.config.config import config

# Import all dataset processing modules
from app.dataset.data_analyzer import SCATSDataAnalyzer, analyze_scats_data
from app.dataset.data_reshaper import SCATSDataReshaper, reshape_scats_data
from app.dataset.time_converter import SCATSTimeConverter
from app.dataset.feature_engineer import SCATSFeatureEngineer, engineer_features
from app.dataset.data_splitter import SCATSDataSplitter, split_scats_data
from app.dataset.validate_processed_data import DataValidator, validate_processed_data

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

class DataProcessingPipeline:
    """
    Comprehensive data processing pipeline for SCATS traffic data.
    
    This class integrates all data processing steps into a single pipeline:
    1. Data Analysis - Analyze SCATS traffic data quality and structure
    2. Data Reshaping - Convert from wide to long format
    3. Time Conversion - Handle Excel dates and add time features
    4. Feature Engineering - Create advanced features for ML models
    5. Data Splitting - Split data chronologically for train/validation/test sets
    
    Features:
    - Progress tracking with detailed logging
    - Data validation checkpoints
    - Summary statistics output
    - Error handling and recovery
    - Comprehensive logging
    """
    
    def __init__(self, args=None):
        """
        Initialize the data processing pipeline.
        
        Args:
            args: Command-line arguments.
        """
        self.args = args
        
        # Initialize paths
        self.excel_file = args.excel_file if args and args.excel_file else config.raw_data['scats_excel']
        self.output_dir = args.output_dir if args and args.output_dir else os.path.dirname(config.processed_data['scats_complete'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define output paths
        self.site_ref_path = os.path.join(self.output_dir, 'scats_site_reference.csv')
        self.traffic_path = os.path.join(self.output_dir, 'scats_traffic.csv')
        self.train_path = os.path.join(self.output_dir, 'train_data.csv')
        self.val_path = os.path.join(self.output_dir, 'val_data.csv')
        self.test_path = os.path.join(self.output_dir, 'test_data.csv')
        
        # For backward compatibility
        self.reshaped_path = self.traffic_path
        self.enhanced_path = self.traffic_path
        
        # Initialize data containers
        self.raw_data = None
        self.reshaped_data = None
        self.site_reference = None
        self.enhanced_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Initialize processing statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'steps_completed': 0,
            'total_steps': 5,  # Analysis, Reshaping, Feature Engineering, Splitting, Validation
            'records_processed': 0,
            'errors': [],
            'warnings': [],
            'step_durations': {},
        }
        
        # Initialize validation checkpoints
        self.validation_checkpoints = {
            'analysis': False,
            'reshaping': False,
            'feature_engineering': False,
            'data_splitting': False,
            'final_validation': False
        }
        
        # Set up logger
        self.logger = logger
        
    def run_pipeline(self):
        """
        Run the complete data processing pipeline.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        self.stats['start_time'] = time.time()
        
        try:
            self.logger.info("=== STARTING SCATS DATA PROCESSING PIPELINE ===")
            self.logger.info(f"Excel file: {self.excel_file}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            # STEP 1: Data Analysis
            if not self.args or not self.args.skip_analysis:
                self._run_data_analysis()
            else:
                self.logger.info("\n=== STEP 1: SKIPPING DATA ANALYSIS ===")
            
            # STEP 2: Data Reshaping
            if not self.args or not self.args.skip_reshape:
                if not self.args or not self.args.use_existing:
                    self._run_data_reshaping()
                else:
                    self._load_existing_reshaped_data()
            else:
                self.logger.info("\n=== STEP 2: SKIPPING DATA RESHAPING ===")
                if not self.args or not self.args.use_existing:
                    self._load_existing_reshaped_data()
            
            # STEP 3: Feature Engineering
            if not self.args or not self.args.skip_feature_engineering:
                self._run_feature_engineering()
            else:
                self.logger.info("\n=== STEP 3: SKIPPING FEATURE ENGINEERING ===")
                self._load_existing_enhanced_data()
            
            # STEP 4: Data Splitting
            if not self.args or not self.args.skip_splitting:
                self._run_data_splitting()
            else:
                self.logger.info("\n=== STEP 4: SKIPPING DATA SPLITTING ===")
            
            # STEP 5: Final Validation
            if not self.args or not self.args.skip_validation:
                self._run_validation()
            else:
                self.logger.info("\n=== STEP 5: SKIPPING FINAL VALIDATION ===")
            
            # Generate final summary
            self._generate_summary()
            
            return True
            
        except Exception as e:
            self.stats['errors'].append(str(e))
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            self.stats['end_time'] = time.time()
            self.stats['total_duration'] = self.stats['end_time'] - self.stats['start_time']
    
    def _run_data_analysis(self):
        """Run the data analysis step."""
        step_start = time.time()
        self.logger.info("\n=== STEP 1: DATA ANALYSIS ===")
        self.logger.info("Analyzing SCATS data quality and structure...")
        
        try:
            # For direct Excel analysis, we'll use pandas to load the Excel file
            try:
                self.raw_data = pd.read_excel(self.excel_file)
                logger.info(f"Loaded Excel file with {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            except Exception as e:
                raise ProcessingError(f"Failed to load Excel file: {str(e)}")
                
            # Create a temporary analyzer for the raw Excel data
            # We'll implement the analysis methods directly since SCATSDataAnalyzer
            # is designed to work with the processed CSV data, not raw Excel
            analyzer = SCATSDataAnalyzer()
            analyzer.data_df = self.raw_data
            
            # Analyze data structure
            self.logger.info("\nAnalyzing data structure...")
            # Get basic structure information
            rows, cols = self.raw_data.shape
            unique_sites = self.raw_data['SCATS_ID'].nunique() if 'SCATS_ID' in self.raw_data.columns else 0
            self.logger.info(f"Data structure: {rows} rows, {cols} columns")
            self.logger.info(f"SCATS sites: {unique_sites} unique sites")
            
            # Analyze date range
            self.logger.info("\nAnalyzing date range...")
            # For Excel files, we need to extract the date range differently
            if 'Date' in self.raw_data.columns:
                date_col = 'Date'
            elif 'DateTime' in self.raw_data.columns:
                date_col = 'DateTime'
            else:
                # Try to find a date column
                date_cols = [col for col in self.raw_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                date_col = date_cols[0] if date_cols else None
            
            if date_col:
                start_date = self.raw_data[date_col].min()
                end_date = self.raw_data[date_col].max()
                days = (end_date - start_date).days + 1 if hasattr(start_date, 'days') else 'unknown'
                self.logger.info(f"Date range: {start_date} to {end_date}")
                self.logger.info(f"Total days: {days}")
            else:
                self.logger.info("No date column found in the Excel file")
            
            # Check data completeness
            self.logger.info("\nChecking data completeness...")
            missing_values = self.raw_data.isna().sum().sum()
            missing_percentage = (missing_values / (rows * cols)) * 100
            self.logger.info(f"Missing values: {missing_values} ({missing_percentage:.2f}%)")
            
            # Display basic statistics
            self.logger.info("\nCalculating basic statistics...")
            # Find traffic count column
            traffic_cols = [col for col in self.raw_data.columns if 'traffic' in col.lower() or 'count' in col.lower() or 'volume' in col.lower()]
            if traffic_cols:
                traffic_col = traffic_cols[0]
                stats = {
                    'mean': self.raw_data[traffic_col].mean(),
                    'median': self.raw_data[traffic_col].median(),
                    'min': self.raw_data[traffic_col].min(),
                    'max': self.raw_data[traffic_col].max(),
                    'std': self.raw_data[traffic_col].std()
                }
                self.logger.info(f"Traffic count statistics for {traffic_col}:")
                self.logger.info(f"  - Mean: {stats['mean']:.2f}")
                self.logger.info(f"  - Median: {stats['median']:.2f}")
                self.logger.info(f"  - Min: {stats['min']}")
                self.logger.info(f"  - Max: {stats['max']}")
                self.logger.info(f"  - Std Dev: {stats['std']:.2f}")
            else:
                self.logger.info("No traffic count column identified in the Excel file")
            
            # Validation checkpoint
            self.validation_checkpoints['analysis'] = True
            self.stats['steps_completed'] += 1
            
            self.logger.info("\nData analysis completed successfully")
            
        except Exception as e:
            self.stats['errors'].append(f"Analysis error: {str(e)}")
            self.logger.error(f"Data analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Data analysis failed: {str(e)}")
        
        finally:
            step_end = time.time()
            self.stats['step_durations']['analysis'] = step_end - step_start
    
    def _run_data_reshaping(self):
        """Run the data reshaping step."""
        step_start = time.time()
        self.logger.info("\n=== STEP 2: DATA RESHAPING ===")
        self.logger.info("Reshaping SCATS data from wide to long format...")
        
        try:
            # Use the reshape_scats_data convenience function
            self.reshaped_data, self.site_reference, validation = reshape_scats_data(
                excel_file=self.excel_file,
                output_path=self.reshaped_path,
                site_ref_path=self.site_ref_path
            )
            
            if self.reshaped_data is None:
                raise ProcessingError("Failed to reshape data")
            
            # Record statistics
            self.stats['records_processed'] = len(self.reshaped_data)
            
            # Display reshaping results
            self.logger.info("\nReshaping results:")
            self.logger.info(f"Total records: {len(self.reshaped_data)}")
            self.logger.info(f"Unique sites: {self.reshaped_data['SCATS_ID'].nunique()}")
            self.logger.info(f"Date range: {self.reshaped_data['DateTime'].min()} to {self.reshaped_data['DateTime'].max()}")
            self.logger.info(f"Validation success: {validation['success']}")
            
            # Validation checkpoint
            self.validation_checkpoints['reshaping'] = validation['success']
            self.stats['steps_completed'] += 1
            
            self.logger.info("\nData reshaping completed successfully")
            self.logger.info(f"Saved reshaped data to: {self.reshaped_path}")
            self.logger.info(f"Saved site reference to: {self.site_ref_path}")
            
        except Exception as e:
            self.stats['errors'].append(f"Reshaping error: {str(e)}")
            self.logger.error(f"Data reshaping failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Data reshaping failed: {str(e)}")
        
        finally:
            step_end = time.time()
            self.stats['step_durations']['reshaping'] = step_end - step_start
    
    def _load_existing_reshaped_data(self):
        """Load existing reshaped data."""
        self.logger.info("\n=== STEP 2: USING EXISTING RESHAPED DATA ===")
        self.logger.info(f"Using existing reshaped data from: {self.reshaped_path}")
        
        try:
            # Check if the files exist
            if not os.path.exists(self.reshaped_path) or not os.path.exists(self.site_ref_path):
                raise ProcessingError("Existing reshaped data files not found")
            
            # Load reshaped data
            self.reshaped_data = pd.read_csv(self.reshaped_path)
            self.reshaped_data['DateTime'] = pd.to_datetime(self.reshaped_data['DateTime'])
            
            # Load site reference
            self.site_reference = pd.read_csv(self.site_ref_path)
            
            # Record statistics
            self.stats['records_processed'] = len(self.reshaped_data)
            
            # Display loading results
            self.logger.info(f"Loaded {len(self.reshaped_data)} records from {self.reshaped_path}")
            self.logger.info(f"Loaded {len(self.site_reference)} sites from {self.site_ref_path}")
            
            # Validation checkpoint
            self.validation_checkpoints['reshaping'] = True
            self.stats['steps_completed'] += 1
            
        except Exception as e:
            self.stats['errors'].append(f"Loading error: {str(e)}")
            self.logger.error(f"Failed to load existing reshaped data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Failed to load existing reshaped data: {str(e)}")
    
    def _run_feature_engineering(self):
        """Run the feature engineering step."""
        step_start = time.time()
        self.logger.info("\n=== STEP 3: FEATURE ENGINEERING ===")
        self.logger.info("Adding advanced features to SCATS data...")
        
        try:
            # Make sure reshaped data is loaded
            if self.reshaped_data is None:
                self._load_existing_reshaped_data()
            
            # Determine which features to add
            add_all = True
            if self.args:
                add_all = not (self.args.lag_features or self.args.rolling_stats or self.args.time_features or 
                           self.args.peak_indicators or self.args.site_normalization or self.args.prediction_targets)
            
            # Create feature engineer
            self.logger.info("Creating feature engineer...")
            feature_engineer = SCATSFeatureEngineer(self.reshaped_data)
            
            # Add features based on command-line arguments
            if add_all:
                self.logger.info("Adding all features...")
                success = feature_engineer.add_all_features()
                if not success:
                    raise ProcessingError("Failed to add all features")
            else:
                # Add individual feature types as requested
                if self.args.lag_features:
                    self.logger.info("Adding lag features...")
                    feature_engineer.add_lag_features()
                
                if self.args.rolling_stats:
                    self.logger.info("Adding rolling statistics...")
                    feature_engineer.add_rolling_statistics()
                
                if self.args.time_features:
                    self.logger.info("Adding time-based features...")
                    feature_engineer.add_time_based_features()
                
                if self.args.peak_indicators:
                    self.logger.info("Adding peak hour indicators...")
                    feature_engineer.add_peak_hour_indicators()
                
                if self.args.site_normalization:
                    self.logger.info("Adding site-specific normalization...")
                    feature_engineer.add_site_normalization()
                
                if self.args.prediction_targets:
                    self.logger.info("Adding prediction targets...")
                    feature_engineer.add_prediction_targets()
            
            # Get the enhanced dataframe
            self.enhanced_data = feature_engineer.get_feature_dataframe()
            
            # Save the enhanced dataset
            self.logger.info(f"Saving enhanced dataset to {self.enhanced_path}...")
            self.enhanced_data.to_csv(self.enhanced_path, index=False)
            self.logger.info(f"Saved enhanced dataset with {len(self.enhanced_data)} records and {len(self.enhanced_data.columns)} columns")
            
            # Display feature engineering results
            self.logger.info("\nFeature engineering results:")
            self.logger.info(f"Total records: {len(self.enhanced_data)}")
            self.logger.info(f"Total features: {len(self.enhanced_data.columns)}")
            
            # List all features
            self.logger.info("\nAvailable features:")
            for col in self.enhanced_data.columns:
                self.logger.info(f"- {col}")
            
            # Validation checkpoint
            self.validation_checkpoints['feature_engineering'] = True
            self.stats['steps_completed'] += 1
            
            self.logger.info("\nFeature engineering completed successfully")
            
        except Exception as e:
            self.stats['errors'].append(f"Feature engineering error: {str(e)}")
            self.logger.error(f"Feature engineering failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Feature engineering failed: {str(e)}")
        
        finally:
            step_end = time.time()
            self.stats['step_durations']['feature_engineering'] = step_end - step_start
    
    def _load_existing_enhanced_data(self):
        """Load existing enhanced data."""
        try:
            if os.path.exists(self.enhanced_path):
                self.logger.info(f"Loading enhanced data from {self.enhanced_path}...")
                self.enhanced_data = pd.read_csv(self.enhanced_path)
                self.enhanced_data['DateTime'] = pd.to_datetime(self.enhanced_data['DateTime'])
                self.logger.info(f"Loaded {len(self.enhanced_data)} records from {self.enhanced_path}")
                
                # Validation checkpoint
                self.validation_checkpoints['feature_engineering'] = True
                self.stats['steps_completed'] += 1
                
            elif self.reshaped_data is not None:
                self.logger.info(f"Enhanced data not found, using reshaped data...")
                self.enhanced_data = self.reshaped_data
            else:
                self._load_existing_reshaped_data()
                self.enhanced_data = self.reshaped_data
        
        except Exception as e:
            self.stats['errors'].append(f"Loading error: {str(e)}")
            self.logger.error(f"Failed to load enhanced data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Failed to load enhanced data: {str(e)}")
    
    def _run_data_splitting(self):
        """Run the data splitting step."""
        step_start = time.time()
        self.logger.info("\n=== STEP 4: DATA SPLITTING ===")
        self.logger.info("Splitting data into train/validation/test sets...")
        
        try:
            # Make sure enhanced data is loaded
            if self.enhanced_data is None:
                self._load_existing_enhanced_data()
            
            # Set split ratios
            train_ratio = self.args.train_ratio if self.args and hasattr(self.args, 'train_ratio') else 0.7
            val_ratio = self.args.val_ratio if self.args and hasattr(self.args, 'val_ratio') else 0.15
            test_ratio = self.args.test_ratio if self.args and hasattr(self.args, 'test_ratio') else 0.15
            
            # Split data chronologically
            self.logger.info("Splitting data chronologically...")
            self.train_data, self.val_data, self.test_data, split_info = split_scats_data(
                self.enhanced_data,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                output_dir=self.output_dir
            )
            
            if self.train_data is None:
                raise ProcessingError("Failed to split data")
            
            # Display split results
            self.logger.info("\nSplit results:")
            self.logger.info(f"Train set: {len(self.train_data)} records, {self.train_data['DateTime'].min()} to {self.train_data['DateTime'].max()}")
            self.logger.info(f"Validation set: {len(self.val_data)} records, {self.val_data['DateTime'].min()} to {self.val_data['DateTime'].max()}")
            self.logger.info(f"Test set: {len(self.test_data)} records, {self.test_data['DateTime'].min()} to {self.test_data['DateTime'].max()}")
            
            # Verify site representation
            self.logger.info(f"\nSite representation in splits:")
            total_sites = self.train_data['SCATS_ID'].nunique()
            self.logger.info(f"Total sites: {total_sites}")
            self.logger.info(f"Sites in train set: {split_info['train']['sites']} ({split_info['train']['sites']/total_sites*100:.1f}%)")
            self.logger.info(f"Sites in validation set: {split_info['val']['sites']} ({split_info['val']['sites']/total_sites*100:.1f}%)")
            self.logger.info(f"Sites in test set: {split_info['test']['sites']} ({split_info['test']['sites']/total_sites*100:.1f}%)")
            
            # Verify split proportions
            self.logger.info(f"\nSplit proportions:")
            self.logger.info(f"Target: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
            total_records = len(self.train_data) + len(self.val_data) + len(self.test_data)
            actual_train_ratio = len(self.train_data) / total_records
            actual_val_ratio = len(self.val_data) / total_records
            actual_test_ratio = len(self.test_data) / total_records
            self.logger.info(f"Actual: train={actual_train_ratio:.2f}, val={actual_val_ratio:.2f}, test={actual_test_ratio:.2f}")
            
            # Validation checkpoint
            self.validation_checkpoints['data_splitting'] = True
            self.stats['steps_completed'] += 1
            
            self.logger.info("\nData splitting completed successfully")
            self.logger.info(f"Saved train data to: {self.train_path}")
            self.logger.info(f"Saved validation data to: {self.val_path}")
            self.logger.info(f"Saved test data to: {self.test_path}")
            
        except Exception as e:
            self.stats['errors'].append(f"Data splitting error: {str(e)}")
            self.logger.error(f"Data splitting failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Data splitting failed: {str(e)}")
        
        finally:
            step_end = time.time()
            self.stats['step_durations']['data_splitting'] = step_end - step_start
    
    def _run_validation(self):
        """Run the final validation step."""
        step_start = time.time()
        self.logger.info("\n=== STEP 5: FINAL VALIDATION ===")
        self.logger.info("Validating processed datasets...")
        
        try:
            # Run the validation
            validation_results = validate_processed_data(
                processed_dir=self.output_dir,
                reshaped_path=self.reshaped_path,
                site_ref_path=self.site_ref_path,
                enhanced_path=self.enhanced_path,
                train_path=self.train_path,
                val_path=self.val_path,
                test_path=self.test_path
            )
            
            # Check if validation passed - explicitly convert to boolean to handle any type issues
            overall_validation = bool(validation_results.get('overall_validation', False))
            
            # Always increment steps completed for validation step
            self.stats['steps_completed'] += 1
            
            if overall_validation:
                self.logger.info("\nValidation PASSED! Dataset is ready for training.")
                self.validation_checkpoints['final_validation'] = True
            else:
                self.logger.warning("\nValidation FAILED! Dataset may not be suitable for training.")
                self.logger.warning("Please check the validation results for details.")
                self.stats['warnings'].append("Final validation failed")
                
            # Log validation results path
            validation_results_path = os.path.join(self.output_dir, 'validation_results.txt')
            self.logger.info(f"Detailed validation results saved to: {validation_results_path}")
            
        except Exception as e:
            self.stats['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Validation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ProcessingError(f"Validation failed: {str(e)}")
        
        finally:
            step_end = time.time()
            self.stats['step_durations']['validation'] = step_end - step_start
    
    def _generate_summary(self):
        """Generate a summary of the data processing pipeline."""
        self.logger.info("\n=== DATA PROCESSING PIPELINE SUMMARY ===")
        
        # Calculate total duration
        self.stats['end_time'] = time.time()
        self.stats['total_duration'] = self.stats['end_time'] - self.stats['start_time']
        
        # Display summary statistics
        self.logger.info(f"Processing time: {self.stats['total_duration']:.2f} seconds")
        self.logger.info(f"Steps completed: {self.stats['steps_completed']} of {self.stats['total_steps']}")
        self.logger.info(f"Records processed: {self.stats['records_processed']}")
        
        # Display step durations
        self.logger.info("\nStep durations:")
        for step, duration in self.stats['step_durations'].items():
            self.logger.info(f"  - {step}: {duration:.2f} seconds")
        
        # Display validation checkpoints
        self.logger.info("\nValidation checkpoints:")
        for checkpoint, status in self.validation_checkpoints.items():
            self.logger.info(f"  - {checkpoint.replace('_', ' ').title()}: {'Passed' if status else 'Not performed'}")
        
        # Display errors and warnings
        if self.stats['errors']:
            self.logger.info("\nErrors encountered:")
            for error in self.stats['errors']:
                self.logger.info(f"  - {error}")
        
        if self.stats['warnings']:
            self.logger.info("\nWarnings encountered:")
            for warning in self.stats['warnings']:
                self.logger.info(f"  - {warning}")
        
        # Display output files
        self.logger.info("\nOutput files:")
        if os.path.exists(self.reshaped_path):
            self.logger.info(f"  - Reshaped data: {self.reshaped_path}")
        if os.path.exists(self.site_ref_path):
            self.logger.info(f"  - Site reference: {self.site_ref_path}")
        if os.path.exists(self.enhanced_path):
            self.logger.info(f"  - Enhanced data: {self.enhanced_path}")
        if os.path.exists(self.train_path):
            self.logger.info(f"  - Train data: {self.train_path}")
        if os.path.exists(self.val_path):
            self.logger.info(f"  - Validation data: {self.val_path}")
        if os.path.exists(self.test_path):
            self.logger.info(f"  - Test data: {self.test_path}")
    
        # Display final validation status
        validation_results_path = os.path.join(self.output_dir, 'validation_results.txt')
        if os.path.exists(validation_results_path):
            self.logger.info(f"\nValidation results: {validation_results_path}")
            if self.validation_checkpoints['final_validation']:
                self.logger.info("DATASET IS READY FOR TRAINING!")
            elif not self.args.skip_validation:
                self.logger.warning("Dataset validation failed or was not performed.")
                self.logger.warning("Please check validation results before using for training.")
            else:
                self.logger.info("Validation was skipped. Run with --validate to perform validation.")
                self.logger.info("Use caution when using this dataset for training without validation.")
        
        # Save summary to file
        summary_path = os.path.join(self.output_dir, 'processing_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write("=== SCATS DATA PROCESSING PIPELINE SUMMARY ===\n\n")
                f.write(f"Processing time: {self.stats['total_duration']:.2f} seconds\n")
                f.write(f"Steps completed: {self.stats['steps_completed']} of {self.stats['total_steps']}\n")
                f.write(f"Records processed: {self.stats['records_processed']}\n\n")
            
                # Write step durations
                f.write("Step durations:\n")
                for step, duration in self.stats['step_durations'].items():
                    f.write(f"  - {step}: {duration:.2f} seconds\n")
                f.write("\n")
            
                # Write validation checkpoints
                f.write("Validation checkpoints:\n")
                for checkpoint, status in self.validation_checkpoints.items():
                    f.write(f"  - {checkpoint.replace('_', ' ').title()}: {'Passed' if status else 'Not performed'}\n")
                f.write("\n")
            
                # Write output files
                f.write("Output files:\n")
                if os.path.exists(self.reshaped_path):
                    f.write(f"  - Reshaped data: {self.reshaped_path}\n")
                if os.path.exists(self.site_ref_path):
                    f.write(f"  - Site reference: {self.site_ref_path}\n")
                if os.path.exists(self.enhanced_path):
                    f.write(f"  - Enhanced data: {self.enhanced_path}\n")
                if os.path.exists(self.train_path):
                    f.write(f"  - Train data: {self.train_path}\n")
                if os.path.exists(self.val_path):
                    f.write(f"  - Validation data: {self.val_path}\n")
                if os.path.exists(self.test_path):
                    f.write(f"  - Test data: {self.test_path}\n")
            
                # Write validation status
                validation_results_path = os.path.join(self.output_dir, 'validation_results.txt')
                if os.path.exists(validation_results_path):
                    # Check validation results file for overall validation status
                    validation_status = "FAILED OR NOT PERFORMED"
                    try:
                        with open(validation_results_path, 'r') as vf:
                            for line in vf:
                                if "OVERALL VALIDATION: PASSED" in line:
                                    validation_status = "PASSED - DATASET IS READY FOR TRAINING!"
                                    self.validation_checkpoints['final_validation'] = True
                                    break
                    except Exception:
                        pass
                    
                    f.write("\nValidation Status: ")
                    f.write(f"{validation_status}\n")
                    f.write(f"Detailed validation results: {validation_results_path}\n")
        
            self.logger.info(f"\nSummary saved to: {summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {str(e)}")
        
        self.logger.info("\n=== DATA PROCESSING PIPELINE COMPLETED ===")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Complete SCATS data processing pipeline')
    parser.add_argument('--excel-file', type=str, help='Path to SCATS Excel file (if not using default)')
    parser.add_argument('--output-dir', type=str, help='Directory to save processed data (if not using default)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip data analysis step')
    parser.add_argument('--skip-reshape', action='store_true', help='Skip data reshaping step')
    parser.add_argument('--skip-feature-engineering', action='store_true', help='Skip feature engineering step')
    parser.add_argument('--skip-splitting', action='store_true', help='Skip data splitting step')
    parser.add_argument('--skip-validation', action='store_true', help='Skip final validation step')
    parser.add_argument('--use-existing', action='store_true', help='Use existing processed data files')

    # Feature engineering options
    parser.add_argument('--lag-features', action='store_true', help='Add lag features only')
    parser.add_argument('--rolling-stats', action='store_true', help='Add rolling statistics only')
    parser.add_argument('--time-features', action='store_true', help='Add time-based features only')
    parser.add_argument('--peak-indicators', action='store_true', help='Add peak hour indicators only')
    parser.add_argument('--site-normalization', action='store_true', help='Add site-specific normalization only')
    parser.add_argument('--prediction-targets', action='store_true', help='Add prediction targets only')
    
    # Data splitting options
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data for training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of data for validation set (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of data for test set (default: 0.15)')
    
    return parser.parse_args()

def main():
    """Main function to run the data processing pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and run the pipeline
    pipeline = DataProcessingPipeline(args)
    success = pipeline.run_pipeline()
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
