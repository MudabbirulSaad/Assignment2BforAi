#!/usr/bin/env python3
"""
SCATS Traffic Data Validation Script

This script validates the processed SCATS traffic data to ensure it meets the expected criteria.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import argparse

# Add the parent directory to sys.path to allow importing app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the logger and config
from app.core.logging import logger
from app.config.config import config

class DataValidator:
    """Class for validating processed SCATS traffic data."""
    
    def __init__(self, processed_dir=None, reshaped_path=None, site_ref_path=None, 
                 enhanced_path=None, train_path=None, val_path=None, test_path=None):
        """Initialize the data validator."""
        self.processed_dir = processed_dir or os.path.dirname(config.processed_data['scats_complete'])
        
        # Define expected file paths
        self.traffic_path = os.path.join(self.processed_dir, 'scats_traffic.csv')
        self.site_ref_path = site_ref_path or os.path.join(self.processed_dir, 'scats_site_reference.csv')
        
        # For backward compatibility
        self.reshaped_path = reshaped_path or self.traffic_path
        self.enhanced_path = enhanced_path or self.traffic_path
        self.train_path = train_path or os.path.join(self.processed_dir, 'train_data.csv')
        self.val_path = val_path or os.path.join(self.processed_dir, 'val_data.csv')
        self.test_path = test_path or os.path.join(self.processed_dir, 'test_data.csv')
        
        # Initialize validation results
        self.validation_results = {
            'file_existence': {},
            'record_counts': {},
            'column_validation': {},
            'date_range_validation': {},
            'site_validation': {},
            'split_validation': {},
            'feature_validation': {}
        }
    
    def validate_all(self):
        """Run all validation checks."""
        logger.info("=== VALIDATING PROCESSED SCATS TRAFFIC DATA ===")
        
        # Check file existence
        self.validate_file_existence()
        
        # Load datasets
        reshaped_df = self.load_dataset(self.reshaped_path, 'reshaped')
        site_ref_df = self.load_dataset(self.site_ref_path, 'site_reference')
        enhanced_df = self.load_dataset(self.enhanced_path, 'enhanced')
        train_df = self.load_dataset(self.train_path, 'train')
        val_df = self.load_dataset(self.val_path, 'validation')
        test_df = self.load_dataset(self.test_path, 'test')
        
        # Skip further validation if any dataset failed to load
        if any(df is None for df in [reshaped_df, site_ref_df, enhanced_df, train_df, val_df, test_df]):
            logger.error("Skipping further validation due to missing datasets")
            return self.validation_results
        
        # Validate record counts
        self.validate_record_counts(reshaped_df, enhanced_df, train_df, val_df, test_df)
        
        # Validate columns
        self.validate_columns(reshaped_df, enhanced_df, train_df)
        
        # Validate date range
        self.validate_date_range(reshaped_df, train_df, val_df, test_df)
        
        # Validate sites
        self.validate_sites(reshaped_df, site_ref_df, train_df, val_df, test_df)
        
        # Validate splits
        self.validate_splits(train_df, val_df, test_df)
        
        # Validate features
        self.validate_features(enhanced_df)
        
        # Print summary
        self.print_validation_summary()
        
        return self.validation_results
    
    def validate_file_existence(self):
        """Check if all expected files exist."""
        logger.info("\n1. Validating file existence...")
        
        files_to_check = {
            'reshaped': self.reshaped_path,
            'site_reference': self.site_ref_path,
            'enhanced': self.enhanced_path,
            'train': self.train_path,
            'validation': self.val_path,
            'test': self.test_path
        }
        
        for name, path in files_to_check.items():
            exists = os.path.exists(path)
            self.validation_results['file_existence'][name] = exists
            if exists:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  - {name} file exists: {exists} (Size: {size_mb:.2f} MB)")
            else:
                logger.error(f"  - {name} file does not exist: {path}")
    
    def load_dataset(self, path, name):
        """Load a dataset and convert DateTime column to datetime."""
        if not os.path.exists(path):
            logger.error(f"Cannot load {name} dataset: File does not exist")
            return None
        
        try:
            df = pd.read_csv(path)
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            return df
        except Exception as e:
            logger.error(f"Error loading {name} dataset: {str(e)}")
            return None
    
    def validate_record_counts(self, reshaped_df, enhanced_df, train_df, val_df, test_df):
        """Validate record counts in datasets."""
        logger.info("\n2. Validating record counts...")
        
        # Expected record counts
        expected_reshaped_count = 4192 * 96  # 4192 rows in Excel * 96 time intervals
        expected_enhanced_count = expected_reshaped_count
        expected_total_split_count = expected_reshaped_count
        
        # Actual record counts
        reshaped_count = len(reshaped_df)
        enhanced_count = len(enhanced_df)
        train_count = len(train_df)
        val_count = len(val_df)
        test_count = len(test_df)
        total_split_count = train_count + val_count + test_count
        
        # Record validation results
        self.validation_results['record_counts'] = {
            'reshaped': {
                'expected': expected_reshaped_count,
                'actual': reshaped_count,
                'valid': abs(reshaped_count - expected_reshaped_count) / expected_reshaped_count < 0.05  # Within 5%
            },
            'enhanced': {
                'expected': expected_enhanced_count,
                'actual': enhanced_count,
                'valid': abs(enhanced_count - expected_enhanced_count) / expected_enhanced_count < 0.05  # Within 5%
            },
            'splits': {
                'expected': expected_total_split_count,
                'actual': total_split_count,
                'valid': abs(total_split_count - expected_total_split_count) / expected_total_split_count < 0.05  # Within 5%
            }
        }
        
        # Log results
        logger.info(f"  - Reshaped records: {reshaped_count} (Expected: ~{expected_reshaped_count})")
        logger.info(f"  - Enhanced records: {enhanced_count} (Expected: ~{expected_enhanced_count})")
        logger.info(f"  - Train records: {train_count} ({train_count/total_split_count*100:.1f}%)")
        logger.info(f"  - Validation records: {val_count} ({val_count/total_split_count*100:.1f}%)")
        logger.info(f"  - Test records: {test_count} ({test_count/total_split_count*100:.1f}%)")
        logger.info(f"  - Total split records: {total_split_count} (Expected: ~{expected_total_split_count})")
    
    def validate_columns(self, reshaped_df, enhanced_df, train_df):
        """Validate columns in datasets."""
        logger.info("\n3. Validating columns...")
        
        # Expected columns in reshaped data
        expected_reshaped_columns = ['SCATS_ID', 'DateTime', 'Traffic_Count', 'Location', 'Latitude', 'Longitude']
        
        # Expected additional columns in enhanced data
        expected_enhanced_additional = [
            'Hour', 'DayOfWeek', 'IsWeekend', 'IsPeakHour',
            'Traffic_Count_t-1', 'Traffic_Count_t-4', 'Traffic_Count_t-96',
            'Rolling_1h_Mean', 'Rolling_3h_Mean',
            'Traffic_Count_Normalized', 'Target_t+1', 'Target_t+4'
        ]
        
        # Check reshaped columns
        reshaped_has_expected = all(col in reshaped_df.columns for col in expected_reshaped_columns)
        
        # Check enhanced columns
        enhanced_has_expected = all(col in enhanced_df.columns for col in expected_reshaped_columns)
        enhanced_has_additional = any(col in enhanced_df.columns for col in expected_enhanced_additional)
        
        # Check train columns (should have same columns as enhanced)
        train_has_expected = all(col in train_df.columns for col in expected_reshaped_columns)
        train_has_additional = any(col in train_df.columns for col in expected_enhanced_additional)
        
        # Record validation results
        self.validation_results['column_validation'] = {
            'reshaped': reshaped_has_expected,
            'enhanced': enhanced_has_expected and enhanced_has_additional,
            'train': train_has_expected and train_has_additional
        }
        
        # Log results
        logger.info(f"  - Reshaped data has expected columns: {reshaped_has_expected}")
        logger.info(f"  - Enhanced data has expected columns: {enhanced_has_expected}")
        logger.info(f"  - Enhanced data has additional feature columns: {enhanced_has_additional}")
        logger.info(f"  - Train data has expected columns: {train_has_expected}")
        logger.info(f"  - Train data has additional feature columns: {train_has_additional}")
    
    def validate_date_range(self, reshaped_df, train_df, val_df, test_df):
        """Validate date range in datasets."""
        logger.info("\n4. Validating date ranges...")
        
        # Expected date range
        expected_start = pd.Timestamp('2006-10-01')
        expected_end = pd.Timestamp('2006-10-31')
        
        # Actual date range
        reshaped_start = reshaped_df['DateTime'].min()
        reshaped_end = reshaped_df['DateTime'].max()
        
        # Split date ranges
        train_start = train_df['DateTime'].min()
        train_end = train_df['DateTime'].max()
        val_start = val_df['DateTime'].min()
        val_end = val_df['DateTime'].max()
        test_start = test_df['DateTime'].min()
        test_end = test_df['DateTime'].max()
        
        # Validate chronological splits
        splits_chronological = (
            train_end < val_start and
            val_end < test_start
        )
        
        # Record validation results
        self.validation_results['date_range_validation'] = {
            'reshaped': {
                'expected_start': expected_start,
                'expected_end': expected_end,
                'actual_start': reshaped_start,
                'actual_end': reshaped_end,
                'valid': (
                    reshaped_start.date() == expected_start.date() and
                    reshaped_end.date() == expected_end.date()
                )
            },
            'splits_chronological': splits_chronological
        }
        
        # Log results
        logger.info(f"  - Reshaped data date range: {reshaped_start} to {reshaped_end}")
        logger.info(f"  - Expected date range: {expected_start} to {expected_end}")
        logger.info(f"  - Train data date range: {train_start} to {train_end}")
        logger.info(f"  - Validation data date range: {val_start} to {val_end}")
        logger.info(f"  - Test data date range: {test_start} to {test_end}")
        logger.info(f"  - Splits are chronological: {splits_chronological}")
    
    def validate_sites(self, reshaped_df, site_ref_df, train_df, val_df, test_df):
        """Validate SCATS sites in datasets."""
        logger.info("\n5. Validating SCATS sites...")
        
        # Count unique sites
        reshaped_sites = reshaped_df['SCATS_ID'].nunique()
        site_ref_sites = len(site_ref_df)
        train_sites = train_df['SCATS_ID'].nunique()
        val_sites = val_df['SCATS_ID'].nunique()
        test_sites = test_df['SCATS_ID'].nunique()
        
        # Check if all sites are in all splits
        all_sites_in_all_splits = (
            train_sites == reshaped_sites and
            val_sites == reshaped_sites and
            test_sites == reshaped_sites
        )
        
        # Check if site reference contains all sites in reshaped data
        # It's OK if site reference has more sites than reshaped data
        # This happens because site reference contains all possible sites,
        # but reshaped data only contains sites with actual traffic data
        reshaped_site_ids = set(reshaped_df['SCATS_ID'].unique())
        site_ref_site_ids = set(site_ref_df['SCATS_ID'].unique() if 'SCATS_ID' in site_ref_df.columns else [])
        site_ref_match = reshaped_site_ids.issubset(site_ref_site_ids)
        
        # Record validation results
        self.validation_results['site_validation'] = {
            'reshaped_sites': reshaped_sites,
            'site_ref_sites': site_ref_sites,
            'train_sites': train_sites,
            'val_sites': val_sites,
            'test_sites': test_sites,
            'site_ref_matches_reshaped': site_ref_match,
            'all_sites_in_all_splits': all_sites_in_all_splits
        }
        
        # Log results
        logger.info(f"  - Unique sites in reshaped data: {self.validation_results['site_validation']['reshaped_sites']}")
        logger.info(f"  - Sites in site reference: {self.validation_results['site_validation']['site_ref_sites']}")
        logger.info(f"  - Site reference contains all reshaped data sites: {self.validation_results['site_validation']['site_ref_matches_reshaped']}")
        logger.info(f"  - Sites in train data: {self.validation_results['site_validation']['train_sites']}")
        logger.info(f"  - Sites in validation data: {self.validation_results['site_validation']['val_sites']}")
        logger.info(f"  - Sites in test data: {self.validation_results['site_validation']['test_sites']}")
        logger.info(f"  - All sites in all splits: {self.validation_results['site_validation']['all_sites_in_all_splits']}")

    def validate_splits(self, train_df, val_df, test_df):
        """Validate data splits."""
        logger.info("\n6. Validating data splits...")
        
        # Calculate split proportions
        total_records = len(train_df) + len(val_df) + len(test_df)
        train_ratio = len(train_df) / total_records
        val_ratio = len(val_df) / total_records
        test_ratio = len(test_df) / total_records
        
        # Expected split proportions
        expected_train_ratio = 0.7
        expected_val_ratio = 0.15
        expected_test_ratio = 0.15
        
        # Check if splits are within acceptable range (±5%)
        train_ratio_valid = abs(train_ratio - expected_train_ratio) < 0.05
        val_ratio_valid = abs(val_ratio - expected_val_ratio) < 0.05
        test_ratio_valid = abs(test_ratio - expected_test_ratio) < 0.05
        
        # Record validation results
        self.validation_results['split_validation'] = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_ratio_valid': train_ratio_valid,
            'val_ratio_valid': val_ratio_valid,
            'test_ratio_valid': test_ratio_valid,
            'all_ratios_valid': train_ratio_valid and val_ratio_valid and test_ratio_valid
        }
        
        # Log results
        logger.info(f"  - Train ratio: {train_ratio:.2f} (Expected: {expected_train_ratio:.2f})")
        logger.info(f"  - Validation ratio: {val_ratio:.2f} (Expected: {expected_val_ratio:.2f})")
        logger.info(f"  - Test ratio: {test_ratio:.2f} (Expected: {expected_test_ratio:.2f})")
        logger.info(f"  - All split ratios within acceptable range: {train_ratio_valid and val_ratio_valid and test_ratio_valid}")
    
    def validate_features(self, enhanced_df):
        """Validate features in enhanced data."""
        logger.info("\n7. Validating features...")
        
        # Check for different feature types
        has_lag_features = any(col.startswith('Traffic_Count_t-') for col in enhanced_df.columns)
        has_rolling_stats = any(col.startswith('Rolling_') for col in enhanced_df.columns)
        has_time_features = all(col in enhanced_df.columns for col in ['Hour', 'DayOfWeek', 'IsWeekend'])
        has_peak_indicators = 'IsPeakHour' in enhanced_df.columns
        has_normalization = any(col in enhanced_df.columns for col in ['Traffic_Count_ZScore', 'Traffic_Count_MinMax', 'Traffic_Count_PctOfMax'])
        has_targets = any(col.startswith('Target_t+') for col in enhanced_df.columns)
        
        # Record validation results
        self.validation_results['feature_validation'] = {
            'lag_features': has_lag_features,
            'rolling_stats': has_rolling_stats,
            'time_features': has_time_features,
            'peak_indicators': has_peak_indicators,
            'normalization': has_normalization,
            'targets': has_targets,
            'all_features': (
                has_lag_features and has_rolling_stats and has_time_features and
                has_peak_indicators and has_normalization and has_targets
            )
        }
        
        # Log results
        logger.info(f"  - Has lag features: {has_lag_features}")
        logger.info(f"  - Has rolling statistics: {has_rolling_stats}")
        logger.info(f"  - Has time-based features: {has_time_features}")
        logger.info(f"  - Has peak hour indicators: {has_peak_indicators}")
        logger.info(f"  - Has normalization features: {has_normalization}")
        logger.info(f"  - Has prediction targets: {has_targets}")
        logger.info(f"  - Has all expected feature types: {self.validation_results['feature_validation']['all_features']}")
    
    def print_validation_summary(self):
        """Print a summary of validation results."""
        logger.info("\n=== VALIDATION SUMMARY ===")
        
        # File existence
        all_files_exist = all(self.validation_results['file_existence'].values())
        logger.info(f"1. File Existence: {'PASSED' if all_files_exist else 'FAILED'}")
        
        # Record counts
        record_counts_valid = all(result['valid'] for result in self.validation_results['record_counts'].values())
        logger.info(f"2. Record Counts: {'PASSED' if record_counts_valid else 'FAILED'}")
        
        # Column validation
        columns_valid = all(self.validation_results['column_validation'].values())
        logger.info(f"3. Column Validation: {'PASSED' if columns_valid else 'FAILED'}")
        
        # Date range validation
        date_range_valid = (
            self.validation_results['date_range_validation']['reshaped']['valid'] and
            self.validation_results['date_range_validation']['splits_chronological']
        )
        logger.info(f"4. Date Range Validation: {'PASSED' if date_range_valid else 'FAILED'}")
        
        # Site validation
        site_info = self.validation_results['site_validation']
        sites_valid = site_info['site_ref_matches_reshaped'] and site_info['all_sites_in_all_splits']
        logger.info(f"5. Site Validation: {'PASSED' if sites_valid else 'FAILED'}")
        
        # Split validation
        splits_valid = self.validation_results['split_validation']['all_ratios_valid']
        logger.info(f"6. Split Validation: {'PASSED' if splits_valid else 'FAILED'}")
        
        # Feature validation
        feature_info = self.validation_results['feature_validation']
        features_valid = feature_info['all_features']
        logger.info(f"7. Feature Validation: {'PASSED' if features_valid else 'FAILED'}")
        
        # Overall validation
        all_valid = (
            all_files_exist and
            record_counts_valid and
            columns_valid and
            date_range_valid and
            self.validation_results['site_validation']['site_ref_matches_reshaped'] and  
            self.validation_results['site_validation']['all_sites_in_all_splits'] and
            splits_valid and
            feature_info['all_features']
        )
        # Set the overall validation result in the validation_results dictionary
        self.validation_results['overall_validation'] = all_valid
        logger.info(f"\nOVERALL VALIDATION: {'PASSED' if all_valid else 'FAILED'}")
        
        # Save validation results to file
        self.save_validation_results()
    
    def save_validation_results(self):
        """Save validation results to a file."""
        validation_path = os.path.join(self.processed_dir, 'validation_results.txt')
        
        try:
            with open(validation_path, 'w') as f:
                f.write("=== SCATS TRAFFIC DATA VALIDATION RESULTS ===\n\n")
                
                # File existence
                all_files_exist = all(self.validation_results['file_existence'].values())
                f.write(f"1. File Existence: {'PASSED' if all_files_exist else 'FAILED'}\n")
                for name, exists in self.validation_results['file_existence'].items():
                    f.write(f"  - {name}: {'EXISTS' if exists else 'MISSING'}\n")
                
                # Record counts
                record_counts_valid = all(result['valid'] for result in self.validation_results['record_counts'].values())
                f.write(f"\n2. Record Counts: {'PASSED' if record_counts_valid else 'FAILED'}\n")
                for dataset, info in self.validation_results['record_counts'].items():
                    f.write(f"  - {dataset}: {info['actual']} records (Expected: ~{info['expected']})\n")
                
                # Column validation
                columns_valid = all(self.validation_results['column_validation'].values())
                f.write(f"\n3. Column Validation: {'PASSED' if columns_valid else 'FAILED'}\n")
                for dataset, valid in self.validation_results['column_validation'].items():
                    f.write(f"  - {dataset}: {'VALID' if valid else 'INVALID'}\n")
                
                # Date range validation
                date_range_valid = (
                    self.validation_results['date_range_validation']['reshaped']['valid'] and
                    self.validation_results['date_range_validation']['splits_chronological']
                )
                f.write(f"\n4. Date Range Validation: {'PASSED' if date_range_valid else 'FAILED'}\n")
                reshaped_info = self.validation_results['date_range_validation']['reshaped']
                f.write(f"  - Reshaped data: {reshaped_info['actual_start']} to {reshaped_info['actual_end']}\n")
                f.write(f"  - Expected: {reshaped_info['expected_start']} to {reshaped_info['expected_end']}\n")
                f.write(f"  - Splits chronological: {'YES' if self.validation_results['date_range_validation']['splits_chronological'] else 'NO'}\n")
                
                # Site validation
                sites_valid = (
                    self.validation_results['site_validation']['site_ref_matches_reshaped'] and
                    self.validation_results['site_validation']['all_sites_in_all_splits']
                )
                f.write(f"\n5. Site Validation: {'PASSED' if sites_valid else 'FAILED'}\n")
                site_info = self.validation_results['site_validation']
                f.write(f"  - Reshaped data: {site_info['reshaped_sites']} sites\n")
                f.write(f"  - Site reference: {site_info['site_ref_sites']} sites\n")
                f.write(f"  - Train data: {site_info['train_sites']} sites\n")
                f.write(f"  - Validation data: {site_info['val_sites']} sites\n")
                f.write(f"  - Test data: {site_info['test_sites']} sites\n")
                f.write(f"  - All sites in all splits: {'YES' if site_info['all_sites_in_all_splits'] else 'NO'}\n")
                
                # Split validation
                splits_valid = self.validation_results['split_validation']['all_ratios_valid']
                f.write(f"\n6. Split Validation: {'PASSED' if splits_valid else 'FAILED'}\n")
                split_info = self.validation_results['split_validation']
                f.write(f"  - Train ratio: {split_info['train_ratio']:.2f} (Valid: {'YES' if split_info['train_ratio_valid'] else 'NO'})\n")
                f.write(f"  - Validation ratio: {split_info['val_ratio']:.2f} (Valid: {'YES' if split_info['val_ratio_valid'] else 'NO'})\n")
                f.write(f"  - Test ratio: {split_info['test_ratio']:.2f} (Valid: {'YES' if split_info['test_ratio_valid'] else 'NO'})\n")
                
                # Feature validation
                features_valid = self.validation_results['feature_validation']['all_features']
                f.write(f"\n7. Feature Validation: {'PASSED' if features_valid else 'FAILED'}\n")
                feature_info = self.validation_results['feature_validation']
                f.write(f"  - Lag features: {'PRESENT' if feature_info['lag_features'] else 'MISSING'}\n")
                f.write(f"  - Rolling statistics: {'PRESENT' if feature_info['rolling_stats'] else 'MISSING'}\n")
                f.write(f"  - Time-based features: {'PRESENT' if feature_info['time_features'] else 'MISSING'}\n")
                f.write(f"  - Peak hour indicators: {'PRESENT' if feature_info['peak_indicators'] else 'MISSING'}\n")
                f.write(f"  - Normalization features: {'PRESENT' if feature_info['normalization'] else 'MISSING'}\n")
                f.write(f"  - Prediction targets: {'PRESENT' if feature_info['targets'] else 'MISSING'}\n")
                
                # Overall validation
                all_valid = (
                    all_files_exist and record_counts_valid and columns_valid and
                    date_range_valid and sites_valid and splits_valid and features_valid
                )
                f.write(f"\nOVERALL VALIDATION: {'PASSED' if all_valid else 'FAILED'}\n")
                
                # Timestamp
                f.write(f"\nValidation performed at: {datetime.now()}\n")
            
            logger.info(f"Validation results saved to: {validation_path}")
        
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")

def main():
    """Main function to run the data validation."""
    logger.info("Starting SCATS traffic data validation...")
    
    validator = DataValidator()
    results = validator.validate_all()
    
    return 0

def validate_processed_data(processed_dir=None, reshaped_path=None, site_ref_path=None, 
                         enhanced_path=None, train_path=None, val_path=None, test_path=None):
    """Convenience function to validate processed data.
    
    Args:
        processed_dir (str, optional): Directory containing processed data files.
        reshaped_path (str, optional): Path to reshaped data file.
        site_ref_path (str, optional): Path to site reference file.
        enhanced_path (str, optional): Path to enhanced data file.
        train_path (str, optional): Path to train data file.
        val_path (str, optional): Path to validation data file.
        test_path (str, optional): Path to test data file.
        
    Returns:
        dict: Validation results with overall_validation key.
    """
    logger.info("Starting SCATS traffic data validation...")
    
    validator = DataValidator(
        processed_dir=processed_dir,
        reshaped_path=reshaped_path,
        site_ref_path=site_ref_path,
        enhanced_path=enhanced_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path
    )
    
    results = validator.validate_all()
    
    # Add overall_validation key for the processing pipeline
    if 'overall_validation' not in results:
        # Check if all validation checks passed
        # File existence validation
        file_existence = all(results['file_existence'].values()) if 'file_existence' in results else False
        
        # Record counts validation
        record_counts_valid = False
        if 'record_counts' in results:
            reshaped_count = results['record_counts'].get('reshaped_count', 0)
            enhanced_count = results['record_counts'].get('enhanced_count', 0)
            total_split_count = results['record_counts'].get('total_split_count', 0)
            expected_count = results['record_counts'].get('expected_count', 0)
            
            # Consider valid if counts are within 1% of expected
            if expected_count > 0:
                reshaped_valid = abs(reshaped_count - expected_count) / expected_count < 0.01
                enhanced_valid = abs(enhanced_count - expected_count) / expected_count < 0.01
                splits_valid = abs(total_split_count - expected_count) / expected_count < 0.01
                record_counts_valid = reshaped_valid and enhanced_valid and splits_valid
            else:
                record_counts_valid = True
        
        # Column validation
        columns_valid = False
        if 'column_validation' in results:
            reshaped_columns = results['column_validation'].get('reshaped_columns_valid', False)
            enhanced_columns = results['column_validation'].get('enhanced_columns_valid', False)
            train_columns = results['column_validation'].get('train_columns_valid', False)
            columns_valid = reshaped_columns and enhanced_columns and train_columns
        
        # Date range validation
        date_range_valid = False
        if 'date_range_validation' in results:
            date_range_valid = results['date_range_validation'].get('valid', False)
        
        # Site validation
        sites_valid = False
        if 'site_validation' in results:
            site_ref_match = results['site_validation'].get('site_ref_matches_reshaped', False)
            all_sites_in_splits = results['site_validation'].get('all_sites_in_all_splits', False)
            sites_valid = site_ref_match and all_sites_in_splits
        
        # Split validation
        splits_valid = False
        if 'split_validation' in results:
            splits_valid = results['split_validation'].get('valid', False)
        
        # Feature validation
        features_valid = False
        if 'feature_validation' in results:
            features_valid = results['feature_validation'].get('all_features', False)
        
        # Set overall validation result
        results['overall_validation'] = (file_existence and record_counts_valid and columns_valid and 
                                       date_range_valid and sites_valid and splits_valid and features_valid)
    
    return results

if __name__ == "__main__":
    sys.exit(main())
