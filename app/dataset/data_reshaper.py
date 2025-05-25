#!/usr/bin/env python3
"""
SCATS Traffic Data Reshaper

This module provides functionality to reshape SCATS traffic data from wide format
(96 columns representing 15-minute intervals) to long format (one row per interval).
It preserves all critical metadata including SCATS_ID, location, and geographic coordinates.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional

# Import the logger, config, and time converter
from app.core.logging import logger
from app.config.config import config
from app.dataset.time_converter import SCATSTimeConverter

class SCATSDataReshaper:
    """
    Class for reshaping SCATS traffic data from wide to long format.
    
    This class handles the conversion of wide-format SCATS data (96 columns for time intervals)
    to long format (one row per interval) while preserving all critical metadata.
    """
    
    def __init__(self, data_df=None, excel_file=None):
        """
        Initialize the data reshaper with optional dataframe or Excel file path.
        
        Args:
            data_df (pd.DataFrame, optional): Wide-format SCATS data DataFrame.
            excel_file (str, optional): Path to SCATS Excel file. If None, uses the path from config.
        """
        self.data_df = data_df
        self.excel_file = excel_file or config.raw_data['scats_excel']
        self.long_df = None
        self.site_reference = None
        
        # Get configuration values from config
        self.expected_col_count = config.scats_data['expected_col_count']
        self.expected_metadata_cols = config.scats_data['metadata_cols']
        self.expected_traffic_cols = config.scats_data['traffic_cols']
        
        # Expected SCATS site IDs for validation
        self.expected_sites = config.scats_data['expected_sites']
        
        # Metadata column names
        self.metadata_cols = config.scats_data['metadata_col_names']
        
        # Create time converter for Excel date conversion
        self.time_converter = SCATSTimeConverter()
    
    def load_data(self, reload=False):
        """
        Load the SCATS data from Excel file if not already provided.
        
        Args:
            reload (bool, optional): Whether to reload data even if already loaded.
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if self.data_df is not None and not reload:
            logger.info("Data already loaded, skipping load_data step")
            return True
        
        try:
            logger.info(f"Loading SCATS Excel file: {self.excel_file}")
            
            # Load all sheets from the Excel file
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
            logger.info(f"Excel file contains {len(excel_data)} sheets: {list(excel_data.keys())}")
            
            # Check if 'Data' sheet exists
            if 'Data' in excel_data:
                logger.info("Found 'Data' sheet in Excel file")
                self.data_df = excel_data['Data']
            else:
                # If no 'Data' sheet, try to find the sheet with the most columns
                max_cols = 0
                sheet_name = None
                for name, df in excel_data.items():
                    if df.shape[1] > max_cols:
                        max_cols = df.shape[1]
                        sheet_name = name
                
                if sheet_name:
                    logger.info(f"Using sheet '{sheet_name}' with {max_cols} columns")
                    self.data_df = excel_data[sheet_name]
                else:
                    logger.error("Could not find a suitable sheet in the Excel file")
                    return False
            
            # Basic validation
            if self.data_df.shape[0] == 0:
                logger.error("Selected sheet is empty")
                return False
            
            # Skip header rows if needed
            # Look for rows that might contain column headers
            for i in range(min(10, self.data_df.shape[0])):
                row = self.data_df.iloc[i]
                # Check if this row has values that look like headers
                if any(str(val).lower() in ['scats', 'id', 'location', 'date', 'v1', 'v01'] 
                       for val in row if val is not None and pd.notna(val)):
                    logger.info(f"Found potential header row at index {i}")
                    # Use this row as header
                    header_row = self.data_df.iloc[i]
                    self.data_df = self.data_df.iloc[i+1:].reset_index(drop=True)
                    self.data_df.columns = header_row
                    break
            
            # Log the actual structure we found
            logger.info(f"Data structure: {self.data_df.shape[0]} rows, {self.data_df.shape[1]} columns")
            logger.info(f"First few column names: {list(self.data_df.columns)[:5]}")
            
            logger.info(f"Successfully loaded Excel file with {self.data_df.shape[0]} data records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def clean_data(self):
        """
        Clean and process the data before reshaping.
        
        Returns:
            bool: True if cleaning was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        try:
            logger.info("Cleaning data...")
            
            # If we have unnamed columns, try to identify the structure
            unnamed_columns = 0
            for col in self.data_df.columns:
                if 'Unnamed' in str(col) or pd.isna(col):
                    unnamed_columns += 1
            
            if unnamed_columns > self.data_df.shape[1] / 2:
                logger.info("Excel file has mostly unnamed columns, attempting to identify structure")
                
                # Look for rows that might contain column headers
                for i in range(min(10, self.data_df.shape[0])):
                    row = self.data_df.iloc[i]
                    # Check if this row has values that look like headers
                    if any(str(val).lower() in ['scats', 'id', 'location', 'date'] for val in row if val is not None and pd.notna(val)):
                        logger.info(f"Found potential header row at index {i}")
                        # Use this row as header
                        header_row = self.data_df.iloc[i]
                        self.data_df = self.data_df.iloc[i+1:].reset_index(drop=True)
                        self.data_df.columns = header_row
                        break
            
            # Find SCATS ID column
            scats_id_col = None
            for col in self.data_df.columns:
                col_str = str(col).lower()
                if ('scats' in col_str and ('number' in col_str or 'id' in col_str or '#' in col_str)) or col_str == 'id':
                    scats_id_col = col
                    logger.info(f"Found SCATS ID column: {col}")
                    break
            
            if scats_id_col:
                # Convert SCATS ID to string to preserve leading zeros
                self.data_df[scats_id_col] = self.data_df[scats_id_col].astype(str)
                logger.info(f"Converted {scats_id_col} to string format")
            else:
                logger.warning("Could not find SCATS ID column, using first column as ID")
                scats_id_col = self.data_df.columns[0]
                self.data_df[scats_id_col] = self.data_df[scats_id_col].astype(str)
            
            # Find other important columns
            date_col = None
            location_col = None
            lat_col = None
            lng_col = None
            melway_col = None
            
            for col in self.data_df.columns:
                col_str = str(col).lower()
                if 'date' in col_str:
                    date_col = col
                    logger.info(f"Found date column: {col}")
                elif 'location' in col_str or 'description' in col_str:
                    location_col = col
                    logger.info(f"Found location column: {col}")
                elif 'latitude' in col_str or 'lat' in col_str:
                    lat_col = col
                    logger.info(f"Found latitude column: {col}")
                elif 'longitude' in col_str or 'lng' in col_str or 'long' in col_str:
                    lng_col = col
                    logger.info(f"Found longitude column: {col}")
                elif 'melway' in col_str or 'cd_melway' in col_str:
                    melway_col = col
                    logger.info(f"Found Melway reference column: {col}")
            
            # Map the found columns to standard names
            column_mapping = {}
            if scats_id_col:
                column_mapping[scats_id_col] = 'SCATS_ID'
            if location_col:
                column_mapping[location_col] = 'Location'
            if lat_col:
                column_mapping[lat_col] = 'Latitude'
            if lng_col:
                column_mapping[lng_col] = 'Longitude'
            if melway_col:
                column_mapping[melway_col] = 'CD_MELWAY'
            if date_col:
                column_mapping[date_col] = 'Excel_Date'
            
            # Apply column mapping
            if column_mapping:
                self.data_df = self.data_df.rename(columns=column_mapping)
                logger.info(f"Renamed columns: {column_mapping}")
            
            # Find traffic data columns (V columns)
            v_columns = []
            for col in self.data_df.columns:
                col_str = str(col)
                # Check for V00, V01, ..., V95 pattern
                if col_str.startswith('V') and col_str[1:].strip().isdigit():
                    v_columns.append(col)
                # Also check for numeric columns after the metadata columns
                elif isinstance(col, (int, float)) and len(v_columns) < self.expected_traffic_cols:
                    v_columns.append(col)
            
            logger.info(f"Found {len(v_columns)} traffic data columns")
            
            # Validate that we have enough traffic data columns
            if len(v_columns) < 90:  # Allow some flexibility, but we need most of the 96 intervals
                logger.warning(f"Expected at least 90 traffic data columns, but found only {len(v_columns)}")
            
            # Convert Excel date column if present
            if 'Excel_Date' in self.data_df.columns:
                try:
                    # Keep the original Excel date for later conversion
                    self.data_df['Excel_Date_Original'] = self.data_df['Excel_Date']
                    logger.info("Preserved original Excel date values")
                except Exception as date_err:
                    logger.warning(f"Could not preserve Excel date: {str(date_err)}")
            
            logger.info("Data cleaning completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def reshape_to_long_format(self):
        """
        Reshape the data from wide format (96 columns) to long format (one row per interval).
        
        Returns:
            bool: True if reshaping was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        try:
            logger.info("Reshaping data from wide to long format...")
            
            # Identify metadata and traffic columns
            metadata_columns = []
            traffic_columns = []
            
            # Check for standard metadata columns
            for col in self.metadata_cols:
                if col in self.data_df.columns:
                    metadata_columns.append(col)
            
            # Find traffic data columns (V columns)
            for col in self.data_df.columns:
                col_str = str(col)
                # Check for V00, V01, ..., V95 pattern
                if col_str.startswith('V') and col_str[1:].strip().isdigit():
                    traffic_columns.append(col)
                # Also check for numeric columns that might be traffic data
                elif isinstance(col, (int, float)) and col not in metadata_columns:
                    traffic_columns.append(col)
            
            # Sort traffic columns if they're numeric
            try:
                traffic_columns = sorted(traffic_columns, key=lambda x: int(str(x).replace('V', '')))
            except:
                # If sorting fails, keep the original order
                pass
            
            # Ensure we have the essential columns
            required_cols = ['SCATS_ID', 'Excel_Date']
            missing_cols = [col for col in required_cols if col not in self.data_df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Create a mapping of interval index to time
            interval_mapping = {}
            for i in range(96):
                interval_time = self.time_converter.interval_to_time(i)
                interval_mapping[i] = interval_time
            
            # Create records for the long format
            records = []
            
            # Process each row in the wide format
            for _, row in self.data_df.iterrows():
                # Extract metadata
                scats_id = row['SCATS_ID']
                excel_date = row['Excel_Date_Original'] if 'Excel_Date_Original' in row else row['Excel_Date']
                
                # Extract other metadata if available
                metadata = {'SCATS_ID': scats_id}
                for col in metadata_columns:
                    if col != 'SCATS_ID' and col != 'Excel_Date' and col in row:
                        metadata[col] = row[col]
                
                # Process each traffic interval
                for i, col in enumerate(traffic_columns):
                    if i >= 96:  # Only process the first 96 intervals
                        break
                    
                    # Get traffic count for this interval
                    traffic_count = row[col]
                    
                    # Convert Excel date + interval to datetime
                    # Handle both numeric Excel dates and already converted datetime objects
                    if isinstance(excel_date, (int, float)) and not pd.isna(excel_date):
                        dt = self.time_converter.excel_to_datetime(excel_date)
                        if dt is None:
                            logger.warning(f"Could not convert Excel date {excel_date} for SCATS_ID {scats_id}")
                            continue
                    elif isinstance(excel_date, datetime):
                        # If it's already a datetime, use it directly
                        dt = excel_date
                    else:
                        logger.warning(f"Invalid Excel date format: {type(excel_date)} for SCATS_ID {scats_id}")
                        continue
                    
                    # Add time component for this interval
                    interval_time = interval_mapping[i]
                    dt = dt.replace(hour=interval_time.hour, minute=interval_time.minute)
                    
                    # Extract time features
                    time_features = self.time_converter.extract_time_features(dt)
                    
                    # Create record
                    record = {
                        'SCATS_ID': scats_id,
                        'DateTime': dt,
                        'Traffic_Count': traffic_count,
                        'IntervalOfDay': i
                    }
                    
                    # Add time features
                    for key, value in time_features.items():
                        if key not in record:
                            record[key] = value
                    
                    # Add other metadata
                    for key, value in metadata.items():
                        if key not in record:
                            record[key] = value
                    
                    records.append(record)
            
            # Create DataFrame from records
            self.long_df = pd.DataFrame(records)
            
            # Create site reference DataFrame
            site_cols = ['SCATS_ID', 'Location', 'Latitude', 'Longitude', 'CD_MELWAY']
            available_cols = [col for col in site_cols if col in self.long_df.columns]
            self.site_reference = self.long_df[available_cols].drop_duplicates().reset_index(drop=True)
            
            logger.info(f"Reshaped data to long format: {len(self.long_df)} records")
            logger.info(f"Created site reference with {len(self.site_reference)} unique sites")
            
            return True
            
        except Exception as e:
            logger.error(f"Error reshaping data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def validate_reshape_output(self):
        """
        Validate the reshaped data to ensure it meets requirements.
        
        Returns:
            dict: Validation results
        """
        if self.long_df is None:
            logger.error("Data not reshaped. Call reshape_to_long_format() first.")
            return {'success': False, 'message': 'Data not reshaped'}
        
        try:
            logger.info("Validating reshaped data...")
            
            validation = {
                'success': True,
                'total_records': len(self.long_df),
                'unique_sites': self.long_df['SCATS_ID'].nunique(),
                'date_range': {
                    'min': self.long_df['DateTime'].min(),
                    'max': self.long_df['DateTime'].max()
                },
                'unique_days': self.long_df['Date'].nunique() if 'Date' in self.long_df.columns else None,
                'missing_data': self.long_df['Traffic_Count'].isna().sum(),
                'site_validation': {}
            }
            
            # Check for expected columns
            expected_columns = ['SCATS_ID', 'DateTime', 'Traffic_Count', 'Date', 'Time', 
                               'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'IntervalOfDay']
            
            missing_columns = [col for col in expected_columns if col not in self.long_df.columns]
            validation['missing_columns'] = missing_columns
            
            if missing_columns:
                logger.warning(f"Missing expected columns: {missing_columns}")
                validation['success'] = False
            
            # Check for expected SCATS sites
            actual_sites = sorted(self.long_df['SCATS_ID'].unique())
            validation['site_validation']['expected_sites'] = len(self.expected_sites)
            validation['site_validation']['actual_sites'] = len(actual_sites)
            validation['site_validation']['missing_sites'] = [site for site in self.expected_sites if site not in actual_sites]
            validation['site_validation']['extra_sites'] = [site for site in actual_sites if site not in self.expected_sites]
            
            if validation['site_validation']['missing_sites']:
                logger.warning(f"Missing expected SCATS sites: {validation['site_validation']['missing_sites']}")
                validation['success'] = False
            
            # Check for expected record count
            expected_records = len(self.expected_sites) * 31 * 96  # sites × days × intervals
            validation['expected_records'] = expected_records
            
            if validation['total_records'] < expected_records * 0.9:  # Allow 10% missing
                logger.warning(f"Expected at least {expected_records * 0.9} records, but found only {validation['total_records']}")
                validation['success'] = False
            
            # Check for missing data
            if validation['missing_data'] > validation['total_records'] * 0.01:  # Allow 1% missing
                logger.warning(f"High proportion of missing traffic data: {validation['missing_data']} ({validation['missing_data']/validation['total_records']*100:.2f}%)")
                validation['success'] = False
            
            # Check for geographic information
            if 'Latitude' not in self.long_df.columns or 'Longitude' not in self.long_df.columns:
                logger.warning("Missing geographic coordinates (Latitude/Longitude)")
                validation['success'] = False
            
            # Log validation results
            logger.info(f"=== RESHAPE VALIDATION RESULTS ===")
            logger.info(f"Total records: {validation['total_records']}")
            logger.info(f"Unique SCATS sites: {validation['unique_sites']}")
            logger.info(f"Date range: {validation['date_range']['min']} to {validation['date_range']['max']}")
            logger.info(f"Missing traffic data: {validation['missing_data']} ({validation['missing_data']/validation['total_records']*100:.2f}%)")
            logger.info(f"Validation success: {validation['success']}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating reshaped data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'message': str(e)}
    
    def save_reshaped_data(self, output_path=None, site_ref_path=None):
        """
        Save the reshaped data to CSV files.
        
        Args:
            output_path (str, optional): Path to save the reshaped data. If None, uses the path from config.
            site_ref_path (str, optional): Path to save the site reference data. If None, uses the path from config.
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if self.long_df is None:
            logger.error("Data not reshaped. Call reshape_to_long_format() first.")
            return False
        
        try:
            # Use default paths if not provided
            if output_path is None:
                output_path = config.processed_data['scats_complete']
            
            if site_ref_path is None:
                site_ref_path = config.processed_data['site_reference']
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(site_ref_path), exist_ok=True)
            
            # Save reshaped data
            self.long_df.to_csv(output_path, index=False)
            logger.info(f"Saved reshaped data to {output_path}")
            
            # Save site reference
            self.site_reference.to_csv(site_ref_path, index=False)
            logger.info(f"Saved site reference to {site_ref_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving reshaped data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_data(self):
        """
        Process the data from loading to saving in one step.
        
        Returns:
            tuple: (long_df, site_reference, validation) if successful, (None, None, None) otherwise
        """
        # Load data
        if not self.load_data():
            return None, None, None
        
        # Clean data
        if not self.clean_data():
            return None, None, None
        
        # Reshape to long format
        if not self.reshape_to_long_format():
            return None, None, None
        
        # Validate reshape output
        validation = self.validate_reshape_output()
        
        # Save reshaped data
        self.save_reshaped_data()
        
        return self.long_df, self.site_reference, validation


def reshape_scats_data(excel_file=None, output_path=None, site_ref_path=None):
    """
    Convenience function to reshape SCATS traffic data from wide to long format.
    
    Args:
        excel_file (str, optional): Path to SCATS Excel file. If None, uses the path from config.
        output_path (str, optional): Path to save the reshaped data. If None, uses the path from config.
        site_ref_path (str, optional): Path to save the site reference data. If None, uses the path from config.
        
    Returns:
        tuple: (long_df, site_reference, validation) if successful, (None, None, None) otherwise
    """
    # Create reshaper
    reshaper = SCATSDataReshaper(excel_file=excel_file)
    
    # Process data
    long_df, site_reference, validation = reshaper.process_data()
    
    # Save to specified paths if provided
    if long_df is not None and (output_path is not None or site_ref_path is not None):
        reshaper.save_reshaped_data(output_path, site_ref_path)
    
    return long_df, site_reference, validation


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Reshape SCATS traffic data from wide to long format')
    parser.add_argument('--excel-file', type=str, help='Path to SCATS Excel file')
    parser.add_argument('--output-path', type=str, help='Path to save the reshaped data')
    parser.add_argument('--site-ref-path', type=str, help='Path to save the site reference data')
    args = parser.parse_args()
    
    # Reshape data
    logger.info("=== SCATS TRAFFIC DATA RESHAPING ===")
    long_df, site_reference, validation = reshape_scats_data(
        excel_file=args.excel_file,
        output_path=args.output_path,
        site_ref_path=args.site_ref_path
    )
    
    if long_df is not None:
        logger.info("Data reshaping completed successfully")
        logger.info(f"Reshaped data: {len(long_df)} records")
        logger.info(f"Site reference: {len(site_reference)} sites")
        logger.info(f"Validation success: {validation['success']}")
    else:
        logger.error("Data reshaping failed")
