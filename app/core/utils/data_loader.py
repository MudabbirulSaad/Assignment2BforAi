#!/usr/bin/env python3
"""
SCATS Traffic Data Loader

This module provides functions to load and process SCATS traffic data from Excel files.
It handles data validation, extraction, and transformation according to project requirements.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path

# Import the logger
from app.core.logging import logger
from app.config.config import config

class SCATSDataLoader:
    """
    Class for loading and processing SCATS traffic data.
    
    This class handles loading data from Excel files, validating structure,
    extracting metadata and traffic data, and converting to appropriate formats.
    """
    
    def __init__(self, excel_file=None):
        """
        Initialize the data loader with the Excel file path.
        
        Args:
            excel_file (str, optional): Path to the SCATS Excel file. If None, uses the path from config.
        """
        self.excel_file = excel_file or config.raw_data['scats_excel']
        self.data_df = None
        self.final_df = None
        self.site_reference = None
        
        # Get configuration values from config
        self.expected_col_count = config.scats_data['expected_col_count']
        self.expected_metadata_cols = config.scats_data['metadata_cols']
        self.expected_traffic_cols = config.scats_data['traffic_cols']
        
        # Expected SCATS site IDs for validation
        self.expected_sites = config.scats_data['expected_sites']
        
        # Metadata column names
        self.metadata_cols = config.scats_data['metadata_col_names']
    
    def load_excel_file(self):
        """
        Load the SCATS Excel file and validate its structure.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
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
        Clean and process the data.
        
        Returns:
            bool: True if cleaning was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_excel_file() first.")
            return False
        
        try:
            logger.info("Cleaning data...")
            
            # If we have unnamed columns, try to identify the structure
            unnamed_columns = self.data_df.columns.str.contains('Unnamed').sum()
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
            
            # Convert date column if present
            if date_col:
                try:
                    self.data_df[date_col] = pd.to_datetime(self.data_df[date_col], errors='coerce')
                    logger.info(f"Converted {date_col} to datetime format")
                except Exception as date_err:
                    logger.warning(f"Could not convert {date_col} to datetime: {str(date_err)}")
            
            # Find traffic data columns (V columns)
            v_columns = []
            for col in self.data_df.columns:
                col_str = str(col)
                if col_str.startswith('V') and col_str[1:].strip().isdigit():
                    v_columns.append(col)
            
            logger.info(f"Found {len(v_columns)} traffic data columns")
            
            # Drop rows with missing SCATS ID
            if scats_id_col:
                initial_count = len(self.data_df)
                self.data_df = self.data_df.dropna(subset=[scats_id_col])
                logger.info(f"Dropped {initial_count - len(self.data_df)} rows with missing SCATS ID")
            
            # Replace NaN values with None for better compatibility
            self.data_df = self.data_df.where(pd.notnull(self.data_df), None)
            
            logger.info(f"Cleaned data: {len(self.data_df)} records remain")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return False
    

    
    def reshape_to_long_format(self):
        """
        Reshape data from wide to long format.
        
        Returns:
            bool: True if reshaping was successful, False otherwise
        """
        if self.data_df is None:
            logger.error("Data not loaded. Call load_excel_file() first.")
            return False
        
        try:
            logger.info("Reshaping data to long format...")
            
            # Find traffic data columns (V columns)
            v_columns = []
            for col in self.data_df.columns:
                col_str = str(col)
                if col_str.startswith('V') and col_str[1:].strip().isdigit():
                    v_columns.append(col)
            
            # Sort V columns numerically
            v_columns = sorted(v_columns, key=lambda x: int(str(x)[1:].strip()))
            
            if not v_columns:
                logger.error("No traffic data columns (V columns) found")
                # Try to identify potential traffic data columns
                numeric_cols = self.data_df.select_dtypes(include=[np.number]).columns.tolist()
                logger.info(f"Found {len(numeric_cols)} numeric columns that might contain traffic data")
                if len(numeric_cols) > 10:  # If we have many numeric columns, use them as traffic data
                    v_columns = numeric_cols[5:101]  # Use a reasonable range for traffic data
                    logger.info(f"Using {len(v_columns)} numeric columns as traffic data")
                else:
                    return False
            
            # Create time intervals mapping
            time_mapping = {}
            for i, col in enumerate(v_columns):
                # Calculate time for each 15-minute interval
                hours = (i * 15) // 60
                minutes = (i * 15) % 60
                time_str = f"{hours:02d}:{minutes:02d}"
                time_mapping[col] = time_str
            
            logger.info(f"Created time mapping for {len(v_columns)} intervals")
            
            # Identify key columns
            scats_id_col = None
            location_col = None
            lat_col = None
            lng_col = None
            date_col = None
            melway_col = None
            
            for col in self.data_df.columns:
                col_str = str(col).lower()
                if ('scats' in col_str and ('number' in col_str or 'id' in col_str or '#' in col_str)) or col_str == 'id':
                    scats_id_col = col
                elif 'location' in col_str or 'description' in col_str:
                    location_col = col
                elif 'latitude' in col_str or 'lat' in col_str:
                    lat_col = col
                elif 'longitude' in col_str or 'lng' in col_str or 'long' in col_str:
                    lng_col = col
                elif 'date' in col_str:
                    date_col = col
                elif 'melway' in col_str or 'cd_melway' in col_str:
                    melway_col = col
            
            # If we couldn't find SCATS ID column, use the first non-V column
            if not scats_id_col:
                for col in self.data_df.columns:
                    if col not in v_columns:
                        scats_id_col = col
                        logger.warning(f"Using {col} as SCATS ID column")
                        break
            
            # If we couldn't find date column, create a default date
            default_date = pd.Timestamp(2006, 10, 1)  # Default to October 1, 2006
            if not date_col:
                logger.warning("No date column found, using default date")
            
            # Create records list
            records = []
            
            # Process each row
            for _, row in self.data_df.iterrows():
                # Skip rows with missing SCATS ID
                if scats_id_col is None or pd.isna(row[scats_id_col]):
                    continue
                    
                scats_id = str(row[scats_id_col]).strip()
                # Skip empty SCATS IDs
                if not scats_id:
                    continue
                    
                # Get date value if available
                date_value = row[date_col] if date_col and not pd.isna(row[date_col]) else default_date
                
                # Get location and coordinates if available
                location = row[location_col] if location_col and not pd.isna(row[location_col]) else None
                latitude = row[lat_col] if lat_col and not pd.isna(row[lat_col]) else None
                longitude = row[lng_col] if lng_col and not pd.isna(row[lng_col]) else None
                melway = row[melway_col] if melway_col and not pd.isna(row[melway_col]) else None
                
                # Process each time interval
                for i, v_col in enumerate(v_columns):
                    # Skip if column doesn't exist in row
                    if v_col not in row:
                        continue
                        
                    traffic_count = row[v_col]
                    
                    # Skip if traffic count is missing or not numeric
                    if pd.isna(traffic_count) or not isinstance(traffic_count, (int, float)):
                        continue
                    
                    # Get time components
                    time_str = time_mapping[v_col]
                    hours, minutes = map(int, time_str.split(':'))
                    
                    # Create datetime
                    try:
                        if isinstance(date_value, pd.Timestamp):
                            dt = pd.Timestamp(date_value.year, date_value.month, date_value.day, 
                                            hours, minutes)
                        elif isinstance(date_value, str):
                            # Try to parse string date
                            date_obj = pd.to_datetime(date_value).date()
                            dt = pd.Timestamp(date_obj.year, date_obj.month, date_obj.day, hours, minutes)
                        else:
                            # Use default date
                            dt = pd.Timestamp(2006, 10, 1, hours, minutes)
                    except Exception as dt_err:
                        logger.warning(f"Error creating datetime for {date_value}: {str(dt_err)}")
                        dt = pd.Timestamp(2006, 10, 1, hours, minutes)
                    
                    # Calculate day of week and weekend flag
                    day_of_week = dt.weekday()
                    is_weekend = day_of_week >= 5
                    
                    # Create record
                    record = {
                        'SCATS_ID': scats_id,
                        'DateTime': dt,
                        'Traffic_Count': float(traffic_count),  # Convert to float for consistency
                        'Date': dt.date(),
                        'Time': dt.time(),
                        'DayOfWeek': day_of_week,
                        'Hour': hours,
                        'Minute': minutes,
                        'IsWeekend': is_weekend,
                        'IntervalOfDay': i,
                    }
                    
                    # Add optional fields if available
                    if location is not None:
                        record['Location'] = location
                    if latitude is not None:
                        record['Latitude'] = float(latitude)
                    if longitude is not None:
                        record['Longitude'] = float(longitude)
                    if melway is not None:
                        record['CD_MELWAY'] = str(melway)
                    
                    records.append(record)
            
            # Create final dataframe
            if not records:
                logger.error("No valid records found after processing")
                return False
                
            self.final_df = pd.DataFrame(records)
            
            # Create site reference dataframe
            site_cols = ['SCATS_ID']
            if 'Location' in self.final_df.columns:
                site_cols.append('Location')
            if 'Latitude' in self.final_df.columns:
                site_cols.append('Latitude')
            if 'Longitude' in self.final_df.columns:
                site_cols.append('Longitude')
            if 'CD_MELWAY' in self.final_df.columns:
                site_cols.append('CD_MELWAY')
                
            self.site_reference = self.final_df[site_cols].drop_duplicates()
            
            logger.info(f"Reshaped data to long format: {len(self.final_df)} records")
            logger.info(f"Created site reference with {len(self.site_reference)} unique sites")
            return True
            
        except Exception as e:
            logger.error(f"Error reshaping to long format: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    

    
    def validate_data_quality(self):
        """
        Validate the quality of the processed data.
        
        Returns:
            bool: True if validation passed, False otherwise
        """
        if self.final_df is None:
            logger.error("Final dataframe is not prepared. Call reshape_to_long_format() first.")
            return False
        
        try:
            # Calculate expected record count
            expected_sites = len(self.expected_sites)
            unique_dates = self.final_df['Date'].nunique()
            expected_records = expected_sites * unique_dates * self.expected_traffic_cols
            
            # Validation checks
            logger.info("=== DATASET VALIDATION ===")
            logger.info(f"Total records: {len(self.final_df)}")
            logger.info(f"Unique SCATS sites: {self.final_df['SCATS_ID'].nunique()}")
            logger.info(f"Date range: {self.final_df['Date'].min()} to {self.final_df['Date'].max()}")
            logger.info(f"Unique days: {unique_dates}")
            logger.info(f"Expected records: {expected_sites} sites × {unique_dates} days × {self.expected_traffic_cols} intervals = {expected_records}")
            
            # Check for missing data
            missing_data = self.final_df['Traffic_Count'].isna().sum()
            missing_pct = missing_data / len(self.final_df) * 100
            logger.info(f"Missing traffic data: {missing_data} ({missing_pct:.2f}%)")
            
            # Validate SCATS IDs match expected list
            actual_sites = sorted(self.final_df['SCATS_ID'].unique())
            sites_match = set(self.expected_sites) == set(actual_sites)
            logger.info(f"SCATS sites validation: {sites_match}")
            
            if not sites_match:
                missing_sites = set(self.expected_sites) - set(actual_sites)
                extra_sites = set(actual_sites) - set(self.expected_sites)
                
                if missing_sites:
                    logger.warning(f"Missing expected sites: {missing_sites}")
                if extra_sites:
                    logger.warning(f"Unexpected extra sites: {extra_sites}")
            
            # Check for negative traffic counts
            negative_counts = (self.final_df['Traffic_Count'] < 0).sum()
            if negative_counts > 0:
                logger.warning(f"Found {negative_counts} negative traffic counts")
            
            # Overall validation result
            validation_passed = (
                len(self.final_df) > 0 and
                self.final_df['SCATS_ID'].nunique() > 0 and
                missing_pct < 5.0  # Allow up to 5% missing data
            )
            
            if validation_passed:
                logger.info("Data validation PASSED")
            else:
                logger.error("Data validation FAILED")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return False
    
    def save_processed_data(self, output_dir=None):
        """
        Save the processed data to CSV files.
        
        Args:
            output_dir (str, optional): Directory to save the output files. If None, uses the path from config.
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if self.final_df is None or self.site_reference is None:
            logger.error("Processed data is not prepared. Call previous methods first.")
            return False
        
        try:
            # Determine output directory
            if output_dir is None:
                # Get the directory part of the processed data path
                scats_complete_path = config.processed_data['scats_complete']
                output_dir = os.path.dirname(scats_complete_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save traffic dataset
            traffic_path = os.path.join(output_dir, 'scats_traffic.csv')
            self.final_df.to_csv(traffic_path, index=False)
            
            # Save site reference
            site_ref_path = os.path.join(output_dir, 'scats_site_reference.csv')
            self.site_reference.to_csv(site_ref_path, index=False)
            
            logger.info("=== DATASET PREPARATION COMPLETE ===")
            logger.info("Files created:")
            logger.info(f"- scats_traffic.csv ({len(self.final_df)} records)")
            logger.info(f"- scats_site_reference.csv ({len(self.site_reference)} records)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def process_data(self):
        """
        Process the SCATS data from loading to saving in one method.
        
        Returns:
            bool: True if the entire process was successful, False otherwise
        """
        steps = [
            self.load_excel_file,
            self.clean_data,
            self.reshape_to_long_format,
            self.validate_data_quality,
            self.save_processed_data
        ]
        
        for step_func in steps:
            step_name = step_func.__name__
            logger.info(f"Starting step: {step_name}")
            
            if not step_func():
                logger.error(f"Failed at step: {step_name}")
                return False
            
            logger.info(f"Completed step: {step_name}")
        
        logger.info("Data processing completed successfully")
        return True
    
    def get_processed_data(self):
        """
        Get the processed data.
        
        Returns:
            tuple: (final_df, site_reference) if data is processed, (None, None) otherwise
        """
        if self.final_df is None or self.site_reference is None:
            logger.warning("Processed data is not available. Call process_data() first.")
            return None, None
        
        return self.final_df, self.site_reference


# Function to load SCATS data using the class
def load_scats_data(excel_file=None):
    """
    Load SCATS traffic data from Excel file.
    
    Args:
        excel_file (str, optional): Path to the SCATS Excel file. If None, uses the path from config.
        
    Returns:
        tuple: (final_df, site_reference) if successful, (None, None) otherwise
    """
    loader = SCATSDataLoader(excel_file)
    if loader.process_data():
        return loader.get_processed_data()
    return None, None


# Example usage
if __name__ == "__main__":
    # Use the path from config
    data, site_ref = load_scats_data()
    
    if data is not None:
        print(f"Successfully loaded {len(data)} records from {data['SCATS_ID'].nunique()} SCATS sites")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    else:
        print("Failed to load SCATS data")
