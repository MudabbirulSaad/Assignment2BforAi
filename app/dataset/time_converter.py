#!/usr/bin/env python3
"""
SCATS Traffic Time Converter Module

This module provides utilities for handling time conversions in the SCATS traffic dataset.
It includes functions for Excel date conversion, 15-minute interval mapping, and time feature extraction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from typing import Union, Tuple, List, Dict, Any

# Import the logger and config
from app.core.logging import logger
from app.config.config import config

class SCATSTimeConverter:
    """
    Class for handling time conversions in SCATS traffic data.
    
    This class provides methods to:
    - Convert Excel dates to Python datetime
    - Map 15-minute intervals to time of day
    - Extract time features (hour, day of week, etc.)
    - Identify peak hours
    """
    
    # Excel date system constants
    EXCEL_DATE_SYSTEM_1900_BASE = datetime(1900, 1, 1)
    EXCEL_DATE_SYSTEM_1904_BASE = datetime(1904, 1, 1)
    EXCEL_LEAP_YEAR_BUG_DAYS = 2  # Excel incorrectly treats 1900 as a leap year
    
    # Time interval constants
    MINUTES_PER_INTERVAL = 15
    INTERVALS_PER_HOUR = 60 // MINUTES_PER_INTERVAL
    INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR
    
    def __init__(self, timezone: str = 'Australia/Melbourne'):
        """
        Initialize the time converter with a timezone.
        
        Args:
            timezone (str, optional): Timezone to use for conversions. Default is 'Australia/Melbourne'.
        """
        self.timezone = pytz.timezone(timezone)
        
        # Get peak hours from config
        self.am_peak_start = config.scats_data['peak_hours']['am_start']
        self.am_peak_end = config.scats_data['peak_hours']['am_end']
        self.pm_peak_start = config.scats_data['peak_hours']['pm_start']
        self.pm_peak_end = config.scats_data['peak_hours']['pm_end']
    
    def excel_to_datetime(self, excel_date: Union[float, int], date_system: str = '1900') -> datetime:
        """
        Convert Excel date number to Python datetime.
        
        Excel stores dates as sequential serial numbers. Jan 1, 1900 is serial number 1 in the 1900 system.
        Excel has a leap year bug where it incorrectly treats 1900 as a leap year, so we need to adjust.
        
        Args:
            excel_date (float or int): Excel date number
            date_system (str, optional): Excel date system ('1900' or '1904'). Default is '1900'.
            
        Returns:
            datetime: Python datetime object
        """
        if pd.isna(excel_date):
            return None
        
        # Choose the appropriate base date
        if date_system == '1900':
            # In the 1900 date system, Excel day 1 is 1900-01-01
            # Excel incorrectly treats 1900 as a leap year, adding Feb 29, 1900 which didn't exist
            # For dates after Feb 28, 1900 (Excel date 59), we need to adjust
            
            # Convert to integer and fractional parts
            days = int(excel_date)
            fraction = excel_date - days
            
            # Adjust for Excel's leap year bug
            if days <= 0:
                return None  # Invalid date
            elif days == 60:  # Feb 29, 1900 (doesn't exist)
                # Return Feb 28 plus the time component
                base_date = datetime(1900, 2, 28)
                delta_seconds = timedelta(seconds=round(fraction * 86400))
                return base_date + delta_seconds
            elif days < 60:
                # Dates from Jan 1 to Feb 28, 1900
                base_date = datetime(1900, 1, 1)
                delta_days = timedelta(days=days - 1)  # Subtract 1 because Excel day 1 is Jan 1, 1900
                delta_seconds = timedelta(seconds=round(fraction * 86400))
                return base_date + delta_days + delta_seconds
            else:  # days > 60
                # Dates after Feb 28, 1900 - need to subtract 1 day to account for non-existent Feb 29
                base_date = datetime(1900, 1, 1)
                delta_days = timedelta(days=days - 2)  # Subtract 2: 1 for Excel offset, 1 for leap year bug
                delta_seconds = timedelta(seconds=round(fraction * 86400))
                return base_date + delta_days + delta_seconds
        else:  # '1904' system
            # In the 1904 date system, Excel day 0 is 1904-01-01
            base_date = self.EXCEL_DATE_SYSTEM_1904_BASE
            
            # Convert to datetime
            days = int(excel_date)
            fraction = excel_date - days
            
            delta_days = timedelta(days=days)
            delta_seconds = timedelta(seconds=round(fraction * 86400))
            
            return base_date + delta_days + delta_seconds
    
    def excel_to_datetime_vectorized(self, excel_dates: Union[pd.Series, np.ndarray], 
                                    date_system: str = '1900') -> pd.Series:
        """
        Convert an array or Series of Excel dates to Python datetimes.
        
        Args:
            excel_dates (pd.Series or np.ndarray): Excel date numbers
            date_system (str, optional): Excel date system ('1900' or '1904'). Default is '1900'.
            
        Returns:
            pd.Series: Series of Python datetime objects
        """
        # Convert to pandas Series if it's not already
        if not isinstance(excel_dates, pd.Series):
            excel_dates = pd.Series(excel_dates)
        
        # Apply the conversion function
        return excel_dates.apply(lambda x: self.excel_to_datetime(x, date_system))
    
    def interval_to_time(self, interval: int) -> time:
        """
        Convert a 15-minute interval number (0-95) to time of day.
        
        Args:
            interval (int): Interval number (0-95)
                0 = 00:00-00:15, 1 = 00:15-00:30, ..., 95 = 23:45-00:00
                
        Returns:
            time: Python time object representing the start of the interval
        """
        if interval < 0 or interval >= self.INTERVALS_PER_DAY:
            raise ValueError(f"Interval must be between 0 and {self.INTERVALS_PER_DAY-1}")
        
        # Calculate hours and minutes
        hours = interval // self.INTERVALS_PER_HOUR
        minutes = (interval % self.INTERVALS_PER_HOUR) * self.MINUTES_PER_INTERVAL
        
        return time(hours, minutes)
    
    def time_to_interval(self, t: Union[time, datetime]) -> int:
        """
        Convert a time or datetime to 15-minute interval number (0-95).
        
        Args:
            t (time or datetime): Time to convert
                
        Returns:
            int: Interval number (0-95)
        """
        # Extract hours and minutes
        if isinstance(t, datetime):
            hours = t.hour
            minutes = t.minute
        else:
            hours = t.hour
            minutes = t.minute
        
        # Calculate interval
        interval = hours * self.INTERVALS_PER_HOUR + minutes // self.MINUTES_PER_INTERVAL
        
        return interval
    
    def extract_time_features(self, dt: datetime) -> Dict[str, Any]:
        """
        Extract time features from a datetime object.
        
        Args:
            dt (datetime): Datetime object
                
        Returns:
            dict: Dictionary of time features
        """
        if dt is None:
            return {
                'Date': None,
                'Time': None,
                'Hour': None,
                'Minute': None,
                'DayOfWeek': None,
                'DayOfMonth': None,
                'Month': None,
                'Year': None,
                'IsWeekend': None,
                'IntervalOfDay': None,
                'IsPeakHour': None,
                'IsMorningPeak': None,
                'IsEveningPeak': None,
                'TimeOfDay': None
            }
        
        # Basic time components
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        
        # Derived features
        is_weekend = day_of_week >= 5  # Saturday or Sunday
        interval_of_day = self.time_to_interval(dt)
        
        # Peak hour indicators
        is_morning_peak = self.am_peak_start <= hour <= self.am_peak_end
        is_evening_peak = self.pm_peak_start <= hour <= self.pm_peak_end
        is_peak_hour = is_morning_peak or is_evening_peak
        
        # Time of day category
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        return {
            'Date': dt.date(),
            'Time': dt.time(),
            'Hour': hour,
            'Minute': minute,
            'DayOfWeek': day_of_week,
            'DayOfMonth': dt.day,
            'Month': dt.month,
            'Year': dt.year,
            'IsWeekend': is_weekend,
            'IntervalOfDay': interval_of_day,
            'IsPeakHour': is_peak_hour,
            'IsMorningPeak': is_morning_peak,
            'IsEveningPeak': is_evening_peak,
            'TimeOfDay': time_of_day
        }
    
    def extract_time_features_vectorized(self, datetimes: pd.Series) -> pd.DataFrame:
        """
        Extract time features from a Series of datetime objects.
        
        Args:
            datetimes (pd.Series): Series of datetime objects
                
        Returns:
            pd.DataFrame: DataFrame of time features
        """
        if datetimes.empty:
            return pd.DataFrame()
        
        # Create a new DataFrame
        df = pd.DataFrame()
        
        # Extract basic time components
        df['Date'] = datetimes.dt.date
        df['Time'] = datetimes.dt.time
        df['Hour'] = datetimes.dt.hour
        df['Minute'] = datetimes.dt.minute
        df['DayOfWeek'] = datetimes.dt.dayofweek
        df['DayOfMonth'] = datetimes.dt.day
        df['Month'] = datetimes.dt.month
        df['Year'] = datetimes.dt.year
        
        # Derived features
        df['IsWeekend'] = df['DayOfWeek'] >= 5
        
        # Calculate interval of day
        df['IntervalOfDay'] = df['Hour'] * self.INTERVALS_PER_HOUR + df['Minute'] // self.MINUTES_PER_INTERVAL
        
        # Peak hour indicators
        df['IsMorningPeak'] = (df['Hour'] >= self.am_peak_start) & (df['Hour'] <= self.am_peak_end)
        df['IsEveningPeak'] = (df['Hour'] >= self.pm_peak_start) & (df['Hour'] <= self.pm_peak_end)
        df['IsPeakHour'] = df['IsMorningPeak'] | df['IsEveningPeak']
        
        # Time of day category
        conditions = [
            (df['Hour'] >= 5) & (df['Hour'] < 12),
            (df['Hour'] >= 12) & (df['Hour'] < 17),
            (df['Hour'] >= 17) & (df['Hour'] < 21)
        ]
        choices = ['morning', 'afternoon', 'evening']
        df['TimeOfDay'] = np.select(conditions, choices, default='night')
        
        return df
    
    def is_peak_hour(self, dt: Union[datetime, time]) -> bool:
        """
        Check if a datetime or time is during peak hours.
        
        Args:
            dt (datetime or time): Datetime or time to check
                
        Returns:
            bool: True if during peak hours, False otherwise
        """
        # Extract hour
        if isinstance(dt, datetime):
            hour = dt.hour
        else:
            hour = dt.hour
        
        # Check if hour is in morning or evening peak
        return (self.am_peak_start <= hour <= self.am_peak_end) or (self.pm_peak_start <= hour <= self.pm_peak_end)
    
    def create_time_interval_mapping(self) -> Dict[int, str]:
        """
        Create a mapping of interval numbers to time strings.
        
        Returns:
            dict: Dictionary mapping interval numbers to time strings
        """
        mapping = {}
        for interval in range(self.INTERVALS_PER_DAY):
            start_time = self.interval_to_time(interval)
            
            # Calculate end time
            end_minutes = (start_time.minute + self.MINUTES_PER_INTERVAL) % 60
            end_hour = (start_time.hour + (start_time.minute + self.MINUTES_PER_INTERVAL) // 60) % 24
            end_time = time(end_hour, end_minutes)
            
            # Format as strings
            start_str = start_time.strftime('%H:%M')
            end_str = end_time.strftime('%H:%M')
            
            mapping[interval] = f"{start_str}-{end_str}"
        
        return mapping
    
    def add_time_features_to_df(self, df: pd.DataFrame, datetime_col: str = 'DateTime') -> pd.DataFrame:
        """
        Add time features to a DataFrame with a datetime column.
        
        Args:
            df (pd.DataFrame): DataFrame with a datetime column
            datetime_col (str, optional): Name of the datetime column. Default is 'DateTime'.
                
        Returns:
            pd.DataFrame: DataFrame with added time features
        """
        if datetime_col not in df.columns:
            logger.error(f"Column '{datetime_col}' not found in DataFrame")
            return df
        
        # Ensure the column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
            except Exception as e:
                logger.error(f"Could not convert '{datetime_col}' to datetime: {str(e)}")
                return df
        
        # Extract time features
        time_features = self.extract_time_features_vectorized(df[datetime_col])
        
        # Add features to the original DataFrame
        # Only add columns that don't already exist
        for col in time_features.columns:
            if col not in df.columns:
                df[col] = time_features[col]
        
        return df


def convert_excel_date(excel_date: Union[float, int], date_system: str = '1900') -> datetime:
    """
    Convenience function to convert an Excel date to a Python datetime.
    
    Args:
        excel_date (float or int): Excel date number
        date_system (str, optional): Excel date system ('1900' or '1904'). Default is '1900'.
        
    Returns:
        datetime: Python datetime object
    """
    converter = SCATSTimeConverter()
    return converter.excel_to_datetime(excel_date, date_system)


def extract_time_features(dt: datetime) -> Dict[str, Any]:
    """
    Convenience function to extract time features from a datetime object.
    
    Args:
        dt (datetime): Datetime object
            
    Returns:
        dict: Dictionary of time features
    """
    converter = SCATSTimeConverter()
    return converter.extract_time_features(dt)


def add_time_features(df: pd.DataFrame, datetime_col: str = 'DateTime') -> pd.DataFrame:
    """
    Convenience function to add time features to a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with a datetime column
        datetime_col (str, optional): Name of the datetime column. Default is 'DateTime'.
            
    Returns:
        pd.DataFrame: DataFrame with added time features
    """
    converter = SCATSTimeConverter()
    return converter.add_time_features_to_df(df, datetime_col)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SCATS time conversion utilities')
    parser.add_argument('--excel-date', type=float, help='Excel date to convert')
    parser.add_argument('--interval', type=int, help='Interval number to convert to time')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    args = parser.parse_args()
    
    # Create converter
    converter = SCATSTimeConverter()
    
    if args.test_all:
        # Test Excel date conversion
        test_dates = [1, 59, 60, 61, 43466.5]  # Jan 1 1900, Feb 28 1900, Mar 1 1900, Mar 2 1900, Oct 15 2018 12:00
        for excel_date in test_dates:
            dt = converter.excel_to_datetime(excel_date)
            print(f"Excel date {excel_date} -> {dt}")
        
        # Test interval conversion
        test_intervals = [0, 4, 32, 68, 95]  # 00:00, 01:00, 08:00, 17:00, 23:45
        for interval in test_intervals:
            t = converter.interval_to_time(interval)
            print(f"Interval {interval} -> {t}")
            
            # Test round trip
            interval_back = converter.time_to_interval(t)
            print(f"  Round trip: {t} -> {interval_back}")
        
        # Test time features
        test_datetimes = [
            datetime(2006, 10, 2, 8, 30),  # Monday morning peak
            datetime(2006, 10, 3, 17, 45),  # Tuesday evening peak
            datetime(2006, 10, 7, 12, 0),   # Saturday noon
            datetime(2006, 10, 8, 23, 15)   # Sunday night
        ]
        for dt in test_datetimes:
            features = converter.extract_time_features(dt)
            print(f"\nTime features for {dt}:")
            for key, value in features.items():
                print(f"  {key}: {value}")
        
        # Test interval mapping
        mapping = converter.create_time_interval_mapping()
        print("\nInterval mapping (sample):")
        for interval in [0, 4, 32, 68, 95]:
            print(f"  {interval}: {mapping[interval]}")
        
    elif args.excel_date is not None:
        # Convert Excel date
        dt = converter.excel_to_datetime(args.excel_date)
        print(f"Excel date {args.excel_date} -> {dt}")
        
        # Extract time features
        features = converter.extract_time_features(dt)
        print("\nTime features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
            
    elif args.interval is not None:
        # Convert interval to time
        t = converter.interval_to_time(args.interval)
        print(f"Interval {args.interval} -> {t}")
        
        # Check if it's a peak hour
        is_peak = converter.is_peak_hour(t)
        print(f"Is peak hour: {is_peak}")
    
    else:
        # Run a simple demo
        print("SCATS Time Converter Demo")
        print("========================")
        
        # Demo Excel date conversion
        excel_date = 39356.5  # October 15, 2007 12:00
        dt = converter.excel_to_datetime(excel_date)
        print(f"\nExcel date {excel_date} -> {dt}")
        
        # Demo time features
        features = converter.extract_time_features(dt)
        print("\nTime features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # Demo interval mapping
        print("\nSample 15-minute intervals:")
        for hour in [0, 8, 12, 17, 23]:
            for minute in [0, 15, 30, 45]:
                t = time(hour, minute)
                interval = converter.time_to_interval(t)
                print(f"  {t} -> Interval {interval}")
        
        print("\nRun with --test-all for more comprehensive tests")
