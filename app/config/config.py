#!/usr/bin/env python3
"""
TBRGS Configuration Module

This module provides basic configuration settings for the Traffic-Based Route Guidance System (TBRGS).
It includes essential paths and parameters for the current stage of development.
"""

import os
from pathlib import Path

class TBRGSConfig:
    """
    Basic configuration class for the Traffic-Based Route Guidance System (TBRGS).
    
    This class provides centralized configuration for:
    - File paths for data
    - Basic project settings
    - Traffic flow to travel time conversion parameters
    - Data processing parameters
    """
    
    def __init__(self):
        # Base directories
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.app_dir = self.project_root / 'app'
        self.data_dir = self.project_root.parent / 'dataset'
        
        # Initialize configurations
        self._init_data_paths()
        self._init_traffic_conversion_params()
        self._init_route_params()
        self._init_data_processing_params()
    
    def _init_data_paths(self):
        """
        Initialize data file paths for raw and processed data.
        """
        # Raw data paths
        self.raw_data = {
            'scats_excel': str(self.app_dir / 'dataset' / 'raw' / 'Scats Data October 2006.xls')
        }
        
        # Processed data paths (to be created)
        self.processed_data = {
            'scats_complete': str(self.app_dir / 'dataset' / 'processed' / 'scats_traffic.csv'),
            'site_reference': str(self.app_dir / 'dataset' / 'processed' / 'scats_site_reference.csv'),
            'train_data': str(self.app_dir / 'dataset' / 'processed' / 'train_data.csv'),
            'val_data': str(self.app_dir / 'dataset' / 'processed' / 'val_data.csv'),
            'test_data': str(self.app_dir / 'dataset' / 'processed' / 'test_data.csv')
        }
    
    def _init_route_params(self):
        """
        Initialize route calculation parameters.
        """
        self.routing = {
            'algorithms': ['astar', 'bfs', 'dfs', 'gbfs', 'iddfs', 'bdwa'],
            'default_edge_cost': 60,  # Default edge cost in seconds when no data available
            'max_segment_time': 15 * 60  # Maximum time for a single segment (15 minutes)
        }
    
    def _init_traffic_conversion_params(self):
        """
        Initialize traffic flow to travel time conversion parameters.
        """
        self.traffic_conversion = {
            # Quadratic equation parameters: flow = a * (speed)^2 + b * (speed)
            'a': -1.4648375,
            'b': 93.75,
            'speed_limit': 60,  # km/h
            'flow_threshold': 351,  # vehicles/hour for speed limit cap
            'intersection_delay': 30  # seconds per controlled intersection
        }
    
    def _init_data_processing_params(self):
        """
        Initialize data processing parameters.
        """
        # SCATS data structure parameters
        self.scats_data = {
            'expected_col_count': 106,
            'metadata_cols': 10,
            'traffic_cols': 96,
            'metadata_col_names': [
                'SCATS_ID', 'Location', 'CD_MELWAY', 'Latitude', 'Longitude', 
                'VR_Internal1', 'VR_Internal2', 'VR_Internal3', 'Survey_Type', 'Excel_Date'
            ],
            'expected_sites': [
                '0970', '2000', '2200', '2820', '2825', '2827', '2846', 
                '3001', '3002', '3120', '3122', '3126', '3127', '3180', 
                '3662', '3682', '3685', '3804', '3812', '4030', '4032', 
                '4034', '4035', '4040', '4043', '4051', '4057', '4063', 
                '4262', '4263', '4264', '4266', '4270', '4272', '4273', 
                '4321', '4324', '4335', '4812', '4821'
            ],
            'time_splits': {
                'train_end': '2006-10-21',
                'val_end': '2006-10-26'
                # test is everything after val_end
            },
            'peak_hours': {
                'am_start': 7,  # 7:00 AM
                'am_end': 9,    # 9:00 AM
                'pm_start': 17, # 5:00 PM
                'pm_end': 19    # 7:00 PM
            }
        }

# Create a singleton instance
config = TBRGSConfig()
