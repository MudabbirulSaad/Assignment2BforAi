#!/usr/bin/env python3
"""
TBRGS Time-Based Traffic Prediction Module

This module implements time-based traffic predictions for the SCATS router.
It provides functions to predict traffic flows based on the time of day.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Add the parent directory to the path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.core.logging import TBRGSLogger
from app.core.integration.flow_speed_converter import flow_to_speed

# Initialize logger
logger = TBRGSLogger.get_logger("integration.time_predictions")

def get_time_based_flow(hour: int) -> tuple:
    """
    Get traffic flow based on time of day.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        tuple: (base_flow, time_period)
    """
    if 7 <= hour < 9:
        # Morning peak (higher flow)
        base_flow = 500  # Vehicles per hour
        time_period = "Morning Peak (7-9 AM)"
    elif 16 <= hour < 19:
        # Evening peak (highest flow)
        base_flow = 600  # Vehicles per hour
        time_period = "Evening Peak (4-7 PM)"
    elif 9 <= hour < 16:
        # Midday (moderate flow)
        base_flow = 300  # Vehicles per hour
        time_period = "Midday (9 AM - 4 PM)"
    else:
        # Off-peak (low flow)
        base_flow = 150  # Vehicles per hour
        time_period = "Night/Off-Peak"
    
    return base_flow, time_period

def predict_traffic_flows(site_ids: list, prediction_time: Optional[datetime] = None) -> Dict[str, float]:
    """
    Predict traffic flows for all SCATS sites at the specified time.
    
    Args:
        site_ids: List of SCATS site IDs
        prediction_time: Time for which to predict traffic (default: current time)
        
    Returns:
        Dict[str, float]: Dictionary mapping SCATS IDs to predicted flows
    """
    if prediction_time is None:
        prediction_time = datetime.now()
    
    # Get hour of day (0-23)
    hour = prediction_time.hour
    
    # Get base flow and time period
    base_flow, time_period = get_time_based_flow(hour)
    
    # Log the time period and base flow being used
    logger.info(f"Using {time_period} traffic pattern for {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Base flow: {base_flow} vehicles/hour")
    
    # Initialize predictions dictionary
    predictions = {}
    
    # Generate predictions for each site
    for site_id in site_ids:
        # Add some randomness based on site ID to make each site unique
        # Use a consistent seed based on site ID for reproducibility
        site_seed = sum(ord(c) for c in site_id)
        np.random.seed(site_seed)
        flow = base_flow * np.random.uniform(0.8, 1.2)
        
        # Store prediction
        predictions[site_id] = flow
    
    # Log a sample of predictions
    sample_sites = list(predictions.keys())[:3] if len(predictions) > 3 else list(predictions.keys())
    for site_id in sample_sites:
        flow = predictions[site_id]
        speed = flow_to_speed(flow)
        logger.info(f"Sample prediction - Site {site_id}: Flow={flow:.1f} veh/h, Speed={speed:.1f} km/h")
    
    return predictions

def calculate_travel_time(distance: float, flow: float) -> float:
    """
    Calculate travel time based on distance and traffic flow.
    
    Args:
        distance: Distance in kilometers
        flow: Traffic flow in vehicles per hour
        
    Returns:
        float: Travel time in seconds
    """
    # Convert flow to speed
    speed = flow_to_speed(flow)
    
    # Calculate travel time (seconds) with 30 sec intersection delay
    travel_time = (distance / speed) * 3600 + 30
    
    return travel_time
