#!/usr/bin/env python3
"""
TBRGS Travel Time Calculator Module

This module implements the calculation of travel times between SCATS sites
based on predicted traffic flows, geographic distances, and intersection delays.

It provides functionality to:
1. Calculate geographic distances between SCATS sites
2. Convert predicted traffic flows to travel speeds
3. Calculate travel times based on distance and speed
4. Include intersection delays in travel time calculations
5. Handle uncertainty in travel time predictions
"""

import os
import math
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from app.config.config import config
from app.core.logging import TBRGSLogger
from app.core.integration.flow_speed_converter import flow_to_speed, get_traffic_regime

# Initialize logger
logger = TBRGSLogger.get_logger("integration.travel_time")

class TravelTimeCalculator:
    """
    Calculator class for travel times between SCATS sites.
    
    This class calculates travel times based on:
    - Geographic distances between SCATS sites
    - Predicted traffic flows converted to speeds
    - Intersection delays
    
    Attributes:
        site_reference_df (pd.DataFrame): DataFrame containing SCATS site reference data
        site_locations (dict): Dictionary mapping SCATS_ID to (latitude, longitude) coordinates
        intersection_delay (float): Delay at each controlled intersection in seconds
        default_edge_cost (float): Default travel time when no data is available
        site_connections (dict): Dictionary mapping site pairs to connection information
    """
    
    def __init__(self, site_reference_path=None):
        """
        Initialize the travel time calculator.
        
        Args:
            site_reference_path (str, optional): Path to the SCATS site reference CSV file.
                If None, uses the path from config.
        """
        # Load parameters from config
        self.intersection_delay = config.traffic_conversion['intersection_delay']  # seconds
        self.default_edge_cost = config.routing['default_edge_cost']  # seconds
        
        # Load site reference data
        if site_reference_path is None:
            site_reference_path = config.processed_data['site_reference']
        
        self._load_site_reference(site_reference_path)
        self._build_site_connections()
        
        logger.info(f"Travel Time Calculator initialized with {len(self.site_locations)} SCATS sites")
    
    def _load_site_reference(self, site_reference_path):
        """
        Load SCATS site reference data from CSV file.
        
        Args:
            site_reference_path (str): Path to the SCATS site reference CSV file
        """
        try:
            # Load the site reference data
            self.site_reference_df = pd.read_csv(site_reference_path)
            
            # Create a dictionary mapping SCATS_ID to (latitude, longitude) coordinates
            # We'll use the average coordinates for each SCATS site to get a central point
            self.site_locations = {}
            
            # Group by SCATS_ID and calculate average coordinates
            for scats_id, group in self.site_reference_df.groupby('SCATS_ID'):
                # Calculate average latitude and longitude for this site
                avg_lat = group['Latitude'].astype(float).mean()
                avg_lon = group['Longitude'].astype(float).mean()
                
                # Store the average coordinates
                self.site_locations[scats_id] = (avg_lat, avg_lon)
                logger.debug(f"Loaded site {scats_id} at coordinates ({avg_lat:.6f}, {avg_lon:.6f})")
            
            logger.info(f"Loaded {len(self.site_locations)} unique SCATS sites from reference data")
        except Exception as e:
            logger.error(f"Error loading site reference data: {e}")
            # Initialize with empty data
            self.site_reference_df = pd.DataFrame()
            self.site_locations = {}
            
            # For testing purposes, add some default sites if the file can't be loaded
            self._add_default_test_sites()
    
    def _add_default_test_sites(self):
        """
        Add default test sites with hardcoded coordinates for testing purposes.
        This is used when the site reference file cannot be loaded.
        """
        # Add some default test sites with hardcoded coordinates
        test_sites = {
            # SCATS ID: (latitude, longitude)
            '0970': (-37.86730, 145.09151),  # WARRIGAL_RD/HIGH STREET_RD
            '2000': (-37.85192, 145.09432),  # WARRIGAL_RD/TOORAK_RD
            '3002': (-37.81514, 145.02655),  # DENMARK_ST/BARKERS_RD
            '4035': (-37.81830, 145.05811),  # BARKERS_RD/BURKE_RD
        }
        
        # Add the test sites to the site locations dictionary
        self.site_locations.update(test_sites)
        logger.warning(f"Added {len(test_sites)} default test sites with hardcoded coordinates")
    
    def _build_site_connections(self):
        """
        Build a dictionary of site connections based on the site reference data.
        This identifies which sites are directly connected to each other.
        """
        self.site_connections = {}
        
        # For now, we'll assume all sites can potentially connect to each other
        # In a real-world implementation, this would be based on the actual road network
        # But for the simplified model, we'll calculate distances between all sites
        
        unique_sites = list(self.site_locations.keys())
        logger.info(f"Building connections between {len(unique_sites)} unique sites")
        
        for i, site1 in enumerate(unique_sites):
            for site2 in unique_sites[i+1:]:
                # Calculate distance between sites
                distance = self.calculate_distance(site1, site2)
                
                # Store connection information
                connection_key = f"{site1}-{site2}"
                self.site_connections[connection_key] = {
                    'distance': distance,  # in kilometers
                    'direct_connection': True  # Simplified assumption
                }
                
                # Store reverse connection as well
                reverse_key = f"{site2}-{site1}"
                self.site_connections[reverse_key] = {
                    'distance': distance,  # in kilometers
                    'direct_connection': True  # Simplified assumption
                }
        
        logger.info(f"Built {len(self.site_connections)} site connections")
    
    def calculate_distance(self, site1_id, site2_id):
        """
        Calculate the geographic distance between two SCATS sites.
        
        Args:
            site1_id (str): SCATS ID of the first site
            site2_id (str): SCATS ID of the second site
            
        Returns:
            float: Distance in kilometers between the sites
        """
        # Get coordinates for both sites
        if site1_id not in self.site_locations or site2_id not in self.site_locations:
            logger.warning(f"Missing site coordinates for {site1_id} or {site2_id}")
            return 0.0
        
        site1_coords = self.site_locations[site1_id]
        site2_coords = self.site_locations[site2_id]
        
        # Check if coordinates are valid
        if not all(isinstance(coord, (int, float)) for coord in site1_coords + site2_coords):
            logger.warning(f"Invalid coordinates for sites {site1_id} or {site2_id}")
            return 0.0
        
        # If sites are the same, return 0
        if site1_id == site2_id:
            return 0.0
        
        # Calculate geodesic distance (as the crow flies)
        try:
            distance = geodesic(site1_coords, site2_coords).kilometers
            
            # Apply a road network factor to account for the fact that roads aren't straight lines
            # Typical values range from 1.2 to 1.5 depending on the road network
            road_factor = 1.3
            adjusted_distance = distance * road_factor
            
            # Ensure we don't return extremely small distances that might be due to rounding errors
            return max(0.1, adjusted_distance) if adjusted_distance > 0 else 0.1
        except Exception as e:
            logger.error(f"Error calculating distance between {site1_id} and {site2_id}: {e}")
            return 0.1  # Return a small default distance
    
    def calculate_travel_time(self, site1_id, site2_id, flow=None, uncertainty=0.0):
        """
        Calculate the travel time between two SCATS sites.
        
        Args:
            site1_id (str): SCATS ID of the origin site
            site2_id (str): SCATS ID of the destination site
            flow (float, optional): Predicted traffic flow in vehicles/hour.
                If None, uses default edge cost.
            uncertainty (float, optional): Uncertainty factor (0.0-1.0) for the prediction.
                Higher values increase the travel time estimate.
                
        Returns:
            dict: Travel time information including:
                - travel_time: Total travel time in seconds
                - distance: Distance in kilometers
                - speed: Travel speed in km/h
                - regime: Traffic regime (green, yellow, red)
                - uncertainty: Uncertainty factor applied
        """
        # Get connection information
        connection_key = f"{site1_id}-{site2_id}"
        
        if connection_key not in self.site_connections:
            logger.warning(f"No connection information for {connection_key}")
            return {
                'travel_time': self.default_edge_cost,
                'distance': 0.0,
                'speed': 0.0,
                'regime': 'unknown',
                'uncertainty': uncertainty
            }
        
        # Get distance between sites
        distance = self.site_connections[connection_key]['distance']
        
        # If no flow provided, use a reasonable default speed (40 km/h)
        if flow is None:
            default_speed = 40.0  # km/h - reasonable urban speed
            travel_time = (distance / default_speed) * 3600  # Convert to seconds
            travel_time += self.intersection_delay  # Add intersection delay
            
            # Apply uncertainty
            uncertainty_factor = 1.0 + uncertainty
            adjusted_travel_time = travel_time * uncertainty_factor
            
            return {
                'travel_time': adjusted_travel_time,
                'distance': distance,
                'speed': default_speed,
                'regime': 'default',
                'uncertainty': uncertainty
            }
        
        # Convert flow to speed
        speed = flow_to_speed(flow)
        regime = get_traffic_regime(flow)
        
        # Calculate base travel time (hours) = distance (km) / speed (km/h)
        # Convert to seconds
        if speed > 0:
            travel_time_seconds = (distance / speed) * 3600
        else:
            # If speed is 0 (extreme congestion), use a reasonable default
            travel_time_seconds = (distance / 10.0) * 3600  # Assume 10 km/h in heavy congestion
            
        # Cap maximum travel time per segment to a reasonable value (15 minutes)
        max_segment_time = 15 * 60  # 15 minutes in seconds
        travel_time_seconds = min(travel_time_seconds, max_segment_time)
        
        # Add intersection delay
        travel_time_seconds += self.intersection_delay
        
        # Apply uncertainty factor
        # Higher uncertainty increases the travel time estimate
        uncertainty_factor = 1.0 + uncertainty
        adjusted_travel_time = travel_time_seconds * uncertainty_factor
        
        return {
            'travel_time': adjusted_travel_time,
            'distance': distance,
            'speed': speed,
            'regime': regime,
            'uncertainty': uncertainty
        }
    
    def calculate_route_travel_time(self, route, flows=None, uncertainty=0.0):
        """
        Calculate the total travel time for a route consisting of multiple SCATS sites.
        
        Args:
            route (list): List of SCATS site IDs representing the route
            flows (dict, optional): Dictionary mapping site pairs to traffic flows.
                If None, uses default edge costs.
            uncertainty (float, optional): Uncertainty factor (0.0-1.0) for the prediction.
                
        Returns:
            dict: Route travel time information including:
                - total_time: Total travel time in seconds
                - segment_times: List of travel times for each segment
                - total_distance: Total distance in kilometers
                - average_speed: Average speed in km/h
                - segments: Detailed information for each route segment
        """
        if len(route) < 2:
            logger.warning("Route must contain at least 2 sites")
            return {
                'total_time': 0.0,
                'segment_times': [],
                'total_distance': 0.0,
                'average_speed': 0.0,
                'segments': []
            }
        
        total_time = 0.0
        total_distance = 0.0
        segment_times = []
        segments = []
        
        # Calculate travel time for each segment of the route
        for i in range(len(route) - 1):
            origin = route[i]
            destination = route[i+1]
            segment_key = f"{origin}-{destination}"
            
            # Get flow for this segment if available
            segment_flow = None
            if flows is not None and segment_key in flows:
                segment_flow = flows[segment_key]
            
            # Calculate travel time for this segment
            segment_info = self.calculate_travel_time(
                origin, destination, flow=segment_flow, uncertainty=uncertainty
            )
            
            # Add to totals
            segment_time = segment_info['travel_time']
            segment_times.append(segment_time)
            total_time += segment_time
            total_distance += segment_info['distance']
            
            # Store segment information
            segments.append({
                'origin': origin,
                'destination': destination,
                'travel_time': segment_time,
                'distance': segment_info['distance'],
                'speed': segment_info['speed'],
                'regime': segment_info['regime']
            })
        
        # Calculate average speed
        if total_time > 0:
            # Convert time from seconds to hours for speed calculation
            average_speed = total_distance / (total_time / 3600)
        else:
            average_speed = 0.0
        
        return {
            'total_time': total_time,
            'segment_times': segment_times,
            'total_distance': total_distance,
            'average_speed': average_speed,
            'segments': segments
        }
    
    def get_travel_time_with_prediction_interval(self, site1_id, site2_id, flow, confidence=0.95):
        """
        Calculate travel time with prediction intervals to account for uncertainty.
        
        Args:
            site1_id (str): SCATS ID of the origin site
            site2_id (str): SCATS ID of the destination site
            flow (float): Predicted traffic flow in vehicles/hour
            confidence (float, optional): Confidence level for prediction interval (0.0-1.0)
                
        Returns:
            dict: Travel time information including:
                - expected: Expected travel time in seconds
                - lower_bound: Lower bound of travel time prediction
                - upper_bound: Upper bound of travel time prediction
                - confidence: Confidence level used
        """
        # Calculate base travel time
        base_result = self.calculate_travel_time(site1_id, site2_id, flow)
        expected_time = base_result['travel_time']
        
        # Define uncertainty based on traffic regime
        # Higher uncertainty for congested regimes
        regime = base_result['regime']
        if regime == 'green':
            uncertainty_factor = 0.1  # 10% uncertainty for free-flowing traffic
        elif regime == 'yellow':
            uncertainty_factor = 0.2  # 20% uncertainty for approaching congestion
        elif regime == 'red':
            uncertainty_factor = 0.3  # 30% uncertainty for congested traffic
        else:
            uncertainty_factor = 0.2  # Default uncertainty
        
        # Adjust uncertainty based on confidence level
        # For 95% confidence, we use approximately 2 standard deviations
        if confidence >= 0.95:
            z_score = 2.0
        elif confidence >= 0.90:
            z_score = 1.645
        elif confidence >= 0.80:
            z_score = 1.28
        else:
            z_score = 1.0
        
        # Calculate prediction interval
        margin = expected_time * uncertainty_factor * z_score
        lower_bound = max(0, expected_time - margin)
        upper_bound = expected_time + margin
        
        return {
            'expected': expected_time,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence
        }
    
    def format_travel_time(self, seconds):
        """
        Format travel time in seconds to a human-readable string.
        
        Args:
            seconds (float): Travel time in seconds
            
        Returns:
            str: Formatted travel time (e.g., "5m 30s" or "1h 15m")
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"


# Create a singleton instance for easy import
calculator = TravelTimeCalculator()


def calculate_travel_time(origin, destination, flow=None, uncertainty=0.0):
    """
    Convenience function to calculate travel time between two SCATS sites.
    
    Args:
        origin (str): SCATS ID of the origin site
        destination (str): SCATS ID of the destination site
        flow (float, optional): Predicted traffic flow in vehicles/hour
        uncertainty (float, optional): Uncertainty factor (0.0-1.0)
        
    Returns:
        dict: Travel time information
    """
    return calculator.calculate_travel_time(origin, destination, flow, uncertainty)


def calculate_route_travel_time(route, flows=None, uncertainty=0.0):
    """
    Convenience function to calculate travel time for a route.
    
    Args:
        route (list): List of SCATS site IDs representing the route
        flows (dict, optional): Dictionary mapping site pairs to traffic flows
        uncertainty (float, optional): Uncertainty factor (0.0-1.0)
        
    Returns:
        dict: Route travel time information
    """
    return calculator.calculate_route_travel_time(route, flows, uncertainty)


def get_travel_time_with_prediction_interval(origin, destination, flow, confidence=0.95):
    """
    Convenience function to get travel time with prediction intervals.
    
    Args:
        origin (str): SCATS ID of the origin site
        destination (str): SCATS ID of the destination site
        flow (float): Predicted traffic flow in vehicles/hour
        confidence (float, optional): Confidence level (0.0-1.0)
        
    Returns:
        dict: Travel time information with prediction intervals
    """
    return calculator.get_travel_time_with_prediction_interval(
        origin, destination, flow, confidence
    )


if __name__ == "__main__":
    # Create a standalone test script that doesn't depend on external files
    import logging
    import sys
    import os
    from pathlib import Path
    
    # Suppress all logging except warnings and errors
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    print("=" * 80)
    print("TBRGS Travel Time Calculator Test")
    print("=" * 80)
    
    # Create a test calculator with hardcoded test sites
    test_calculator = TravelTimeCalculator()
    
    # If the site locations dictionary is empty, add test sites manually
    if not test_calculator.site_locations:
        test_calculator._add_default_test_sites()
    
    # Print site coordinates for verification
    print("\nSite Coordinates:")
    print("-" * 60)
    print(f"{'SCATS ID':8} | {'Latitude':15} | {'Longitude':15}")
    print("-" * 60)
    test_sites = ['0970', '2000', '3002', '4035']
    for site_id in test_sites:
        if site_id in test_calculator.site_locations:
            coords = test_calculator.site_locations[site_id]
            print(f"{site_id:8} | {coords[0]:15.6f} | {coords[1]:15.6f}")
    
    # Manually build site connections for testing
    test_calculator.site_connections = {}
    for i, site1 in enumerate(test_sites):
        for j, site2 in enumerate(test_sites):
            if i != j:
                # Calculate distance between sites
                distance = test_calculator.calculate_distance(site1, site2)
                
                # Store connection information
                connection_key = f"{site1}-{site2}"
                test_calculator.site_connections[connection_key] = {
                    'distance': distance,  # in kilometers
                    'direct_connection': True  # Simplified assumption
                }
    
    # Test distance calculation
    print("\nDistance Calculation:")
    print("-" * 50)
    print(f"{'Origin':8} | {'Destination':8} | {'Distance (km)':15}")
    print("-" * 50)
    for i, site1 in enumerate(test_sites):
        for site2 in test_sites[i+1:]:
            distance = test_calculator.calculate_distance(site1, site2)
            print(f"{site1:8} | {site2:8} | {distance:15.2f}")
    
    # Test travel time calculation with different flows
    print("\nTravel Time Calculation:")
    print("-" * 70)
    print(f"{'Flow (veh/h)':12} | {'Travel Time':12} | {'Speed (km/h)':12} | {'Regime':10}")
    print("-" * 70)
    test_flows = [100, 351, 1000, 1500]
    for flow in test_flows:
        result = test_calculator.calculate_travel_time('0970', '2000', flow)
        formatted_time = test_calculator.format_travel_time(result['travel_time'])
        print(f"{flow:12} | {formatted_time:12} | {result['speed']:12.2f} | {result['regime']:10}")
    
    # Test route travel time calculation
    test_route = ['0970', '2000', '3002', '4035']
    print("\nRoute Travel Time Calculation:")
    # Create sample flows for testing
    test_route_flows = {
        '0970-2000': 300,
        '2000-3002': 800,
        '3002-4035': 1200
    }
    route_result = test_calculator.calculate_route_travel_time(test_route, test_route_flows)
    print(f"Route: {' → '.join(test_route)}")
    print(f"Total Distance: {route_result['total_distance']:.2f} km")
    print(f"Total Travel Time: {test_calculator.format_travel_time(route_result['total_time'])}")
    print(f"Average Speed: {route_result['average_speed']:.2f} km/h")
    
    print("\nSegment Details:")
    print("-" * 80)
    print(f"{'Segment':15} | {'Time':10} | {'Distance':10} | {'Speed':10} | {'Regime':10}")
    print("-" * 80)
    for segment in route_result['segments']:
        segment_name = f"{segment['origin']}-{segment['destination']}"
        time = test_calculator.format_travel_time(segment['travel_time'])
        print(f"{segment_name:15} | {time:10} | {segment['distance']:10.2f} | {segment['speed']:10.2f} | {segment['regime']:10}")
    
    # Test prediction intervals
    print("\nTravel Time with Prediction Intervals:")
    print("-" * 70)
    print(f"{'Flow (veh/h)':12} | {'Expected':12} | {'Lower Bound':12} | {'Upper Bound':12}")
    print("-" * 70)
    for flow in [300, 1000, 1500]:
        interval = test_calculator.get_travel_time_with_prediction_interval('0970', '2000', flow)
        expected = test_calculator.format_travel_time(interval['expected'])
        lower = test_calculator.format_travel_time(interval['lower_bound'])
        upper = test_calculator.format_travel_time(interval['upper_bound'])
        print(f"{flow:12} | {expected:12} | {lower:12} | {upper:12}")
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
