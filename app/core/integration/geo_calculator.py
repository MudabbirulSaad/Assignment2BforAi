#!/usr/bin/env python3
"""
TBRGS Geographic Distance Calculator Module

This module implements various methods for calculating geographic distances
between coordinates, with a focus on accurate distance calculations for
the Traffic-Based Route Guidance System.

It provides functionality to:
1. Calculate Euclidean distance between coordinates
2. Calculate Haversine distance for more accurate geographic distances
3. Validate geographic coordinates
4. Cache distance calculations for performance
5. Calculate distances for entire routes
"""

import math
import numpy as np
from functools import lru_cache
from typing import Tuple, List, Dict, Union, Optional
from app.config.config import config
from app.core.logging import TBRGSLogger

# Initialize logger
logger = TBRGSLogger.get_logger("integration.geo_calculator")

# Type aliases for clarity
Coordinate = Tuple[float, float]  # (latitude, longitude)
Route = List[Coordinate]  # List of coordinates representing a route

class GeoCalculator:
    """
    Geographic distance calculator for TBRGS.
    
    This class provides methods to calculate distances between geographic coordinates
    using different algorithms, with caching for performance.
    
    Attributes:
        earth_radius (float): Earth radius in kilometers
        road_factor (float): Factor to account for road network vs straight-line distance
        cache_size (int): Size of the LRU cache for distance calculations
    """
    
    def __init__(self, earth_radius: float = 6371.0, road_factor: float = 1.3, cache_size: int = 1024):
        """
        Initialize the geographic calculator.
        
        Args:
            earth_radius (float): Earth radius in kilometers (default: 6371.0 km)
            road_factor (float): Factor to account for road network vs straight-line distance
                                 (default: 1.3, typical values range from 1.2 to 1.5)
            cache_size (int): Size of the LRU cache for distance calculations (default: 1024)
        """
        self.earth_radius = earth_radius
        self.road_factor = road_factor
        self.cache_size = cache_size
        
        logger.info(f"Geographic Calculator initialized with earth radius={earth_radius} km, " +
                   f"road factor={road_factor}, cache size={cache_size}")
    
    def validate_coordinates(self, coords: Coordinate) -> bool:
        """
        Validate if the given coordinates are valid geographic coordinates.
        
        Args:
            coords (Coordinate): Tuple of (latitude, longitude)
            
        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        if not isinstance(coords, tuple) or len(coords) != 2:
            logger.warning(f"Invalid coordinate format: {coords}, expected (latitude, longitude) tuple")
            return False
        
        lat, lon = coords
        
        # Check if coordinates are numeric
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            logger.warning(f"Non-numeric coordinates: ({lat}, {lon})")
            return False
        
        # Check if coordinates are within valid ranges
        if lat < -90 or lat > 90:
            logger.warning(f"Latitude {lat} out of range [-90, 90]")
            return False
        
        if lon < -180 or lon > 180:
            logger.warning(f"Longitude {lon} out of range [-180, 180]")
            return False
        
        return True
    
    def euclidean_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """
        Calculate the Euclidean distance between two coordinates.
        
        This is a simple straight-line distance calculation that doesn't account for
        Earth's curvature. It's faster but less accurate for large distances.
        
        Args:
            coord1 (Coordinate): First coordinate (latitude, longitude)
            coord2 (Coordinate): Second coordinate (latitude, longitude)
            
        Returns:
            float: Euclidean distance in kilometers
        """
        # Validate coordinates
        if not self.validate_coordinates(coord1) or not self.validate_coordinates(coord2):
            return 0.0
        
        # Convert to radians for calculation
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        # Calculate x, y, z coordinates on a unit sphere
        x1 = math.cos(lat1) * math.cos(lon1)
        y1 = math.cos(lat1) * math.sin(lon1)
        z1 = math.sin(lat1)
        
        x2 = math.cos(lat2) * math.cos(lon2)
        y2 = math.cos(lat2) * math.sin(lon2)
        z2 = math.sin(lat2)
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # Convert to kilometers (chord length * earth radius)
        distance_km = distance * self.earth_radius
        
        return distance_km
    
    @lru_cache(maxsize=1024)
    def _cached_haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Cached implementation of the Haversine formula.
        
        This private method is wrapped with lru_cache for performance.
        
        Args:
            lat1, lon1, lat2, lon2: Coordinates in radians
            
        Returns:
            float: Haversine distance in kilometers
        """
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Distance in kilometers
        distance = self.earth_radius * c
        
        return distance
    
    def haversine_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """
        Calculate the Haversine distance between two coordinates.
        
        This is a more accurate distance calculation that accounts for Earth's curvature.
        It's the preferred method for geographic distance calculations.
        
        Args:
            coord1 (Coordinate): First coordinate (latitude, longitude)
            coord2 (Coordinate): Second coordinate (latitude, longitude)
            
        Returns:
            float: Haversine distance in kilometers
        """
        # Validate coordinates
        if not self.validate_coordinates(coord1) or not self.validate_coordinates(coord2):
            return 0.0
        
        # Convert to radians for calculation
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        # Use cached implementation
        distance = self._cached_haversine(lat1, lon1, lat2, lon2)
        
        return distance
    
    def road_distance(self, coord1: Coordinate, coord2: Coordinate, method: str = 'haversine') -> float:
        """
        Calculate the estimated road distance between two coordinates.
        
        This applies a road factor to the geographic distance to account for the fact that
        roads are not straight lines. The road factor is typically between 1.2 and 1.5.
        
        Args:
            coord1 (Coordinate): First coordinate (latitude, longitude)
            coord2 (Coordinate): Second coordinate (latitude, longitude)
            method (str): Distance calculation method ('euclidean' or 'haversine')
            
        Returns:
            float: Estimated road distance in kilometers
        """
        # Calculate base geographic distance
        if method == 'euclidean':
            base_distance = self.euclidean_distance(coord1, coord2)
        else:
            base_distance = self.haversine_distance(coord1, coord2)
        
        # Apply road factor to account for road network
        road_distance = base_distance * self.road_factor
        
        # Ensure minimum distance for very close points
        return max(0.1, road_distance) if road_distance > 0 else 0.1
    
    def route_distance(self, route: Route, method: str = 'haversine') -> Dict[str, Union[float, List[float]]]:
        """
        Calculate the total distance of a route and segment distances.
        
        Args:
            route (Route): List of coordinates representing the route
            method (str): Distance calculation method ('euclidean' or 'haversine')
            
        Returns:
            Dict: Dictionary containing total distance and segment distances
        """
        if len(route) < 2:
            logger.warning("Route must contain at least 2 coordinates")
            return {'total_distance': 0.0, 'segment_distances': []}
        
        # Calculate segment distances
        segment_distances = []
        for i in range(len(route) - 1):
            segment_distance = self.road_distance(route[i], route[i+1], method)
            segment_distances.append(segment_distance)
        
        # Calculate total distance
        total_distance = sum(segment_distances)
        
        return {
            'total_distance': total_distance,
            'segment_distances': segment_distances
        }
    
    def clear_cache(self) -> None:
        """Clear the distance calculation cache."""
        self._cached_haversine.cache_clear()
        logger.info("Distance calculation cache cleared")


# Create a singleton instance for easy import
calculator = GeoCalculator()


def euclidean_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Convenience function to calculate Euclidean distance between coordinates.
    
    Args:
        coord1 (Coordinate): First coordinate (latitude, longitude)
        coord2 (Coordinate): Second coordinate (latitude, longitude)
        
    Returns:
        float: Euclidean distance in kilometers
    """
    return calculator.euclidean_distance(coord1, coord2)


def haversine_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Convenience function to calculate Haversine distance between coordinates.
    
    Args:
        coord1 (Coordinate): First coordinate (latitude, longitude)
        coord2 (Coordinate): Second coordinate (latitude, longitude)
        
    Returns:
        float: Haversine distance in kilometers
    """
    return calculator.haversine_distance(coord1, coord2)


def road_distance(coord1: Coordinate, coord2: Coordinate, method: str = 'haversine') -> float:
    """
    Convenience function to calculate estimated road distance between coordinates.
    
    Args:
        coord1 (Coordinate): First coordinate (latitude, longitude)
        coord2 (Coordinate): Second coordinate (latitude, longitude)
        method (str): Distance calculation method ('euclidean' or 'haversine')
        
    Returns:
        float: Estimated road distance in kilometers
    """
    return calculator.road_distance(coord1, coord2, method)


def route_distance(route: Route, method: str = 'haversine') -> Dict[str, Union[float, List[float]]]:
    """
    Convenience function to calculate the total distance of a route.
    
    Args:
        route (Route): List of coordinates representing the route
        method (str): Distance calculation method ('euclidean' or 'haversine')
        
    Returns:
        Dict: Dictionary containing total distance and segment distances
    """
    return calculator.route_distance(route, method)


def validate_coordinates(coords: Coordinate) -> bool:
    """
    Convenience function to validate geographic coordinates.
    
    Args:
        coords (Coordinate): Tuple of (latitude, longitude)
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    return calculator.validate_coordinates(coords)


if __name__ == "__main__":
    # Suppress logging for cleaner output
    import logging
    logging.getLogger('tbrgs').setLevel(logging.WARNING)
    
    print("=" * 80)
    print("TBRGS Geographic Distance Calculator Test")
    print("=" * 80)
    
    # Test coordinates (Melbourne area)
    melbourne_cbd = (-37.8136, 144.9631)  # Melbourne CBD
    flinders_st = (-37.8183, 144.9671)    # Flinders Street Station
    southern_cross = (-37.8183, 144.9522) # Southern Cross Station
    richmond = (-37.8232, 145.0078)       # Richmond Station
    
    # Test coordinate validation
    print("\nCoordinate Validation:")
    print("-" * 50)
    test_coords = [
        melbourne_cbd,
        flinders_st,
        (91.0, 145.0),     # Invalid latitude
        (-37.8, 181.0),    # Invalid longitude
        (-37.8, "invalid") # Invalid type
    ]
    
    for coords in test_coords:
        try:
            is_valid = validate_coordinates(coords)
            print(f"Coordinates {coords}: {'Valid' if is_valid else 'Invalid'}")
        except Exception as e:
            print(f"Coordinates {coords}: Error - {e}")
    
    # Test distance calculations
    print("\nDistance Calculations:")
    print("-" * 70)
    print(f"{'Method':12} | {'From':25} | {'To':25} | {'Distance (km)':15}")
    print("-" * 70)
    
    # Calculate distances using different methods
    locations = [
        ("CBD", melbourne_cbd),
        ("Flinders St", flinders_st),
        ("Southern Cross", southern_cross),
        ("Richmond", richmond)
    ]
    
    for i, (name1, coord1) in enumerate(locations):
        for name2, coord2 in locations[i+1:]:
            euclidean = euclidean_distance(coord1, coord2)
            haversine = haversine_distance(coord1, coord2)
            road = road_distance(coord1, coord2)
            
            print(f"{'Euclidean':12} | {name1:25} | {name2:25} | {euclidean:15.3f}")
            print(f"{'Haversine':12} | {name1:25} | {name2:25} | {haversine:15.3f}")
            print(f"{'Road':12} | {name1:25} | {name2:25} | {road:15.3f}")
            print("-" * 70)
    
    # Test route distance calculation
    print("\nRoute Distance Calculation:")
    route = [melbourne_cbd, flinders_st, southern_cross, richmond]
    route_names = ["CBD", "Flinders St", "Southern Cross", "Richmond"]
    
    result = route_distance(route)
    
    print(f"Route: {' → '.join(route_names)}")
    print(f"Total Distance: {result['total_distance']:.3f} km")
    print("\nSegment Details:")
    print("-" * 70)
    print(f"{'Segment':30} | {'Distance (km)':15}")
    print("-" * 70)
    
    for i, distance in enumerate(result['segment_distances']):
        segment = f"{route_names[i]} → {route_names[i+1]}"
        print(f"{segment:30} | {distance:15.3f}")
    
    # Test cache performance
    print("\nCache Performance Test:")
    import time
    
    # Clear cache
    calculator.clear_cache()
    
    # Test without cache
    start_time = time.time()
    for _ in range(10000):
        calculator.haversine_distance(melbourne_cbd, richmond)
    no_cache_time = time.time() - start_time
    
    # Test with cache
    start_time = time.time()
    for _ in range(10000):
        calculator.haversine_distance(melbourne_cbd, richmond)
    with_cache_time = time.time() - start_time
    
    print(f"Time without cache (first run): {no_cache_time:.6f} seconds")
    print(f"Time with cache (second run): {with_cache_time:.6f} seconds")
    print(f"Speedup factor: {no_cache_time / with_cache_time:.2f}x")
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
