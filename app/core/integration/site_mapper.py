#!/usr/bin/env python3
"""
TBRGS SCATS Site Mapper Module

This module implements the mapping between SCATS site IDs and graph nodes
for the Traffic-Based Route Guidance System, addressing coordinate mapping issues
between SCATS data and actual intersection locations.

It provides functionality to:
1. Map SCATS IDs to graph nodes from Part A
2. Address coordinate mapping issues between SCATS and actual locations
3. Implement manual coordinate corrections for key intersections
4. Validate and verify coordinates
5. Map SCATS sites to intersections with distance calculations
6. Provide fallback mapping for unmapped sites
"""

import os
import math
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
# Use relative imports
import sys
import os

# Add the parent directory to the path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.config.config import config
from app.core.logging import TBRGSLogger
from app.core.integration.geo_calculator import haversine_distance, validate_coordinates

# Initialize logger
logger = TBRGSLogger.get_logger("integration.site_mapper")

# Type aliases for clarity
Coordinate = Tuple[float, float]  # (latitude, longitude)
NodeID = str  # Graph node ID
SCATSID = str  # SCATS site ID

class SCATSSiteMapper:
    """
    Mapper class for SCATS sites to graph nodes.
    
    This class handles the mapping between SCATS site IDs and graph nodes,
    addressing coordinate mapping issues between SCATS data and actual locations.
    
    Attributes:
        site_reference_df (pd.DataFrame): DataFrame containing SCATS site reference data
        site_locations (Dict[SCATSID, Coordinate]): Dictionary mapping SCATS IDs to coordinates
        corrected_coordinates (Dict[SCATSID, Coordinate]): Dictionary of manually corrected coordinates
        node_positions (Dict[NodeID, Coordinate]): Dictionary mapping node IDs to coordinates
        site_to_node_map (Dict[SCATSID, NodeID]): Dictionary mapping SCATS IDs to node IDs
        node_to_site_map (Dict[NodeID, SCATSID]): Dictionary mapping node IDs to SCATS IDs
        max_mapping_distance (float): Maximum distance (km) for automatic mapping
    """
    
    def __init__(self, site_reference_path: Optional[str] = None, 
                 corrections_path: Optional[str] = None,
                 max_mapping_distance: float = 0.2):
        """
        Initialize the SCATS site mapper.
        
        Args:
            site_reference_path (str, optional): Path to the SCATS site reference CSV file.
                If None, uses the path from config.
            corrections_path (str, optional): Path to the coordinate corrections JSON file.
                If None, uses the default path.
            max_mapping_distance (float): Maximum distance (km) for automatic mapping
        """
        # Set maximum mapping distance
        self.max_mapping_distance = max_mapping_distance
        
        # Initialize data structures
        self.site_reference_df = None
        self.site_locations = {}
        self.corrected_coordinates = {}
        self.node_positions = {}
        self.site_to_node_map = {}
        self.node_to_site_map = {}
        
        # Load site reference data
        if site_reference_path is None:
            site_reference_path = config.processed_data['site_reference']
        self._load_site_reference(site_reference_path)
        
        # Load coordinate corrections
        if corrections_path is None:
            # Default path for corrections file
            corrections_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data',
                'coordinate_corrections.json'
            )
        self._load_coordinate_corrections(corrections_path)
        
        # Ensure the corrections directory exists
        os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
        
        logger.info(f"SCATS Site Mapper initialized with {len(self.site_locations)} sites")
    
    def _load_site_reference(self, site_reference_path: str) -> None:
        """
        Load SCATS site reference data from CSV file.
        
        Args:
            site_reference_path (str): Path to the SCATS site reference CSV file
        """
        try:
            self.site_reference_df = pd.read_csv(site_reference_path)
            
            # Create a dictionary mapping SCATS_ID to (latitude, longitude) coordinates
            # We'll use the average coordinates for each SCATS site to get a central point
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
    
    def _load_coordinate_corrections(self, corrections_path: str) -> None:
        """
        Load coordinate corrections from JSON file.
        
        Args:
            corrections_path (str): Path to the coordinate corrections JSON file
        """
        try:
            if os.path.exists(corrections_path):
                with open(corrections_path, 'r') as f:
                    self.corrected_coordinates = json.load(f)
                logger.info(f"Loaded {len(self.corrected_coordinates)} coordinate corrections")
            else:
                # Create a default corrections file with known issues
                self._create_default_corrections(corrections_path)
        except Exception as e:
            logger.error(f"Error loading coordinate corrections: {e}")
            self.corrected_coordinates = {}
    
    def _create_default_corrections(self, corrections_path: str) -> None:
        """
        Create a default coordinate corrections file with known issues.
        
        Args:
            corrections_path (str): Path to save the coordinate corrections JSON file
        """
        # Create default corrections based on known issues
        # Format: SCATS_ID: [corrected_latitude, corrected_longitude]
        default_corrections = {
            # Example corrections for key intersections with known issues
            # These are based on the assignment note about SCATS lat/lng not aligning with Google Maps
            "0970": [-37.86730, 145.09151],  # WARRIGAL_RD/HIGH STREET_RD
            "2000": [-37.85192, 145.09432],  # WARRIGAL_RD/TOORAK_RD
            "3002": [-37.81514, 145.02655],  # DENMARK_ST/BARKERS_RD
            "4035": [-37.81830, 145.05811],  # BARKERS_RD/BURKE_RD
        }
        
        # Save the default corrections
        try:
            os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
            with open(corrections_path, 'w') as f:
                json.dump(default_corrections, f, indent=2)
            
            self.corrected_coordinates = default_corrections
            logger.info(f"Created default coordinate corrections file with {len(default_corrections)} entries")
        except Exception as e:
            logger.error(f"Error creating default coordinate corrections: {e}")
            self.corrected_coordinates = {}
    
    def save_coordinate_corrections(self, corrections_path: Optional[str] = None) -> None:
        """
        Save coordinate corrections to JSON file.
        
        Args:
            corrections_path (str, optional): Path to save the coordinate corrections JSON file.
                If None, uses the default path.
        """
        if corrections_path is None:
            # Default path for corrections file
            corrections_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data',
                'coordinate_corrections.json'
            )
        
        try:
            os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
            with open(corrections_path, 'w') as f:
                json.dump(self.corrected_coordinates, f, indent=2)
            logger.info(f"Saved {len(self.corrected_coordinates)} coordinate corrections to {corrections_path}")
        except Exception as e:
            logger.error(f"Error saving coordinate corrections: {e}")
    
    def get_site_coordinate(self, scats_id: SCATSID) -> Optional[Coordinate]:
        """
        Get the coordinate for a SCATS site, using corrected coordinates if available.
        
        Args:
            scats_id (SCATSID): SCATS site ID
            
        Returns:
            Optional[Coordinate]: Tuple of (latitude, longitude) or None if site not found
        """
        # Check if we have a corrected coordinate for this site
        if scats_id in self.corrected_coordinates:
            return tuple(self.corrected_coordinates[scats_id])
        
        # Otherwise use the original coordinate from the site reference data
        return self.site_locations.get(scats_id)
    
    def add_coordinate_correction(self, scats_id: SCATSID, corrected_coord: Coordinate) -> None:
        """
        Add or update a coordinate correction for a SCATS site.
        
        Args:
            scats_id (SCATSID): SCATS site ID
            corrected_coord (Coordinate): Corrected coordinate (latitude, longitude)
        """
        # Validate the coordinate
        if not validate_coordinates(corrected_coord):
            logger.error(f"Invalid coordinate for correction: {corrected_coord}")
            return
        
        # Add or update the correction
        self.corrected_coordinates[scats_id] = corrected_coord
        logger.info(f"Added coordinate correction for site {scats_id}: {corrected_coord}")
    
    def load_graph_nodes(self, node_positions: Dict[NodeID, Coordinate]) -> None:
        """
        Load graph node positions from Part A routing.
        
        Args:
            node_positions (Dict[NodeID, Coordinate]): Dictionary mapping node IDs to coordinates
        """
        self.node_positions = node_positions
        logger.info(f"Loaded {len(self.node_positions)} graph node positions")
    
    def map_sites_to_nodes(self) -> None:
        """
        Map SCATS sites to graph nodes based on geographic proximity.
        
        This method creates a bidirectional mapping between SCATS site IDs and graph node IDs.
        """
        if not self.node_positions:
            logger.warning("No graph nodes loaded. Call load_graph_nodes() first.")
            return
        
        # Clear existing mappings
        self.site_to_node_map = {}
        self.node_to_site_map = {}
        
        # Map each SCATS site to the closest graph node
        for scats_id, site_coord in self.site_locations.items():
            # Get corrected coordinate if available
            site_coord = self.get_site_coordinate(scats_id)
            
            if not site_coord:
                logger.warning(f"No coordinate available for site {scats_id}")
                continue
            
            # Find the closest graph node
            closest_node = None
            min_distance = float('inf')
            
            for node_id, node_coord in self.node_positions.items():
                distance = haversine_distance(site_coord, node_coord)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node_id
            
            # Only map if the distance is within the maximum mapping distance
            if min_distance <= self.max_mapping_distance:
                self.site_to_node_map[scats_id] = closest_node
                self.node_to_site_map[closest_node] = scats_id
                logger.debug(f"Mapped site {scats_id} to node {closest_node} (distance: {min_distance:.3f} km)")
            else:
                logger.warning(f"Site {scats_id} is too far from any graph node (min distance: {min_distance:.3f} km)")
        
        logger.info(f"Mapped {len(self.site_to_node_map)} SCATS sites to graph nodes")
    
    def create_graph_nodes_from_sites(self) -> Dict[NodeID, Coordinate]:
        """
        Create graph nodes from SCATS sites when no existing graph is available.
        
        This is a fallback method when no Part A graph is provided.
        
        Returns:
            Dict[NodeID, Coordinate]: Dictionary mapping node IDs to coordinates
        """
        # Create a node for each SCATS site
        node_positions = {}
        
        for scats_id in self.site_locations:
            # Use the site ID as the node ID
            node_id = f"N{scats_id}"
            
            # Get corrected coordinate if available
            coord = self.get_site_coordinate(scats_id)
            
            if coord:
                node_positions[node_id] = coord
                
                # Create the mapping
                self.site_to_node_map[scats_id] = node_id
                self.node_to_site_map[node_id] = scats_id
        
        # Store the node positions
        self.node_positions = node_positions
        
        logger.info(f"Created {len(node_positions)} graph nodes from SCATS sites")
        return node_positions
    
    def get_node_for_site(self, scats_id: SCATSID) -> Optional[NodeID]:
        """
        Get the graph node ID for a SCATS site.
        
        Args:
            scats_id (SCATSID): SCATS site ID
            
        Returns:
            Optional[NodeID]: Graph node ID or None if not mapped
        """
        return self.site_to_node_map.get(scats_id)
    
    def get_site_for_node(self, node_id: NodeID) -> Optional[SCATSID]:
        """
        Get the SCATS site ID for a graph node.
        
        Args:
            node_id (NodeID): Graph node ID
            
        Returns:
            Optional[SCATSID]: SCATS site ID or None if not mapped
        """
        return self.node_to_site_map.get(node_id)
    
    def verify_mapping_quality(self) -> Dict[str, Any]:
        """
        Verify the quality of the mapping between SCATS sites and graph nodes.
        
        Returns:
            Dict[str, Any]: Dictionary with mapping quality metrics
        """
        if not self.site_to_node_map or not self.node_positions:
            logger.warning("No mapping available to verify")
            return {
                'mapped_sites': 0,
                'total_sites': len(self.site_locations),
                'mapping_coverage': 0.0,
                'average_distance': 0.0,
                'max_distance': 0.0,
                'unmapped_sites': list(self.site_locations.keys())
            }
        
        # Calculate mapping metrics
        mapped_sites = len(self.site_to_node_map)
        total_sites = len(self.site_locations)
        mapping_coverage = mapped_sites / total_sites if total_sites > 0 else 0.0
        
        # Calculate distances between sites and mapped nodes
        distances = []
        for scats_id, node_id in self.site_to_node_map.items():
            site_coord = self.get_site_coordinate(scats_id)
            node_coord = self.node_positions.get(node_id)
            
            if site_coord and node_coord:
                distance = haversine_distance(site_coord, node_coord)
                distances.append(distance)
        
        # Calculate distance statistics
        average_distance = sum(distances) / len(distances) if distances else 0.0
        max_distance = max(distances) if distances else 0.0
        
        # Get list of unmapped sites
        unmapped_sites = [site for site in self.site_locations if site not in self.site_to_node_map]
        
        return {
            'mapped_sites': mapped_sites,
            'total_sites': total_sites,
            'mapping_coverage': mapping_coverage,
            'average_distance': average_distance,
            'max_distance': max_distance,
            'unmapped_sites': unmapped_sites
        }
    
    def export_mapping(self, output_path: str) -> None:
        """
        Export the mapping between SCATS sites and graph nodes to a JSON file.
        
        Args:
            output_path (str): Path to save the mapping JSON file
        """
        mapping_data = {
            'site_to_node': self.site_to_node_map,
            'node_to_site': self.node_to_site_map,
            'corrected_coordinates': self.corrected_coordinates,
            'mapping_quality': self.verify_mapping_quality()
        }
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            logger.info(f"Exported mapping to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting mapping: {e}")
    
    def import_mapping(self, input_path: str) -> bool:
        """
        Import the mapping between SCATS sites and graph nodes from a JSON file.
        
        Args:
            input_path (str): Path to the mapping JSON file
            
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            with open(input_path, 'r') as f:
                mapping_data = json.load(f)
            
            self.site_to_node_map = mapping_data.get('site_to_node', {})
            self.node_to_site_map = mapping_data.get('node_to_site', {})
            self.corrected_coordinates = mapping_data.get('corrected_coordinates', {})
            
            logger.info(f"Imported mapping from {input_path} with {len(self.site_to_node_map)} mappings")
            return True
        except Exception as e:
            logger.error(f"Error importing mapping: {e}")
            return False


# Create a singleton instance for easy import
mapper = SCATSSiteMapper()


def get_node_for_site(scats_id: SCATSID) -> Optional[NodeID]:
    """
    Convenience function to get the graph node ID for a SCATS site.
    
    Args:
        scats_id (SCATSID): SCATS site ID
        
    Returns:
        Optional[NodeID]: Graph node ID or None if not mapped
    """
    # For debugging
    result = mapper.get_node_for_site(scats_id)
    if result is None:
        logger.debug(f"No node mapping found for SCATS ID {scats_id}. Current mappings: {mapper.site_to_node_map}")
    return result


def get_site_for_node(node_id: NodeID) -> Optional[SCATSID]:
    """
    Convenience function to get the SCATS site ID for a graph node.
    
    Args:
        node_id (NodeID): Graph node ID
        
    Returns:
        Optional[SCATSID]: SCATS site ID or None if not mapped
    """
    return mapper.get_site_for_node(node_id)


def get_site_coordinate(scats_id: SCATSID) -> Optional[Coordinate]:
    """
    Convenience function to get the coordinate for a SCATS site.
    
    Args:
        scats_id (SCATSID): SCATS site ID
        
    Returns:
        Optional[Coordinate]: Tuple of (latitude, longitude) or None if site not found
    """
    return mapper.get_site_coordinate(scats_id)


if __name__ == "__main__":
    # Suppress logging for cleaner output
    import logging
    logging.getLogger('tbrgs').setLevel(logging.WARNING)
    
    print("=" * 80)
    print("TBRGS SCATS Site Mapper Test")
    print("=" * 80)
    
    # Create a test mapper
    test_mapper = SCATSSiteMapper()
    
    # Print site coordinates
    print("\nSCATS Site Coordinates:")
    print("-" * 70)
    print(f"{'SCATS ID':8} | {'Original Latitude':15} | {'Original Longitude':15} | {'Corrected':8}")
    print("-" * 70)
    
    # Get a sample of sites to display
    sample_sites = list(test_mapper.site_locations.keys())[:10]
    for scats_id in sample_sites:
        orig_coord = test_mapper.site_locations.get(scats_id, (None, None))
        corr_coord = test_mapper.get_site_coordinate(scats_id)
        
        is_corrected = "Yes" if scats_id in test_mapper.corrected_coordinates else "No"
        
        if orig_coord and len(orig_coord) == 2:
            print(f"{scats_id:8} | {orig_coord[0]:15.6f} | {orig_coord[1]:15.6f} | {is_corrected:8}")
    
    # Create sample graph nodes for testing
    print("\nCreating Sample Graph Nodes:")
    node_positions = {}
    
    # Create nodes based on SCATS sites with small random offsets
    import random
    random.seed(42)  # For reproducibility
    
    for scats_id, coord in test_mapper.site_locations.items():
        # Add small random offset (up to 50 meters)
        lat_offset = random.uniform(-0.0005, 0.0005)
        lon_offset = random.uniform(-0.0005, 0.0005)
        
        node_coord = (coord[0] + lat_offset, coord[1] + lon_offset)
        node_id = f"N{scats_id}"
        
        node_positions[node_id] = node_coord
    
    # Add some additional nodes not based on SCATS sites
    for i in range(5):
        node_id = f"X{i+1}"
        lat = -37.8 + random.uniform(-0.1, 0.1)
        lon = 145.0 + random.uniform(-0.1, 0.1)
        node_positions[node_id] = (lat, lon)
    
    print(f"Created {len(node_positions)} sample graph nodes")
    
    # Load the graph nodes
    test_mapper.load_graph_nodes(node_positions)
    
    # Map sites to nodes
    test_mapper.map_sites_to_nodes()
    
    # Print mapping results
    print("\nMapping Results:")
    mapping_quality = test_mapper.verify_mapping_quality()
    print(f"Mapped {mapping_quality['mapped_sites']} out of {mapping_quality['total_sites']} sites " +
          f"({mapping_quality['mapping_coverage']*100:.1f}% coverage)")
    print(f"Average distance: {mapping_quality['average_distance']*1000:.1f} meters")
    print(f"Maximum distance: {mapping_quality['max_distance']*1000:.1f} meters")
    
    # Print some sample mappings
    print("\nSample Mappings:")
    print("-" * 70)
    print(f"{'SCATS ID':8} | {'Node ID':8} | {'Distance (m)':12}")
    print("-" * 70)
    
    for scats_id in sample_sites:
        node_id = test_mapper.get_node_for_site(scats_id)
        
        if node_id:
            site_coord = test_mapper.get_site_coordinate(scats_id)
            node_coord = test_mapper.node_positions.get(node_id)
            
            if site_coord and node_coord:
                distance = haversine_distance(site_coord, node_coord) * 1000  # Convert to meters
                print(f"{scats_id:8} | {node_id:8} | {distance:12.1f}")
            else:
                print(f"{scats_id:8} | {node_id:8} | {'N/A':12}")
        else:
            print(f"{scats_id:8} | {'N/A':8} | {'N/A':12}")
    
    # Test exporting and importing the mapping
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        export_path = tmp.name
    
    test_mapper.export_mapping(export_path)
    
    # Create a new mapper and import the mapping
    new_mapper = SCATSSiteMapper()
    new_mapper.import_mapping(export_path)
    
    # Verify the imported mapping
    print("\nImported Mapping Verification:")
    print(f"Original mappings: {len(test_mapper.site_to_node_map)}")
    print(f"Imported mappings: {len(new_mapper.site_to_node_map)}")
    print(f"Mapping match: {test_mapper.site_to_node_map == new_mapper.site_to_node_map}")
    
    # Clean up the temporary file
    try:
        os.remove(export_path)
    except:
        pass
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
