#!/usr/bin/env python3
"""
TBRGS Enhanced Map Visualization Module

This module provides improved map visualization capabilities for the TBRGS project,
with a focus on detailed route path visualization using Plotly.
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from traefik.app.core.logging import TBRGSLogger

# Initialize logger
logger = TBRGSLogger.get_logger("gui.enhanced_map_visualizer")

class EnhancedMapVisualizer:
    """
    Enhanced map visualization class for the TBRGS project.
    
    This class provides improved map visualization capabilities using Plotly,
    with a focus on detailed route path visualization.
    """
    
    def __init__(self, scats_sites=None, default_center=(-37.8136, 144.9631)):
        """
        Initialize the enhanced map visualizer.
        
        Args:
            scats_sites: Dictionary of SCATS sites with coordinates and metadata
            default_center: Default map center coordinates (latitude, longitude)
        """
        self.scats_sites = scats_sites or {}
        self.default_center = default_center
        
        # Plotly colors (named colors work well)
        self.plotly_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
        
        logger.info("Enhanced map visualizer initialized")
    
    def create_plotly_map(self, origin_site, destination_site, routes, site_data, traffic_flows=None):
        """
        Create an interactive map visualization using Plotly and Mapbox.
        
        This method creates a detailed map visualization with proper route paths
        showing all intermediate points, similar to Google Maps.
        
        Args:
            origin_site: Origin SCATS site ID
            destination_site: Destination SCATS site ID
            routes: List of route information dictionaries
            site_data: Dictionary of site data
            traffic_flows: Dictionary mapping SCATS IDs to traffic flow values
            
        Returns:
            str: Path to the HTML file containing the map
        """
        try:
            logger.info(f"Creating enhanced Plotly map for route from {origin_site} to {destination_site}")
            
            # Create dataframes for sites and routes
            sites_df = pd.DataFrame()
            
            # Process site data for plotting
            site_ids = []
            lats = []
            lons = []
            names = []
            site_types = []
            
            for site_id, site_info in site_data.items():
                try:
                    lat = float(site_info['latitude'])
                    lon = float(site_info['longitude'])
                    name = site_info.get('name', 'Unknown')
                    
                    site_ids.append(site_id)
                    lats.append(lat)
                    lons.append(lon)
                    names.append(name)
                    
                    # Mark origin and destination sites
                    if site_id == origin_site:
                        site_types.append('origin')
                    elif site_id == destination_site:
                        site_types.append('destination')
                    else:
                        site_types.append('regular')
                        
                except Exception as e:
                    logger.warning(f"Error processing site {site_id}: {e}")
            
            # Create sites dataframe
            sites_df = pd.DataFrame({
                'site_id': site_ids,
                'lat': lats,
                'lon': lons,
                'name': names,
                'type': site_types
            })
            
            # Create the figure
            fig = go.Figure()
            
            # Get origin and destination coordinates
            origin_coords = None
            dest_coords = None
            
            if origin_site in site_data:
                origin_data = site_data[origin_site]
                origin_coords = (float(origin_data['latitude']), float(origin_data['longitude']))
                
            if destination_site in site_data:
                dest_data = site_data[destination_site]
                dest_coords = (float(dest_data['latitude']), float(dest_data['longitude']))
            
            # Add all SCATS sites as small dots
            regular_sites = sites_df[sites_df['type'] == 'regular']
            if not regular_sites.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=regular_sites['lat'],
                    lon=regular_sites['lon'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.7
                    ),
                    text=regular_sites['site_id'] + ': ' + regular_sites['name'],
                    hoverinfo='text',
                    name='SCATS Sites'
                ))
            
            # Add origin site with special marker
            if origin_coords:
                fig.add_trace(go.Scattermapbox(
                    lat=[origin_coords[0]],
                    lon=[origin_coords[1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='green',
                        opacity=1.0
                    ),
                    text=[f'Origin: {origin_site} - {site_data[origin_site].get("name", "Unknown")}'],
                    hoverinfo='text',
                    name='Origin'
                ))
            
            # Add destination site with special marker
            if dest_coords:
                fig.add_trace(go.Scattermapbox(
                    lat=[dest_coords[0]],
                    lon=[dest_coords[1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        opacity=1.0
                    ),
                    text=[f'Destination: {destination_site} - {site_data[destination_site].get("name", "Unknown")}'],
                    hoverinfo='text',
                    name='Destination'
                ))
            
            # Process and add routes with detailed path visualization
            if routes:
                logger.info(f"Processing {len(routes)} routes for visualization")
                
                for i, route in enumerate(routes[:5]):  # Limit to 5 routes
                    # Get route information
                    travel_time = route.get('travel_time', 0)  # Travel time in seconds
                    algorithm = route.get('algorithm', 'Unknown')
                    distance = route.get('distance', 0)  # Distance in km
                    
                    # Convert travel time from seconds to minutes for display
                    travel_time_min = travel_time / 60.0
                    
                    # Format the route name with travel time in minutes
                    route_name = f"Route {i+1}: {travel_time_min:.1f} min ({algorithm})"
                    
                    # Extract path and segments information
                    path = route.get('path', [])
                    segments = route.get('segments', [])
                    
                    # Log route details for debugging
                    logger.info(f"Route {i+1} details: algorithm={algorithm}, travel_time={travel_time}, path_nodes={len(path)}, segments={len(segments) if segments else 0}")
                    
                    # Initialize route coordinates
                    route_lats = []
                    route_lons = []
                    route_texts = []
                    
                    # First try to use the detailed path if available
                    if path and len(path) >= 2:
                        logger.info(f"Using detailed path with {len(path)} nodes for route {i+1}")
                        
                        # Start with origin if not already in path
                        if origin_coords and str(path[0]) != origin_site:
                            route_lats.append(origin_coords[0])
                            route_lons.append(origin_coords[1])
                            route_texts.append(f"Origin: {origin_site}")
                        
                        # Process each node in the path
                        for node_id in path:
                            node_id_str = str(node_id)
                            if node_id_str in site_data:
                                lat = float(site_data[node_id_str]['latitude'])
                                lon = float(site_data[node_id_str]['longitude'])
                                name = site_data[node_id_str].get('name', 'Unknown')
                                
                                route_lats.append(lat)
                                route_lons.append(lon)
                                route_texts.append(f"Node {node_id_str}: {name}")
                            else:
                                logger.warning(f"Could not find coordinates for node {node_id_str}")
                        
                        # End with destination if not already in path
                        if dest_coords and str(path[-1]) != destination_site:
                            route_lats.append(dest_coords[0])
                            route_lons.append(dest_coords[1])
                            route_texts.append(f"Destination: {destination_site}")
                    
                    # If we couldn't get coordinates from path, try using segments
                    elif segments and len(segments) > 0:
                        logger.info(f"Using segments for route {i+1} with {len(segments)} segments")
                        
                        # Start with origin
                        if origin_coords:
                            route_lats.append(origin_coords[0])
                            route_lons.append(origin_coords[1])
                            route_texts.append(f"Origin: {origin_site}")
                        
                        # Process each segment
                        for segment in segments:
                            from_node = segment.get('from_node')
                            to_node = segment.get('to_node')
                            segment_time = segment.get('travel_time', 0)
                            
                            # Get from_node coordinates if not already added
                            if from_node and str(from_node) in site_data:
                                from_data = site_data[str(from_node)]
                                from_lat = float(from_data['latitude'])
                                from_lon = float(from_data['longitude'])
                                
                                # Only add if it's not the same as the last point
                                if not route_lats or (from_lat != route_lats[-1] or from_lon != route_lons[-1]):
                                    route_lats.append(from_lat)
                                    route_lons.append(from_lon)
                                    route_texts.append(f"Node {from_node}: {from_data.get('name', 'Unknown')}")
                            
                            # Get to_node coordinates
                            if to_node and str(to_node) in site_data:
                                to_data = site_data[str(to_node)]
                                to_lat = float(to_data['latitude'])
                                to_lon = float(to_data['longitude'])
                                
                                route_lats.append(to_lat)
                                route_lons.append(to_lon)
                                route_texts.append(f"Node {to_node}: {to_data.get('name', 'Unknown')} ({segment_time:.1f} min)")
                        
                        # End with destination if not already added
                        if dest_coords and (not route_lats or dest_coords[0] != route_lats[-1] or dest_coords[1] != route_lons[-1]):
                            route_lats.append(dest_coords[0])
                            route_lons.append(dest_coords[1])
                            route_texts.append(f"Destination: {destination_site}")
                    
                    # Fallback to direct route if we still don't have any points
                    else:
                        logger.warning(f"Fallback to direct route for route {i+1}")
                        if origin_coords and dest_coords:
                            route_lats = [origin_coords[0], dest_coords[0]]
                            route_lons = [origin_coords[1], dest_coords[1]]
                            route_texts = [f"Origin: {origin_site}", f"Destination: {destination_site}"]
                    
                    # Add the route line if we have coordinates
                    if route_lats and route_lons and len(route_lats) >= 2:
                        # Add the main route line
                        fig.add_trace(go.Scattermapbox(
                            lat=route_lats,
                            lon=route_lons,
                            mode='lines+markers',
                            line=dict(
                                width=4,
                                color=self.plotly_colors[i % len(self.plotly_colors)]
                            ),
                            marker=dict(
                                size=8,
                                color=self.plotly_colors[i % len(self.plotly_colors)],
                                opacity=0.8
                            ),
                            text=route_texts,
                            hoverinfo='text',
                            name=route_name
                        ))
                        
                        # Add route number label at the midpoint
                        mid_idx = len(route_lats) // 2
                        fig.add_trace(go.Scattermapbox(
                            lat=[route_lats[mid_idx]],
                            lon=[route_lons[mid_idx]],
                            mode='markers+text',
                            marker=dict(
                                size=20,
                                color=self.plotly_colors[i % len(self.plotly_colors)],
                                opacity=0.9
                            ),
                            text=[f"{i+1}"],
                            textfont=dict(size=12, color='white'),
                            textposition='middle center',
                            hoverinfo='text',
                            hovertext=[f"Route {i+1}: {travel_time:.1f} min"],
                            name=f"Route {i+1} Label",
                            showlegend=False
                        ))
                    else:
                        logger.warning(f"Not enough coordinates to draw route {i+1}")
            
            # If traffic flows are provided, add them to the map
            if traffic_flows:
                logger.info(f"Adding traffic flow visualization for {len(traffic_flows)} sites")
                
                # Create traffic flow data
                flow_site_ids = []
                flow_lats = []
                flow_lons = []
                flow_values = []
                flow_texts = []
                
                for site_id, flow in traffic_flows.items():
                    if site_id in site_data:
                        site_info = site_data[site_id]
                        lat = float(site_info['latitude'])
                        lon = float(site_info['longitude'])
                        name = site_info.get('name', 'Unknown')
                        
                        flow_site_ids.append(site_id)
                        flow_lats.append(lat)
                        flow_lons.append(lon)
                        flow_values.append(flow)
                        flow_texts.append(f"Traffic at {site_id}: {flow:.1f} vehicles/hour")
                
                # Add traffic flow markers
                if flow_lats:
                    # Normalize flow values for sizing (min 5, max 20)
                    min_flow = min(flow_values) if flow_values else 0
                    max_flow = max(flow_values) if flow_values else 1000
                    normalized_sizes = [max(5, min(20, 5 + (flow - min_flow) / (max_flow - min_flow) * 15)) for flow in flow_values]
                    
                    # Create a colorscale for the flows
                    fig.add_trace(go.Scattermapbox(
                        lat=flow_lats,
                        lon=flow_lons,
                        mode='markers',
                        marker=dict(
                            size=normalized_sizes,
                            color=flow_values,
                            colorscale='RdYlGn_r',  # Red for high traffic, green for low
                            opacity=0.7,
                            colorbar=dict(title="Traffic Flow"),
                            showscale=True
                        ),
                        text=flow_texts,
                        hoverinfo='text',
                        name='Traffic Flow',
                        visible='legendonly'  # Hide by default, can be toggled
                    ))
            
            # Set up the layout with Mapbox
            # Calculate center and zoom level based on route coordinates
            all_lats = sites_df['lat'].tolist()
            all_lons = sites_df['lon'].tolist()
            
            # Set center to midpoint between origin and destination
            if origin_coords and dest_coords:
                center_lat = (origin_coords[0] + dest_coords[0]) / 2
                center_lon = (origin_coords[1] + dest_coords[1]) / 2
            else:
                center_lat = sum(all_lats) / len(all_lats) if all_lats else -37.8136
                center_lon = sum(all_lons) / len(all_lons) if all_lons else 144.9631
            
            # Calculate zoom level based on distance between points
            zoom = 12  # Default zoom level
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",  # Use OpenStreetMap style (free, no token needed)
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                title=dict(
                    text=f"Route from {origin_site} to {destination_site}",
                    x=0.5,
                    y=0.98
                )
            )
            
            # Save the map to an HTML file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            map_filename = f"plotly_map_{origin_site}_to_{destination_site}_{timestamp}.html"
            map_path = os.path.join(output_dir, map_filename)
            
            # Save the figure to HTML
            fig.write_html(map_path, include_plotlyjs='cdn')
            logger.info(f"Saved enhanced Plotly map to {map_path}")
            
            # Print to console for easy access
            print(f"\n\n=== MAP SAVED TO: {map_path} ===\n\n")
            
            return map_path
            
        except Exception as e:
            logger.error(f"Error creating enhanced Plotly map: {e}")
            return None
