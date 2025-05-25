#!/usr/bin/env python3
"""
TBRGS Gradio Web Interface

This module implements a web-based graphical user interface for the
Traffic-Based Route Guidance System using Gradio and Folium for map visualization.
It integrates with the SCATS router and route predictor to provide an interactive
interface for users to find optimal routes between SCATS sites.
"""

import os
import sys
import json
import time
import gradio as gr
import folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import tempfile
from pathlib import Path
import webbrowser

# Add the parent directory to the path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import TBRGS components
from core.logging import TBRGSLogger
from core.integration.scats_router import SCATSRouter
from core.integration.site_mapper import get_node_for_site, get_site_for_node, get_site_coordinate
from core.integration.ml_route_integration import create_ml_route_integration
from config.config import config
from gui.map_visualizer import EnhancedMapVisualizer

# Initialize logger
logger = TBRGSLogger.get_logger("gui.gradio_app")

class TBRGSGradioApp:
    """
    Gradio web interface for the Traffic-Based Route Guidance System.
    
    This class integrates the SCATS router and route predictor with a Gradio
    web interface, providing an interactive way to find optimal routes between
    SCATS sites and visualize them on a map.
    """
    
    def __init__(self):
        """Initialize the TBRGS Gradio application."""
        try:
            # Initialize the SCATS router
            logger.info("Initializing SCATS router...")
            self.router = SCATSRouter()
            
            # Get available SCATS sites
            self.scats_sites = self.router.site_data
            self.site_options = [(f"{site_id} - {data['name']}", site_id) 
                                for site_id, data in self.scats_sites.items()]
            
            # Sort site options alphabetically
            self.site_options.sort(key=lambda x: x[0])
            
            # Get available routing algorithms
            self.algorithms = self.router.route_predictor.routing_algorithms
            
            # Default map center (Melbourne CBD)
            self.default_center = (-37.8136, 144.9631)
            
            # Initialize map visualizer
            self.map_visualizer = EnhancedMapVisualizer(self.scats_sites, self.default_center)
            
            # Available ML models
            self.ml_models = {
                "LSTM": "LSTM",
                "GRU": "GRU",
                "CNN-RNN": "CNN-RNN",
                "Ensemble": "ensemble"
            }
            
            # Default ML model
            self.current_model = "GRU"
            self.use_ensemble = False
            
            # Initialize ML-Route integration with default model
            self._setup_ml_integration(self.current_model, self.use_ensemble)
            
            logger.info(f"TBRGS Gradio App initialized with {len(self.scats_sites)} SCATS sites")
        
        except Exception as e:
            logger.error(f"Error initializing TBRGS Gradio App: {e}")
            raise
            
    def _setup_ml_integration(self, model_type: str, use_ensemble: bool = False):
        """Set up the ML-Route integration with the specified model.
        
        Args:
            model_type (str): The ML model type to use (LSTM, GRU, CNN-RNN)
            use_ensemble (bool): Whether to use ensemble prediction
        """
        try:
            # Update current model settings
            self.current_model = model_type
            self.use_ensemble = use_ensemble
            
            # Create ML-Route integration
            if use_ensemble:
                logger.info("Setting up ML-Route integration with ensemble prediction")
                self.ml_integration = create_ml_route_integration(use_ensemble=True)
            else:
                logger.info(f"Setting up ML-Route integration with {model_type} model")
                self.ml_integration = create_ml_route_integration(model_type=model_type, use_ensemble=False)
                
            logger.info("ML-Route integration setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up ML-Route integration: {e}")
            raise
    
    def create_map(self, routes=None, origin_site=None, destination_site=None):
        """
        Create a Folium map with routes and SCATS sites using the MapVisualizer.
        
        Args:
            routes: List of route information dictionaries
            origin_site: Origin SCATS site ID
            destination_site: Destination SCATS site ID
            
        Returns:
            str: Path to the HTML file containing the map
        """
        try:
            # Get traffic flow predictions if routes are provided
            traffic_flows = None
            if routes and origin_site and destination_site:
                # Get current time for traffic predictions
                current_time = datetime.now()
                # Get traffic predictions from the router
                traffic_flows = self.router._get_traffic_predictions(current_time)
            
            # Use the map visualizer to create a comprehensive route map
            map_path = self.map_visualizer.create_route_map(
                origin_site=origin_site,
                destination_site=destination_site,
                routes=routes,
                site_data=self.scats_sites,
                traffic_flows=traffic_flows,
                get_site_for_node=get_site_for_node,
                get_node_coordinates=lambda node_id: self.router.graph.get_node_coordinates(node_id) if hasattr(self.router, 'graph') else None
            )
            
            return map_path
        
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return None
    
    def find_routes(self, origin_site_display, destination_site_display, time_option, model_type, max_routes=3, show_traffic=True, show_all_sites=False, use_plotly=True):
        """
        Find optimal routes between SCATS sites.
        
        Args:
            origin_site_display: Origin SCATS site display name
            destination_site_display: Destination SCATS site display name
            time_option: Time option for traffic prediction
            model_type: ML model type to use for traffic prediction
            max_routes: Maximum number of routes to return
            show_traffic: Whether to show traffic flow visualization
            show_all_sites: Whether to show all SCATS sites
            use_plotly: Whether to use Plotly for map visualization
            
        Returns:
            tuple: (map_html, routes_markdown, routes_data)
        """
        """
        Find optimal routes between SCATS sites.
        
        Args:
            origin_site_display: Origin SCATS site display name
            destination_site_display: Destination SCATS site display name
            time_option: Time option for traffic prediction
            max_routes: Maximum number of routes to return
            show_traffic: Whether to show traffic flow visualization
            show_all_sites: Whether to show all SCATS sites
            use_plotly: Whether to use Plotly for map visualization (alternative to Folium)
            
        Returns:
            tuple: (map_html, routes_markdown, routes_data)
        """
        try:
            # Validate inputs
            if not origin_site_display or not destination_site_display:
                return None, "Please select both origin and destination sites.", None
            
            # Extract SCATS site IDs from display names
            origin_site = self.get_site_id(origin_site_display)
            destination_site = self.get_site_id(destination_site_display)
            
            if not origin_site or not destination_site:
                return None, "Could not extract site IDs from the selected options.", None
            
            if origin_site == destination_site:
                return None, "Origin and destination sites must be different.", None
                
            # Convert time option to actual datetime
            prediction_time = None
            if time_option == "Current Time":
                prediction_time = datetime.now()
            elif time_option == "Morning Peak (8:00 AM)":
                now = datetime.now()
                prediction_time = datetime(now.year, now.month, now.day, 8, 0, 0)
            elif time_option == "Midday (12:00 PM)":
                now = datetime.now()
                prediction_time = datetime(now.year, now.month, now.day, 12, 0, 0)
            elif time_option == "Evening Peak (5:00 PM)":
                now = datetime.now()
                prediction_time = datetime(now.year, now.month, now.day, 17, 0, 0)
            elif time_option == "Night (10:00 PM)":
                now = datetime.now()
                prediction_time = datetime(now.year, now.month, now.day, 22, 0, 0)
            
            logger.info(f"Using prediction time: {prediction_time}")
            
            # Set up ML integration with selected model
            use_ensemble = model_type == "Ensemble"
            if use_ensemble != self.use_ensemble or (not use_ensemble and model_type != self.current_model):
                logger.info(f"Switching to model: {model_type}")
                self._setup_ml_integration(model_type, use_ensemble)
            
            # Get routes
            # Use prediction_time directly (it's already a datetime object)
            logger.info(f"Getting routes with prediction time: {prediction_time}")
                
            routes = self.router.get_routes(
                origin_scats=origin_site, 
                destination_scats=destination_site, 
                prediction_time=prediction_time,
                max_routes=max_routes
            )
            
            if not routes:
                return None, "No routes found between the selected sites.", None
                
            # Ensure routes have the correct structure
            for i, route in enumerate(routes):
                # Make sure path exists and is not empty
                if 'path' not in route or not route['path']:
                    # Try to get path from scats_path if available
                    if 'scats_path' in route and route['scats_path']:
                        route['path'] = route['scats_path']
                        logger.info(f"Using scats_path as path for route {i+1}")
                    else:
                        logger.warning(f"Route {i+1} has no path data")
            
            # Add debug output
            logger.info(f"Found {len(routes)} routes between {origin_site} and {destination_site}")
            
            # Log detailed route information
            for i, route in enumerate(routes):
                path = route.get('path', [])
                travel_time = route.get('travel_time', 0)
                algorithm = route.get('algorithm', 'Unknown')
                logger.info(f"Route {i+1}: {algorithm}, {len(path)} nodes, {travel_time:.1f} min")
                logger.info(f"Route {i+1} path: {path}")
            
            # Debug site data
            if origin_site in self.scats_sites:
                logger.info(f"Origin site data: {self.scats_sites[origin_site]}")
            else:
                logger.warning(f"Origin site {origin_site} not found in site data")
                
            if destination_site in self.scats_sites:
                logger.info(f"Destination site data: {self.scats_sites[destination_site]}")
            else:
                logger.warning(f"Destination site {destination_site} not found in site data")
            
            # Get traffic flow predictions based on display options
            traffic_flows = None
            if show_traffic and prediction_time:
                # Use the ML integration to get predictions instead of directly calling _get_traffic_predictions
                try:
                    # Check if the router has an _ml_get_traffic_predictions method
                    if hasattr(self.router, '_ml_get_traffic_predictions'):
                        traffic_flows = self.router._ml_get_traffic_predictions(prediction_time)
                    # Otherwise, try to update traffic predictions which will generate the predictions internally
                    elif hasattr(self.router, 'update_traffic_predictions'):
                        self.router.update_traffic_predictions(prediction_time)
                        # We don't have direct access to the flows, but the graph is updated
                        traffic_flows = {}
                    logger.info(f"Generated traffic predictions for visualization")
                except Exception as e:
                    logger.warning(f"Could not generate traffic predictions: {e}")
                    traffic_flows = {}
            
            # Create a simple HTML map as fallback
            fallback_html = None
            try:
                # Create a basic Folium map
                m = folium.Map(location=self.default_center, zoom_start=12)
                
                # Add markers for origin and destination
                if origin_site in self.scats_sites:
                    site_data = self.scats_sites[origin_site]
                    folium.Marker(
                        location=[site_data['latitude'], site_data['longitude']],
                        popup=f"Origin: {origin_site}",
                        icon=folium.Icon(color="green", icon="play"),
                    ).add_to(m)
                
                if destination_site in self.scats_sites:
                    site_data = self.scats_sites[destination_site]
                    folium.Marker(
                        location=[site_data['latitude'], site_data['longitude']],
                        popup=f"Destination: {destination_site}",
                        icon=folium.Icon(color="red", icon="stop"),
                    ).add_to(m)
                
                # Save fallback map
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    m.save(tmp.name)
                    fallback_html = tmp.name
                    logger.info(f"Created fallback map at {fallback_html}")
            except Exception as e:
                logger.error(f"Error creating fallback map: {e}")
            
            # Try to create map with the MapVisualizer
            try:
                logger.info(f"Attempting to create map with MapVisualizer using {'Plotly' if use_plotly else 'Folium'}...")
                # Create a function to get node coordinates that handles errors
                def get_node_coordinates(node_id):
                    try:
                        if hasattr(self.router, 'graph'):
                            coords = self.router.graph.get_node_coordinates(node_id)
                            if coords:
                                return coords
                        # Fallback: Try to get coordinates from site data
                        site_id = get_site_for_node(node_id)
                        if site_id and site_id in self.scats_sites:
                            return [self.scats_sites[site_id]['latitude'], self.scats_sites[site_id]['longitude']]
                        return None
                    except Exception as e:
                        logger.error(f"Error getting coordinates for node {node_id}: {e}")
                        return None
                
                # Create map using Plotly visualization
                logger.info("Using Enhanced Plotly for map visualization")
                map_path = self.map_visualizer.create_plotly_map(
                    origin_site=origin_site,
                    destination_site=destination_site,
                    routes=routes,
                    site_data=self.scats_sites,
                    traffic_flows=traffic_flows
                )
                
                logger.info(f"Map created at {map_path}")
            except Exception as e:
                logger.error(f"Error creating map with MapVisualizer: {e}")
                map_path = fallback_html
                
            # Read the HTML content from the map file
            map_html = None
            if map_path and os.path.exists(map_path):
                try:
                    with open(map_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    logger.info(f"Read map HTML content from {map_path}, size: {len(map_html)} bytes")
                    
                    # Create an iframe HTML to display the map
                    map_html = f'''
                    <iframe id="map-frame" style="width:100%; height:600px; border:none;" srcdoc='{map_html.replace("'", "&apos;")}' title="Route Map"></iframe>
                    '''
                except Exception as e:
                    logger.error(f"Error reading map HTML: {e}")
                    map_html = f'''
                    <div style="width:100%; height:600px; display:flex; justify-content:center; align-items:center; background-color:#f0f0f0; border-radius:5px;">
                        <div style="text-align:center;">
                            <h3>Error displaying map</h3>
                            <p>Could not load the map visualization. Please try again.</p>
                            <p>Error: {str(e)}</p>
                        </div>
                    </div>
                    '''
            else:
                map_html = f'''
                <div style="width:100%; height:600px; display:flex; justify-content:center; align-items:center; background-color:#f0f0f0; border-radius:5px;">
                    <div style="text-align:center;">
                        <h3>No Map Available</h3>
                        <p>Could not generate a map for the selected route.</p>
                    </div>
                </div>
                '''
            
            # Create markdown output
            markdown_output = f"<div style=\"padding: 15px; background-color: #e3f2fd; border-radius: 10px; border-left: 5px solid #2196f3;\"><h3 style=\"color: #1565c0; margin-top: 0;\">Routes from {origin_site} to {destination_site}</h3>"
            
            for i, route in enumerate(routes):
                # Get route information
                travel_time = route.get('travel_time', 0)  # Travel time in seconds
                path = route.get('path', [])
                algorithm = route.get('algorithm', 'Unknown')
                distance = route.get('distance', 0)  # Distance in km
                average_speed = route.get('average_speed', 0)  # Average speed in km/h
                
                # Convert travel time from seconds to minutes for display
                travel_time_min = travel_time / 60.0
                
                # Create route card with color coding
                markdown_output += f"<div style=\"margin-top: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 8px; border-left: 4px solid {['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#e91e63'][i % 5]};\">"
                
                # Add route header with travel time in minutes
                markdown_output += f"<h4 style=\"color: #424242; margin-top: 0;\">Route {i+1}: {travel_time_min:.1f} minutes</h4>"
                
                # Add algorithm, distance and speed information
                markdown_output += f"<p style=\"margin: 5px 0;\"><strong>Algorithm:</strong> <span style=\"color: #616161;\">{algorithm}</span></p>"
                markdown_output += f"<p style=\"margin: 5px 0;\"><strong>Distance:</strong> <span style=\"color: #616161;\">{distance:.2f} km</span></p>"
                markdown_output += f"<p style=\"margin: 5px 0;\"><strong>Average Speed:</strong> <span style=\"color: #616161;\">{average_speed:.1f} km/h</span></p>"
                
                # Add path details
                markdown_output += "**Path:** "
                path_str = []
                
                for node_id in path:
                    # Try to get SCATS site for node
                    scats_id = get_site_for_node(node_id)
                    if scats_id:
                        path_str.append(scats_id)
                    else:
                        path_str.append(node_id)
                
                markdown_output += f"<p style=\"margin: 5px 0;\"><strong>Path:</strong> <span style=\"color: #616161;\">{' → '.join(map(str, path))}</span></p>"
                
                # Add segment details if available
                if 'segments' in route:
                    markdown_output += "<p style=\"margin: 5px 0;\"><strong>Segments:</strong></p><ul style=\"margin: 5px 0; padding-left: 20px;\">"
                    for segment in route['segments']:
                        from_node = segment.get('from_node', '')
                        to_node = segment.get('to_node', '')
                        segment_time = segment.get('travel_time', 0)
                        
                        # Try to get SCATS sites for nodes
                        from_scats = get_site_for_node(from_node) or from_node
                        to_scats = get_site_for_node(to_node) or to_node
                        
                        markdown_output += f"<li style=\"margin: 2px 0;\">{from_scats} → {to_scats}: <strong>{segment_time:.1f}</strong> minutes</li>"
                    
                    markdown_output += "</ul></div>"
            
            markdown_output += "</div>"
            return map_html, markdown_output, routes
        
        except Exception as e:
            logger.error(f"Error finding routes: {e}")
            error_html = f'''
            <div style="width:100%; height:600px; display:flex; justify-content:center; align-items:center; background-color:#f0f0f0; border-radius:5px;">
                <div style="text-align:center;">
                    <h3>Error Finding Routes</h3>
                    <p>{str(e)}</p>
                </div>
            </div>
            '''
            return error_html, f"<div style=\"padding: 15px; background-color: #ffebee; border-radius: 10px; border-left: 5px solid #f44336;\"><h4 style=\"color: #d32f2f; margin-top: 0;\">Error Finding Routes</h4><p style=\"color: #b71c1c;\">{str(e)}</p></div>", None
    
    def create_interface(self):
        """
        Create the Gradio interface.
        
        Returns:
            gr.Blocks: Gradio interface
        """
        with gr.Blocks(title="TBRGS - Traffic-Based Route Guidance System", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")) as interface:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div style="text-align: center; margin-bottom: 10px">
                            <h1 style="margin-bottom: 5px; color: #2c3e50;">Traffic-Based Route Guidance System</h1>
                            <h3 style="margin-top: 0; color: #7f8c8d;">Find optimal routes between SCATS sites based on traffic conditions</h3>
                        </div>
                        """
                    )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📍 Route Selection")
                        # Input controls with improved styling
                        with gr.Group():
                            origin_dropdown = gr.Dropdown(
                                choices=[site[0] for site in self.site_options],
                                label="Origin SCATS Site",
                                info="Select the starting point",
                                container=True,
                                scale=1
                            )
                            
                            destination_dropdown = gr.Dropdown(
                                choices=[site[0] for site in self.site_options],
                                label="Destination SCATS Site",
                                info="Select the destination",
                                container=True,
                                scale=1
                            )
                    
                    with gr.Group():
                        gr.Markdown("### ⏱️ Time & Parameters")
                        # Create time options for different times of day
                        time_options = [
                            "Current Time",
                            "Morning Peak (8:00 AM)",
                            "Midday (12:00 PM)",
                            "Evening Peak (5:00 PM)",
                            "Night (10:00 PM)"
                        ]
                        
                        with gr.Group():
                            time_dropdown = gr.Dropdown(
                                choices=time_options,
                                label="Prediction Time",
                                info="Select time for traffic prediction",
                                value="Current Time",
                                container=True
                            )
                            
                            max_routes = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=3,
                                step=1,
                                label="Maximum Routes",
                                info="Maximum number of routes to find",
                                container=True
                            )
                    
                    with gr.Group():
                        gr.Markdown("### 🧠 ML Model Selection")
                        with gr.Group():
                            model_dropdown = gr.Dropdown(
                                choices=list(self.ml_models.keys()),
                                label="Traffic Prediction Model",
                                info="Select the ML model for traffic prediction",
                                value="GRU",
                                container=True
                            )
                            
                            gr.Markdown("""
                            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.9em;">
                                <p><strong>Model Information:</strong></p>
                                <ul>
                                    <li><strong>LSTM</strong>: Long Short-Term Memory - Good for capturing long-term dependencies</li>
                                    <li><strong>GRU</strong>: Gated Recurrent Unit - Faster and more efficient than LSTM</li>
                                    <li><strong>CNN-RNN</strong>: Hybrid model combining CNN and RNN - Best overall performance</li>
                                    <li><strong>Ensemble</strong>: Combines predictions from all models - Most robust but slower</li>
                                </ul>
                            </div>
                            """)
                    
                    with gr.Group():
                        gr.Markdown("### 🗺️ Map Options")
                        with gr.Group():
                            with gr.Row():
                                show_traffic = gr.Checkbox(
                                    label="Show Traffic Flow",
                                    info="Display traffic flow visualization on the map",
                                    value=True,
                                    container=True
                                )
                                
                                show_all_sites = gr.Checkbox(
                                    label="Show All SCATS Sites",
                                    info="Display all SCATS sites on the map",
                                    value=False,
                                    container=True
                                )
                            
                            use_plotly = gr.Checkbox(
                                label="Use Enhanced Map Visualization",
                                info="Use Plotly for detailed route visualization with intermediate points",
                                value=True,
                                container=True
                            )
                    
                    find_button = gr.Button("🔍 Find Routes", variant="primary")
                    
                    # Quick route selection with improved styling
                    with gr.Group():
                        gr.Markdown("### ⚡ Quick Route Selection")
                        with gr.Group():
                            with gr.Row():
                                quick_route1 = gr.Button("2000 → 3002", variant="secondary")
                                quick_route2 = gr.Button("0970 → 2200", variant="secondary")
                            
                            with gr.Row():
                                quick_route3 = gr.Button("1010 → 2000", variant="secondary")
                                quick_route4 = gr.Button("0970 → 3002", variant="secondary")
                
                with gr.Column(scale=2):
                    # Output displays with improved styling
                    with gr.Group():
                        gr.Markdown("### 🗺️ Route Map")
                        map_output = gr.HTML(
                            value="<div style=\"width:100%; height:600px; display:flex; justify-content:center; align-items:center; background-color:#f0f0f0; border-radius:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);\"><div style=\"text-align:center;\"><h3 style=\"color:#2c3e50;\">No Map Available</h3><p style=\"color:#7f8c8d;\">Select origin and destination sites and click 'Find Routes' to generate a map.</p></div></div>"
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 📊 Route Information")
                        route_info = gr.Markdown(
                            value="<div style=\"padding: 15px; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);\"><p style=\"color: #7f8c8d; text-align: center;\">Select origin and destination sites and click 'Find Routes' to see route details.</p></div>"
                        )
                    
                    routes_json = gr.JSON(
                        label="Route Data (JSON)",
                        visible=False
                    )
            
            # Set up event handlers
            find_button.click(
                fn=self.find_routes,
                inputs=[
                    origin_dropdown,
                    destination_dropdown,
                    time_dropdown,
                    model_dropdown,
                    max_routes,
                    show_traffic,
                    show_all_sites,
                    use_plotly
                ],
                outputs=[
                    map_output,
                    route_info,
                    routes_json
                ]
            )
            
            # Quick route selection handlers
            quick_route1.click(
                fn=lambda: self.set_quick_route("2000", "3002"),
                outputs=[origin_dropdown, destination_dropdown]
            )
            
            quick_route2.click(
                fn=lambda: self.set_quick_route("0970", "2200"),
                outputs=[origin_dropdown, destination_dropdown]
            )
            
            quick_route3.click(
                fn=lambda: self.set_quick_route("1010", "2000"),
                outputs=[origin_dropdown, destination_dropdown]
            )
            
            quick_route4.click(
                fn=lambda: self.set_quick_route("0970", "3002"),
                outputs=[origin_dropdown, destination_dropdown]
            )
        
        return interface
    
    def get_site_id(self, site_display):
        """
        Extract the SCATS site ID from the display string.
        
        Args:
            site_display: Display string from dropdown (e.g., "2000 - Main St")
            
        Returns:
            str: SCATS site ID
        """
        if not site_display:
            return None
        
        # Find the site ID in the options
        for display, site_id in self.site_options:
            if display == site_display:
                return site_id
        
        # If not found, try to extract the ID from the display string
        if " - " in site_display:
            return site_display.split(" - ")[0].strip()
        
        return site_display
    
    def set_quick_route(self, origin, destination):
        """
        Set up a quick route selection.
        
        Args:
            origin: Origin SCATS site ID
            destination: Destination SCATS site ID
            
        Returns:
            tuple: (origin_display, destination_display)
        """
        # Find the display strings for the site IDs
        origin_display = None
        destination_display = None
        
        for display, site_id in self.site_options:
            if site_id == origin:
                origin_display = display
            if site_id == destination:
                destination_display = display
        
        return origin_display, destination_display
    
    def launch(self, share=False, server_name="127.0.0.1", server_port=7860):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a shareable link
            server_name: Server name or IP address
            server_port: Server port
            
        Returns:
            gr.Blocks: Launched interface
        """
        interface = self.create_interface()
        return interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


def main():
    """Main function to launch the TBRGS Gradio application."""
    try:
        # Create and launch the app
        app = TBRGSGradioApp()
        app.launch()
    
    except Exception as e:
        logger.error(f"Error launching TBRGS Gradio App: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
