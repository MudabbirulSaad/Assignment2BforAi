import os
import pandas as pd
import numpy as np
import math
import json
from sklearn.preprocessing import MinMaxScaler

# Load configuration from JSON file
with open('config/default_config.json', 'r') as f:
    config = json.load(f)

# --- CONFIG ---
RAW_DATA_PATH = config['paths']['raw_data']
PROCESSED_CSV_PATH = config['paths']['processed_data']
SEQUENCE_DATA_PATH = config['paths']['sequence_data']
NODES_CSV_PATH = config['paths']['nodes_data']
EDGES_CSV_PATH = config['paths']['edges_data']

def reshape_data(filepath):
    try:
        df = pd.read_excel(filepath, header=None, engine='openpyxl')

        # Time headers: row 0, columns 10+ (K onward)
        time_intervals = df.iloc[0, 10:].tolist()

        records = []

        # Iterate from row 2 onward (index 2+)
        for i in range(2, df.shape[0]):
            date_cell = df.iloc[i, 9]  # Column J = index 9
            try:
                date = pd.to_datetime(str(date_cell)).date()
            except:
                print(f"[WARN] Skipping invalid date at row {i}: {date_cell}")
                continue

            for j, time in enumerate(time_intervals):
                try:
                    time_obj = pd.to_datetime(str(time)).time()
                    timestamp = pd.Timestamp.combine(date, time_obj)
                    flow = df.iloc[i, 10 + j]
                    records.append({"Timestamp": timestamp, "Flow": flow})
                except Exception as e:
                    print(f"[WARN] Row {i}, Col {j}: {e}")

        df_long = pd.DataFrame(records).sort_values("Timestamp").set_index("Timestamp")
        print(f"[INFO] Time-series created with {len(df_long)} rows.")
        return df_long

    except Exception as e:
        print("[ERROR] Reshaping failed:", e)
        return None

def clean_and_normalize(df):
    try:
        df = df.resample("15min").mean()
        df = df.fillna(method="ffill")
        df = df[df["Flow"] >= 0]

        scaler = MinMaxScaler()
        df["Flow_norm"] = scaler.fit_transform(df[["Flow"]])
        print("[INFO] Data cleaned and normalized.")
        return df
    except Exception as e:
        print("[ERROR] Cleaning failed:", e)
        return None

def create_sequences(values, window_size):
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y)

def split_data(df, window_size=16):
    values = df["Flow_norm"].values
    split = int(len(values) * 0.8)
    X_train, y_train = create_sequences(values[:split], window_size)
    X_test, y_test = create_sequences(values[split:], window_size)

    X_train = X_train.reshape((-1, window_size, 1))
    X_test = X_test.reshape((-1, window_size, 1))
    return X_train, y_train, X_test, y_test

def save_all(df, X_train, y_train, X_test, y_test):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    
    df.to_csv(PROCESSED_CSV_PATH)
    print(f"[INFO] Saved CSV: {PROCESSED_CSV_PATH}")

    np.savez(SEQUENCE_DATA_PATH,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             data_min=df['Flow'].min(),
             data_max=df['Flow'].max())
    print(f"[INFO] Saved sequence data: {SEQUENCE_DATA_PATH}")

def extract_scats_sites(filepath):
    """
    Extract SCATS site information from the raw data file.
    """
    try:
        df = pd.read_excel(filepath, header=None, engine='openpyxl')
        
        # SCATS site information is in the first few columns of the Excel file
        # Extract site IDs from column A and site names from columns B-I
        scats_sites = {}
        
        # Start from row 2 (index 2) to skip headers
        for i in range(2, df.shape[0]):
            try:
                # Extract site ID from column A
                site_id = int(df.iloc[i, 0])
                
                # Extract site name components from columns B-I
                # Typically these are road names, directions, etc.
                site_components = []
                for j in range(1, 9):  # Columns B through I
                    if pd.notna(df.iloc[i, j]) and df.iloc[i, j] != '':
                        site_components.append(str(df.iloc[i, j]).strip())
                
                # Create a site name from the components
                # Format as ROAD1/ROAD2 for intersection
                if len(site_components) >= 2:
                    site_name = f"{site_components[0]}/{site_components[1]}"
                else:
                    site_name = site_components[0] if site_components else f"SITE_{site_id}"
                
                # Generate approximate coordinates based on site ID
                # This is a simplified approach - in a real system, you would use actual coordinates
                # We're using the site ID to generate pseudo-random but consistent coordinates
                # within the Boroondara area
                base_lat = -37.82  # Base latitude for Boroondara area
                base_lon = 145.05  # Base longitude for Boroondara area
                
                # Use the site ID to generate a deterministic offset
                lat_offset = (site_id % 100) * 0.001
                lon_offset = (site_id // 100) * 0.002
                
                latitude = base_lat - lat_offset
                longitude = base_lon + lon_offset
                
                # Store the site information
                scats_sites[site_id] = (site_name, latitude, longitude)
                
            except Exception as e:
                print(f"[WARN] Error processing site at row {i}: {e}")
                continue
        
        # If we couldn't extract any sites, use a fallback set for demonstration
        if not scats_sites:
            print("[WARN] Could not extract SCATS sites from data, using fallback set")
            # Fallback set of SCATS sites in the Boroondara area
            scats_sites = {
                # Format: site_id: (name, latitude, longitude)
                2000: ("WARRIGAL_RD/TOORAK_RD", -37.8553, 145.0899),
                2001: ("BURKE_RD/TOORAK_RD", -37.8524, 145.0574),
                2002: ("GLENFERRIE_RD/TOORAK_RD", -37.8503, 145.0234),
                2003: ("KOOYONG_RD/TOORAK_RD", -37.8487, 145.0095),
                2004: ("AUBURN_RD/TOORAK_RD", -37.8513, 145.0366),
                2005: ("TOORONGA_RD/TOORAK_RD", -37.8538, 145.0733),
                3000: ("DENMARK_ST/KOONUNG_RD", -37.7975, 145.0814),
                3001: ("DENMARK_ST/DONCASTER_RD", -37.7928, 145.0819),
                3002: ("DENMARK_ST/BARKERS_RD", -37.8036, 145.0808),
                3003: ("DENMARK_ST/HIGHFIELD_RD", -37.8098, 145.0802),
                3004: ("DENMARK_ST/WHITEHORSE_RD", -37.8169, 145.0795),
                4000: ("WARRIGAL_RD/CANTERBURY_RD", -37.8248, 145.0899),
                4001: ("WARRIGAL_RD/RIVERSDALE_RD", -37.8315, 145.0899),
                4002: ("WARRIGAL_RD/HIGHFIELD_RD", -37.8098, 145.0899),
                4003: ("WARRIGAL_RD/WHITEHORSE_RD", -37.8169, 145.0899),
                4004: ("WARRIGAL_RD/CAMBERWELL_RD", -37.8419, 145.0899),
            }
            print(f"[INFO] Created fallback set with {len(scats_sites)} SCATS sites")
        
        # Create a DataFrame for nodes
        nodes_df = pd.DataFrame([
            {"id": site_id, "name": info[0], "latitude": info[1], "longitude": info[2]}
            for site_id, info in scats_sites.items()
        ])
        
        # Create edges based on proximity
        edges = []
        
        # Helper function to calculate distance between two points
        def haversine_distance(lat1, lon1, lat2, lon2):
            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371000  # Radius of Earth in meters
            return c * r
        
        # Create a list of all sites with their coordinates
        all_sites = [(site_id, name, lat, lon) for site_id, (name, lat, lon) in scats_sites.items()]
        
        # Create a simplified road network by connecting nearby sites
        # This ensures we have a connected graph even if we can't extract road names properly
        for i, (site1_id, name1, lat1, lon1) in enumerate(all_sites):
            # Connect to the 3 nearest neighbors
            distances = []
            for j, (site2_id, name2, lat2, lon2) in enumerate(all_sites):
                if i != j:  # Don't connect to self
                    distance = haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append((j, site2_id, distance))
            
            # Sort by distance and connect to the 3 nearest
            distances.sort(key=lambda x: x[2])  # Sort by distance
            for _, site2_id, distance in distances[:3]:  # Connect to 3 nearest
                # Add edges in both directions
                edges.append({"source": site1_id, "target": site2_id, "distance": distance})
                edges.append({"source": site2_id, "target": site1_id, "distance": distance})
        
        # If we have road names, also try to connect sites on the same road
        try:
            # Extract road names from site names
            road_groups = {}
            for site_id, (name, lat, lon) in scats_sites.items():
                if '/' in name:
                    road_parts = name.split('/')
                    for road_name in road_parts:
                        if road_name not in road_groups:
                            road_groups[road_name] = []
                        road_groups[road_name].append((site_id, lat, lon))
            
            # Connect sites on the same road
            for road, sites in road_groups.items():
                if len(sites) > 1:  # Only process roads with multiple sites
                    # Sort sites by distance from a reference point
                    ref_lat, ref_lon = sites[0][1], sites[0][2]  # Use first site as reference
                    sites_with_dist = [(site_id, lat, lon, haversine_distance(ref_lat, ref_lon, lat, lon)) 
                                      for site_id, lat, lon in sites]
                    sites_with_dist.sort(key=lambda x: x[3])  # Sort by distance from reference
                    
                    # Connect adjacent sites on the same road
                    for i in range(len(sites_with_dist) - 1):
                        site1_id, lat1, lon1, _ = sites_with_dist[i]
                        site2_id, lat2, lon2, _ = sites_with_dist[i + 1]
                        
                        # Calculate actual distance
                        distance = haversine_distance(lat1, lon1, lat2, lon2)
                        
                        # Add edges in both directions
                        edges.append({"source": site1_id, "target": site2_id, "distance": distance})
                        edges.append({"source": site2_id, "target": site1_id, "distance": distance})
        except Exception as e:
            print(f"[WARN] Error connecting sites on the same road: {e}")
            # Continue with the nearest-neighbor connections we already made
        
        # Remove duplicate edges
        edges_df = pd.DataFrame(edges).drop_duplicates(subset=['source', 'target'])
        
        return nodes_df, edges_df
    
    except Exception as e:
        print("[ERROR] Extracting SCATS sites failed:", e)
        return None, None

def save_graph_data(nodes_df, edges_df):
    """
    Save the graph data to CSV files.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(NODES_CSV_PATH), exist_ok=True)
        
        # Save nodes and edges to CSV
        nodes_df.to_csv(NODES_CSV_PATH, index=False)
        edges_df.to_csv(EDGES_CSV_PATH, index=False)
        
        print(f"[INFO] Saved graph data: {len(nodes_df)} nodes and {len(edges_df)} edges")
        print(f"[INFO] Nodes file: {NODES_CSV_PATH}")
        print(f"[INFO] Edges file: {EDGES_CSV_PATH}")
        
        return True
    except Exception as e:
        print("[ERROR] Saving graph data failed:", e)
        return False

def main():
    # Process traffic flow data
    df = reshape_data(RAW_DATA_PATH)
    if df is not None:
        df_clean = clean_and_normalize(df)
        if df_clean is not None:
            X_train, y_train, X_test, y_test = split_data(df_clean)
            save_all(df_clean, X_train, y_train, X_test, y_test)
    
    # Extract and save graph data
    nodes_df, edges_df = extract_scats_sites(RAW_DATA_PATH)
    if nodes_df is not None and edges_df is not None:
        save_graph_data(nodes_df, edges_df)

if __name__ == "__main__":
    main()
