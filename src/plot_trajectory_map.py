#!/usr/bin/env python3
"""
Plot RTK-IMU trajectory on an interactive map using Folium
"""

import numpy as np
import pandas as pd
import folium
from folium import plugins
import sys
import os
from datetime import datetime

def read_pos_file(filename):
    """Read RTKPOS solution file"""
    
    # Read the file, skipping the header
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if len(parts) >= 15:
                week = int(parts[0])
                tow = float(parts[1])
                lat = float(parts[2])
                lon = float(parts[3])
                height = float(parts[4])
                Q = int(parts[5])
                ns = int(parts[6])
                sdn = float(parts[7])
                sde = float(parts[8])
                sdu = float(parts[9])
                
                # Convert GPS time to datetime
                # GPS epoch: 1980-01-06 00:00:00
                gps_epoch = datetime(1980, 1, 6)
                # Add weeks and seconds
                timestamp = gps_epoch + pd.Timedelta(weeks=week, seconds=tow)
                
                data.append({
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'height': height,
                    'Q': Q,
                    'ns': ns,
                    'sdn': sdn,
                    'sde': sde,
                    'sdu': sdu
                })
    
    return pd.DataFrame(data)

def get_quality_color(Q):
    """Get color based on solution quality"""
    colors = {
        1: 'green',   # Fixed
        2: 'yellow',  # Float
        3: 'orange',  # SBAS
        4: 'red',     # DGPS
        5: 'blue',    # Single
        6: 'gray'     # PPP
    }
    return colors.get(Q, 'black')

def get_quality_name(Q):
    """Get quality name"""
    names = {
        1: 'Fixed',
        2: 'Float', 
        3: 'SBAS',
        4: 'DGPS',
        5: 'Single',
        6: 'PPP'
    }
    return names.get(Q, 'Unknown')

def create_trajectory_map(df, output_file='trajectory_map.html'):
    """Create interactive map with trajectory"""
    
    # Calculate center of the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Group points by quality
    quality_groups = df.groupby('Q')
    
    for Q, group in quality_groups:
        # Create feature group for each quality type
        fg = folium.FeatureGroup(name=f'{get_quality_name(Q)} ({len(group)} points)')
        
        # Add points
        for idx, row in group.iterrows():
            popup_text = f"""
            Time: {row['timestamp']}<br>
            Lat: {row['latitude']:.8f}<br>
            Lon: {row['longitude']:.8f}<br>
            Height: {row['height']:.3f} m<br>
            Quality: {get_quality_name(row['Q'])}<br>
            Satellites: {row['ns']}<br>
            Std N: {row['sdn']:.3f} m<br>
            Std E: {row['sde']:.3f} m<br>
            Std U: {row['sdu']:.3f} m
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=300),
                color=get_quality_color(row['Q']),
                fill=True,
                fillColor=get_quality_color(row['Q']),
                fillOpacity=0.7
            ).add_to(fg)
        
        fg.add_to(m)
    
    # Add trajectory line
    if len(df) > 1:
        points = df[['latitude', 'longitude']].values.tolist()
        folium.PolyLine(
            points,
            color='red',
            weight=2,
            opacity=0.8,
            name='Trajectory'
        ).add_to(m)
    
    # Add start and end markers
    if len(df) > 0:
        # Start point
        folium.Marker(
            location=[df.iloc[0]['latitude'], df.iloc[0]['longitude']],
            popup=f"Start: {df.iloc[0]['timestamp']}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # End point
        folium.Marker(
            location=[df.iloc[-1]['latitude'], df.iloc[-1]['longitude']],
            popup=f"End: {df.iloc[-1]['timestamp']}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measurement tool
    plugins.MeasureControl().add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Map saved to: {output_file}")
    
    # Print summary statistics
    print("\nTrajectory Summary:")
    print(f"Total points: {len(df)}")
    print(f"Duration: {df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']}")
    print(f"Distance covered: ~{calculate_distance(df):.1f} m")
    print("\nSolution quality breakdown:")
    for Q, count in df['Q'].value_counts().sort_index().items():
        print(f"  {get_quality_name(Q)}: {count} ({100*count/len(df):.1f}%)")

def calculate_distance(df):
    """Calculate approximate total distance traveled"""
    if len(df) < 2:
        return 0.0
        
    # Convert to radians
    lat = np.radians(df['latitude'].values)
    lon = np.radians(df['longitude'].values)
    
    # Calculate differences
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # Earth radius in meters
    R = 6371000
    distances = R * c
    
    return np.sum(distances)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectory_map.py <pos_file> [output_html]")
        sys.exit(1)
    
    pos_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'trajectory_map.html'
    
    # Check if file exists
    if not os.path.exists(pos_file):
        print(f"Error: File {pos_file} not found")
        sys.exit(1)
    
    # Read position data
    print(f"Reading position file: {pos_file}")
    df = read_pos_file(pos_file)
    
    if len(df) == 0:
        print("Error: No valid data found in file")
        sys.exit(1)
    
    # Create map
    create_trajectory_map(df, output_file)
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
    except:
        pass

if __name__ == "__main__":
    main()