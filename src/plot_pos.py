#!/usr/bin/env python3
"""
Plot RTKLib POS file results with map visualization
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import folium
from folium import plugins
import webbrowser


def read_pos_file(filename):
    """Read RTKLib POS file"""
    data = {
        'time': [],
        'lat': [],
        'lon': [],
        'height': [],
        'Q': [],
        'ns': [],
        'sdn': [],
        'sde': [],
        'sdu': [],
        'ratio': []
    }
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            try:
                # Parse fields
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
                ratio = float(parts[14])
                
                # Store data
                data['time'].append(tow)
                data['lat'].append(lat)
                data['lon'].append(lon)
                data['height'].append(height)
                data['Q'].append(Q)
                data['ns'].append(ns)
                data['sdn'].append(sdn)
                data['sde'].append(sde)
                data['sdu'].append(sdu)
                data['ratio'].append(ratio)
                
            except (ValueError, IndexError):
                continue
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def plot_trajectory(data, output_dir=None):
    """Plot trajectory and solution quality"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Horizontal trajectory
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(data['lon'], data['lat'], c=data['Q'], 
                         cmap='RdYlGn', s=10, alpha=0.7)
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title('Horizontal Trajectory')
    ax1.grid(True)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Solution Quality')
    
    # 2. Height profile
    ax2 = plt.subplot(2, 3, 2)
    time_rel = data['time'] - data['time'][0]
    ax2.plot(time_rel, data['height'], 'b-', linewidth=1)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Height [m]')
    ax2.set_title('Height Profile')
    ax2.grid(True)
    
    # 3. Solution quality over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_rel, data['Q'], 'g.', markersize=4)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Solution Quality')
    ax3.set_title('Solution Quality vs Time')
    ax3.set_ylim(0, 6)
    ax3.grid(True)
    
    # 4. Number of satellites
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_rel, data['ns'], 'b.', markersize=4)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Number of Satellites')
    ax4.set_title('Satellite Count')
    ax4.grid(True)
    
    # 5. Position uncertainties
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_rel, data['sdn'], 'r-', label='North', linewidth=1)
    ax5.plot(time_rel, data['sde'], 'g-', label='East', linewidth=1)
    ax5.plot(time_rel, data['sdu'], 'b-', label='Up', linewidth=1)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Std Dev [m]')
    ax5.set_title('Position Uncertainties')
    ax5.legend()
    ax5.grid(True)
    ax5.set_ylim(bottom=0)
    
    # 6. AR ratio
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(time_rel, data['ratio'], 'k.', markersize=4)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('AR Ratio')
    ax6.set_title('Ambiguity Resolution Ratio')
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output directory specified
    if output_dir:
        output_path = os.path.join(output_dir, 'pos_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def create_map(data, output_dir=None):
    """Create interactive map with trajectory"""
    
    # Calculate center of trajectory
    center_lat = data['lat'].mean()
    center_lon = data['lon'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18)
    
    # Prepare trajectory points
    points = []
    colors = []
    
    # Color mapping for solution quality
    color_map = {
        1: '#00FF00',  # Fixed: Green
        2: '#FFFF00',  # Float: Yellow  
        5: '#FF0000',  # Single: Red
    }
    
    for i in range(len(data['lat'])):
        points.append([data['lat'][i], data['lon'][i]])
        colors.append(color_map.get(data['Q'][i], '#808080'))
    
    # Add trajectory as colored polyline segments
    for i in range(len(points) - 1):
        folium.PolyLine(
            points[i:i+2],
            color=colors[i],
            weight=3,
            opacity=0.8
        ).add_to(m)
    
    # Add start and end markers
    folium.Marker(
        points[0],
        popup='Start',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        points[-1],
        popup='End',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 150px; height: 100px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px">
    <p style="margin: 0;"><b>Solution Quality</b></p>
    <p style="margin: 0;"><span style="color: #00FF00;">●</span> Fixed</p>
    <p style="margin: 0;"><span style="color: #FFFF00;">●</span> Float</p>
    <p style="margin: 0;"><span style="color: #FF0000;">●</span> Single</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    map_file = 'trajectory_map.html'
    if output_dir:
        map_file = os.path.join(output_dir, map_file)
    
    m.save(map_file)
    print(f"Map saved to: {map_file}")
    
    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(map_file)}')


def print_statistics(data):
    """Print solution statistics"""
    total = len(data['Q'])
    
    # Count solution types
    # Q=1: Fix, Q=2: Float, Q=5: Single
    q_fix = np.sum(data['Q'] == 1)
    q_float = np.sum(data['Q'] == 2) 
    q_single = np.sum(data['Q'] == 5)
    
    print("\n=== Solution Statistics ===")
    print(f"Total epochs: {total}")
    print(f"Fixed solutions: {q_fix} ({q_fix/total*100:.1f}%)")
    print(f"Float solutions: {q_float} ({q_float/total*100:.1f}%)")
    print(f"Single solutions: {q_single} ({q_single/total*100:.1f}%)")
    
    # Position statistics
    print(f"\nPosition range:")
    print(f"  Latitude:  {data['lat'].min():.8f} - {data['lat'].max():.8f} deg")
    print(f"  Longitude: {data['lon'].min():.8f} - {data['lon'].max():.8f} deg")
    print(f"  Height:    {data['height'].min():.3f} - {data['height'].max():.3f} m")
    
    # Satellite statistics
    print(f"\nSatellite count:")
    print(f"  Min: {data['ns'].min()}")
    print(f"  Max: {data['ns'].max()}")
    print(f"  Mean: {data['ns'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Plot RTKLib POS file results')
    parser.add_argument('pos_file', help='Path to POS file')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--no-map', action='store_true', help='Skip map generation')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pos_file):
        print(f"Error: File '{args.pos_file}' not found")
        return
    
    print(f"Reading POS file: {args.pos_file}")
    data = read_pos_file(args.pos_file)
    
    if len(data['time']) == 0:
        print("Error: No valid data found in POS file")
        return
    
    # Print statistics
    print_statistics(data)
    
    # Plot results
    plot_trajectory(data, args.output)
    
    # Create map
    if not args.no_map:
        create_map(data, args.output)


if __name__ == '__main__':
    main()