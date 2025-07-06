#!/usr/bin/env python3
"""
Analyze and plot RTK-IMU trajectory with detailed statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

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
                gps_epoch = datetime(1980, 1, 6)
                timestamp = gps_epoch + pd.Timedelta(weeks=week, seconds=tow)
                
                data.append({
                    'timestamp': timestamp,
                    'gps_tow': tow,
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

def calculate_local_coords(df):
    """Convert lat/lon to local ENU coordinates"""
    # Use first point as reference
    ref_lat = np.radians(df.iloc[0]['latitude'])
    ref_lon = np.radians(df.iloc[0]['longitude'])
    ref_h = df.iloc[0]['height']
    
    # Convert all points
    lat = np.radians(df['latitude'].values)
    lon = np.radians(df['longitude'].values)
    h = df['height'].values
    
    # Calculate local coordinates
    dlon = lon - ref_lon
    dlat = lat - ref_lat
    
    # Approximate conversion to meters
    R = 6371000  # Earth radius
    east = R * np.cos(ref_lat) * dlon
    north = R * dlat
    up = h - ref_h
    
    df['east'] = east
    df['north'] = north
    df['up'] = up
    
    return df

def plot_trajectory_analysis(df, output_prefix='trajectory'):
    """Create comprehensive trajectory analysis plots"""
    
    # Calculate local coordinates
    df = calculate_local_coords(df)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 2D trajectory (North-East)
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['east'], df['north'], c=df.index, cmap='viridis', s=20)
    ax1.plot(df['east'], df['north'], 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('2D Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Add start and end markers
    ax1.plot(df['east'].iloc[0], df['north'].iloc[0], 'go', markersize=10, label='Start')
    ax1.plot(df['east'].iloc[-1], df['north'].iloc[-1], 'ro', markersize=10, label='End')
    ax1.legend()
    
    # 2. Height profile
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['gps_tow'] - df['gps_tow'].iloc[0], df['height'], 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Height Profile')
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df['gps_tow'] - df['gps_tow'].iloc[0], df['Q'], 'r.', markersize=8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Solution Quality')
    ax3.set_title('Solution Quality')
    ax3.set_yticks([1, 2, 3, 4, 5, 6])
    ax3.set_yticklabels(['Fixed', 'Float', 'SBAS', 'DGPS', 'Single', 'PPP'])
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of satellites
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(df['gps_tow'] - df['gps_tow'].iloc[0], df['ns'], 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Number of Satellites')
    ax4.set_title('Satellite Count')
    ax4.grid(True, alpha=0.3)
    
    # 5. Position uncertainties
    ax5 = plt.subplot(2, 3, 5)
    time_s = df['gps_tow'] - df['gps_tow'].iloc[0]
    ax5.plot(time_s, df['sdn'], 'b-', label='North', linewidth=1.5)
    ax5.plot(time_s, df['sde'], 'r-', label='East', linewidth=1.5)
    ax5.plot(time_s, df['sdu'], 'g-', label='Up', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Standard Deviation (m)')
    ax5.set_title('Position Uncertainties')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Velocity estimation (from position differences)
    ax6 = plt.subplot(2, 3, 6)
    dt = np.diff(df['gps_tow'].values)
    dx = np.diff(df['east'].values)
    dy = np.diff(df['north'].values)
    dz = np.diff(df['up'].values)
    
    # Avoid division by zero
    mask = dt > 0
    vx = np.zeros_like(dx)
    vy = np.zeros_like(dy)
    vz = np.zeros_like(dz)
    vx[mask] = dx[mask] / dt[mask]
    vy[mask] = dy[mask] / dt[mask]
    vz[mask] = dz[mask] / dt[mask]
    
    speed = np.sqrt(vx**2 + vy**2)
    
    ax6.plot(time_s[1:], speed, 'b-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Horizontal Speed (m/s)')
    ax6.set_title('Estimated Speed')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to: {output_prefix}_analysis.png")
    
    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"Duration: {df['gps_tow'].iloc[-1] - df['gps_tow'].iloc[0]:.1f} seconds")
    print(f"Total horizontal distance: {np.sum(np.sqrt(np.diff(df['east'])**2 + np.diff(df['north'])**2)):.2f} m")
    print(f"Height range: {df['height'].min():.2f} to {df['height'].max():.2f} m")
    print(f"Height variation: {df['height'].max() - df['height'].min():.2f} m")
    print(f"Average number of satellites: {df['ns'].mean():.1f}")
    
    # Quality statistics
    print("\nSolution Quality:")
    quality_counts = df['Q'].value_counts().sort_index()
    quality_names = {1: 'Fixed', 2: 'Float', 3: 'SBAS', 4: 'DGPS', 5: 'Single', 6: 'PPP'}
    for q, count in quality_counts.items():
        print(f"  {quality_names.get(q, f'Q={q}')}: {count} ({100*count/len(df):.1f}%)")
    
    # Position uncertainty statistics
    print("\nPosition Uncertainty (average std dev):")
    print(f"  North: {df['sdn'].mean():.3f} m")
    print(f"  East: {df['sde'].mean():.3f} m")
    print(f"  Up: {df['sdu'].mean():.3f} m")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectory_analysis.py <pos_file> [output_prefix]")
        sys.exit(1)
    
    pos_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'trajectory'
    
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
    
    # Create analysis plots
    plot_trajectory_analysis(df, output_prefix)

if __name__ == "__main__":
    main()