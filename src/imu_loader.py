"""
IMU data loader for RTKLib-py with ISAM2 integration
Loads IMU data for tight coupling with GNSS
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from rtkcmn import trace, gtime_t, gpst2time, time2gpst

class ImuLoader:
    """Load and preprocess IMU data for RTKLib-py"""
    
    def __init__(self, nav):
        """Initialize IMU loader
        
        Args:
            nav: RTKLib navigation structure with IMU configuration
        """
        self.nav = nav
        self.imu_data = None
        self.timestamps = None
        self.accelerations = None
        self.angular_velocities = None
        
        # Load IMU data if file is specified
        if hasattr(nav, 'imu_file') and nav.imu_file:
            self.load_imu_data(nav.imu_file)
            
    def load_imu_data(self, imu_file):
        """Load IMU data from CSV file
        
        Expected format depends on dataset type
        """
        trace(3, f"Loading IMU data from: {imu_file}\n")
        
        if not Path(imu_file).exists():
            trace(2, f"IMU file not found: {imu_file}\n")
            return False
            
        try:
            # Try to detect format by reading the header
            with open(imu_file, 'r') as f:
                header = f.readline()
            
            # Check if PPC-Dataset format by looking for characteristic columns
            if 'GPS TOW' in header and 'GPS Week' in header:
                self._load_ppc_dataset(imu_file)
            else:
                # Generic CSV format
                self._load_generic_csv(imu_file)
                
            trace(3, f"Loaded {len(self.timestamps)} IMU measurements\n")
            return True
            
        except Exception as e:
            trace(2, f"Failed to load IMU data: {e}\n")
            return False
            
    def _load_ppc_dataset(self, imu_file):
        """Load PPC-Dataset IMU format
        
        Format: GPS TOW (s), GPS Week, Acc X (m/s^2), Acc Y (m/s^2), Acc Z (m/s^2), 
                Ang Rate X (deg/s), Ang Rate Y (deg/s), Ang Rate Z (deg/s)
        """
        # Read CSV with proper column names
        df = pd.read_csv(imu_file)
        
        # Get column names (they have spaces, so we need to handle carefully)
        col_names = df.columns.tolist()
        
        # Extract GPS time columns
        gps_tow_col = [c for c in col_names if 'GPS TOW' in c][0]
        gps_week_col = [c for c in col_names if 'GPS Week' in c][0]
        
        # Extract acceleration columns
        acc_x_col = [c for c in col_names if 'Acc X' in c][0]
        acc_y_col = [c for c in col_names if 'Acc Y' in c][0]
        acc_z_col = [c for c in col_names if 'Acc Z' in c][0]
        
        # Extract angular rate columns  
        gyro_x_col = [c for c in col_names if 'Ang Rate X' in c][0]
        gyro_y_col = [c for c in col_names if 'Ang Rate Y' in c][0]
        gyro_z_col = [c for c in col_names if 'Ang Rate Z' in c][0]
        
        # Convert GPS time to timestamps
        # GPS TOW is already in seconds, but need to add GPS leap seconds (18s as of 2024)
        # to match the GNSS time conversion
        gps_leap_seconds = 18.0
        self.timestamps = df[gps_tow_col].values + gps_leap_seconds
        
        # Extract IMU measurements
        self.accelerations = df[[acc_x_col, acc_y_col, acc_z_col]].values
        
        # Convert angular velocities from deg/s to rad/s
        deg2rad = np.pi / 180.0
        self.angular_velocities = df[[gyro_x_col, gyro_y_col, gyro_z_col]].values * deg2rad
        
        # Store full dataframe for additional info
        self.imu_data = df
        
    def _load_generic_csv(self, imu_file):
        """Load generic CSV IMU format
        
        Expected columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        """
        df = pd.read_csv(imu_file)
        
        # Extract data
        self.timestamps = df['timestamp'].values
        self.accelerations = df[['acc_x', 'acc_y', 'acc_z']].values
        self.angular_velocities = df[['gyro_x', 'gyro_y', 'gyro_z']].values
        
        self.imu_data = df
        
    def get_measurements_between(self, t_start, t_end):
        """Get IMU measurements between two timestamps
        
        Args:
            t_start: Start time (GPS seconds)
            t_end: End time (GPS seconds)
            
        Returns:
            List of (timestamp, accel, gyro) tuples
        """
        if self.timestamps is None:
            return []
            
        # Find measurements in time range
        mask = (self.timestamps > t_start) & (self.timestamps <= t_end)
        indices = np.where(mask)[0]
        
        measurements = []
        for i in indices:
            measurements.append((
                self.timestamps[i],
                self.accelerations[i],
                self.angular_velocities[i]
            ))
            
        return measurements
        
    def get_time_range(self):
        """Get time range of IMU data
        
        Returns:
            (start_time, end_time) in GPS seconds
        """
        if self.timestamps is None:
            return (0.0, 0.0)
            
        return (self.timestamps[0], self.timestamps[-1])
        
    def get_imu_rate(self):
        """Estimate IMU data rate
        
        Returns:
            Data rate in Hz
        """
        if self.timestamps is None or len(self.timestamps) < 2:
            return 100.0  # Default
            
        # Compute average time difference
        dt = np.mean(np.diff(self.timestamps))
        return 1.0 / dt if dt > 0 else 100.0