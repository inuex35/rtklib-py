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
            # Check if PPC-Dataset format
            if 'PPC-Dataset' in str(imu_file):
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
        
        Format: GPSWeek,GPSTime,ax,ay,az,wx,wy,wz,status
        """
        # Read CSV
        df = pd.read_csv(imu_file)
        
        # Convert GPS time to Unix timestamp
        import gnss_lib_py as glp
        
        gps_weeks = df['GPSWeek'].values
        gps_tows = df['GPSTime'].values
        
        # Convert to Unix timestamps
        unix_ms = glp.tow_to_unix_millis(gps_weeks, gps_tows)
        self.timestamps = unix_ms / 1000.0
        
        # Extract IMU measurements and apply axis inversion (Y and Z axes)
        self.accelerations = df[['ax', 'ay', 'az']].values
        self.accelerations[:, 1] = self.accelerations[:, 1]  # Invert Y axis
        self.accelerations[:, 2] = -self.accelerations[:, 2]  # Invert Z axis
        
        self.angular_velocities = df[['wx', 'wy', 'wz']].values
        self.angular_velocities[:, 1] = -self.angular_velocities[:, 1]  # Invert Y axis
        self.angular_velocities[:, 2] = -self.angular_velocities[:, 2]  # Invert Z axis
        
        # Store full dataframe for additional info
        self.imu_data = df
        
    def _load_generic_csv(self, imu_file):
        """Load generic CSV IMU format
        
        Expected columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        """
        df = pd.read_csv(imu_file)
        
        # Extract data and apply axis inversion (Y and Z axes)
        self.timestamps = df['timestamp'].values
        self.accelerations = df[['acc_x', 'acc_y', 'acc_z']].values
        self.accelerations[:, 1] = -self.accelerations[:, 1]  # Invert Y axis
        self.accelerations[:, 2] = -self.accelerations[:, 2]  # Invert Z axis
        
        self.angular_velocities = df[['gyro_x', 'gyro_y', 'gyro_z']].values
        self.angular_velocities[:, 1] = -self.angular_velocities[:, 1]  # Invert Y axis
        self.angular_velocities[:, 2] = -self.angular_velocities[:, 2]  # Invert Z axis
        
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