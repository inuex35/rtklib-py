#!/usr/bin/env python3
"""
IMU Data Loader for RTKLib-ISAM2 Integration
Handles loading and preprocessing of IMU data for tight coupling
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# Import ImuMeasurement from rtk_imu_isam
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rtk_imu_isam import ImuMeasurement

class ImuDataLoader:
    """Load and preprocess IMU data from various formats"""
    
    def __init__(self, config: dict):
        """Initialize IMU data loader
        
        Args:
            config: Configuration containing:
                - imu_file: Path to IMU data file
                - imu_rate: Expected IMU rate (Hz)
                - time_offset: Time offset between IMU and GNSS (seconds)
                - format: File format ('csv', 'binary', etc.)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load IMU data
        self.imu_data = self._load_imu_data(config['imu_file'])
        
        # Time synchronization
        self.time_offset = config.get('time_offset', 0.0)
        self.expected_rate = config.get('imu_rate', 100.0)
        
        # Validate data
        self._validate_data()
        
    def _load_imu_data(self, imu_file: str) -> pd.DataFrame:
        """Load IMU data from file
        
        Args:
            imu_file: Path to IMU data file
            
        Returns:
            DataFrame with columns: timestamp, ax, ay, az, wx, wy, wz
        """
        file_path = Path(imu_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"IMU file not found: {imu_file}")
            
        self.logger.info(f"Loading IMU data from: {imu_file}")
        
        # Determine format and load accordingly
        if file_path.suffix == '.csv':
            return self._load_csv_imu(file_path)
        elif file_path.suffix == '.txt':
            return self._load_text_imu(file_path)
        else:
            raise ValueError(f"Unsupported IMU file format: {file_path.suffix}")
            
    def _load_csv_imu(self, file_path: Path) -> pd.DataFrame:
        """Load IMU data from CSV file
        
        Expected format:
        timestamp, ax, ay, az, wx, wy, wz
        or
        time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        or PPC-Dataset format:
        GPS TOW (s), GPS Week, Acc X (m/s^2), Acc Y (m/s^2), Acc Z (m/s^2), Ang Rate X (deg/s), Ang Rate Y (deg/s), Ang Rate Z (deg/s)
        """
        # Try to infer column names
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Check for PPC-Dataset format
        if 'GPS TOW (s)' in df.columns:
            # PPC-Dataset format
            # Convert GPS time to timestamp using gnss_lib_py
            try:
                import gnss_lib_py as glp
                gps_week = df['GPS Week'].values
                gps_tow = df['GPS TOW (s)'].values
                
                # Convert GPS week and TOW to Unix milliseconds
                unix_millis = glp.tow_to_unix_millis(gps_week, gps_tow)
                
                # Convert milliseconds to seconds
                timestamps = unix_millis / 1000.0
            except ImportError:
                # Fallback to manual conversion if gnss_lib_py not available
                gps_week = df['GPS Week'].values
                gps_tow = df['GPS TOW (s)'].values
                # GPS epoch: January 6, 1980 00:00:00 UTC = 315964800 Unix timestamp
                # Each week is 604800 seconds
                gps_epoch = 315964800
                # Add leap seconds (as of 2024, GPS-UTC = 18 seconds)
                leap_seconds = 18
                timestamps = gps_epoch + gps_week * 604800 + gps_tow - leap_seconds
            
            # Create new dataframe with standardized columns
            new_df = pd.DataFrame({
                'timestamp': timestamps,
                'ax': df['Acc X (m/s^2)'].values,
                'ay': df['Acc Y (m/s^2)'].values,
                'az': df['Acc Z (m/s^2)'].values,
                'wx': np.radians(df['Ang Rate X (deg/s)'].values),  # Convert to rad/s
                'wy': np.radians(df['Ang Rate Y (deg/s)'].values),
                'wz': np.radians(df['Ang Rate Z (deg/s)'].values)
            })
            df = new_df
        else:
            # Standard format - standardize column names
            column_mapping = {
                'time': 'timestamp',
                'accel_x': 'ax', 'accel_y': 'ay', 'accel_z': 'az',
                'gyro_x': 'wx', 'gyro_y': 'wy', 'gyro_z': 'wz',
                'acc_x': 'ax', 'acc_y': 'ay', 'acc_z': 'az',
                'gyr_x': 'wx', 'gyr_y': 'wy', 'gyr_z': 'wz',
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Check required columns
            required_cols = ['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                # Try alternative format
                if len(df.columns) >= 7:
                    # Assume columns are in order: time, ax, ay, az, wx, wy, wz
                    df.columns = ['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'] + [f'extra_{i}' for i in range(len(df.columns) - 7)]
                    df = df[required_cols]
                else:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                    
        # Select only required columns
        df = df[['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']]
        
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        self.logger.info(f"Loaded {len(df)} IMU measurements")
        
        return df
        
    def _load_text_imu(self, file_path: Path) -> pd.DataFrame:
        """Load IMU data from space-delimited text file"""
        # Similar to CSV but with different delimiter
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        
        # Assume standard column order
        if df.shape[1] >= 7:
            df.columns = ['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'] + [f'extra_{i}' for i in range(df.shape[1] - 7)]
            df = df[['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']]
        else:
            raise ValueError(f"Insufficient columns in IMU file: {df.shape[1]}")
            
        return df
        
    def _validate_data(self):
        """Validate loaded IMU data"""
        if self.imu_data.empty:
            raise ValueError("No IMU data loaded")
            
        # Check for NaN values
        nan_count = self.imu_data.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in IMU data")
            # Remove rows with NaN
            self.imu_data.dropna(inplace=True)
            
        # Check data rate
        if len(self.imu_data) > 1:
            time_diffs = np.diff(self.imu_data['timestamp'])
            mean_dt = np.mean(time_diffs)
            actual_rate = 1.0 / mean_dt
            
            if abs(actual_rate - self.expected_rate) > 5.0:
                self.logger.warning(
                    f"IMU rate mismatch: expected {self.expected_rate} Hz, "
                    f"got {actual_rate:.1f} Hz"
                )
                
        # Check for reasonable values (basic sanity check)
        accel_mag = np.linalg.norm(self.imu_data[['ax', 'ay', 'az']].values, axis=1)
        if np.mean(accel_mag) < 5.0 or np.mean(accel_mag) > 15.0:
            self.logger.warning(
                f"Unusual accelerometer magnitudes: mean = {np.mean(accel_mag):.2f} m/s²"
            )
            
    def get_measurements_between(self, start_time: float, end_time: float) -> List[ImuMeasurement]:
        """Get IMU measurements between two timestamps
        
        Args:
            start_time: Start timestamp (GPS seconds)
            end_time: End timestamp (GPS seconds)
            
        Returns:
            List of ImuMeasurement objects
        """
        # Apply time offset
        start_time_imu = start_time - self.time_offset
        end_time_imu = end_time - self.time_offset
        
        # Filter data
        mask = (self.imu_data['timestamp'] >= start_time_imu) & \
               (self.imu_data['timestamp'] <= end_time_imu)
        
        filtered_data = self.imu_data[mask]
        
        # Convert to ImuMeasurement objects
        measurements = []
        for _, row in filtered_data.iterrows():
            meas = ImuMeasurement(
                timestamp=row['timestamp'] + self.time_offset,  # Convert back to GPS time
                accelerometer=np.array([row['ax'], - row['ay'], - row['az']]),
                gyroscope=np.array([row['wx'], - row['wy'], - row['wz']])
            )
            measurements.append(meas)
            
        return measurements
        
    def get_measurement_at_time(self, timestamp: float, time_tolerance: float = 0.01) -> Optional[ImuMeasurement]:
        """Get IMU measurement closest to specified timestamp
        
        Args:
            timestamp: Desired timestamp (GPS seconds)
            time_tolerance: Maximum time difference to accept (seconds)
            
        Returns:
            ImuMeasurement object or None if no data within tolerance
        """
        # Apply time offset
        timestamp_imu = timestamp - self.time_offset
        
        # Find closest measurement
        time_diffs = np.abs(self.imu_data['timestamp'] - timestamp_imu)
        min_idx = np.argmin(time_diffs)
        min_diff = time_diffs.iloc[min_idx]
        
        if min_diff > time_tolerance:
            return None
            
        row = self.imu_data.iloc[min_idx]
        
        return ImuMeasurement(
            timestamp=row['timestamp'] + self.time_offset,
            accelerometer=np.array([row['ax'], row['ay'], row['az']]),
            gyroscope=np.array([row['wx'], row['wy'], row['wz']])
        )
        
    def get_static_measurements(self, duration: float = 10.0) -> List[ImuMeasurement]:
        """Get initial static measurements for bias estimation
        
        Args:
            duration: Duration of static period (seconds)
            
        Returns:
            List of ImuMeasurement objects from static period
        """
        if self.imu_data.empty:
            return []
            
        start_time = self.imu_data['timestamp'].iloc[0]
        end_time = start_time + duration
        
        return self.get_measurements_between(
            start_time + self.time_offset,
            end_time + self.time_offset
        )
        
    def get_time_range(self) -> Tuple[float, float]:
        """Get time range of IMU data
        
        Returns:
            (start_time, end_time) in GPS seconds
        """
        if self.imu_data.empty:
            return (0.0, 0.0)
            
        start_time = self.imu_data['timestamp'].iloc[0] + self.time_offset
        end_time = self.imu_data['timestamp'].iloc[-1] + self.time_offset
        
        return (start_time, end_time)
        
    def resample(self, target_rate: float) -> 'ImuDataLoader':
        """Resample IMU data to target rate
        
        Args:
            target_rate: Target sampling rate (Hz)
            
        Returns:
            New ImuDataLoader with resampled data
        """
        if target_rate >= self.expected_rate:
            self.logger.warning(
                f"Target rate {target_rate} Hz is higher than current rate {self.expected_rate} Hz"
            )
            
        # Create time index
        self.imu_data.set_index('timestamp', inplace=True)
        
        # Resample
        target_period = f'{int(1000/target_rate)}ms'
        resampled = self.imu_data.resample(target_period).mean()
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        # Create new loader with resampled data
        new_config = self.config.copy()
        new_config['imu_rate'] = target_rate
        
        new_loader = ImuDataLoader.__new__(ImuDataLoader)
        new_loader.config = new_config
        new_loader.logger = self.logger
        new_loader.imu_data = resampled
        new_loader.time_offset = self.time_offset
        new_loader.expected_rate = target_rate
        
        return new_loader