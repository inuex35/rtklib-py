"""
Data loader for GNSS and IMU data
"""

import os
import sys
import numpy as np
from typing import Optional, List, Tuple
import pandas as pd


class DataLoader:
    """Unified data loader for GNSS and IMU data"""
    
    def __init__(self, config):
        self.config = config
        self.nav = None
        self.rov_obs = None
        self.base_obs = None
        self.imu_data = None
        
        # Add parent directories to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
    def load_all(self):
        """Load all data files"""
        self.load_navigation()
        self.load_observations()
        if self.config.use_imu and self.config.imufile:
            self.load_imu()
            
    def load_navigation(self):
        """Load navigation data"""
        # Add src directory to path if needed
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            
        import rtkcmn as gn
        from rtkpos import rtkinit
        
        # Try to import existing config or use minimal config
        try:
            import __ppk_config as cfg
        except ImportError:
            # Create a minimal config object for rtkinit
            class MinimalConfig:
                def __init__(self, isam_config):
                    self.mode = isam_config.mode
                    self.navsys = isam_config.navsys
                    self.elmin = isam_config.elmin * np.pi / 180.0  # Convert to radians
                    self.snrmin = isam_config.snrmin
                    self.filtertype = isam_config.filtertype
                    self.trace = isam_config.trace_level
                    # Add other required RTKLib config parameters
                    self.nf = 2  # Number of frequencies
                    self.soltype = 0  # Forward
                    self.tidecorr = 0
                    self.ionoopt = 1  # Broadcast model
                    self.troopt = 1  # Saastamoinen model
                    self.sateph = 0  # Broadcast ephemeris
                    self.maxdtoe = 7200.0
                    self.maxgdop = 30.0
                    self.thresar = 3.0
                    self.maxout = 5
                    self.minlock = 0
                    self.minfix = 10
                    self.glomodear = 1
                    self.gpsmodear = 1
                    self.bdsmodear = 1
                    self.arfilter = 1
                    self.maxage = 30.0
                    self.syncsol = 0
                    self.slipthres = 0.05
                    self.rejionno = 30.0
                    self.rejgdop = 30.0
                    self.niter = 1
                    self.baselen = 0.0
                    self.basesig = 0.0
                    # Add missing attributes that might be needed
                    self.gnss_t = (0, 0)  # Start/end time
                    self.rb = [0.0, 0.0, 0.0]  # Base position
                    
            cfg = MinimalConfig(self.config)
        
        # Initialize navigation structure
        gn.tracelevel(self.config.trace_level)
        self.nav = rtkinit(cfg)
        self.nav.maxepoch = self.config.maxepoch
        
        # Set base position if provided
        if self.config.basepos and self.config.basepos != [0.0, 0.0, 0.0]:
            self.nav.rb = self.config.basepos
            
    def load_observations(self):
        """Load GNSS observations"""
        import rinex as rn
        
        # Change to data directory for file access
        original_dir = os.getcwd()
        os.chdir(self.config.datadir)
        
        try:
            # Load rover observations
            print(f'Loading rover observations: {self.config.rovfile}')
            # Create rnx_decode with nav.opt or create minimal options
            class MinimalOpt:
                def __init__(self):
                    self.ts = gn.Gtime()
                    self.te = gn.Gtime()
                    self.ti = 30.0
                    self.tu = 0
                    
            opt = MinimalOpt() if not hasattr(self.nav, 'opt') else self.nav.opt
            rov = rn.rnx_decode(opt)
            
            max_obs = None
            if self.config.filtertype != 'backward' and self.config.maxepoch > 0:
                max_obs = self.config.maxepoch
                
            rov.decode_obsfile(self.nav, self.config.rovfile, max_obs)
            self.rov_obs = rov
            
            # Load base observations
            if self.config.basefile:
                print(f'Loading base observations: {self.config.basefile}')
                base = rn.rnx_decode(opt)
                base.decode_obsfile(self.nav, self.config.basefile, None)
                self.base_obs = base
                
                # Use base position if not set
                if self.nav.rb[0] == 0:
                    self.nav.rb = base.pos
                    
            # Load navigation data
            print(f'Loading navigation data: {self.config.navfile}')
            self.rov_obs.decode_nav(self.config.navfile, self.nav)
            
        finally:
            os.chdir(original_dir)
            
    def load_imu(self):
        """Load IMU data"""
        if not self.config.imufile:
            return
            
        imu_path = os.path.join(self.config.datadir, self.config.imufile)
        print(f'Loading IMU data: {imu_path}')
        
        try:
            # Load IMU CSV file
            imu_df = pd.read_csv(imu_path)
            
            # Expected columns: time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            required_cols = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in imu_df.columns]
            if missing_cols:
                # Try alternative column names
                alt_names = {
                    'time': ['timestamp', 't', 'gps_time'],
                    'acc_x': ['ax', 'accel_x', 'acceleration_x'],
                    'acc_y': ['ay', 'accel_y', 'acceleration_y'],
                    'acc_z': ['az', 'accel_z', 'acceleration_z'],
                    'gyro_x': ['gx', 'gyro_x', 'angular_velocity_x'],
                    'gyro_y': ['gy', 'gyro_y', 'angular_velocity_y'],
                    'gyro_z': ['gz', 'gyro_z', 'angular_velocity_z']
                }
                
                # Rename columns if alternatives found
                for std_col in missing_cols:
                    for alt_col in alt_names.get(std_col, []):
                        if alt_col in imu_df.columns:
                            imu_df = imu_df.rename(columns={alt_col: std_col})
                            break
                            
            # Store IMU data
            self.imu_data = {
                'timestamps': imu_df['time'].values,
                'accelerometer': imu_df[['acc_x', 'acc_y', 'acc_z']].values,
                'gyroscope': imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
            }
            
            print(f'Loaded {len(self.imu_data["timestamps"])} IMU measurements')
            
        except Exception as e:
            print(f'Warning: Failed to load IMU data: {e}')
            self.imu_data = None
            
    def get_imu_measurements(self, t_start: float, t_end: float) -> Optional[dict]:
        """Get IMU measurements between two timestamps"""
        if self.imu_data is None:
            return None
            
        # Find measurements in time range
        mask = (self.imu_data['timestamps'] >= t_start) & (self.imu_data['timestamps'] <= t_end)
        
        if not np.any(mask):
            return None
            
        return {
            'timestamps': self.imu_data['timestamps'][mask],
            'accelerometer': self.imu_data['accelerometer'][mask],
            'gyroscope': self.imu_data['gyroscope'][mask]
        }
        
    def get_epoch_count(self) -> int:
        """Get number of epochs to process"""
        if self.rov_obs:
            return self.rov_obs.obs.n
        return 0
        
    def get_epoch_data(self, index: int) -> dict:
        """Get all data for a specific epoch"""
        epoch_data = {
            'time': None,
            'rover_obs': None,
            'base_obs': None,
            'nav': self.nav
        }
        
        if self.rov_obs and index < self.rov_obs.obs.n:
            epoch_data['time'] = self.rov_obs.obs.data[index].time
            epoch_data['rover_obs'] = self.rov_obs.obs.data[index]
            
        if self.base_obs:
            # Find matching base observation
            for base_obs in self.base_obs.obs.data:
                if abs(base_obs.time.time - epoch_data['time'].time) < 0.1:
                    epoch_data['base_obs'] = base_obs
                    break
                    
        return epoch_data