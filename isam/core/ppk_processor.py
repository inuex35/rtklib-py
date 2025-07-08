"""
Main PPK processor using ISAM2
"""

import os
import sys
import numpy as np
import gtsam
from gtsam import symbol
from typing import Optional, List
from copy import deepcopy

from .config import ISAMConfig
from .data_loader import DataLoader
from .optimizer import ISAMOptimizer
from ..factors import create_gnss_factor, create_imu_factor, create_motion_factor


class PPKProcessor:
    """Main PPK processor with ISAM2 optimization"""
    
    def __init__(self, config: ISAMConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.optimizer = ISAMOptimizer(config)
        self.solutions = []
        
        # State tracking
        self.prev_time = None
        self.prev_pose_key = None
        self.prev_vel_key = None
        self.prev_bias_key = None
        
        # IMU preintegration
        self.imu_params = None
        self.imu_preintegrated = None
        
        # Initialize IMU if enabled
        if config.use_imu:
            self._init_imu()
            
    def _init_imu(self):
        """Initialize IMU parameters"""
        # Create IMU preintegration parameters
        self.imu_params = gtsam.PreintegrationParams.MakeSharedU(self.config.imu_gravity)
        
        # Set noise parameters
        acc_noise = self.config.imu_accel_noise_sigma * np.ones(3)
        gyro_noise = self.config.imu_gyro_noise_sigma * np.ones(3)
        self.imu_params.setAccelerometerCovariance(np.diag(acc_noise ** 2))
        self.imu_params.setGyroscopeCovariance(np.diag(gyro_noise ** 2))
        
        # Integration uncertainty
        if self.config.imu_integration_sigma > 0:
            self.imu_params.setIntegrationCovariance(
                self.config.imu_integration_sigma * np.eye(3)
            )
            
    def process(self):
        """Process all epochs"""
        print("Loading data...")
        self.data_loader.load_all()
        
        n_epochs = self.data_loader.get_epoch_count()
        if self.config.maxepoch > 0:
            n_epochs = min(n_epochs, self.config.maxepoch)
            
        print(f"Processing {n_epochs} epochs...")
        
        # Process each epoch
        for i in range(n_epochs):
            self._process_epoch(i)
            
            # Print progress
            if (i + 1) % 10 == 0:
                error = self.optimizer.calculate_error()
                print(f"Epoch {i+1}/{n_epochs}, Error: {error:.3f}")
                
            # Marginalize old states periodically
            if (i + 1) % 100 == 0:
                self.optimizer.marginalize_old_states(keep_last_n=50)
                
        print(f"Processing complete. {len(self.solutions)} solutions generated.")
        return self.solutions
        
    def _process_epoch(self, epoch_idx: int):
        """Process single epoch"""
        # Get epoch data
        epoch_data = self.data_loader.get_epoch_data(epoch_idx)
        time = epoch_data['time']
        
        # Initialize first epoch
        if self.prev_time is None:
            self._initialize_first_epoch(epoch_data)
            self.prev_time = time
            return
            
        # Add IMU factors if available
        if self.config.use_imu and self.data_loader.imu_data is not None:
            self._add_imu_factors(self.prev_time, time)
            
        # Create new state variables
        pose_key = symbol('X', self.optimizer.pose_count)
        vel_key = symbol('V', self.optimizer.vel_count)
        clock_bias_key = symbol('C', self.optimizer.clock_count)
        clock_drift_key = symbol('D', self.optimizer.clock_count)
        
        self.optimizer.pose_count += 1
        self.optimizer.vel_count += 1
        self.optimizer.clock_count += 1
        
        # Add GNSS factors
        self._add_gnss_factors(epoch_data, pose_key, vel_key, clock_bias_key, clock_drift_key)
        
        # Add motion constraints
        if self.config.use_nhc:
            nhc_factor = create_motion_factor(
                'non_holonomic',
                sigma_lateral=self.config.nhc_sigma_lateral,
                sigma_vertical=self.config.nhc_sigma_vertical
            )
            self.optimizer.add_motion_constraint(nhc_factor, [pose_key, vel_key])
            
        # Add initial values (prediction from previous state)
        self._add_initial_values(pose_key, vel_key, clock_bias_key, clock_drift_key, time)
        
        # Update ISAM2
        self.optimizer.update()
        
        # Save solution
        self._save_solution(epoch_data, epoch_idx)
        
        # Update previous state
        self.prev_time = time
        self.prev_pose_key = pose_key
        self.prev_vel_key = vel_key
        
    def _initialize_first_epoch(self, epoch_data):
        """Initialize first epoch with priors"""
        # Get initial position from single point positioning
        nav = epoch_data['nav']
        rover_obs = epoch_data['rover_obs']
        
        # Simple SPP for initial position
        from pntpos import pntpos
        sol = pntpos(rover_obs, nav)
        
        if sol.stat == 0:
            print("Warning: Failed to get initial position")
            pos = np.array([0, 0, 0])
        else:
            pos = sol.rr[:3]
            
        # Create initial pose
        init_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(pos))
        
        # Add pose prior
        pose_cov = np.eye(6)
        pose_cov[:3, :3] *= 100.0  # 10m position uncertainty
        pose_cov[3:, 3:] *= 1.0    # 1 rad rotation uncertainty
        
        self.prev_pose_key = self.optimizer.add_pose_prior(init_pose, pose_cov)
        
        # Add velocity prior
        init_vel = np.zeros(3)
        vel_sigma = np.array([10.0, 10.0, 10.0])  # 10 m/s uncertainty
        self.prev_vel_key = self.optimizer.add_velocity_prior(init_vel, vel_sigma)
        
        # Add IMU bias prior if using IMU
        if self.config.use_imu:
            init_bias = gtsam.imuBias.ConstantBias()
            bias_sigma = np.array([
                self.config.imu_accel_bias_rw_sigma,
                self.config.imu_accel_bias_rw_sigma,
                self.config.imu_accel_bias_rw_sigma,
                self.config.imu_gyro_bias_rw_sigma,
                self.config.imu_gyro_bias_rw_sigma,
                self.config.imu_gyro_bias_rw_sigma
            ])
            self.prev_bias_key = self.optimizer.add_bias_prior(init_bias, bias_sigma)
            
            # Initialize IMU preintegration
            self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(
                self.imu_params, init_bias
            )
            
        # Add clock states
        clock_bias = 0.0  # Initial clock bias
        clock_drift = 0.0  # Initial clock drift
        clock_sigma = np.array([1e-6, 1e-9])  # Clock uncertainties
        
        self.optimizer.add_clock_state(clock_bias, clock_drift, clock_sigma)
        
        # Update optimizer
        self.optimizer.update()
        
    def _add_imu_factors(self, t_prev, t_curr):
        """Add IMU preintegration factors"""
        if self.imu_preintegrated is None:
            return
            
        # Get IMU measurements between epochs
        imu_data = self.data_loader.get_imu_measurements(
            t_prev.time, t_curr.time
        )
        
        if imu_data is None or len(imu_data['timestamps']) == 0:
            return
            
        # Add measurements to preintegration
        for i in range(len(imu_data['timestamps']) - 1):
            dt = imu_data['timestamps'][i+1] - imu_data['timestamps'][i]
            acc = imu_data['accelerometer'][i]
            gyro = imu_data['gyroscope'][i]
            
            self.imu_preintegrated.integrateMeasurement(acc, gyro, dt)
            
        # Create IMU factor
        pose_i_key = self.prev_pose_key
        vel_i_key = self.prev_vel_key
        pose_j_key = symbol('X', self.optimizer.pose_count)
        vel_j_key = symbol('V', self.optimizer.vel_count)
        bias_key = self.prev_bias_key
        
        imu_factor = gtsam.ImuFactor(
            pose_i_key, vel_i_key,
            pose_j_key, vel_j_key,
            bias_key,
            self.imu_preintegrated
        )
        
        self.optimizer.graph.add(imu_factor)
        
        # Reset preintegration for next interval
        prev_bias = self.optimizer.get_bias(self.optimizer.bias_count - 1)
        if prev_bias is None:
            prev_bias = gtsam.imuBias.ConstantBias()
            
        self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(
            self.imu_params, prev_bias
        )
        
    def _add_gnss_factors(self, epoch_data, pose_key, vel_key, clock_bias_key, clock_drift_key):
        """Add GNSS measurement factors"""
        rover_obs = epoch_data['rover_obs']
        nav = epoch_data['nav']
        
        if rover_obs is None:
            return
            
        # Process each satellite observation
        for i in range(rover_obs.n):
            sat = rover_obs.data[i].sat
            
            # Get satellite position
            sat_pos, sat_vel, svh = nav.satpos(rover_obs.data[i].time, sat)
            
            if svh != 0:  # Skip unhealthy satellites
                continue
                
            # Add pseudorange factor
            if rover_obs.data[i].P[0] > 0:
                pr_factor = create_gnss_factor(
                    'pseudorange',
                    sat_pos=sat_pos[:3],
                    measured_range=rover_obs.data[i].P[0],
                    sigma=self.config.pseudorange_sigma
                )
                self.optimizer.add_gnss_factor(pr_factor, [pose_key, clock_bias_key])
                
            # Add Doppler factor if available
            if rover_obs.data[i].D[0] != 0:
                doppler_factor = create_gnss_factor(
                    'doppler',
                    sat_pos=sat_pos[:3],
                    sat_vel=sat_vel[:3],
                    measured_doppler=rover_obs.data[i].D[0],
                    wavelength=nav.lam[sat][0],
                    sigma=self.config.doppler_sigma
                )
                self.optimizer.add_gnss_factor(doppler_factor, [pose_key, vel_key, clock_drift_key])
                
    def _add_initial_values(self, pose_key, vel_key, clock_bias_key, clock_drift_key, time):
        """Add initial values for new state variables"""
        # Predict from previous state
        if self.prev_pose_key is not None:
            prev_pose = self.optimizer.get_pose(self.optimizer.pose_count - 2)
            prev_vel = self.optimizer.get_velocity(self.optimizer.vel_count - 2)
            
            if prev_pose and prev_vel:
                dt = time.time - self.prev_time.time
                
                # Simple constant velocity prediction
                new_pos = prev_pose.translation() + prev_vel * dt
                new_pose = gtsam.Pose3(prev_pose.rotation(), new_pos)
                
                self.optimizer.initial_values.insert(pose_key, new_pose)
                self.optimizer.initial_values.insert(vel_key, prev_vel)
            else:
                # Default initialization
                self.optimizer.initial_values.insert(pose_key, gtsam.Pose3())
                self.optimizer.initial_values.insert(vel_key, gtsam.Point3(0, 0, 0))
        else:
            # First epoch - should not happen
            self.optimizer.initial_values.insert(pose_key, gtsam.Pose3())
            self.optimizer.initial_values.insert(vel_key, gtsam.Point3(0, 0, 0))
            
        # Clock states
        self.optimizer.initial_values.insert(clock_bias_key, 0.0)
        self.optimizer.initial_values.insert(clock_drift_key, 0.0)
        
    def _save_solution(self, epoch_data, epoch_idx):
        """Save solution for current epoch"""
        # Get optimized values
        pose = self.optimizer.get_pose(self.optimizer.pose_count - 1)
        vel = self.optimizer.get_velocity(self.optimizer.vel_count - 1)
        
        if pose is None:
            return
            
        # Create solution structure (RTKLib compatible)
        import rtkcmn as gn
        sol = gn.Sol()
        sol.time = epoch_data['time']
        sol.rr[:3] = pose.translation()
        sol.rr[3:6] = vel if vel is not None else np.zeros(3)
        sol.stat = gn.SOLQ_FIX  # Assume fixed for now
        
        # Compute position variance
        try:
            marginals = gtsam.Marginals(
                self.optimizer.isam.getFactorsUnsafe(),
                self.optimizer.current_estimate
            )
            pose_cov = marginals.marginalCovariance(
                symbol('X', self.optimizer.pose_count - 1)
            )
            sol.qr[:3] = np.sqrt(np.diag(pose_cov[:3, :3]))
        except:
            sol.qr[:3] = np.ones(3)  # Default uncertainty
            
        self.solutions.append(sol)
        
    def save_results(self, output_path: Optional[str] = None):
        """Save results to file"""
        if output_path is None:
            output_path = self.config.get_output_path()
            
        # Import postpos to use savesol function
        from postpos import savesol
        savesol(self.solutions, output_path)
        
        print(f"Solutions saved to: {output_path}")
        
        # Save statistics if needed
        if self.config.trace_level > 0:
            stat_path = output_path + '.stat'
            with open(stat_path, 'w') as f:
                f.write(f"PPK ISAM2 Processing Statistics\n")
                f.write(f"Total epochs: {len(self.solutions)}\n")
                f.write(f"Final error: {self.optimizer.calculate_error():.3f}\n")
                
            print(f"Statistics saved to: {stat_path}")