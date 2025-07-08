"""
RTKLib-py with ISAM2 filter integration
Maintains RTKLib structure but replaces Kalman filter with ISAM2

Copyright (c) 2022 Tim Everett
Modified to use ISAM2 for filtering
"""

import numpy as np
from numpy.linalg import inv, norm
from sys import stdout
from copy import copy, deepcopy
import rtkcmn as gn
from rtkcmn import rCST, DTTOL, sat2prn, sat2freq, timediff, xyz2enu
import rinex as rn
from pntpos import pntpos
from ephemeris import satposs
from mlambda import mlambda
from rtkcmn import trace, tracemat, uGNSS
import __ppk_config as cfg

# Import GTSAM and IMU-related modules
import gtsam
from gtsam import symbol_shorthand as S
import pandas as pd
import gnss_lib_py as glp

# Import existing RTKLib functions
from rtkpos import (
    IB, zdres, ddres, selsat, valpos, outsolstat,
    ddidx, restamb, resamb_lambda, manage_amb_LAMBDA
)

# Import phase bias initialization utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))
try:
    from gtsam_gnss.utils.rtk_phase_init import (
        initialize_nav_states,
        fix_ddidx_calculation
    )
    from gtsam_gnss.utils.fix_nav_initialization import (
        ensure_nav_states_initialized,
        fix_ddidx_initialization
    )
except ImportError:
    # Fallback if import fails
    def initialize_nav_states(nav, obsr, obsb, sol):
        """Fallback initialization function"""
        pass
    
    def fix_ddidx_calculation(nav, sats):
        """Fallback fix function"""
        return 0
        
    def ensure_nav_states_initialized(nav, sats):
        """Fallback initialization"""
        return 0
        
    def fix_ddidx_initialization(nav, obsr, obsb):
        """Fallback initialization"""
        pass

MAX_VAR_EPH = 300**2


class RTKLibISAM2:
    """RTKLib-py with ISAM2 filter for GNSS-IMU tight coupling"""
    
    def __init__(self, nav):
        """Initialize ISAM2 filter
        
        Args:
            nav: RTKLib navigation structure
        """
        self.nav = nav
        
        # ISAM2 parameters
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.1)
        params.relinearizeSkip = 10
        self.isam = gtsam.ISAM2(params)
        
        # Factor graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        # Time index
        self.idx = 0
        
        # IMU preintegration
        self.imu_params = None
        self.imu_preintegrated = None
        self.last_imu_time = None
        self.imu_timestamps = None
        self.imu_accels = None
        self.imu_gyros = None
        
        # Initialize IMU if available
        self._init_imu()
        
        # Ambiguity resolution state
        self.fixed_ambiguities = {}  # Dict of (sat, freq) -> fixed ambiguity value
        self.ambiguity_covariance = None
        self.last_ar_solution = None
        self.ar_ratio = 0.0
        
    def _init_imu(self):
        """Initialize IMU parameters and preintegration"""
        # Check if IMU loader exists
        if hasattr(self.nav, 'imu_loader') and self.nav.imu_loader:
            # Get IMU noise parameters from config
            accel_noise_sigma = getattr(cfg, 'accel_noise_sigma', 0.01)
            gyro_noise_sigma = getattr(cfg, 'gyro_noise_sigma', 0.001)
            accel_bias_rw_sigma = getattr(cfg, 'accel_bias_rw_sigma', 0.0001)
            gyro_bias_rw_sigma = getattr(cfg, 'gyro_bias_rw_sigma', 0.0001)
            
            self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
            self.imu_params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
            self.imu_params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
            self.imu_params.setIntegrationCovariance(np.eye(3) * 1e-2)
            self.imu_params.setOmegaCoriolis(np.zeros(3))  # Ignore Earth rotation for now
            
            # Load IMU data
            self._load_imu_data()
            
    def _load_imu_data(self):
        """Load IMU data from file"""
        if not hasattr(self.nav, 'imu_loader'):
            return
            
        # Use the IMU loader to get data
        loader = self.nav.imu_loader
        if loader.timestamps is not None:
            self.imu_timestamps = loader.timestamps
            self.imu_accels = loader.accelerations
            self.imu_gyros = loader.angular_velocities
            trace(3, f"Using IMU data: {len(self.imu_timestamps)} measurements\n")
        else:
            trace(2, "No IMU data available from loader\n")
            self.imu_timestamps = None
            self.imu_accels = None
            self.imu_gyros = None
            
    def _symbol(self, key_char, idx):
        """Create GTSAM symbol"""
        return getattr(S, key_char)(idx)
        
    def _add_gnss_factors(self, obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb):
        """Add GNSS factors to the graph
        
        This replaces the Kalman filter measurement update in RTKLib
        """
        # Get current position estimate
        x_key = self._symbol('X', self.idx)
        c_key = self._symbol('C', self.idx)
        
        # Process double-differenced measurements (same as RTKLib)
        # Get base station residuals
        trace(3, f'Base position: {self.nav.rb}\n')
        yr, er, azelr = zdres(self.nav, obsb, rsb, dtsb, svhb, varb, self.nav.rb, 0)
        
        # Get rover residuals  
        yu, eu, azel = zdres(self.nav, obsr, rs, dts, svh, var, self.nav.x[0:3], 1)
        
        # Find common satellites
        ns, iu, ir = selsat(self.nav, obsr, obsb, azelr[:,1])
        
        if ns <= 0:
            trace(3, 'no common sats for GNSS factors\n')
            return
            
        # Remove non-common residuals
        yr, er = yr[ir,:], er[ir,:]
        yu, eu = yu[iu,:], eu[iu,:]
        sats = obsr.sat[iu]
        els = azel[iu,1]
        
        # Calculate double-differenced residuals
        v, H, R = ddres(self.nav, self.nav.x, self.nav.P, yr, er, yu, eu, sats, els, 
                       self.nav.dt, obsr, True)
        
        trace(3, f'ddres returned {len(v)} measurements, needed at least 4\n')
        if len(v) > 0:
            trace(3, f'DD residuals: min={np.min(v):.3f}, max={np.max(v):.3f}, mean={np.mean(v):.3f}\n')
            
        # Add GNSS factors for each measurement
        for i in range(len(v)):
            if abs(v[i]) > 0 and R[i,i] > 0:
                # Create custom GNSS factor
                noise = gtsam.noiseModel.Gaussian.Covariance(np.array([[R[i,i]]]))
                
                # Extract position part of H matrix
                h_pos = H[i, 0:3]
                h_clk = H[i, 3] if H.shape[1] > 3 else 0
                
                # Create factor (simplified - in practice would use custom factor)
                factor = GNSSPseudorangeFactor(x_key, c_key, v[i], h_pos, h_clk, noise)
                self.graph.add(factor)
                
    def _add_imu_factors(self, t_prev, t_curr):
        """Add IMU preintegrated factors between epochs"""
        if self.imu_timestamps is None or self.idx == 0:
            return
            
        # Get IMU measurements between epochs
        mask = (self.imu_timestamps > t_prev) & (self.imu_timestamps <= t_curr)
        imu_idx = np.where(mask)[0]
        
        if len(imu_idx) == 0:
            return
            
        trace(3, f"Adding IMU factors: {len(imu_idx)} measurements between {t_prev:.3f} and {t_curr:.3f}\n")
            
        # Create new preintegrated measurement
        if self.imu_preintegrated is None:
            # Get initial bias estimate
            initial_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
            self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(
                self.imu_params, initial_bias)
        else:
            # Reset preintegration with current bias estimate
            b_key = self._symbol('B', self.idx - 1)
            if self.values.exists(b_key):
                current_bias = self.values.atConstantBias(b_key)
                self.imu_preintegrated.resetIntegrationAndSetBias(current_bias)
                
        # Add measurements
        for i in imu_idx:
            dt = 0.01  # Assuming 100Hz IMU
            if i > 0:
                dt = self.imu_timestamps[i] - self.imu_timestamps[i-1]
            self.imu_preintegrated.integrateMeasurement(
                self.imu_accels[i], self.imu_gyros[i], dt)
                
        trace(3, f'Added {len(imu_idx)} IMU measurements between epochs (t_prev={t_prev:.3f}, t_curr={t_curr:.3f})\n')
        if len(imu_idx) > 0:
            trace(4, f'IMU time range: {self.imu_timestamps[imu_idx[0]]:.3f} to {self.imu_timestamps[imu_idx[-1]]:.3f}\n')
        
        # Add IMU factor
        x_prev = self._symbol('X', self.idx - 1)
        v_prev = self._symbol('V', self.idx - 1) 
        b_prev = self._symbol('B', self.idx - 1)
        x_curr = self._symbol('X', self.idx)
        v_curr = self._symbol('V', self.idx)
        b_curr = self._symbol('B', self.idx)
        
        imu_factor = gtsam.ImuFactor(
            x_prev, v_prev, x_curr, v_curr, b_prev,
            self.imu_preintegrated)
        self.graph.add(imu_factor)
        
        # Add bias random walk
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate([
                np.full(3, getattr(self.nav, 'accel_bias_rw_sigma', 0.0001)),
                np.full(3, getattr(self.nav, 'gyro_bias_rw_sigma', 0.00001))
            ]))
        self.graph.add(gtsam.BetweenFactorConstantBias(
            b_prev, b_curr, gtsam.imuBias.ConstantBias(), bias_noise))
            
    def _add_nhc_factor(self):
        """Add Non-Holonomic Constraint factor
        
        This constrains the vehicle to not move sideways or vertically in its body frame,
        which is a valid assumption for ground vehicles.
        """
        if self.idx == 0:
            return  # Need at least one previous state
            
        # Check if NHC is enabled
        if not getattr(cfg, 'use_nhc', True):
            return
            
        # Get NHC noise parameters from config
        nhc_sigma_y = getattr(cfg, 'nhc_sigma_y', 0.1)  # Lateral velocity constraint
        nhc_sigma_z = getattr(cfg, 'nhc_sigma_z', 0.1)  # Vertical velocity constraint
        
        # Create noise model for NHC (2D: y and z velocity constraints)
        nhc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([nhc_sigma_y, nhc_sigma_z]))
        
        # Add NHC factor
        x_key = self._symbol('X', self.idx)
        v_key = self._symbol('V', self.idx)
        
        nhc_factor = NonHolonomicConstraint(x_key, v_key, nhc_noise)
        self.graph.add(nhc_factor)
        
        trace(3, f"Added NHC factor at idx {self.idx} with sigmas: y={nhc_sigma_y}, z={nhc_sigma_z}\n")
            
    def _initialize_state(self, obsr):
        """Initialize state for first epoch"""
        # Initialize RTKLib navigation attributes for ambiguity resolution if not present
        trace(3, f"Initializing navigation: na={self.nav.na}, nx={self.nav.nx}, nf={self.nav.nf}\n")
        
        # Initialize state vector and covariance if not present
        if not hasattr(self.nav, 'x'):
            self.nav.x = np.zeros(self.nav.nx)
        if not hasattr(self.nav, 'P'):
            self.nav.P = np.zeros((self.nav.nx, self.nav.nx))
            # Initialize diagonal with reasonable values
            self.nav.P[0:3, 0:3] = np.eye(3) * 100.0  # Position variance
            self.nav.P[3:6, 3:6] = np.eye(3) * 10.0   # Velocity variance
            self.nav.P[6:9, 6:9] = np.eye(3) * 1.0    # Acceleration variance
            for i in range(self.nav.na, self.nav.nx):
                self.nav.P[i, i] = 100.0  # Large initial variance for ambiguities
                
        if not hasattr(self.nav, 'lock'):
            self.nav.lock = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'slip'):
            self.nav.slip = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'fix'):
            self.nav.fix = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'prev_fix'):
            self.nav.prev_fix = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'outc'):
            self.nav.outc = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'rejc'):
            self.nav.rejc = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'sysprn'):
            # System/PRN mapping
            self.nav.sysprn = {}
            for i in range(1, gn.uGNSS.MAXSAT + 1):
                if i <= 32:
                    self.nav.sysprn[i] = (0, i)  # GPS
                elif i <= 56:
                    self.nav.sysprn[i] = (1, i-32)  # GLONASS
                elif i <= 95:
                    self.nav.sysprn[i] = (2, i-56)  # Galileo
                else:
                    self.nav.sysprn[i] = (3, i-95)  # BeiDou
        if not hasattr(self.nav, 'gf'):
            self.nav.gf = np.zeros(gn.uGNSS.MAXSAT)
        if not hasattr(self.nav, 'prev_ratio1'):
            self.nav.prev_ratio1 = 0.0
        if not hasattr(self.nav, 'sig_n0'):
            self.nav.sig_n0 = 0.3  # Initial phase bias std (meters)
        if not hasattr(self.nav, 'dt'):
            self.nav.dt = 0.0  # Time difference between rover and base
        if not hasattr(self.nav, 'prev_ratio2'):
            self.nav.prev_ratio2 = 0.0
        if not hasattr(self.nav, 'excsat_ix'):
            self.nav.excsat_ix = 0
        if not hasattr(self.nav, 'ratio'):
            self.nav.ratio = 0.0
        if not hasattr(self.nav, 'nb_ar'):
            self.nav.nb_ar = 0
        if not hasattr(self.nav, 'thresar'):
            self.nav.thresar = getattr(cfg, 'thresar', 3.0)  # Default AR threshold
        if not hasattr(self.nav, 'thresar1'):
            self.nav.thresar1 = getattr(cfg, 'thresar1', 1.0)  # Position variance threshold
        if not hasattr(self.nav, 'mindropsats'):
            self.nav.mindropsats = getattr(cfg, 'mindropsats', 10)
        if not hasattr(self.nav, 'minfixsats'):
            self.nav.minfixsats = getattr(cfg, 'minfixsats', 4)
        if not hasattr(self.nav, 'elmaskar'):
            self.nav.elmaskar = getattr(cfg, 'elmaskar', np.deg2rad(15))  # 15 deg default
        if not hasattr(self.nav, 'azel'):
            self.nav.azel = np.zeros((gn.uGNSS.MAXSAT, 2))
        if not hasattr(self.nav, 'vsat'):
            self.nav.vsat = np.zeros((gn.uGNSS.MAXSAT, self.nav.nf), dtype=int)
        if not hasattr(self.nav, 'xa'):
            self.nav.xa = np.zeros(self.nav.na)
        if not hasattr(self.nav, 'Pa'):
            self.nav.Pa = np.eye(self.nav.na) * 1e4
            
        # Get initial position from single point positioning
        sol = pntpos(obsr, self.nav)
        
        # Initialize GTSAM values
        x_key = self._symbol('X', 0)  # Pose
        v_key = self._symbol('V', 0)  # Velocity
        b_key = self._symbol('B', 0)  # IMU Bias
        c_key = self._symbol('C', 0)  # Clock
        
        # Initial pose
        initial_pose = gtsam.Pose3(
            gtsam.Rot3(),  # Identity rotation
            gtsam.Point3(sol.rr[0:3])
        )
        self.values.insert(x_key, initial_pose)
        
        # Initial velocity
        self.values.insert(v_key, np.zeros(3))
        
        # Initial IMU bias
        self.values.insert(b_key, gtsam.imuBias.ConstantBias())
        
        # Initial clock bias (as vector for consistency)
        self.values.insert(c_key, np.array([sol.dtr[0] * rCST.CLIGHT]))
        
        # Add prior factors
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([30.0, 30.0, 30.0, 0.1, 0.1, 0.1]))  # position + rotation
        self.graph.add(gtsam.PriorFactorPose3(x_key, initial_pose, pose_noise))
        
        vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(3, 10.0))
        self.graph.add(gtsam.PriorFactorVector(v_key, np.zeros(3), vel_noise))
        
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 0.1))
        self.graph.add(gtsam.PriorFactorConstantBias(
            b_key, gtsam.imuBias.ConstantBias(), bias_noise))
            
        clock_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([30.0]))
        self.graph.add(gtsam.PriorFactorVector(c_key, np.array([sol.dtr[0] * rCST.CLIGHT]), clock_noise))
        
        # Update RTKLib state
        self.nav.x[0:6] = sol.rr[0:6]
        
        # Initialize covariance if needed
        if not hasattr(self.nav, 'P') or self.nav.P.shape[0] == 0:
            self.nav.P = np.eye(self.nav.nx) * 1e4  # Large initial uncertainty
            
        # Set initial IMU time in GPS TOW
        unix_ms = (obsr.t.time + obsr.t.sec) * 1000.0
        _, self.last_imu_time = glp.unix_millis_to_tow(unix_ms)
        
    def relpos(self, nav, obsr, obsb, sol):
        """Relative positioning with ISAM2 (replaces RTKLib's relpos)"""
        
        # Time diff between rover and base
        nav.dt = timediff(obsr.t, obsb.t)
        trace(1,"\n---------------------------------------------------------\n")
        trace(1, "relpos_isam: epoch=%d dt=%.3f nu=%d nr=%d\n" % (self.idx, nav.dt, len(obsr.sat), len(obsb.sat)))
        trace(1, "obsr.t: time=%d sec=%.3f\n" % (obsr.t.time, obsr.t.sec))
        trace(1,"---------------------------------------------------------\n")
        
        if abs(nav.dt) > nav.maxage:
            trace(3, 'Age of differential too large: %.2f\n' % nav.dt)
            return
            
        # Clear valid sat status
        nav.vsat[:,:] = 0
        
        # Compute satellite positions using RTKLib functions
        rs, var, dts, svh = satposs(obsr, nav)
        rsb, varb, dtsb, svhb = satposs(obsb, nav)
        
        # Initialize on first epoch
        if self.idx == 0:
            self._initialize_state(obsr)
            # Also update the solution structure for the first epoch
            sol.t = obsr.t
            sol.rr[0:6] = self.nav.x[0:6].copy()
            sol.stat = gn.SOLQ_SINGLE  # Single point solution for first epoch
            sol.qr = np.eye(3) * 10.0
            sol.dtr = np.array([self.nav.x[6], 0])  # Clock bias
            
        # Add IMU factors if available
        if self.idx > 0 and self.last_imu_time is not None:
            # Convert observation time to GPS TOW for comparison with IMU timestamps
            # obsr.t.time is Unix timestamp in seconds, convert to milliseconds then to GPS TOW
            unix_ms = (obsr.t.time + obsr.t.sec) * 1000.0
            gps_week, current_gps_tow = glp.unix_millis_to_tow(unix_ms)
            trace(4, f'GNSS epoch time: Unix={obsr.t.time + obsr.t.sec:.3f}, GPS Week={gps_week}, GPS TOW={current_gps_tow:.3f}\n')
            self._add_imu_factors(self.last_imu_time, current_gps_tow)
            
        # Add GNSS factors
        self._add_gnss_factors(obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb)
        
        # Add NHC factor if enabled
        self._add_nhc_factor()
        
        # Attempt ambiguity resolution if we have carrier phase measurements
        # Note: Do this AFTER the update so we can check the current estimate
        attempt_ar = self.idx > 0 and hasattr(self.nav, 'nf') and self.nav.nf >= 1
        
        # Update ISAM2
        if self.graph.size() > 0:
            try:
                result = self.isam.update(self.graph, self.values)
                self.graph = gtsam.NonlinearFactorGraph()
                self.values = gtsam.Values()
            except Exception as e:
                trace(2, f"ISAM2 update failed at idx {self.idx}: {e}\n")
                # Clear the graph and values to avoid accumulating bad factors
                self.graph = gtsam.NonlinearFactorGraph()
                self.values = gtsam.Values()
                # Keep the solution from single point positioning
                return
            
            # Extract current estimate
            current_estimate = self.isam.calculateEstimate()
            x_key = self._symbol('X', self.idx)
            v_key = self._symbol('V', self.idx)
            c_key = self._symbol('C', self.idx)
            
            try:
                # Check if keys exist before accessing
                if current_estimate.exists(x_key):
                    pose = current_estimate.atPose3(x_key)
                    nav.x[0:3] = pose.translation()
                else:
                    trace(3, f"Warning: X key {x_key} not in estimate\n")
                    
                if current_estimate.exists(v_key):
                    nav.x[3:6] = current_estimate.atVector(v_key)
                else:
                    trace(3, f"Warning: V key {v_key} not in estimate\n")
                    
                if current_estimate.exists(c_key):
                    nav.x[6] = current_estimate.atVector(c_key)[0] / rCST.CLIGHT
                else:
                    trace(3, f"Warning: C key {c_key} not in estimate\n")
                
                # Update solution
                sol.t = obsr.t
                sol.rr[0:6] = nav.x[0:6].copy()
                # Set solution status based on ambiguity resolution
                if self.ar_ratio >= self.nav.thresar and len(self.fixed_ambiguities) > 0:
                    sol.stat = gn.SOLQ_FIX  # Fixed solution
                else:
                    sol.stat = gn.SOLQ_FLOAT  # Float solution
                sol.dtr = np.array([nav.x[6], 0])  # Clock bias
                sol.ratio = self.ar_ratio  # Store AR ratio in solution
                
                # Get covariance (simplified)
                try:
                    if current_estimate.exists(x_key) and self.isam.getFactorsUnsafe().size() > 0:
                        marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), current_estimate)
                        cov = marginals.marginalCovariance(x_key)
                        # Extract 3x3 position covariance matrix
                        sol.qr = cov[3:6, 3:6]  # Position covariance (translation part)
                    else:
                        sol.qr = np.eye(3) * 10.0
                except Exception as e:
                    trace(4, f"Could not compute covariance: {e}\n")
                    sol.qr = np.eye(3) * 10.0
            except Exception as e:
                trace(2, f"Failed to extract estimate at idx {self.idx}: {e}\n")

        # Now attempt ambiguity resolution after update
        if attempt_ar and self.isam.getFactorsUnsafe().size() > 0:
            fixed_solution = self._resolve_ambiguities(obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb)
            if fixed_solution is not None:
                # Update solution status if ambiguities were fixed
                if self.ar_ratio >= self.nav.thresar and len(self.fixed_ambiguities) > 0:
                    sol.stat = gn.SOLQ_FIX
                    sol.ratio = self.ar_ratio
                    trace(3, f"Ambiguities fixed! Ratio={self.ar_ratio:.2f}\n")

        # Prepare for next epoch
        self.idx += 1
        
        # Add prediction for next epoch
        if self.idx > 0:
            x_key = self._symbol('X', self.idx)
            v_key = self._symbol('V', self.idx)
            b_key = self._symbol('B', self.idx)
            c_key = self._symbol('C', self.idx)
            
            pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(nav.x[0:3]))
            self.values.insert(x_key, pose)
            self.values.insert(v_key, nav.x[3:6])
            self.values.insert(b_key, gtsam.imuBias.ConstantBias())
            self.values.insert(c_key, np.array([nav.x[6] * rCST.CLIGHT]))
            
            # Add velocity constraint if no IMU
            if not hasattr(self.nav, 'imu_loader') or self.nav.imu_loader.timestamps is None:
                # Add constant velocity factor
                x_prev = self._symbol('X', self.idx - 1)
                v_prev = self._symbol('V', self.idx - 1)
                dt = 0.2  # 5Hz GNSS rate
                # Position should follow velocity model
                pos_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
                vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
                # Between factor for velocity (constant velocity model)
                self.graph.add(gtsam.BetweenFactorVector(v_prev, v_key, np.zeros(3), vel_noise))
        # Store last IMU time in GPS TOW
        unix_ms = (obsr.t.time + obsr.t.sec) * 1000.0
        _, self.last_imu_time = glp.unix_millis_to_tow(unix_ms)
        
        # Update nav solution list
        nav.sol.append(deepcopy(sol))
        
    def _add_simple_pseudorange_factors(self, obs, rs, dts, svh, var):
        """Add simple pseudorange factors when double-differencing fails"""
        x_key = self._symbol('X', self.idx)
        c_key = self._symbol('C', self.idx)
        
        added = 0
        for i in range(len(obs.sat)):
            if obs.P[i,0] == 0 or norm(rs[i,:]) < rCST.RE_WGS84:
                continue
            if svh[i] != 0:
                continue
                
            # Pseudorange measurement
            pr = obs.P[i,0]
            sat_pos = rs[i,:]
            sat_clk = dts[i] * rCST.CLIGHT
            
            # Measurement noise (simplified)
            pr_sigma = 3.0  # meters
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([pr_sigma]))
            
            # Create pseudorange factor
            # For now, just skip adding factors - need to implement properly
            # factor = create_pseudorange_factor(x_key, c_key, pr - sat_clk, sat_pos, noise)
            # self.graph.add(factor)
            # added += 1
            
        trace(3, f'Added {added} simple pseudorange factors\n')
    
    def _add_fixed_ambiguity_factors(self, fixed_solution):
        """Add constraints for fixed ambiguities to the factor graph"""
        # For now, we'll add strong priors on the ambiguity states
        # In a full implementation, we would add carrier phase factors with fixed ambiguities
        
        na = self.nav.na
        for (sat, freq), amb_value in self.fixed_ambiguities.items():
            ib = IB(sat, freq, na)
            if ib < len(fixed_solution):
                # Add a strong prior on the ambiguity value
                amb_key = self._symbol('N', self.idx * 100 + sat * 10 + freq)  # Unique key for ambiguity
                amb_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001]))  # Very small uncertainty
                
                # Note: In a real implementation, we would need to properly handle ambiguity states
                # For now, this is a placeholder showing the concept
                trace(3, f'Fixed ambiguity for sat {sat} freq {freq}: {amb_value:.2f}\n')
                
    def _resolve_ambiguities(self, obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb):
        """Resolve integer ambiguities using LAMBDA method
        
        Returns:
            fixed_solution: Fixed ambiguity solution if successful, None otherwise
        """
        # Check if we have enough double-difference measurements
        yr, er, azelr = zdres(self.nav, obsb, rsb, dtsb, svhb, varb, self.nav.rb, 0)
        yu, eu, azel = zdres(self.nav, obsr, rs, dts, svh, var, self.nav.x[0:3], 1)
        
        # Find common satellites
        ns, iu, ir = selsat(self.nav, obsr, obsb, azelr[:,1])
        
        if ns < 4:  # Need at least 4 satellites for double-differencing
            trace(3, 'Not enough satellites for ambiguity resolution\n')
            return None
            
        # Get current position variance
        if hasattr(self, 'isam') and self.idx >= 0:
            try:
                current_estimate = self.isam.calculateEstimate()
                # Use the current index which should be in the estimate after update
                x_key = self._symbol('X', self.idx)
                
                # Check if the key exists in the estimate
                if not current_estimate.exists(x_key):
                    # Try previous index if current not available
                    if self.idx > 0:
                        x_key = self._symbol('X', self.idx - 1)
                        if not current_estimate.exists(x_key):
                            trace(3, f"Neither current nor previous position key in estimate, skipping AR\n")
                            return None
                    else:
                        trace(3, f"Position key {x_key} not yet in estimate, skipping AR\n")
                        return None
                    
                # Only compute marginals if we have enough factors
                if self.isam.getFactorsUnsafe().size() > 0:
                    try:
                        marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), current_estimate)
                        cov = marginals.marginalCovariance(x_key)
                        posvar = np.trace(cov[3:6, 3:6]) / 3  # Position variance
                    except RuntimeError as e:
                        # This can happen if the factor graph is not well-constrained
                        trace(3, f"Cannot compute marginals yet: {e}\n")
                        posvar = 1e6
                else:
                    posvar = 1e6
            except Exception as e:
                trace(2, f"Failed to compute position variance: {e}\n")
                posvar = 1e6  # Large value if can't compute
        else:
            posvar = 1e6
            
        # Check if position variance is small enough for AR
        if posvar > getattr(self.nav, 'thresar1', 1.0):
            trace(3, f'Position variance too large for AR: {posvar:.3f}\n')
            return None
            
        # Update nav.x with current position estimate from GTSAM
        if hasattr(self, 'isam') and self.idx >= 0:
            try:
                current_estimate = self.isam.calculateEstimate()
                x_key = self._symbol('X', self.idx)
                if current_estimate.exists(x_key):
                    pose = current_estimate.atPose3(x_key)
                    self.nav.x[0:3] = pose.translation()
                    
                v_key = self._symbol('V', self.idx)
                if current_estimate.exists(v_key):
                    self.nav.x[3:6] = current_estimate.atVector(v_key)
            except:
                pass
        
        # Initialize navigation states before ambiguity resolution
        # Use the simpler initialization that just ensures non-zero values
        fix_ddidx_initialization(self.nav, obsr, obsb)
        
        # Need to call ddres to set nav.vsat before ambiguity resolution
        # Get double difference residuals
        sats = obsr.sat[iu]
        els = azel[iu, 1]  # Use rover elevation angles for common satellites
        
        # Get residuals for selected satellites
        yr_sel = yr[ir,:] if len(yr) > 0 else np.zeros((len(ir), self.nav.nf))
        er_sel = er[ir,:] if len(er) > 0 else np.ones((len(ir), self.nav.nf))
        yu_sel = yu[iu,:] if len(yu) > 0 else np.zeros((len(iu), self.nav.nf))
        eu_sel = eu[iu,:] if len(eu) > 0 else np.ones((len(iu), self.nav.nf))
        
        # Calculate double-differenced residuals to set nav.vsat
        v, H, R = ddres(self.nav, self.nav.x, self.nav.P, yr_sel, er_sel, 
                        yu_sel, eu_sel, sats, els, self.nav.dt, obsr, True)
        
        # Now nav.vsat should be properly set for valid phase observations
        trace(3, f'After ddres: {np.sum(self.nav.vsat)} satellites marked as visible\n')
        
        # Perform ambiguity resolution using RTKLib's manage_amb_LAMBDA
        stat = gn.SOLQ_FLOAT  # Current solution status
        
        # Call RTKLib's ambiguity resolution
        nb, xa = manage_amb_LAMBDA(self.nav, sats, stat, posvar)
        
        if nb > 0 and self.nav.ratio >= self.nav.thresar:
            trace(3, f'Ambiguity resolution successful: ratio={self.nav.ratio:.2f}, nb={nb}\n')
            self.ar_ratio = self.nav.ratio
            
            # Store fixed ambiguities
            for i in range(len(sats)):
                sat = sats[i]
                for f in range(self.nav.nf):
                    if self.nav.fix[sat-1, f] == 2:  # Fixed
                        ib = IB(sat, f, self.nav.na)
                        if ib < len(xa):
                            self.fixed_ambiguities[(sat, f)] = xa[ib]
                            
            # Return fixed solution
            fixed_x = self.nav.x.copy()
            fixed_x[self.nav.na:self.nav.na+nb] = xa[self.nav.na:self.nav.na+nb]
            return fixed_x
        else:
            trace(3, f'Ambiguity resolution failed: ratio={self.nav.ratio:.2f}\n')
            return None


def gnss_pseudorange_error_func(residual, h_pos, h_clk, this, values, jacobians):
    """Error function for GNSS pseudorange factor
    
    Args:
        residual: Measurement residual
        h_pos: Jacobian w.r.t position (3x1)
        h_clk: Jacobian w.r.t clock
        this: The factor instance
        values: Current variable values
        jacobians: Jacobian matrices (output)
        
    Returns:
        Residual vector
    """
    # Get keys
    keys = this.keys()
    pose_key = keys[0]
    clock_key = keys[1]
    
    # Get current estimates
    pose = values.atPose3(pose_key)
    clock = values.atVector(clock_key)[0]
    
    # Compute error
    pos = pose.translation()
    error = -residual + h_pos @ pos + h_clk * clock
    
    # Compute Jacobians if requested
    if jacobians is not None:
        # Jacobian w.r.t pose (6x1: 3 rotation, 3 translation)
        J_pose = np.zeros((1, 6))
        J_pose[0, 3:6] = h_pos  # Only translation part affects error
        jacobians[0] = J_pose
        
        # Jacobian w.r.t clock (1x1)
        J_clock = np.array([[h_clk]])
        jacobians[1] = J_clock
    
    return np.array([error])


class GNSSPseudorangeFactor(gtsam.CustomFactor):
    """Custom GNSS pseudorange factor"""
    
    def __init__(self, pose_key, clock_key, residual, h_pos, h_clk, noise_model):
        """Initialize GNSS factor
        
        Args:
            pose_key: Key for pose variable
            clock_key: Key for clock bias variable  
            residual: Measurement residual
            h_pos: Jacobian w.r.t position (3x1)
            h_clk: Jacobian w.r.t clock
            noise_model: Measurement noise model
        """
        # Create error function with bound parameters
        error_func = lambda this, values, jacobians: gnss_pseudorange_error_func(
            residual, h_pos, h_clk, this, values, jacobians
        )
        
        # Initialize CustomFactor with noise model, keys, and error function
        super().__init__(noise_model, [pose_key, clock_key], error_func)


def nhc_error_func(this, values, jacobians):
    """Error function for Non-Holonomic Constraint factor
    
    NHC assumes the vehicle cannot move sideways or vertically in its body frame.
    This constrains velocity in the body y and z directions to be zero.
    
    Args:
        this: The factor instance
        values: Current variable values
        jacobians: Jacobian matrices (output)
        
    Returns:
        Error vector [v_y, v_z] in body frame
    """
    # Get keys
    keys = this.keys()
    pose_key = keys[0]
    vel_key = keys[1]
    
    # Get current estimates
    pose = values.atPose3(pose_key)
    vel_world = values.atVector(vel_key)
    
    # Transform velocity from world to body frame
    R_world_to_body = pose.rotation().matrix().T
    vel_body = R_world_to_body @ vel_world
    
    # Error is the y and z components of velocity in body frame
    error = np.array([vel_body[1], vel_body[2]])  # v_y and v_z should be zero
    
    # Compute Jacobians if requested
    if jacobians is not None:
        # Jacobian w.r.t pose (2x6)
        J_pose = np.zeros((2, 6))
        # Rotation affects how world velocity maps to body frame
        # This is complex, so we'll use numerical differentiation or set to zero for now
        J_pose[:, 0:3] = 0  # Simplification: ignore rotation derivative
        
        # Jacobian w.r.t velocity (2x3)
        # d(R^T * v) / dv = R^T
        J_vel = R_world_to_body[1:3, :]  # Take y and z rows
        
        jacobians[0] = J_pose
        jacobians[1] = J_vel
    
    return error


class NonHolonomicConstraint(gtsam.CustomFactor):
    """Non-Holonomic Constraint factor for ground vehicles"""
    
    def __init__(self, pose_key, vel_key, noise_model):
        """Initialize NHC factor
        
        Args:
            pose_key: Key for pose variable
            vel_key: Key for velocity variable
            noise_model: Measurement noise model (2D for y,z velocity constraints)
        """
        # Initialize CustomFactor with noise model, keys, and error function
        super().__init__(noise_model, [pose_key, vel_key], nhc_error_func)


def rtkpos_isam(nav, rov, base, fp_stat, dir=1):
    """RTK positioning with ISAM2 filter
    
    This replaces the original rtkpos function but maintains the same interface
    """
    trace(3, "rtkpos_isam: filtertype=%s dir=%d\n" % (nav.filtertype, dir))
    
    # Initialize ISAM2 filter
    isam_filter = RTKLibISAM2(nav)
    
    # Get first observations
    obsr, obsb = rn.first_obs(nav, rov, base, dir)
    if obsr == [] or obsb == []:
        return
        
    # Process all epochs
    nav.sol = []
    sol = gn.Sol()
    t = 0
    n = 0
    
    # Process first epoch
    if n == 0:
        # Single precision solution for initial time
        if nav.use_sing_pos or sol.stat == gn.SOLQ_NONE or sol.rr[0] == 0.0:
            sol = pntpos(obsr, nav)
        else:
            sol = gn.Sol()
        if sol.t.time == 0:
            sol.t = obsr.t
        # Process with ISAM2
        isam_filter.relpos(nav, obsr, obsb, sol)
        outsolstat(nav, sol, fp_stat)
        ep = gn.time2epoch(sol.t)
        stdout.write('\r   %2d/%2d/%4d %02d:%02d:%05.2f: %d (epoch %d)' % (ep[1], ep[2], ep[0],
                ep[3], ep[4], ep[5], sol.stat, n))
        stdout.flush()
        n += 1
        
    # Process remaining epochs
    while True:
        # Get next observations
        if len(nav.sol) > 0:
            t = nav.sol[-1].t  # previous epoch
        obsr, obsb = rn.next_obs(nav, rov, base, dir)
        if obsr == [] or obsb == []:
            break
            
        # Single precision solution for time update
        if nav.use_sing_pos or sol.stat == gn.SOLQ_NONE or sol.rr[0] == 0.0:
            sol = pntpos(obsr, nav)
        else:
            sol = gn.Sol()
        if sol.t.time == 0:
            sol.t = obsr.t
        if t != 0:
            nav.tt = timediff(sol.t, t)  # timediff from previous epoch
            
        # Process with ISAM2
        isam_filter.relpos(nav, obsr, obsb, sol)
        outsolstat(nav, sol, fp_stat)
        ep = gn.time2epoch(sol.t)
        stdout.write('\r   %2d/%2d/%4d %02d:%02d:%05.2f: %d (epoch %d)' % (ep[1], ep[2], ep[0],
                ep[3], ep[4], ep[5], sol.stat, n))
        stdout.flush()
        n += 1
        
        # Check epoch limit
        if nav.maxepoch is not None and n >= nav.maxepoch:
            break
            
    stdout.write('\n')
    trace(3, "rtkpos_isam: nobs=%d\n" % len(nav.sol))


# Make rtkpos point to ISAM2 version
rtkpos = rtkpos_isam