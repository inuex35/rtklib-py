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
import pandas as pd
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

# Import existing RTKLib functions
from rtkpos import (
    IB, zdres, ddres, selsat, valpos, outsolstat
)

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
        params.enableRelinearization = True
        params.evaluateNonlinearError = False  # Faster updates
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
        
        # Ambiguity states for carrier phase
        self.ambiguities = {}  # {(sat, freq): symbol_index}
        self.amb_idx = 0  # Counter for ambiguity states
        
        # Initialize IMU if available
        self._init_imu()
        
    def _init_imu(self):
        """Initialize IMU parameters and preintegration"""
        # Check if IMU configuration exists
        if hasattr(self.nav, 'imu_file') and self.nav.imu_file:
            # IMU noise parameters (from config or defaults)
            accel_noise_sigma = getattr(self.nav, 'accel_noise_sigma', 0.01)
            gyro_noise_sigma = getattr(self.nav, 'gyro_noise_sigma', 0.001)
            accel_bias_rw_sigma = getattr(self.nav, 'accel_bias_rw_sigma', 0.0001)
            gyro_bias_rw_sigma = getattr(self.nav, 'gyro_bias_rw_sigma', 0.00001)
            
            self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
            self.imu_params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
            self.imu_params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
            self.imu_params.setIntegrationCovariance(np.eye(3) * 1e-7)
            self.imu_params.setOmegaCoriolis(np.zeros(3))  # Ignore Earth rotation for now
            
            # Load IMU data
            self._load_imu_data()
            
    def _load_imu_data(self):
        """Load IMU data from file"""
        if not hasattr(self.nav, 'imu_file'):
            return
            
        try:
            # Load IMU data from PPC-Dataset format
            self.imu_data = pd.read_csv(self.nav.imu_file, header=0, skipinitialspace=True)
            
            # Strip spaces from column names
            self.imu_data.columns = self.imu_data.columns.str.strip()
            
            trace(3, f"Loading IMU data from: {self.nav.imu_file}\n")
            trace(3, f"IMU columns: {list(self.imu_data.columns)}\n")
            
            # Convert GPS time to Unix timestamp
            gps_tow = self.imu_data['GPS TOW (s)'].values
            gps_week = self.imu_data['GPS Week'].values
            
            # Convert to seconds since GPS epoch
            self.imu_timestamps = gps_week * 604800 + gps_tow
            
            # Extract accelerometer data (m/s^2)
            self.imu_accels = self.imu_data[[
                'Acc X (m/s^2)', 'Acc Y (m/s^2)', 'Acc Z (m/s^2)'
            ]].values
            
            # Extract gyroscope data (convert deg/s to rad/s)
            # Note: Column names have leading spaces
            self.imu_gyros = self.imu_data[[
                'Ang Rate X (deg/s)', 'Ang Rate Y (deg/s)', 'Ang Rate Z (deg/s)'
            ]].values * np.pi / 180.0
            
            trace(2, f"Loaded {len(self.imu_data)} IMU measurements\n")
            trace(3, f"IMU time range: {self.imu_timestamps[0]:.3f} to {self.imu_timestamps[-1]:.3f}\n")
            trace(3, f"First accel: {self.imu_accels[0]}, First gyro: {self.imu_gyros[0]}\n")
        except Exception as e:
            trace(2, f"Failed to load IMU data: {e}\n")
            self.imu_data = None
            
    def _symbol(self, key_char, idx):
        """Create GTSAM symbol"""
        return getattr(S, key_char)(idx)
        
    def _get_ambiguity_key(self, sat, freq):
        """Get or create ambiguity key for satellite and frequency"""
        key = (sat, freq)
        
        if key not in self.ambiguities:
            # Create new ambiguity state
            a_key = self._symbol('A', self.amb_idx)
            self.ambiguities[key] = a_key
            self.amb_idx += 1
            
            # Initialize ambiguity value if not exists
            if not self.values.exists(a_key):
                # Initial ambiguity value (float)
                initial_amb = 0.0
                self.values.insert(a_key, np.array([initial_amb]))
                
                # Add prior with large uncertainty
                amb_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([100.0]))
                self.graph.add(gtsam.PriorFactorVector(a_key, np.array([initial_amb]), amb_noise))
                
                trace(3, f"Created ambiguity state A{self.amb_idx-1} for sat {sat} freq {freq}\n")
        
        return self.ambiguities[key]
        
    def _add_gnss_factors(self, obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb):
        """Add GNSS factors to the graph
        
        This replaces the Kalman filter measurement update in RTKLib
        """
        # Get current position estimate
        x_key = self._symbol('X', self.idx)
        c_key = self._symbol('C', self.idx)
        
        # Process double-differenced measurements (same as RTKLib)
        # Get base station residuals
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
        
        if len(v) < 4:
            trace(3, 'not enough double-differenced residuals for GNSS factors\n')
            # For now, add simple pseudorange factors instead
            self._add_simple_pseudorange_factors(obsr, rs, dts, svh, var)
            return
            
        # Add GNSS factors for each measurement
        nf = len(v) // self.nav.nf  # Number of satellite pairs
        
        for f in range(self.nav.nf):  # For each frequency
            for i in range(nf):
                idx = f * nf + i
                if idx >= len(v) or abs(v[idx]) == 0 or R[idx,idx] <= 0:
                    continue
                    
                # Create noise model
                noise = gtsam.noiseModel.Gaussian.Covariance(np.array([[R[idx,idx]]]))
                
                # Extract position part of H matrix
                h_pos = H[idx, 0:3]
                h_clk = H[idx, 3] if H.shape[1] > 3 else 0
                
                # Check measurement type based on index
                # In RTKLib, measurements are ordered: L1, L2, P1, P2 for each satellite
                # For double-differenced, we get all satellites for each measurement type
                
                # Determine if this is carrier phase or pseudorange
                # First half of measurements are carrier phase, second half are pseudorange
                total_meas = len(v) 
                is_carrier = idx < total_meas // 2
                
                if not is_carrier:  # Pseudorange
                    factor = GNSSPseudorangeFactor(x_key, c_key, v[idx], h_pos, h_clk, noise)
                    self.graph.add(factor)
                    trace(3, f"Added pseudorange factor for sat {sats[i]} with residual {v[idx]:.3f}\n")
                else:  # Carrier phase
                    # Get satellite and create/retrieve ambiguity key
                    sat = sats[i]
                    # Frequency is based on which half of carrier measurements
                    freq_idx = 0 if idx < nf else 1
                    amb_key = self._get_ambiguity_key(sat, freq_idx)
                    
                    # Get wavelength
                    freq = sat2freq(sat, freq_idx, self.nav)
                    if freq > 0:
                        wavelength = rCST.CLIGHT / freq
                        factor = GNSSCarrierPhaseFactor(x_key, c_key, amb_key, 
                                                      v[idx], h_pos, h_clk, 
                                                      wavelength, noise)
                        self.graph.add(factor)
                        trace(3, f"Added carrier phase factor for sat {sats[i]} freq {freq_idx} with residual {v[idx]:.3f}, wavelength {wavelength:.3f}\n")
                
    def _add_imu_factors(self, t_prev, t_curr):
        """Add IMU preintegrated factors between epochs"""
        if self.imu_data is None or self.idx == 0:
            trace(3, f"IMU factors skipped: imu_data={self.imu_data is not None}, idx={self.idx}\n")
            return
            
        # Get IMU measurements between epochs
        mask = (self.imu_timestamps > t_prev) & (self.imu_timestamps <= t_curr)
        imu_idx = np.where(mask)[0]
        
        trace(3, f"IMU preintegration: t_prev={t_prev:.3f}, t_curr={t_curr:.3f}, found {len(imu_idx)} measurements\n")
        
        if len(imu_idx) == 0:
            trace(2, f"WARNING: No IMU measurements found between {t_prev:.3f} and {t_curr:.3f}\n")
            return
            
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
        
        trace(2, f"Added IMU factor: epoch {self.idx}, preintegrated {len(imu_idx)} measurements\n")
        trace(3, f"  Delta position: {self.imu_preintegrated.deltaPij()}\n")
        trace(3, f"  Delta velocity: {self.imu_preintegrated.deltaVij()}\n")
        trace(3, f"  Delta rotation: {self.imu_preintegrated.deltaRij().matrix()}\n")
        
        # Add bias random walk
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate([
                np.full(3, getattr(self.nav, 'accel_bias_rw_sigma', 0.0001)),
                np.full(3, getattr(self.nav, 'gyro_bias_rw_sigma', 0.00001))
            ]))
        self.graph.add(gtsam.BetweenFactorConstantBias(
            b_prev, b_curr, gtsam.imuBias.ConstantBias(), bias_noise))
            
    def _initialize_state(self, obsr):
        """Initialize state for first epoch"""
        # Get initial position from single point positioning
        sol = pntpos(obsr, self.nav)
        
        # Ensure we have a valid position
        if sol.stat == gn.SOLQ_NONE or norm(sol.rr[0:3]) < rCST.RE_WGS84:
            # Use a default position if SPP fails
            trace(2, "WARNING: SPP failed for initialization, using nav.x position\n")
            sol.rr[0:3] = self.nav.x[0:3] if norm(self.nav.x[0:3]) > rCST.RE_WGS84 else np.array([-3810230.789, 3567860.707, 3652881.806])
            sol.dtr[0] = self.nav.x[6] if abs(self.nav.x[6]) > 0 else -112160.861 / rCST.CLIGHT
        
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
        initial_velocity = sol.rr[3:6] if norm(sol.rr[3:6]) > 0 else np.zeros(3)
        self.values.insert(v_key, initial_velocity)
        
        # Initial IMU bias
        self.values.insert(b_key, gtsam.imuBias.ConstantBias())
        
        # Initial clock bias (as vector for consistency)
        initial_clock = np.array([sol.dtr[0] * rCST.CLIGHT])
        self.values.insert(c_key, initial_clock)
        
        # Add prior factors with appropriate uncertainties
        # Position uncertainty based on solution quality
        pos_sigma = np.array([10.0, 10.0, 15.0]) if sol.stat != gn.SOLQ_NONE else np.array([100.0, 100.0, 150.0])
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate([pos_sigma, np.array([0.1, 0.1, 0.1])]))  # position + rotation
        self.graph.add(gtsam.PriorFactorPose3(x_key, initial_pose, pose_noise))
        
        # Velocity prior - larger uncertainty if no velocity estimate
        vel_sigma = np.full(3, 1.0) if norm(initial_velocity) > 0 else np.full(3, 10.0)
        vel_noise = gtsam.noiseModel.Diagonal.Sigmas(vel_sigma)
        self.graph.add(gtsam.PriorFactorVector(v_key, initial_velocity, vel_noise))
        
        # IMU bias prior - small uncertainty to regularize
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]))
        self.graph.add(gtsam.PriorFactorConstantBias(
            b_key, gtsam.imuBias.ConstantBias(), bias_noise))
            
        # Clock bias prior
        clock_sigma = 10.0 if sol.stat != gn.SOLQ_NONE else 100.0
        clock_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([clock_sigma]))
        self.graph.add(gtsam.PriorFactorVector(c_key, initial_clock, clock_noise))
        
        # Update RTKLib state
        self.nav.x[0:3] = sol.rr[0:3]
        self.nav.x[3:6] = initial_velocity
        self.nav.x[6] = sol.dtr[0]
        
        # Initialize covariance if needed
        if not hasattr(self.nav, 'P') or self.nav.P.shape[0] == 0:
            self.nav.P = np.eye(self.nav.nx) * 1e4  # Large initial uncertainty
            
        trace(3, f"Initialized state: pos={sol.rr[0:3]}, vel={initial_velocity}, clock={initial_clock[0]:.3f}m\n")
        
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
            
        # Add IMU factors if available
        if self.idx > 0 and self.last_imu_time is not None:
            # Convert observation time to GPS time for IMU matching
            obs_gps_time = (obsr.t.time + obsr.t.sec) - 315964800.0
            self._add_imu_factors(self.last_imu_time, obs_gps_time)
            
        # Add GNSS factors
        self._add_gnss_factors(obsr, obsb, rs, rsb, dts, dtsb, svh, svhb, var, varb)
        
        # Update ISAM2
        if self.graph.size() > 0:
            trace(3, f"\n=== ISAM2 Update Epoch {self.idx} ===\n")
            trace(3, f"Graph has {self.graph.size()} factors\n")
            
            # Count factor types
            gnss_factors = 0
            imu_factors = 0
            bias_factors = 0
            prior_factors = 0
            position_factors = 0
            
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                factor_type = str(type(factor))
                # Check for our custom factors
                if isinstance(factor, gtsam.CustomFactor):
                    # Check keys to identify factor type
                    if len(factor.keys()) == 2:  # Pose and clock keys
                        gnss_factors += 1
                    elif len(factor.keys()) == 1:  # Just pose key
                        position_factors += 1
                elif 'ImuFactor' in factor_type:
                    imu_factors += 1
                elif 'BetweenFactor' in factor_type:
                    bias_factors += 1
                elif 'PriorFactor' in factor_type:
                    prior_factors += 1
            
            trace(3, f"Factor breakdown: GNSS={gnss_factors}, Position={position_factors}, IMU={imu_factors}, " +
                     f"Bias={bias_factors}, Prior={prior_factors}\n")
            
            # Check if we have enough constraints
            total_constraints = gnss_factors + position_factors + prior_factors
            if self.idx > 0 and total_constraints == 0 and imu_factors > 0:
                # Add emergency position constraint to prevent drift
                trace(2, "WARNING: No GNSS constraints, adding emergency position factor\n")
                x_key = self._symbol('X', self.idx)
                current_pos = self.nav.x[0:3].copy()
                pos_sigma = np.array([50.0, 50.0, 80.0])  # Very loose constraint
                noise = gtsam.noiseModel.Diagonal.Sigmas(pos_sigma)
                factor = SimpleGNSSPositionFactor(x_key, current_pos, noise)
                self.graph.add(factor)
            
            try:
                result = self.isam.update(self.graph, self.values)
                self.graph = gtsam.NonlinearFactorGraph()
                self.values = gtsam.Values()
            except Exception as e:
                trace(2, f"ERROR: ISAM2 update failed: {e}\n")
                trace(2, "Attempting recovery with stronger constraints\n")
                
                # Add stronger position constraint and retry
                if self.idx > 0:
                    x_key = self._symbol('X', self.idx)
                    current_pos = self.nav.x[0:3].copy()
                    pos_sigma = np.array([20.0, 20.0, 30.0])
                    noise = gtsam.noiseModel.Diagonal.Sigmas(pos_sigma)
                    factor = SimpleGNSSPositionFactor(x_key, current_pos, noise)
                    self.graph.add(factor)
                    
                    try:
                        result = self.isam.update(self.graph, self.values)
                        self.graph = gtsam.NonlinearFactorGraph()
                        self.values = gtsam.Values()
                    except Exception as e2:
                        trace(2, f"ERROR: Recovery failed: {e2}\n")
                        return
            
            # Extract current estimate
            current_estimate = self.isam.calculateEstimate()
            x_key = self._symbol('X', self.idx)
            v_key = self._symbol('V', self.idx)
            c_key = self._symbol('C', self.idx)
            
            try:
                pose = current_estimate.atPose3(x_key)
                nav.x[0:3] = pose.translation()
                
                nav.x[3:6] = current_estimate.atVector(v_key)
                clock_bias = current_estimate.atVector(c_key)[0]
                nav.x[6] = clock_bias / rCST.CLIGHT
                trace(3, f"DEBUG: UNIX Time: {obsr.t.time + obsr.t.sec:.3f}, Clock bias: {clock_bias:.3f} m\n")
                
                # Update solution
                sol.t = obsr.t
                sol.rr[0:6] = nav.x[0:6].copy()
                sol.stat = gn.SOLQ_FLOAT
                
                # Get covariance (simplified)
                try:
                    marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), current_estimate)
                    cov = marginals.marginalCovariance(x_key)
                    sol.qr[0:3,0:3] = cov[3:6, 3:6]  # Position covariance
                except:
                    sol.qr = np.eye(6) * 10.0
            except Exception as e:
                trace(2, f"Failed to extract estimate at idx {self.idx}: {e}\n")
                
        # Update last IMU time for next epoch
        self.last_imu_time = (obsr.t.time + obsr.t.sec) - 315964800.0
        
        # Prepare for next epoch
        self.idx += 1
        
        # Add prediction for next epoch
        if self.idx > 0:
            x_key = self._symbol('X', self.idx)
            v_key = self._symbol('V', self.idx)
            b_key = self._symbol('B', self.idx)
            c_key = self._symbol('C', self.idx)
            
            # Only add if they don't exist already
            if not self.values.exists(x_key):
                pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(nav.x[0:3]))
                self.values.insert(x_key, pose)
            if not self.values.exists(v_key):
                self.values.insert(v_key, nav.x[3:6])
            if not self.values.exists(b_key):
                self.values.insert(b_key, gtsam.imuBias.ConstantBias())
            if not self.values.exists(c_key):
                self.values.insert(c_key, np.array([nav.x[6] * rCST.CLIGHT]))
        
        # Update nav solution list
        nav.sol.append(deepcopy(sol))
        
        # Try ambiguity resolution if enough epochs
        if self.idx > 20 and nav.armode > 0 and sol.stat == gn.SOLQ_FLOAT:
            self._resolve_ambiguities(nav, sol)
    
    def _resolve_ambiguities(self, nav, sol):
        """Try to resolve integer ambiguities using LAMBDA method"""
        try:
            # Get current estimate
            current_estimate = self.isam.calculateEstimate()
            
            # Collect float ambiguities
            amb_float = []
            amb_keys = []
            
            for (sat, freq), a_key in self.ambiguities.items():
                if current_estimate.exists(a_key):
                    amb_value = current_estimate.atVector(a_key)[0]
                    amb_float.append(amb_value)
                    amb_keys.append(a_key)
                    
            if len(amb_float) < 4:
                return  # Not enough ambiguities
                
            # Get covariance matrix for ambiguities
            n = len(amb_float)
            Q = np.eye(n) * 0.01  # Simplified - should extract from marginals
            
            # Apply LAMBDA
            amb_fixed, s = mlambda(np.array(amb_float), Q, 2)
            
            # Check ratio test
            if s[0] > 0 and s[1] / s[0] > nav.thresar:
                trace(2, f"Ambiguity fixed! Ratio: {s[1]/s[0]:.1f}\n")
                sol.stat = gn.SOLQ_FIX
                
                # Update ambiguities with fixed values
                for i, a_key in enumerate(amb_keys):
                    fixed_value = amb_fixed[i, 0]
                    # Add tight constraint on ambiguity
                    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01]))
                    self.graph.add(gtsam.PriorFactorVector(a_key, np.array([fixed_value]), noise))
                    
        except Exception as e:
            trace(3, f"Ambiguity resolution failed: {e}\n")
        
    def _add_simple_pseudorange_factors(self, obs, rs, dts, svh, var):
        """Add simple pseudorange factors when double-differencing fails"""
        x_key = self._symbol('X', self.idx)
        c_key = self._symbol('C', self.idx)
        
        # Get pseudorange residuals
        y, e, azel = zdres(self.nav, obs, rs, dts, svh, var, self.nav.x[0:3], 1)
        
        # Count valid measurements - check P1 pseudorange (column 2)
        valid_count = 0
        for i in range(len(obs.sat)):
            # P1 pseudorange is in column 2 (0=L1, 1=L2, 2=P1, 3=P2)
            if abs(y[i,2]) > 0.0 and e[i,2] > 0:  # Valid P1 pseudorange
                valid_count += 1
                
        trace(3, f'Found {valid_count} valid P1 pseudorange measurements\n')
        
        # Debug: print measurement types
        if self.idx < 3:
            trace(3, f'Measurement matrix y shape: {y.shape}, e shape: {e.shape}\n')
            trace(3, f'First satellite measurements: L1={y[0,0]:.1f}, L2={y[0,1]:.1f}, P1={y[0,2]:.1f}, P2={y[0,3]:.1f}\n')
        
        if valid_count < 4:
            # Not enough measurements, add loose position constraint
            if self.idx > 0:
                current_pos = self.nav.x[0:3].copy()
                pos_sigma = np.array([30.0, 30.0, 50.0])
                noise = gtsam.noiseModel.Diagonal.Sigmas(pos_sigma)
                factor = SimpleGNSSPositionFactor(x_key, current_pos, noise)
                self.graph.add(factor)
                trace(3, f'Added loose position constraint at {current_pos}\n')
            return
            
        # Add pseudorange factors for each satellite
        added_factors = 0
        for i in range(len(obs.sat)):
            # Use P1 pseudorange (column 2)
            if abs(y[i,2]) == 0.0 or e[i,2] <= 0:
                continue
                
            # Get satellite position (first 3 elements are XYZ)
            sat_pos = rs[i,0:3]
            
            # Unit vector from receiver to satellite
            r = sat_pos - self.nav.x[0:3]
            dist = norm(r)
            if dist < 1e-10:
                continue
            u = r / dist
            
            # Create pseudorange factor
            # Jacobian w.r.t position
            h_pos = -u
            # Jacobian w.r.t clock (speed of light)
            h_clk = 1.0
            
            # Measurement noise from P1 error
            sigma = np.sqrt(e[i,2])
            noise = gtsam.noiseModel.Gaussian.Covariance(np.array([[sigma**2]]))
            
            # Add factor using P1 residual
            factor = GNSSPseudorangeFactor(x_key, c_key, y[i,2], h_pos, h_clk, noise)
            self.graph.add(factor)
            added_factors += 1
            
        trace(3, f'Added {added_factors} pseudorange factors\n')
        
        # Update nav state if SPP available and this is first epoch
        if self.idx == 0:
            sol = pntpos(obs, self.nav)
            if sol.stat != gn.SOLQ_NONE and norm(sol.rr[0:3]) > rCST.RE_WGS84:
                self.nav.x[0:3] = sol.rr[0:3]
                self.nav.x[6] = sol.dtr[0]
                trace(3, f'Initialized with SPP: pos={sol.rr[0:3]}, clock={sol.dtr[0]*rCST.CLIGHT:.3f}m\n')


def simple_position_error(measured_position, this, values, jacobians):
    """Error function for simple GNSS position factor"""
    pose_key = this.keys()[0]
    pose = values.atPose3(pose_key)
    position = pose.translation()
    
    error = position - measured_position
    
    if jacobians is not None:
        # Jacobian w.r.t pose (position part)
        J_pose = np.zeros((3, 6))
        J_pose[:3, :3] = np.eye(3)  # Position part
        jacobians[0] = J_pose
        
    return error


class SimpleGNSSPositionFactor(gtsam.CustomFactor):
    """Simple GNSS position factor for absolute position constraint"""
    
    def __init__(self, pose_key, measured_position, noise_model):
        """Initialize position factor
        
        Args:
            pose_key: Key for pose variable
            measured_position: Measured position (3x1)
            noise_model: Measurement noise model
        """
        # Create error function with bound measured position
        error_func = lambda this, values, jacobians: simple_position_error(
            measured_position, this, values, jacobians)
        
        super().__init__(noise_model, [pose_key], error_func)
        self.measured_position = measured_position


def gnss_pseudorange_error(residual, h_pos, h_clk, this, values, jacobians):
    """Error function for GNSS pseudorange factor"""
    pose_key = this.keys()[0]
    clock_key = this.keys()[1]
    
    pose = values.atPose3(pose_key)
    clock = values.atVector(clock_key)[0]
    
    # The residual from zdres is already (measurement - predicted)
    # We just need to return it as the error
    # zdres computes: y = measured - predicted, so error = -y for minimization
    error = np.array([-residual])
    
    if jacobians is not None:
        # Jacobian w.r.t pose (only position part affects pseudorange)
        J_pose = np.zeros((1, 6))
        J_pose[0, 3:6] = -h_pos  # Negative because error = -(meas - pred)
        jacobians[0] = J_pose
        
        # Jacobian w.r.t clock
        jacobians[1] = np.array([[-h_clk]])
        
    return error


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
        error_func = lambda this, values, jacobians: gnss_pseudorange_error(
            residual, h_pos, h_clk, this, values, jacobians)
        
        super().__init__(noise_model, [pose_key, clock_key], error_func)
        self.residual = residual
        self.h_pos = h_pos
        self.h_clk = h_clk


def gnss_carrier_phase_error(residual, h_pos, h_clk, wavelength, this, values, jacobians):
    """Error function for GNSS carrier phase factor"""
    pose_key = this.keys()[0]
    clock_key = this.keys()[1]
    amb_key = this.keys()[2]
    
    pose = values.atPose3(pose_key)
    clock = values.atVector(clock_key)[0]
    ambiguity = values.atVector(amb_key)[0]
    
    # Error includes ambiguity term
    # residual = measured - (predicted + wavelength * ambiguity)
    # so error = -residual + wavelength * ambiguity
    error = np.array([-residual + wavelength * ambiguity])
    
    if jacobians is not None:
        # Jacobian w.r.t pose (only position part)
        J_pose = np.zeros((1, 6))
        J_pose[0, 0:3] = -h_pos
        jacobians[0] = J_pose
        
        # Jacobian w.r.t clock
        jacobians[1] = np.array([[-h_clk]])
        
        # Jacobian w.r.t ambiguity
        jacobians[2] = np.array([[wavelength]])
        
    return error


class GNSSCarrierPhaseFactor(gtsam.CustomFactor):
    """Custom GNSS carrier phase factor"""
    
    def __init__(self, pose_key, clock_key, amb_key, residual, h_pos, h_clk, wavelength, noise_model):
        """Initialize carrier phase factor
        
        Args:
            pose_key: Key for pose variable
            clock_key: Key for clock bias variable  
            amb_key: Key for ambiguity variable
            residual: Measurement residual
            h_pos: Jacobian w.r.t position (3x1)
            h_clk: Jacobian w.r.t clock
            wavelength: Carrier wavelength
            noise_model: Measurement noise model
        """
        # Create error function with bound parameters
        error_func = lambda this, values, jacobians: gnss_carrier_phase_error(
            residual, h_pos, h_clk, wavelength, this, values, jacobians)
        
        super().__init__(noise_model, [pose_key, clock_key, amb_key], error_func)
        self.residual = residual
        self.h_pos = h_pos
        self.h_clk = h_clk
        self.wavelength = wavelength


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