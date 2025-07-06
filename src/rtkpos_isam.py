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
        
    def _init_imu(self):
        """Initialize IMU parameters and preintegration"""
        # Check if IMU loader exists
        if hasattr(self.nav, 'imu_loader') and self.nav.imu_loader:
            # Get IMU noise parameters from config
            accel_noise_sigma = getattr(cfg, 'accel_noise_sigma', 0.1)
            gyro_noise_sigma = getattr(cfg, 'gyro_noise_sigma', 0.05)
            accel_bias_rw_sigma = getattr(cfg, 'accel_bias_rw_sigma', 0.01)
            gyro_bias_rw_sigma = getattr(cfg, 'gyro_bias_rw_sigma', 0.01)
            
            self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
            self.imu_params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
            self.imu_params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
            self.imu_params.setIntegrationCovariance(np.eye(3) * 1e-3)
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
        
        if len(v) < 4:
            trace(3, 'not enough double-differenced residuals for GNSS factors\n')
            # Add simple position constraint based on single point positioning
            if hasattr(self, 'nav') and self.nav.x[0] != 0:
                x_key = self._symbol('X', self.idx)
                # Use current position with larger uncertainty
                pos_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1]))
                current_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(self.nav.x[0:3]))
                self.graph.add(gtsam.PriorFactorPose3(x_key, current_pose, pos_noise))
                trace(3, 'Added position constraint from single point positioning\n')
            return
            
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
            
    def _initialize_state(self, obsr):
        """Initialize state for first epoch"""
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
                pose = current_estimate.atPose3(x_key)
                nav.x[0:3] = pose.translation()
                
                nav.x[3:6] = current_estimate.atVector(v_key)
                nav.x[6] = current_estimate.atVector(c_key)[0] / rCST.CLIGHT
                
                # Update solution
                sol.t = obsr.t
                sol.rr[0:6] = nav.x[0:6].copy()
                sol.stat = gn.SOLQ_FLOAT
                sol.dtr = np.array([nav.x[6], 0])  # Clock bias
                
                # Get covariance (simplified)
                try:
                    marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), current_estimate)
                    cov = marginals.marginalCovariance(x_key)
                    # Extract 3x3 position covariance matrix
                    sol.qr = cov[3:6, 3:6]  # Position covariance (translation part)
                except:
                    sol.qr = np.eye(3) * 10.0
            except Exception as e:
                trace(2, f"Failed to extract estimate at idx {self.idx}: {e}\n")
                
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
        super().__init__(noise_model, [pose_key, clock_key])
        self.residual = residual
        self.h_pos = h_pos
        self.h_clk = h_clk
        
    def error(self, values):
        """Compute error given current values"""
        pose = values.atPose3(self.keys()[0])
        clock = values.atVector(self.keys()[1])[0]
        
        # Error is negative of residual (since residual = measurement - predicted)
        # In Kalman filter: x_new = x_old + K*(y - h(x))
        # In factor graph: we minimize ||h(x) - y||^2
        pos = pose.translation()
        error = -self.residual + self.h_pos @ pos + self.h_clk * clock
        return np.array([error])


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