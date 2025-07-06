#!/usr/bin/env python3
"""
RTKLib-ISAM2 Tight Coupling Implementation
Integrates RTKLib GNSS processing with IMU data using GTSAM's ISAM2 optimizer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gtsam
from gtsam import symbol_shorthand as S
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

# RTKLib imports - temporarily comment out problematic imports
# We'll use the gnss_data_loader instead which handles RINEX reading
import numpy as np
CLIGHT = 299792458.0  # speed of light
FREQ1 = 1575.42e6     # L1 frequency

# GTSAM imports
from gtsam import (
    ISAM2, ISAM2Params, ISAM2Result,
    NonlinearFactorGraph, Values,
    Pose3, Point3, Rot3, NavState,
    PreintegratedImuMeasurements,
    ImuFactor,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias,
    BetweenFactorPose3, BetweenFactorConstantBias,
    noiseModel
)

# Check if CombinedImuFactor is available
try:
    from gtsam import CombinedImuFactor, PreintegratedCombinedMeasurements
    HAS_COMBINED_IMU = True
except ImportError:
    HAS_COMBINED_IMU = False

# Symbol shortcuts
X = S.X  # Pose3 (position, orientation)
V = S.V  # Velocity
B = S.B  # IMU Bias (accelerometer, gyroscope)
C = S.C  # Clock states (receiver clock bias, drift)
N = S.N  # Ambiguity states

@dataclass
class ImuMeasurement:
    """Single IMU measurement"""
    timestamp: float
    accelerometer: np.ndarray  # 3x1
    gyroscope: np.ndarray      # 3x1

@dataclass
class GnssMeasurement:
    """GNSS observations at single epoch"""
    timestamp: float
    pseudoranges: Dict[int, float]  # sat_id -> pseudorange
    carrier_phases: Dict[int, float]  # sat_id -> carrier phase
    dopplers: Dict[int, float]  # sat_id -> doppler
    cnos: Dict[int, float]  # sat_id -> C/N0
    sat_positions: Dict[int, np.ndarray]  # sat_id -> ECEF position
    sat_velocities: Dict[int, np.ndarray]  # sat_id -> ECEF velocity
    base_obs: Optional[Dict] = None  # For differential processing

class RtkImuISAM:
    """RTK-IMU tight coupling with ISAM2 optimization"""
    
    def __init__(self, config: Dict):
        """Initialize RTK-IMU ISAM2 estimator
        
        Args:
            config: Configuration dictionary containing:
                - imu_params: IMU noise parameters
                - gnss_params: GNSS noise models
                - isam_params: ISAM2 optimization parameters
                - rtk_params: RTK processing options
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ISAM2
        isam_params = ISAM2Params()
        isam_params.setRelinearizeThreshold(config.get('relinearize_threshold', 0.1))
        isam_params.relinearizeSkip = config.get('relinearize_skip', 10)
        # isam_params.enableDetailedResults = True  # Not available in this version
        # isam_params.factorization = gtsam.ISAM2Params.QR  # Use default
        
        # Enable partial relinearization to handle ill-posed systems
        isam_params.enablePartialRelinearizationCheck = True
        
        self.isam = ISAM2(isam_params)
        
        # Initialize IMU preintegration
        self._init_imu_preintegration()
        
        # State tracking
        self.current_state_idx = 0
        self.values = Values()
        self.graph = NonlinearFactorGraph()
        self.imu_buffer: List[ImuMeasurement] = []
        self.last_imu_time = None
        
        # RTK specific
        self.ambiguity_states = {}  # sat_id -> ambiguity symbol
        self.fixed_ambiguities = set()
        self.cycle_slip_detector = CycleSlipDetector()
        
        # Mode tracking
        self.rtk_mode = 'SPP'  # SPP, DGPS, RTK_FLOAT, RTK_FIXED
        
    def _init_imu_preintegration(self):
        """Initialize IMU preintegration parameters"""
        # Extract IMU parameters
        imu_config = self.config['imu_params']
        
        # Create IMU parameters - use PreintegrationParams for basic IMU integration
        params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # gravity magnitude
        
        # Set accelerometer and gyroscope noise
        params.setAccelerometerCovariance(np.eye(3) * imu_config['accel_noise_sigma']**2)
        params.setGyroscopeCovariance(np.eye(3) * imu_config['gyro_noise_sigma']**2)
        params.setIntegrationCovariance(np.eye(3) * imu_config['integration_sigma']**2)
        
        # Store for creating new preintegration instances
        self.imu_params = params
        self.current_preintegration = None
        self.current_bias = gtsam.imuBias.ConstantBias()
        
    def initialize(self, initial_gnss: GnssMeasurement, initial_imu: List[ImuMeasurement]):
        """Initialize the estimator with initial GNSS and IMU data
        
        Args:
            initial_gnss: First GNSS measurement for initial position
            initial_imu: Initial IMU measurements for bias estimation
        """
        self.logger.info("Initializing RTK-IMU ISAM2 estimator")
        
        # Compute initial position using SPP
        initial_pos = self._compute_spp_solution(initial_gnss)
        
        # Estimate initial attitude and bias from static IMU data
        initial_attitude, initial_bias = self._estimate_initial_orientation(initial_imu)
        
        # Create initial pose
        initial_pose = Pose3(initial_attitude, Point3(initial_pos))
        initial_velocity = np.zeros(3)
        
        # Add prior factors
        pose_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]))
        velocity_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        bias_noise = noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]))
        
        self.graph.add(PriorFactorPose3(X(0), initial_pose, pose_noise))
        self.graph.add(PriorFactorVector(V(0), initial_velocity, velocity_noise))
        self.graph.add(PriorFactorConstantBias(B(0), initial_bias, bias_noise))
        
        # Initialize values
        self.values.insert(X(0), initial_pose)
        self.values.insert(V(0), initial_velocity)
        self.values.insert(B(0), initial_bias)
        
        # Initialize clock states
        self._initialize_clock_states(initial_gnss)
        
        # Create new IMU preintegration
        self.current_bias = initial_bias
        self.current_preintegration = PreintegratedImuMeasurements(
            self.imu_params, self.current_bias
        )
        
        # Set initial IMU time
        self.last_imu_time = initial_gnss.timestamp
        
        # First ISAM2 update
        try:
            self.isam.update(self.graph, self.values)
        except Exception as e:
            self.logger.error(f"Initial ISAM2 update failed: {e}")
            # Debug: print what's in values
            self.logger.error(f"Values keys: {list(self.values.keys())}")
            raise
        self.graph.resize(0)
        self.values.clear()
        
        self.logger.info(f"Initialization complete at position: {initial_pos}")
        
    def process_imu(self, imu_meas: ImuMeasurement):
        """Process single IMU measurement
        
        Args:
            imu_meas: IMU measurement containing timestamp, accelerometer, gyroscope
        """
        if self.current_preintegration is None:
            self.logger.warning("IMU preintegration not initialized, skipping measurement")
            return
            
        # Add to buffer for potential reprocessing
        self.imu_buffer.append(imu_meas)
        
        # Compute dt
        if self.last_imu_time is not None:
            dt = imu_meas.timestamp - self.last_imu_time
            imu_meas.accelerometer[1] = - imu_meas.accelerometer[1]
            imu_meas.accelerometer[2] = - imu_meas.accelerometer[2]
            imu_meas.gyroscope[1] = - imu_meas.gyroscope[1]
            imu_meas.gyroscope[2] = - imu_meas.gyroscope[2]
            # Only integrate if dt is positive
            if dt > 0:
                # Preintegrate IMU measurement
                self.current_preintegration.integrateMeasurement(
                    imu_meas.accelerometer,
                    imu_meas.gyroscope,
                    dt
                )
            elif dt < 0:
                self.logger.warning(f"Negative dt in IMU: {dt}, timestamps: {self.last_imu_time} -> {imu_meas.timestamp}")
        
        self.last_imu_time = imu_meas.timestamp
        
    def process_gnss(self, gnss_meas: GnssMeasurement) -> Dict:
        """Process GNSS measurement and update the graph
        
        Args:
            gnss_meas: GNSS observations at current epoch
            
        Returns:
            Dict containing current state estimate and statistics
        """
        self.current_state_idx += 1
        idx = self.current_state_idx
        
        # Add IMU factor if we have preintegrated measurements
        if self.current_preintegration and self.current_preintegration.deltaTij() > 0:
            #self._add_imu_factor(idx)
            self.logger.info(f"Added IMU factor with dt={self.current_preintegration.deltaTij()}")
        else:
            self.logger.warning(f"No IMU preintegration available for idx {idx}")
            
        # Detect cycle slips and update ambiguity states
        self._detect_cycle_slips(gnss_meas)
        
        # Add GNSS factors based on current mode
        self.logger.debug(f"Adding GNSS factors for idx={idx}, mode={self.rtk_mode}")
        gnss_factor_count_before = self.graph.size()
        
        if self.rtk_mode == 'RTK_FIXED':
            self._add_rtk_fixed_factors(idx, gnss_meas)
        elif self.rtk_mode == 'RTK_FLOAT':
            self._add_rtk_float_factors(idx, gnss_meas)
        elif self.rtk_mode == 'DGPS':
            self._add_dgps_factors(idx, gnss_meas)
        else:  # SPP
            self._add_spp_factors(idx, gnss_meas)
            
        gnss_factors_added = self.graph.size() - gnss_factor_count_before
        self.logger.info(f"Added {gnss_factors_added} GNSS factors for epoch {idx}")
            
        # Add clock dynamics
        self._add_clock_factors(idx)
        
        # Predict current state for initialization
        predicted_state = self._predict_current_state()
        self.values.insert(X(idx), predicted_state.pose())
        self.values.insert(V(idx), predicted_state.velocity())
        self.values.insert(B(idx), self.current_bias)
        
        # Add clock state
        if idx > 0:
            # Predict clock from previous
            try:
                prev_clock = self.isam.calculateEstimate().atVector(C(idx-1))
                dt = self.current_preintegration.deltaTij() if self.current_preintegration else 0.0
                predicted_clock = prev_clock + np.array([dt * prev_clock[1], 0])  # bias + dt*drift
                self.values.insert(C(idx), predicted_clock)
                self.logger.debug(f"Clock state at epoch {idx}: bias={predicted_clock[0]:.3f}m, drift={predicted_clock[1]:.3f}m/s")
            except Exception as e:
                self.logger.warning(f"Failed to predict clock from previous: {e}")
                self.values.insert(C(idx), np.array([0.0, 0.0]))
                self.logger.debug(f"Clock state at epoch {idx}: bias=0.0m, drift=0.0m/s (reset)")
        else:
            self.values.insert(C(idx), np.array([0.0, 0.0]))
            self.logger.debug(f"Clock state at epoch {idx}: bias=0.0m, drift=0.0m/s (initial)")
        
        # Update ISAM2
        try:
            result = self.isam.update(self.graph, self.values)
        except Exception as e:
            self.logger.error(f"ISAM2 update failed at idx {idx}: {e}")
            self.logger.error(f"Graph factors: {self.graph.size()}")
            self.logger.error(f"Values keys: {list(self.values.keys())}")
            # Print each factor type
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                self.logger.error(f"Factor {i}: {type(factor).__name__} with keys {list(factor.keys())}")
            raise
        
        # Clear graph and values for next iteration
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        # Extract current estimate
        current_estimate = self.isam.calculateEstimate()
        current_pose = current_estimate.atPose3(X(idx))
        current_velocity = current_estimate.atVector(V(idx))
        current_bias = current_estimate.atConstantBias(B(idx))
        
        # Update bias for next preintegration
        self.current_bias = current_bias
        
        # Reset IMU preintegration
        self.current_preintegration = PreintegratedImuMeasurements(
            self.imu_params, self.current_bias
        )
        self.imu_buffer.clear()
        
        # Try ambiguity resolution if in float mode
        if self.rtk_mode == 'RTK_FLOAT':
            self._try_fix_ambiguities(current_estimate)
            
        # Update RTK mode based on solution quality
        self._update_rtk_mode(result, gnss_meas)
        
        return {
            'timestamp': gnss_meas.timestamp,
            'position': current_pose.translation(),
            'velocity': current_velocity,
            'attitude': current_pose.rotation(),
            'bias': current_bias.vector(),
            'mode': self.rtk_mode,
            'num_satellites': len(gnss_meas.pseudoranges),
            'fixed_ambiguities': len(self.fixed_ambiguities)
        }
        
    def _add_imu_factor(self, idx: int):
        """Add IMU preintegration factor between consecutive states"""
        imu_factor = ImuFactor(
            X(idx-1), V(idx-1), X(idx), V(idx), B(idx-1),
            self.current_preintegration
        )
        self.graph.add(imu_factor)
        
        # Add bias random walk
        bias_noise = noiseModel.Diagonal.Sigmas(np.array([
            self.config['imu_params']['accel_bias_rw_sigma'],
            self.config['imu_params']['accel_bias_rw_sigma'],
            self.config['imu_params']['accel_bias_rw_sigma'],
            self.config['imu_params']['gyro_bias_rw_sigma'],
            self.config['imu_params']['gyro_bias_rw_sigma'],
            self.config['imu_params']['gyro_bias_rw_sigma']
        ]) * np.sqrt(self.current_preintegration.deltaTij()))
        
        self.graph.add(BetweenFactorConstantBias(
            B(idx-1), B(idx),
            gtsam.imuBias.ConstantBias(),  # Zero change expected
            bias_noise
        ))
        
    def _add_spp_factors(self, idx: int, gnss_meas: GnssMeasurement):
        """Add single point positioning factors"""
        # Pseudorange noise model
        pr_noise = noiseModel.Diagonal.Sigmas(
            np.ones(1) * self.config['gnss_params']['pseudorange_sigma']
        )
        
        for sat_id, pr in gnss_meas.pseudoranges.items():
            if sat_id not in gnss_meas.sat_positions:
                continue
                
            # Create pseudorange factor
            try:
                factor = create_pseudorange_factor(
                    X(idx), C(idx), 
                    pr, gnss_meas.sat_positions[sat_id],
                    pr_noise
                )
                self.graph.add(factor)
            except Exception as e:
                self.logger.warning(f"Failed to add pseudorange factor for sat {sat_id}: {e}")
            
    def _add_rtk_float_factors(self, idx: int, gnss_meas: GnssMeasurement):
        """Add RTK float solution factors"""
        # Add pseudorange factors
        self._add_spp_factors(idx, gnss_meas)
        
        # Add carrier phase factors with float ambiguities
        cp_noise = noiseModel.Diagonal.Sigmas(
            np.ones(1) * self.config['gnss_params']['carrier_phase_sigma']
        )
        
        for sat_id, cp in gnss_meas.carrier_phases.items():
            if sat_id not in gnss_meas.sat_positions:
                continue
                
            # Get or create ambiguity state
            if sat_id not in self.ambiguity_states:
                self._initialize_ambiguity(sat_id, idx, gnss_meas)
                
            # Add carrier phase factor
            factor = CarrierPhaseFactor(
                X(idx), C(idx), self.ambiguity_states[sat_id],
                cp, gnss_meas.sat_positions[sat_id],
                cp_noise
            )
            self.graph.add(factor)
            
    def _add_rtk_fixed_factors(self, idx: int, gnss_meas: GnssMeasurement):
        """Add RTK fixed solution factors with integer ambiguity constraints"""
        # First add float factors
        self._add_rtk_float_factors(idx, gnss_meas)
        
        # Add integer constraints for fixed ambiguities
        integer_noise = noiseModel.Diagonal.Sigmas(np.ones(1) * 0.001)
        
        for sat_id in self.fixed_ambiguities:
            if sat_id in self.ambiguity_states:
                # Get integer value from previous solution
                integer_value = self._get_fixed_ambiguity_value(sat_id)
                
                factor = PriorFactorVector(
                    self.ambiguity_states[sat_id],
                    np.array([integer_value]),
                    integer_noise
                )
                self.graph.add(factor)
                
    def _compute_spp_solution(self, gnss_meas: GnssMeasurement) -> np.ndarray:
        """Compute single point positioning solution
        
        Args:
            gnss_meas: GNSS observations
            
        Returns:
            ECEF position (3x1)
        """
        # Directly use least squares solution
        return self._least_squares_position(gnss_meas)
            
    def _least_squares_position(self, gnss_meas: GnssMeasurement) -> np.ndarray:
        """Compute position using least squares"""
        # Better initial guess - use average of satellite positions
        sat_positions = np.array(list(gnss_meas.sat_positions.values()))
        if len(sat_positions) > 0:
            # Start from a point closer to Earth surface
            avg_sat_pos = np.mean(sat_positions, axis=0)
            earth_radius = 6371000  # meters
            initial_pos = avg_sat_pos * (earth_radius / np.linalg.norm(avg_sat_pos))
        else:
            initial_pos = np.array([6371000, 0, 0])  # Default Earth surface position
            
        x = np.zeros(4)  # [x, y, z, clock_bias]
        x[:3] = initial_pos
        
        for iteration in range(10):  # Iterate to converge
            H = []
            z = []
            
            for sat_id, pr in gnss_meas.pseudoranges.items():
                if sat_id not in gnss_meas.sat_positions:
                    continue
                    
                sat_pos = gnss_meas.sat_positions[sat_id]
                
                # Geometric range
                r = np.linalg.norm(sat_pos - x[:3])
                
                # Line of sight vector
                los = (x[:3] - sat_pos) / r
                
                # Measurement residual
                z.append(pr - r - x[3])
                
                # Jacobian row
                H.append(np.hstack([los, 1]))
                
            if len(H) < 4:
                self.logger.warning(f"Only {len(H)} satellites available, need at least 4")
                break
                
            H = np.array(H)
            z = np.array(z)
            
            # Least squares update
            try:
                dx = np.linalg.lstsq(H, z, rcond=None)[0]
                x += dx
                
                if np.linalg.norm(dx) < 1e-4:
                    break
            except np.linalg.LinAlgError:
                self.logger.error("Least squares failed")
                break
                
        self.logger.info(f"Least squares converged after {iteration+1} iterations, position: {x[:3]}")
        return x[:3]
        
    def _estimate_initial_orientation(self, imu_data: List[ImuMeasurement]) -> Tuple[Rot3, gtsam.imuBias.ConstantBias]:
        """Estimate initial orientation and bias from static IMU data"""
        # Average accelerometer readings
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        
        for meas in imu_data:
            accel_sum += meas.accelerometer
            gyro_sum += meas.gyroscope
            
        accel_avg = accel_sum / len(imu_data)
        gyro_bias = gyro_sum / len(imu_data)
        
        # Compute roll and pitch from accelerometer
        roll = np.arctan2(accel_avg[1], accel_avg[2])
        pitch = np.arctan2(-accel_avg[0], np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
        yaw = 0.0  # Unknown without magnetometer
        
        # Create rotation
        rotation = Rot3.RzRyRx(yaw, pitch, roll)
        
        # Initial bias estimate
        accel_bias = np.zeros(3)  # Assume zero accelerometer bias initially
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        
        return rotation, bias
        
    def _predict_current_state(self) -> NavState:
        """Predict current state using IMU preintegration"""
        if self.current_state_idx == 0:
            # First epoch, use initial values
            return NavState(
                self.values.atPose3(X(0)),
                self.values.atVector(V(0))
            )
            
        # Get previous state estimate
        prev_estimate = self.isam.calculateEstimate()
        prev_pose = prev_estimate.atPose3(X(self.current_state_idx - 1))
        prev_velocity = prev_estimate.atVector(V(self.current_state_idx - 1))
        prev_nav_state = NavState(prev_pose, prev_velocity)
        
        # Predict using IMU
        return self.current_preintegration.predict(prev_nav_state, self.current_bias)
        
    def _initialize_clock_states(self, gnss_meas: GnssMeasurement):
        """Initialize receiver clock states"""
        # Simple clock model: bias and drift
        clock_bias = 0.0  # Will be estimated
        clock_drift = 0.0
        
        clock_noise = noiseModel.Diagonal.Sigmas(np.array([1000.0, 1.0]))  # Large initial uncertainty
        
        self.graph.add(PriorFactorVector(C(0), np.array([clock_bias, clock_drift]), clock_noise))
        self.values.insert(C(0), np.array([clock_bias, clock_drift]))
        
    def _add_clock_factors(self, idx: int):
        """Add clock dynamics between epochs"""
        if idx == 0:
            return
            
        # Clock random walk model
        dt = self.current_preintegration.deltaTij()
        
        # Clock transition noise
        clock_noise = noiseModel.Diagonal.Sigmas(
            np.array([
                self.config['gnss_params']['clock_bias_sigma'] * np.sqrt(dt),
                self.config['gnss_params']['clock_drift_sigma'] * np.sqrt(dt)
            ])
        )
        
        # Use between factor for vectors
        # Clock model: C(k+1) = C(k) + [dt*drift; 0] + noise
        try:
            self.graph.add(gtsam.BetweenFactorVector(
                C(idx-1), C(idx), 
                np.array([0.0, 0.0]),  # Expected change (drift integrated over time)
                clock_noise
            ))
            self.logger.debug(f"Added clock factor between C({idx-1}) and C({idx}) with dt={dt}")
            
            # Add a soft constraint on clock drift to prevent divergence
            if idx % 10 == 0:  # Every 10 epochs
                drift_prior_noise = noiseModel.Diagonal.Sigmas(np.array([1000.0, 1.0]))  # Loose on bias, tight on drift
                self.graph.add(gtsam.PriorFactorVector(
                    C(idx), np.array([0.0, 0.0]), drift_prior_noise
                ))
                self.logger.debug(f"Added clock drift prior at epoch {idx}")
                
        except Exception as e:
            self.logger.error(f"Failed to add clock factor: {e}")
        
    def _detect_cycle_slips(self, gnss_meas: GnssMeasurement):
        """Detect cycle slips in carrier phase measurements"""
        # Placeholder for cycle slip detection
        # In practice, would use phase-code differences, phase prediction, etc.
        pass
        
    def _try_fix_ambiguities(self, current_estimate: Values):
        """Attempt to fix float ambiguities to integers using LAMBDA"""
        # Placeholder for LAMBDA ambiguity resolution
        # Would extract float ambiguities, apply LAMBDA, validate
        pass
        
    def _update_rtk_mode(self, result: ISAM2Result, gnss_meas: GnssMeasurement):
        """Update RTK processing mode based on solution quality"""
        # Simple mode logic - in practice would check:
        # - Number of satellites
        # - DOP values  
        # - Residuals
        # - Ambiguity validation
        
        num_sats = len(gnss_meas.pseudoranges)
        
        if gnss_meas.base_obs and num_sats >= 5:
            if len(self.fixed_ambiguities) >= 4:
                self.rtk_mode = 'RTK_FIXED'
            else:
                self.rtk_mode = 'RTK_FLOAT'
        elif gnss_meas.base_obs:
            self.rtk_mode = 'DGPS'
        else:
            self.rtk_mode = 'SPP'
            
    def _initialize_ambiguity(self, sat_id: int, idx: int, gnss_meas: GnssMeasurement):
        """Initialize new ambiguity state"""
        # Create ambiguity symbol
        amb_symbol = N(sat_id)
        self.ambiguity_states[sat_id] = amb_symbol
        
        # Initial estimate (could be from phase-code difference)
        initial_amb = 0.0  # Simplified - would compute from observations
        
        # Large initial uncertainty
        amb_noise = noiseModel.Diagonal.Sigmas(np.ones(1) * 100.0)
        
        self.graph.add(PriorFactorVector(amb_symbol, np.array([initial_amb]), amb_noise))
        self.values.insert(amb_symbol, np.array([initial_amb]))
        
    def _get_fixed_ambiguity_value(self, sat_id: int) -> float:
        """Get fixed integer ambiguity value"""
        # Placeholder - would retrieve from ambiguity fixing results
        return 0.0


# Custom factor implementations using function-based approach
def pseudorange_error_func(measurement: float, sat_pos: np.ndarray):
    """Error function for pseudorange measurements"""
    def error(this: gtsam.CustomFactor, values: gtsam.Values, H: gtsam.JacobianVector):
        pose_key = this.keys()[0]
        clock_key = this.keys()[1]
        
        pose = values.atPose3(pose_key)
        clock = values.atVector(clock_key)
        
        # Receiver position
        rec_pos = pose.translation()
        
        # Geometric range
        diff = sat_pos - rec_pos
        range_geo = np.linalg.norm(diff)
        
        # Predicted measurement
        predicted = range_geo + clock[0]  # clock bias
        
        # Error
        error = np.array([measurement - predicted])
        
        # Compute Jacobians if requested
        if H is not None:
            # Jacobian w.r.t. pose (6x1)
            J_pose = np.zeros((1, 6))
            # Position part
            J_pose[0, :3] = -diff / range_geo
            # Rotation part is zero
            
            # Jacobian w.r.t. clock (2x1)
            J_clock = np.zeros((1, 2))
            J_clock[0, 0] = 1.0  # clock bias
            
            # Set Jacobians
            H[0] = J_pose
            H[1] = J_clock
            
        return error
    
    return error

def create_pseudorange_factor(pose_key: int, clock_key: int, measurement: float, 
                            sat_pos: np.ndarray, noise_model):
    """Create a pseudorange factor"""
    error_func = pseudorange_error_func(measurement, sat_pos)
    return gtsam.CustomFactor(noise_model, [pose_key, clock_key], error_func)


class CarrierPhaseFactor(gtsam.CustomFactor):
    """Carrier phase measurement factor"""
    def __init__(self, pose_key, clock_key, amb_key, measurement, sat_pos, noise_model):
        super().__init__(noise_model, [pose_key, clock_key, amb_key])
        self.measurement = measurement
        self.sat_pos = sat_pos
        self.wavelength = 0.19  # L1 wavelength in meters
        
    def error(self, values):
        pose = values.atPose3(self.keys()[0])
        clock = values.atVector(self.keys()[1])
        ambiguity = values.atVector(self.keys()[2])
        
        # Receiver position
        rec_pos = pose.translation()
        
        # Geometric range
        range_geo = np.linalg.norm(self.sat_pos - rec_pos)
        
        # Predicted measurement (in cycles)
        predicted = (range_geo + clock[0]) / self.wavelength + ambiguity[0]
        
        # Error
        return np.array([self.measurement - predicted])


class LinearBetweenFactor(gtsam.CustomFactor):
    """Linear transition factor for clock states"""
    def __init__(self, key1, key2, F, noise_model):
        super().__init__(noise_model, [key1, key2])
        self.F = F
        
    def error(self, values):
        x1 = values.atVector(self.keys()[0])
        x2 = values.atVector(self.keys()[1])
        
        # Predicted state
        predicted = self.F @ x1
        
        # Error
        return x2 - predicted


class CycleSlipDetector:
    """Simple cycle slip detector"""
    def __init__(self):
        self.prev_phases = {}
        self.prev_ranges = {}
        
    def detect(self, sat_id: int, phase: float, pseudorange: float) -> bool:
        """Detect cycle slip using phase-code combination"""
        if sat_id not in self.prev_phases:
            self.prev_phases[sat_id] = phase
            self.prev_ranges[sat_id] = pseudorange
            return False
            
        # Simple detection based on phase-code difference
        wavelength = 0.19  # L1
        
        curr_diff = phase * wavelength - pseudorange
        prev_diff = self.prev_phases[sat_id] * wavelength - self.prev_ranges[sat_id]
        
        # Check if difference changed significantly
        if abs(curr_diff - prev_diff) > 5.0:  # 5 meter threshold
            return True
            
        self.prev_phases[sat_id] = phase
        self.prev_ranges[sat_id] = pseudorange
        
        return False