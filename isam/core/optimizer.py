"""
ISAM2 optimizer wrapper for GNSS/IMU fusion
"""

import gtsam
import numpy as np
from gtsam import symbol
import rtkcmn as gn


class ISAMOptimizer:
    """ISAM2 optimizer for incremental GNSS/IMU fusion"""
    
    def __init__(self, config):
        self.config = config
        self.isam = None
        self.result = None
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = gtsam.Values()
        
        # Symbol counters
        self.pose_count = 0
        self.vel_count = 0
        self.bias_count = 0
        self.clock_count = 0
        self.amb_count = 0
        
        # Initialize ISAM2
        self._init_isam()
        
    def _init_isam(self):
        """Initialize ISAM2 with parameters"""
        # Use default ISAM2 parameters
        # Most GTSAM versions have different APIs, so we use simplest approach
        params = gtsam.ISAM2Params()
        
        # Try to set parameters if methods exist
        try:
            # These might be properties or methods depending on GTSAM version
            if hasattr(params, 'relinearizeThreshold'):
                params.relinearizeThreshold = self.config.relinearize_threshold
            if hasattr(params, 'relinearizeSkip'):
                params.relinearizeSkip = self.config.relinearize_skip
            if hasattr(params, 'cacheLinearizedFactors'):
                params.cacheLinearizedFactors = self.config.cache_linearized_factors
            if hasattr(params, 'enableDetailedResults'):
                params.enableDetailedResults = True
            if hasattr(params, 'findUnusedFactorSlots'):
                params.findUnusedFactorSlots = True
        except:
            # Use default parameters if setting fails
            pass
        
        self.isam = gtsam.ISAM2(params)
        
    def add_pose_prior(self, pose, covariance, timestamp=None):
        """Add pose prior"""
        key = symbol('X', self.pose_count)
        self.pose_count += 1
        
        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)
        factor = gtsam.PriorFactorPose3(key, pose, noise_model)
        self.graph.add(factor)
        self.initial_values.insert(key, pose)
        
        return key
        
    def add_velocity_prior(self, velocity, sigma, timestamp=None):
        """Add velocity prior"""
        key = symbol('V', self.vel_count)
        self.vel_count += 1
        
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigma)
        factor = gtsam.PriorFactorPoint3(key, gtsam.Point3(velocity), noise_model)
        self.graph.add(factor)
        self.initial_values.insert(key, gtsam.Point3(velocity))
        
        return key
        
    def add_bias_prior(self, bias, sigma):
        """Add IMU bias prior"""
        key = symbol('B', self.bias_count)
        self.bias_count += 1
        
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigma)
        factor = gtsam.PriorFactorConstantBias(key, bias, noise_model)
        self.graph.add(factor)
        self.initial_values.insert(key, bias)
        
        return key
        
    def add_clock_state(self, clock_bias, clock_drift, sigma):
        """Add clock state (bias and drift)"""
        bias_key = symbol('C', self.clock_count)
        drift_key = symbol('D', self.clock_count)
        self.clock_count += 1
        
        # Add initial values
        self.initial_values.insert(bias_key, clock_bias)
        self.initial_values.insert(drift_key, clock_drift)
        
        # Add priors if needed
        if sigma is not None:
            noise_bias = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma[0]]))
            noise_drift = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma[1]]))
            
            self.graph.add(gtsam.PriorFactorDouble(bias_key, clock_bias, noise_bias))
            self.graph.add(gtsam.PriorFactorDouble(drift_key, clock_drift, noise_drift))
        
        return bias_key, drift_key
        
    def add_gnss_factor(self, factor, keys):
        """Add GNSS measurement factor"""
        self.graph.add(factor.create_factor(*keys))
        
    def add_imu_factor(self, factor, keys):
        """Add IMU preintegration factor"""
        self.graph.add(factor.create_factor(*keys))
        
    def add_motion_constraint(self, factor, keys):
        """Add motion constraint factor"""
        self.graph.add(factor.create_factor(*keys))
        
    def update(self):
        """Update ISAM2 with new factors"""
        self.result = self.isam.update(self.graph, self.initial_values)
        self.current_estimate = self.isam.calculateEstimate()
        
        # Clear for next iteration
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        
        return self.result
        
    def get_current_estimate(self):
        """Get current state estimate"""
        return self.current_estimate
        
    def get_pose(self, index):
        """Get pose estimate"""
        key = symbol('X', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atPose3(key)
        return None
        
    def get_velocity(self, index):
        """Get velocity estimate"""
        key = symbol('V', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atPoint3(key)
        return None
        
    def get_bias(self, index):
        """Get IMU bias estimate"""
        key = symbol('B', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atConstantBias(key)
        return None
        
    def get_clock_bias(self, index):
        """Get clock bias estimate"""
        key = symbol('C', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atDouble(key)
        return None
        
    def get_clock_drift(self, index):
        """Get clock drift estimate"""
        key = symbol('D', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atDouble(key)
        return None
        
    def get_ambiguity(self, index):
        """Get carrier phase ambiguity"""
        key = symbol('N', index)
        if self.current_estimate.exists(key):
            return self.current_estimate.atDouble(key)
        return None
        
    def marginalize_old_states(self, keep_last_n=10):
        """Marginalize old states to manage memory"""
        if self.pose_count > keep_last_n * 2:
            # Create list of keys to marginalize
            keys_to_marginalize = gtsam.KeyVector()
            
            margin_until = self.pose_count - keep_last_n
            for i in range(margin_until):
                keys_to_marginalize.append(symbol('X', i))
                keys_to_marginalize.append(symbol('V', i))
                
            # Marginalize
            self.isam.marginalizeLeaves(keys_to_marginalize)
            
    def calculate_error(self):
        """Calculate total error of current estimate"""
        if self.current_estimate:
            return self.isam.getFactorsUnsafe().error(self.current_estimate)
        return 0.0