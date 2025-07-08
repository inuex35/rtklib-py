"""
Motion constraint factors for GTSAM
"""

import gtsam
import numpy as np
from gtsam import symbol


class NonHolonomicFactor:
    """Non-holonomic constraint factor (for ground vehicles)"""
    
    def __init__(self, sigma_lateral, sigma_vertical):
        self.sigma_lateral = sigma_lateral
        self.sigma_vertical = sigma_vertical
        
    def create_factor(self, pose_key, vel_key):
        """Create non-holonomic constraint factor"""
        def error_func(this, values, H):
            pose = values.atPose3(pose_key)
            vel = values.atPoint3(vel_key)
            
            # Transform velocity to body frame
            R_world_to_body = pose.rotation().matrix().T
            vel_body = R_world_to_body @ vel
            
            # Non-holonomic constraint: lateral and vertical velocities should be zero
            error = np.array([vel_body[1], vel_body[2]])  # y and z components
            
            if H:
                # Jacobian w.r.t pose (only rotation affects this)
                H_pose = np.zeros((2, 6))
                # Complex rotation derivative - simplified here
                H[0] = H_pose
                
                # Jacobian w.r.t velocity
                H_vel = R_world_to_body[1:3, :]
                H[1] = H_vel
                
            return error
            
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_lateral, self.sigma_vertical])
        )
        return gtsam.CustomFactor(noise_model, [pose_key, vel_key], error_func)


class VelocityPrior:
    """Velocity prior factor"""
    
    def __init__(self, velocity, sigma):
        self.velocity = velocity
        self.sigma = sigma
        
    def create_factor(self, vel_key):
        """Create velocity prior factor"""
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(self.sigma)
        return gtsam.PriorFactorPoint3(vel_key, gtsam.Point3(self.velocity), noise_model)


def create_motion_factor(constraint_type, **kwargs):
    """Factory function to create motion constraint factors"""
    if constraint_type == 'non_holonomic':
        return NonHolonomicFactor(
            kwargs['sigma_lateral'],
            kwargs['sigma_vertical']
        )
    elif constraint_type == 'velocity_prior':
        return VelocityPrior(
            kwargs['velocity'],
            kwargs['sigma']
        )
    else:
        raise ValueError(f"Unknown motion constraint type: {constraint_type}")