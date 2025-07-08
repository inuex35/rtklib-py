"""
GNSS measurement factors for GTSAM
"""

import gtsam
import numpy as np
from gtsam import symbol
import rtkcmn as gn


class GNSSPseudorangeFactor:
    """Pseudorange measurement factor"""
    
    def __init__(self, sat_pos, measured_range, sigma):
        self.sat_pos = sat_pos
        self.measured_range = measured_range
        self.sigma = sigma
        
    def create_factor(self, pos_key, clock_key):
        """Create GTSAM factor for pseudorange"""
        def error_func(this, values, H):
            pos = values.atPoint3(pos_key)
            clock_bias = values.atDouble(clock_key)
            
            # Compute geometric range
            delta = self.sat_pos - pos
            range_computed = np.linalg.norm(delta) + clock_bias * gn.CLIGHT
            
            # Error
            error = np.array([range_computed - self.measured_range])
            
            if H:
                # Jacobian w.r.t position
                H_pos = delta / np.linalg.norm(delta)
                H[0] = H_pos.reshape(1, 3)
                
                # Jacobian w.r.t clock bias
                H[1] = np.array([[gn.CLIGHT]])
                
            return error
            
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.sigma]))
        return gtsam.CustomFactor(noise_model, [pos_key, clock_key], error_func)


class GNSSCarrierPhaseFactor:
    """Carrier phase measurement factor"""
    
    def __init__(self, sat_pos, measured_phase, wavelength, sigma):
        self.sat_pos = sat_pos
        self.measured_phase = measured_phase
        self.wavelength = wavelength
        self.sigma = sigma
        
    def create_factor(self, pos_key, clock_key, ambiguity_key):
        """Create GTSAM factor for carrier phase"""
        def error_func(this, values, H):
            pos = values.atPoint3(pos_key)
            clock_bias = values.atDouble(clock_key)
            ambiguity = values.atDouble(ambiguity_key)
            
            # Compute geometric range
            delta = self.sat_pos - pos
            range_computed = np.linalg.norm(delta) + clock_bias * gn.CLIGHT
            
            # Convert to phase
            phase_computed = range_computed / self.wavelength + ambiguity
            
            # Error
            error = np.array([phase_computed - self.measured_phase])
            
            if H:
                # Jacobian w.r.t position
                H_pos = delta / (np.linalg.norm(delta) * self.wavelength)
                H[0] = H_pos.reshape(1, 3)
                
                # Jacobian w.r.t clock bias
                H[1] = np.array([[gn.CLIGHT / self.wavelength]])
                
                # Jacobian w.r.t ambiguity
                H[2] = np.array([[1.0]])
                
            return error
            
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.sigma]))
        return gtsam.CustomFactor(noise_model, [pos_key, clock_key, ambiguity_key], error_func)


class GNSSDopplerFactor:
    """Doppler measurement factor"""
    
    def __init__(self, sat_pos, sat_vel, measured_doppler, wavelength, sigma):
        self.sat_pos = sat_pos
        self.sat_vel = sat_vel
        self.measured_doppler = measured_doppler
        self.wavelength = wavelength
        self.sigma = sigma
        
    def create_factor(self, pos_key, vel_key, clock_drift_key):
        """Create GTSAM factor for Doppler"""
        def error_func(this, values, H):
            pos = values.atPoint3(pos_key)
            vel = values.atPoint3(vel_key)
            clock_drift = values.atDouble(clock_drift_key)
            
            # Line of sight vector
            delta = self.sat_pos - pos
            los = delta / np.linalg.norm(delta)
            
            # Relative velocity
            rel_vel = self.sat_vel - vel
            
            # Doppler prediction
            doppler_computed = -np.dot(los, rel_vel) / self.wavelength + clock_drift * gn.CLIGHT / self.wavelength
            
            # Error
            error = np.array([doppler_computed - self.measured_doppler])
            
            if H:
                # Compute Jacobians
                r = np.linalg.norm(delta)
                
                # Jacobian w.r.t position
                H_pos = (rel_vel - np.dot(rel_vel, los) * los) / (r * self.wavelength)
                H[0] = H_pos.reshape(1, 3)
                
                # Jacobian w.r.t velocity
                H_vel = los / self.wavelength
                H[1] = H_vel.reshape(1, 3)
                
                # Jacobian w.r.t clock drift
                H[2] = np.array([[gn.CLIGHT / self.wavelength]])
                
            return error
            
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.sigma]))
        return gtsam.CustomFactor(noise_model, [pos_key, vel_key, clock_drift_key], error_func)


def create_gnss_factor(meas_type, **kwargs):
    """Factory function to create GNSS factors"""
    if meas_type == 'pseudorange':
        return GNSSPseudorangeFactor(
            kwargs['sat_pos'],
            kwargs['measured_range'],
            kwargs['sigma']
        )
    elif meas_type == 'carrier_phase':
        return GNSSCarrierPhaseFactor(
            kwargs['sat_pos'],
            kwargs['measured_phase'],
            kwargs['wavelength'],
            kwargs['sigma']
        )
    elif meas_type == 'doppler':
        return GNSSDopplerFactor(
            kwargs['sat_pos'],
            kwargs['sat_vel'],
            kwargs['measured_doppler'],
            kwargs['wavelength'],
            kwargs['sigma']
        )
    else:
        raise ValueError(f"Unknown GNSS measurement type: {meas_type}")