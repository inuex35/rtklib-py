"""
IMU measurement factors for GTSAM
"""

import gtsam
import numpy as np
from gtsam import symbol


class IMUPreintegrationFactor:
    """IMU preintegration factor wrapper"""
    
    def __init__(self, preintegrated_imu):
        self.preintegrated_imu = preintegrated_imu
        
    def create_factor(self, pose_i_key, vel_i_key, pose_j_key, vel_j_key, bias_key):
        """Create GTSAM IMU factor"""
        return gtsam.ImuFactor(
            pose_i_key, vel_i_key,
            pose_j_key, vel_j_key,
            bias_key,
            self.preintegrated_imu
        )


def create_imu_factor(imu_params, acc_measurements, gyro_measurements, dt_measurements):
    """Create IMU preintegration factor from measurements"""
    
    # Create preintegration object
    preintegrated_imu = gtsam.PreintegratedImuMeasurements(imu_params)
    
    # Add measurements
    for i in range(len(acc_measurements)):
        preintegrated_imu.integrateMeasurement(
            acc_measurements[i],
            gyro_measurements[i],
            dt_measurements[i]
        )
    
    return IMUPreintegrationFactor(preintegrated_imu)