#!/usr/bin/env python3
"""
Example script for RTKLib-ISAM2 tight coupling with PPC-Dataset
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime

from rtk_imu_isam import RtkImuISAM
from gnss_data_loader import GnssDataLoader
from imu_data_loader import ImuDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ppc_dataset_config():
    """Load configuration for PPC-Dataset nagoya/run1"""
    base_path = Path(__file__).parent.parent.parent / "examples/data/PPC-Dataset/nagoya/run1"
    
    config = {
        # GNSS configuration
        'gnss_params': {
            'rinex_obs': str(base_path / "rover.obs"),
            'rinex_nav': [
                str(base_path / "base.nav"),
            ],
            'rinex_base': str(base_path / "base.obs"),
            'elevation_mask': 10.0,  # degrees
            'systems': ['GPS', 'GLONASS'],
            'pseudorange_sigma': 3.0,  # meters
            'carrier_phase_sigma': 0.003,  # meters
            'doppler_sigma': 0.3,  # m/s
            'clock_bias_sigma': 30.0,  # meters
            'clock_drift_sigma': 1.0,  # m/s
        },
        
        # IMU configuration
        'imu_params': {
            'imu_file': str(base_path / "imu.csv"),
            'imu_rate': 100.0,  # Hz
            'time_offset': 0.0,  # IMU to GPS time offset
            'accel_noise_sigma': 0.01,  # m/s²
            'gyro_noise_sigma': 0.001,  # rad/s
            'accel_bias_rw_sigma': 0.0001,  # m/s²/√s
            'gyro_bias_rw_sigma': 0.00001,  # rad/s/√s
            'integration_sigma': 1e-7,
        },
        
        # ISAM2 configuration
        'relinearize_threshold': 0.1,
        'relinearize_skip': 10,
        
        # RTK configuration
        'rtk_params': {
            'min_satellites': 5,
            'min_fix_satellites': 6,
            'ratio_threshold': 3.0,  # For ambiguity validation
        }
    }
    
    return config

def evaluate_trajectory(results, reference_file=None):
    """Evaluate trajectory against reference if available"""
    # Extract trajectory
    timestamps = [r['timestamp'] for r in results]
    positions = np.array([r['position'] for r in results])
    velocities = np.array([r['velocity'] for r in results])
    modes = [r['mode'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('RTK-IMU ISAM2 Results')
    
    # Plot XYZ positions
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[i, 0]
        ax.plot(timestamps, positions[:, i])
        ax.set_ylabel(f'{label} Position (m)')
        ax.grid(True)
        
    axes[2, 0].set_xlabel('Time (s)')
    
    # Plot velocities
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[i, 1]
        ax.plot(timestamps, velocities[:, i])
        ax.set_ylabel(f'{label} Velocity (m/s)')
        ax.grid(True)
        
    axes[2, 1].set_xlabel('Time (s)')
    
    # Plot mode timeline
    fig2, ax = plt.subplots(figsize=(12, 3))
    mode_mapping = {'SPP': 0, 'DGPS': 1, 'RTK_FLOAT': 2, 'RTK_FIXED': 3}
    mode_values = [mode_mapping.get(m, 0) for m in modes]
    
    ax.plot(timestamps, mode_values, 'o-')
    ax.set_yticks(list(mode_mapping.values()))
    ax.set_yticklabels(list(mode_mapping.keys()))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RTK Mode')
    ax.grid(True)
    ax.set_title('RTK Mode Timeline')
    
    # If reference available, compute errors
    if reference_file and Path(reference_file).exists():
        logger.info(f"Loading reference trajectory from: {reference_file}")
        # Load and interpolate reference
        # (Implementation depends on reference format)
        
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    logger.info("\n=== Trajectory Statistics ===")
    logger.info(f"Total epochs: {len(results)}")
    logger.info(f"Time span: {timestamps[-1] - timestamps[0]:.1f} seconds")
    
    mode_counts = {}
    for mode in modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
    logger.info("\nMode distribution:")
    for mode, count in mode_counts.items():
        percentage = 100 * count / len(modes)
        logger.info(f"  {mode}: {count} epochs ({percentage:.1f}%)")
        
    # Compute position change statistics
    pos_changes = np.diff(positions, axis=0)
    distances = np.linalg.norm(pos_changes, axis=1)
    
    logger.info(f"\nTotal distance traveled: {np.sum(distances):.1f} m")
    logger.info(f"Average speed: {np.mean(np.linalg.norm(velocities, axis=1)):.2f} m/s")

def main():
    """Main execution function"""
    logger.info("Starting RTK-IMU ISAM2 tight coupling example")
    
    # Load configuration
    config = load_ppc_dataset_config()
    
    # Check if data files exist
    gnss_file = Path(config['gnss_params']['rinex_obs'])
    imu_file = Path(config['imu_params']['imu_file'])
    
    if not gnss_file.exists():
        logger.error(f"GNSS observation file not found: {gnss_file}")
        logger.info("Please ensure PPC-Dataset is properly extracted in examples/data/PPC-Dataset/")
        return
        
    if not imu_file.exists():
        logger.error(f"IMU data file not found: {imu_file}")
        return
    
    # Initialize data loaders
    logger.info("Loading GNSS data...")
    logger.info(f"GNSS config: {config['gnss_params']}")
    gnss_loader = GnssDataLoader(config['gnss_params'])
    
    logger.info("Loading IMU data...")
    imu_loader = ImuDataLoader(config['imu_params'])
    
    # Get time range
    gnss_start, gnss_end = gnss_loader.get_time_range()
    imu_start, imu_end = imu_loader.get_time_range()
    
    # Use overlapping time range
    start_time = max(gnss_start, imu_start)
    end_time = min(gnss_end, imu_end)
    
    logger.info(f"GNSS time range: {gnss_start:.1f} to {gnss_end:.1f} seconds")
    logger.info(f"IMU time range: {imu_start:.1f} to {imu_end:.1f} seconds")
    logger.info(f"Processing time range: {start_time:.1f} to {end_time:.1f} seconds")
    
    if start_time >= end_time:
        logger.error("No overlapping time range between GNSS and IMU data")
        return
    
    # Initialize RTK-IMU ISAM2 estimator
    rtk_imu = RtkImuISAM(config)
    
    # Get initial measurements
    logger.info("Getting initial GNSS measurements...")
    initial_gnss = gnss_loader.get_measurements_at_time(start_time)
    if not initial_gnss:
        logger.error("No initial GNSS measurements available")
        return
    logger.info(f"Got {len(initial_gnss.pseudoranges)} satellites in initial GNSS")
        
    initial_imu = imu_loader.get_static_measurements(duration=5.0)
    if not initial_imu:
        logger.warning("No static IMU data for initialization, using zero bias")
        initial_imu = []
    
    # Initialize estimator
    rtk_imu.initialize(initial_gnss, initial_imu)
    
    # Process data
    results = []
    current_time = start_time
    gnss_rate = gnss_loader.get_observation_rate()
    gnss_period = 1.0 / gnss_rate
    
    logger.info(f"Starting main processing loop (GNSS rate: {gnss_rate} Hz)")
    
    # Process all epochs
    epoch_count = 0
    
    while current_time <= end_time:
        # Get IMU measurements since last GNSS epoch
        imu_measurements = imu_loader.get_measurements_between(
            current_time - gnss_period,
            current_time
        )
        
        # Process IMU measurements
        for imu_meas in imu_measurements:
            rtk_imu.process_imu(imu_meas)
            
        # Get GNSS measurements
        gnss_meas = gnss_loader.get_measurements_at_time(current_time)
        
        if gnss_meas:
            # Process GNSS and get state estimate
            result = rtk_imu.process_gnss(gnss_meas)
            results.append(result)
            epoch_count += 1
            
            # Log progress
            logger.info(
                f"Processed epoch {len(results)}, "
                f"mode: {result['mode']}, "
                f"sats: {result['num_satellites']}, "
                f"position: {result['position'][:2]}"  # Just X,Y for brevity
            )
        
        # Move to next epoch
        current_time += gnss_period
    
    logger.info(f"Processing complete. Total epochs: {len(results)}")
    
    # Save results
    output_file = f"rtk_imu_isam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r['position'] = r['position'].tolist()
            json_r['velocity'] = r['velocity'].tolist()
            json_r['attitude'] = r['attitude'].matrix().tolist()
            json_r['bias'] = r['bias'].tolist()
            json_results.append(json_r)
            
        json.dump(json_results, f, indent=2)
        
    logger.info(f"Results saved to: {output_file}")
    
    # Evaluate and plot results
    evaluate_trajectory(results)

if __name__ == "__main__":
    main()