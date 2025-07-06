#!/usr/bin/env python3
"""
GNSS Data Loader for RTKLib-ISAM2 Integration
Handles loading and preprocessing of RINEX files for tight coupling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path

# RTKLib imports - using georinex instead for RINEX reading
# Define necessary constants
CLIGHT = 299792458.0  # speed of light
FREQ1 = 1575.42e6     # L1 frequency

# For more advanced RINEX handling
try:
    import georinex as gr
    HAS_GEORINEX = True
except ImportError:
    HAS_GEORINEX = False

from dataclasses import dataclass

# Import GnssMeasurement from rtk_imu_isam
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rtk_imu_isam import GnssMeasurement
from gnss_nav_reader import GnssNavReader

@dataclass
class SatelliteObservation:
    """Single satellite observation"""
    prn: int
    pseudorange_l1: Optional[float] = None
    pseudorange_l2: Optional[float] = None
    carrier_phase_l1: Optional[float] = None
    carrier_phase_l2: Optional[float] = None
    doppler_l1: Optional[float] = None
    doppler_l2: Optional[float] = None
    cno_l1: Optional[float] = None
    cno_l2: Optional[float] = None
    

class GnssDataLoader:
    """Load and preprocess GNSS data from RINEX files"""
    
    def __init__(self, config: Dict):
        """Initialize GNSS data loader
        
        Args:
            config: Configuration containing:
                - rinex_obs: Path to observation RINEX file
                - rinex_nav: Path(s) to navigation RINEX file(s)
                - rinex_base: Optional base station RINEX for RTK
                - elevation_mask: Minimum satellite elevation (degrees)
                - systems: GNSS systems to use (GPS, GLONASS, etc.)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load navigation data using RTKLib reader
        self.nav_reader = None
        if 'rinex_nav' in config and config['rinex_nav']:
            self.nav_reader = GnssNavReader(config['rinex_nav'], config)
        
        # Load observation data
        self.obs_data = self._load_observations(config['rinex_obs'])
        
        # Load base station data if available
        self.base_data = None
        if 'rinex_base' in config and config['rinex_base']:
            self.base_data = self._load_observations(config['rinex_base'])
            
        # Preprocessing parameters
        self.elevation_mask = np.radians(config.get('elevation_mask', 10.0))
        self.systems = config.get('systems', ['GPS'])
        
        
    def _load_observations(self, obs_file):
        """Load observation data from RINEX obs file"""
        self.logger.info(f"Loading observation file: {obs_file}")
        
        if HAS_GEORINEX:
            # Use georinex for RINEX reading
            try:
                obs = gr.load(obs_file)
                return obs
            except Exception as e:
                self.logger.error(f"Failed to load observation file: {e}")
                raise
        else:
            raise RuntimeError("georinex is required for RINEX file reading. Install with: pip install georinex")
            
        
    def get_measurements_at_time(self, timestamp: float, time_tolerance: float = 0.1) -> Optional[GnssMeasurement]:
        """Get GNSS measurements at specified timestamp
        
        Args:
            timestamp: Desired timestamp (GPS seconds)
            time_tolerance: Maximum time difference to accept (seconds)
            
        Returns:
            GnssMeasurement object or None if no data available
        """
        if not HAS_GEORINEX or self.obs_data is None:
            return None
            
        # Convert timestamp to datetime for georinex
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Find closest time in observations
        try:
            obs_times = self.obs_data.time.values
            time_diffs = np.abs((obs_times - np.datetime64(dt)).astype('timedelta64[s]').astype(float))
            min_idx = np.argmin(time_diffs)
            
            if time_diffs[min_idx] > time_tolerance:
                return None
                
            obs_time = obs_times[min_idx]
            
            # Extract observations at this time
            obs_epoch = self.obs_data.sel(time=obs_time)
            
            # Get satellite list
            try:
                sats = obs_epoch.sv.values
            except:
                sats = obs_epoch.satellite.values
                
            # Build measurement object
            gnss_meas = GnssMeasurement(
                timestamp=timestamp,
                pseudoranges={},
                carrier_phases={},
                dopplers={},
                cnos={},
                sat_positions={},
                sat_velocities={}
            )
            
            # Extract measurements for each satellite
            for sat in sats:
                if not isinstance(sat, str) or len(sat) < 3:
                    continue
                    
                # Parse satellite ID
                sys = sat[0]
                prn = int(sat[1:3])
                
                # Convert to satellite number
                if sys == 'G':  # GPS
                    sat_id = prn
                elif sys == 'R':  # GLONASS
                    sat_id = prn + 100
                elif sys == 'E':  # Galileo
                    sat_id = prn + 200
                else:
                    continue
                    
                # Skip if not in requested systems
                if sys == 'G' and 'GPS' not in self.systems:
                    continue
                elif sys == 'R' and 'GLONASS' not in self.systems:
                    continue
                elif sys == 'E' and 'GALILEO' not in self.systems:
                    continue
                    
                # Extract L1 pseudorange
                try:
                    if 'C1C' in obs_epoch:
                        pr = float(obs_epoch['C1C'].sel(sv=sat).values)
                        if not np.isnan(pr) and pr > 0:
                            gnss_meas.pseudoranges[sat_id] = pr
                except:
                    pass
                    
                # Extract L1 carrier phase
                try:
                    if 'L1C' in obs_epoch:
                        cp = float(obs_epoch['L1C'].sel(sv=sat).values)
                        if not np.isnan(cp):
                            # Convert cycles to meters
                            wavelength = CLIGHT / FREQ1
                            gnss_meas.carrier_phases[sat_id] = cp * wavelength
                except:
                    pass
                    
                # Extract L1 Doppler
                try:
                    if 'D1C' in obs_epoch:
                        dop = float(obs_epoch['D1C'].sel(sv=sat).values)
                        if not np.isnan(dop):
                            gnss_meas.dopplers[sat_id] = dop
                except:
                    pass
                    
                # Extract L1 SNR
                try:
                    if 'S1C' in obs_epoch:
                        snr = float(obs_epoch['S1C'].sel(sv=sat).values)
                        if not np.isnan(snr):
                            gnss_meas.cnos[sat_id] = snr
                except:
                    pass
                    
            # Compute satellite positions from navigation data
            if self.nav_reader and self.nav_reader.has_navigation_data():
                try:
                    # Get satellite positions using navigation reader
                    sat_ids = list(gnss_meas.pseudoranges.keys())
                    positions, velocities, clock_biases = self.nav_reader.get_satellite_positions(
                        timestamp, sat_ids, gnss_meas.pseudoranges
                    )
                    
                    # Update GNSS measurement with computed positions
                    gnss_meas.sat_positions.update(positions)
                    gnss_meas.sat_velocities.update(velocities)
                    
                    # Add fallback positions for satellites without valid positions
                    for i, sat_id in enumerate(sat_ids):
                        if sat_id not in positions:
                            self._add_fallback_position(gnss_meas, sat_id, i, len(sat_ids))
                            
                except Exception as e:
                    self.logger.warning(f"Failed to compute satellite positions: {e}")
                    # Use fallback for all satellites
                    for i, sat_id in enumerate(gnss_meas.pseudoranges):
                        self._add_fallback_position(gnss_meas, sat_id, i, len(gnss_meas.pseudoranges))
            else:
                # No navigation data available
                self.logger.warning("No navigation data available for satellite position computation")
                for i, sat_id in enumerate(gnss_meas.pseudoranges):
                    self._add_fallback_position(gnss_meas, sat_id, i, len(gnss_meas.pseudoranges))
                
        except Exception as e:
            self.logger.warning(f"Error extracting measurements: {e}")
            return None
            
        # Add base observations if available
        if self.base_data is not None:
            gnss_meas.base_obs = self._get_base_obs_at_time(timestamp, time_tolerance)
            
        return gnss_meas if gnss_meas.pseudoranges else None
    
    def _add_fallback_position(self, gnss_meas, sat_id, i, n_sats):
        """Add fallback satellite position when navigation data is not available
        
        Args:
            gnss_meas: GnssMeasurement object to update
            sat_id: Satellite ID
            i: Index of satellite in the list
            n_sats: Total number of satellites
        """
        # Use a simple approximation for satellite positions
        # This is only used when navigation data is unavailable
        angle = 2 * np.pi * i / n_sats
        radius = 26560000  # Approximate GPS orbit radius in meters
        
        # Simple circular orbit approximation
        gnss_meas.sat_positions[sat_id] = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])
        
        # Approximate velocity for circular orbit
        orbital_period = 12 * 3600  # 12 hours in seconds
        orbital_velocity = 2 * np.pi * radius / orbital_period
        
        gnss_meas.sat_velocities[sat_id] = np.array([
            -orbital_velocity * np.sin(angle),
            orbital_velocity * np.cos(angle),
            0.0
        ])
        
    def _check_elevation(self, sat_pos: np.ndarray, 
                        approx_pos: np.ndarray = np.zeros(3)) -> bool:
        """Check if satellite elevation is above mask
        
        Args:
            sat_pos: Satellite position in ECEF
            approx_pos: Approximate receiver position (defaults to Earth center)
            
        Returns:
            True if elevation is acceptable
        """
        # If no approximate position, accept all
        if np.linalg.norm(approx_pos) < 1:
            return True
            
        # Compute elevation
        los = sat_pos - approx_pos
        los_norm = los / np.linalg.norm(los)
        
        # Convert to local frame (simplified)
        # In practice, would use proper ECEF to ENU transformation
        up = approx_pos / np.linalg.norm(approx_pos)
        elevation = np.arcsin(np.dot(los_norm, up))
        
        return elevation >= self.elevation_mask
        
    def _get_base_obs_at_time(self, timestamp: float, time_tolerance: float) -> Dict:
        """Get base station observations at specified time"""
        # Simplified implementation - would need to process base data similarly
        return {}
        
    def get_time_range(self) -> Tuple[float, float]:
        """Get time range of observation data
        
        Returns:
            (start_time, end_time) in GPS seconds
        """
        if self.obs_data is None or not HAS_GEORINEX:
            return (0.0, 0.0)
            
        try:
            import gnss_lib_py as glp
            
            times = self.obs_data.time.values
            start_dt = pd.Timestamp(times[0]).to_pydatetime()
            end_dt = pd.Timestamp(times[-1]).to_pydatetime()
            
            # Use gnss_lib_py to convert datetime to Unix milliseconds
            start_unix_ms = glp.datetime_to_unix_millis(start_dt)
            end_unix_ms = glp.datetime_to_unix_millis(end_dt)
            
            # Convert milliseconds to seconds
            start_time = start_unix_ms / 1000.0
            end_time = end_unix_ms / 1000.0
            
            return (start_time, end_time)
        except Exception as e:
            self.logger.warning(f"Error getting time range: {e}")
            # Fallback to direct timestamp conversion
            try:
                times = self.obs_data.time.values
                start_dt = pd.Timestamp(times[0]).to_pydatetime()
                end_dt = pd.Timestamp(times[-1]).to_pydatetime()
                start_time = start_dt.timestamp()
                end_time = end_dt.timestamp()
                return (start_time, end_time)
            except:
                return (0.0, 0.0)
        
    def get_observation_rate(self) -> float:
        """Estimate observation data rate
        
        Returns:
            Data rate in Hz
        """
        if self.obs_data is None or not HAS_GEORINEX:
            return 1.0
            
        try:
            times = self.obs_data.time.values
            if len(times) < 2:
                return 1.0
                
            # Check time between first few epochs
            dt = (pd.Timestamp(times[1]) - pd.Timestamp(times[0])).total_seconds()
            return 1.0 / dt if dt > 0 else 1.0
        except:
            return 1.0