#!/usr/bin/env python3
"""
GNSS Navigation Data Reader for RTKLib-ISAM2 Integration
Reads navigation files and provides satellite positions
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# RTKLib imports
from rtkcmn import gtime_t, epoch2time, gpst2time, tracelevel
from rtkpos import rtkinit
from rinex import rnx_decode
from ephemeris import satposs

# Set trace level to suppress debug messages
tracelevel(0)


class GnssNavReader:
    """Read and process GNSS navigation data"""
    
    def __init__(self, nav_files: List[str], config: Dict = None):
        """Initialize navigation reader
        
        Args:
            nav_files: List of navigation RINEX files
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nav_files = nav_files if isinstance(nav_files, list) else [nav_files]
        self.config = config or {}
        
        # Initialize RTKLib navigation structure manually
        from rtkcmn import Nav
        self.nav = Nav.__new__(Nav)  # Create without calling __init__
        
        # Initialize minimum required attributes
        self.nav.eph = []       # GPS/GAL/BDS/QZS ephemeris
        self.nav.geph = []      # GLONASS ephemeris  
        self.nav.eph_index = {}  # Ephemeris index for fast lookup
        self.nav.ion = np.zeros((2, 4))  # Ionospheric parameters
        self.nav.obs_idx = [[0], [0]]  # Observation indices
        self.nav.glofrq = {}  # GLONASS frequency numbers
        
        # Load navigation data
        self._load_navigation_data()
        
    def _load_navigation_data(self):
        """Load navigation data from RINEX files"""
        # Create RINEX decoder with proper config
        class SimpleConfig:
            def __init__(self):
                self.nf = 2  # Number of frequencies
                self.sig_tbl = {}
                self.skip_sig_tbl = {}
                
        cfg = SimpleConfig()
        decoder = rnx_decode(cfg)
        
        # Load each navigation file
        for nav_file in self.nav_files:
            if not Path(nav_file).exists():
                self.logger.warning(f"Navigation file not found: {nav_file}")
                continue
                
            self.logger.info(f"Loading navigation file: {nav_file}")
            try:
                decoder.decode_nav(nav_file, self.nav)
                self.logger.info(f"Loaded {len(self.nav.eph)} GPS/GAL/BDS/QZS ephemerides")
                self.logger.info(f"Loaded {len(self.nav.geph)} GLONASS ephemerides")
            except Exception as e:
                self.logger.error(f"Failed to load navigation file {nav_file}: {e}")
                
    def get_satellite_positions(self, timestamp: float, sat_ids: List[int], 
                              pseudoranges: Dict[int, float]) -> Tuple[Dict, Dict, Dict]:
        """Get satellite positions at given timestamp
        
        Args:
            timestamp: GPS timestamp in seconds
            sat_ids: List of satellite IDs
            pseudoranges: Dictionary of satellite ID to pseudorange measurements
            
        Returns:
            Tuple of (positions, velocities, clock_biases) dictionaries
        """
        # Create observation structure for RTKLib
        from rtkcmn import Obs
        obs = Obs()
        
        # Convert satellite IDs and pseudoranges
        n_sats = len(sat_ids)
        # Ensure sat_ids are regular Python ints, not numpy int64
        obs.sat = np.array([int(sid) for sid in sat_ids], dtype=int)
        obs.P = np.zeros((n_sats, 2))  # 2 frequencies
        
        # Fill pseudorange data
        for i, sat_id in enumerate(sat_ids):
            if sat_id in pseudoranges:
                obs.P[i, 0] = pseudoranges[sat_id]
        
        # Set observation time
        obs.t = gtime_t()
        obs.t.time = int(timestamp)
        obs.t.sec = timestamp - int(timestamp)
        
        # Compute satellite positions
        rs, var, dts, svh = satposs(obs, self.nav)
        
        # Extract results into dictionaries
        positions = {}
        velocities = {}
        clock_biases = {}
        
        for i, sat_id in enumerate(sat_ids):
            # Check if position is valid
            if np.linalg.norm(rs[i, :3]) > 1e6:  # Must be above Earth's surface
                positions[sat_id] = rs[i, :3].copy()
                velocities[sat_id] = rs[i, 3:6].copy()
                clock_biases[sat_id] = dts[i]
            else:
                self.logger.debug(f"Invalid position for satellite {sat_id}")
                
        return positions, velocities, clock_biases
        
    def has_navigation_data(self) -> bool:
        """Check if navigation data is available"""
        return len(self.nav.eph) > 0 or len(self.nav.geph) > 0