"""
Configuration for ISAM2-based PPK processing
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ISAMConfig:
    """Configuration for ISAM2 PPK processor"""
    
    # Data paths
    datadir: str = ""
    navfile: str = ""
    rovfile: str = ""
    basefile: str = ""
    imufile: Optional[str] = None
    
    # Processing options
    maxepoch: int = 0  # 0 means process all
    trace_level: int = 0
    filtertype: str = "forward"  # forward, backward, combined
    
    # ISAM2 parameters
    relinearize_threshold: float = 0.1
    relinearize_skip: int = 10
    cache_linearized_factors: bool = True
    optimizer_type: str = "gauss_newton"  # gauss_newton or dogleg
    print_stats: bool = False
    
    # IMU parameters
    use_imu: bool = True
    imu_gravity: float = 9.81
    imu_accel_noise_sigma: float = 0.01
    imu_gyro_noise_sigma: float = 0.001
    imu_accel_bias_rw_sigma: float = 0.0001
    imu_gyro_bias_rw_sigma: float = 0.00001
    imu_integration_sigma: float = 0.0
    
    # GNSS noise parameters
    pseudorange_sigma: float = 3.0
    carrier_phase_sigma: float = 0.003
    doppler_sigma: float = 0.1
    
    # Motion constraints
    use_nhc: bool = True  # Non-holonomic constraint
    nhc_sigma_lateral: float = 0.1
    nhc_sigma_vertical: float = 0.1
    
    # Solution output
    output_dir: str = ""
    solution_prefix: str = "ppk_isam"
    
    # RTKLib compatibility options
    mode: int = 2  # 0:single, 1:dgps, 2:kinematic, 3:static
    navsys: int = 7  # GPS+GLO+GAL
    elmin: float = 15.0  # elevation mask (degrees)
    snrmin: float = 0.0  # SNR mask
    
    # Base station position
    basepos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    @classmethod
    def from_rtklib_config(cls, cfg_module):
        """Create ISAMConfig from RTKLib config module"""
        config = cls()
        
        # Copy relevant parameters
        if hasattr(cfg_module, 'filtertype'):
            config.filtertype = cfg_module.filtertype
        if hasattr(cfg_module, 'trace'):
            config.trace_level = cfg_module.trace
        if hasattr(cfg_module, 'mode'):
            config.mode = cfg_module.mode
        if hasattr(cfg_module, 'navsys'):
            config.navsys = cfg_module.navsys
        if hasattr(cfg_module, 'elmin'):
            config.elmin = cfg_module.elmin
        if hasattr(cfg_module, 'snrmin'):
            config.snrmin = cfg_module.snrmin
            
        # IMU parameters
        if hasattr(cfg_module, 'imu_file'):
            config.imufile = cfg_module.imu_file
            config.use_imu = True
            
        return config
        
    def get_output_path(self, suffix=""):
        """Get output file path"""
        if not self.output_dir:
            self.output_dir = self.datadir
            
        filename = f"{self.solution_prefix}{suffix}.pos"
        return os.path.join(self.output_dir, filename)