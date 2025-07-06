"""
Simple test of RTKLib-py with ISAM2 integration
"""

import sys
import os

# Add path for RTKLib modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing RTKLib-py ISAM2 integration...")

# Test imports
try:
    print("1. Testing basic imports...")
    import rtkcmn as gn
    print("   - rtkcmn: OK")
    
    import rinex as rn
    print("   - rinex: OK")
    
    from rtkpos import rtkinit
    print("   - rtkpos.rtkinit: OK")
    
    print("\n2. Testing ISAM2 imports...")
    import gtsam
    print("   - gtsam: OK")
    
    from rtkpos_isam import RTKLibISAM2
    print("   - RTKLibISAM2: OK")
    
    from imu_loader import ImuLoader
    print("   - ImuLoader: OK")
    
    print("\n3. Testing configuration...")
    import config_ppc_isam as cfg
    print("   - config_ppc_isam: OK")
    
    print("\n4. Initializing navigation structure...")
    gn.tracelevel(0)  # Initialize trace level
    nav = rtkinit(cfg)
    print(f"   - nav initialized: na={nav.na}, nx={nav.nx}, nf={nav.nf}")
    
    print("\n5. Testing IMU loader...")
    imu_file = '../../examples/data/PPC-Dataset/nagoya/run1/imu.csv'
    if os.path.exists(imu_file):
        nav.imu_file = imu_file
        imu_loader = ImuLoader(nav)
        if imu_loader.timestamps is not None:
            print(f"   - IMU data loaded: {len(imu_loader.timestamps)} measurements")
            t_start, t_end = imu_loader.get_time_range()
            print(f"   - Time range: {t_start:.1f} to {t_end:.1f} seconds")
            print(f"   - Rate: {imu_loader.get_imu_rate():.1f} Hz")
        else:
            print("   - Failed to load IMU data")
    else:
        print(f"   - IMU file not found: {imu_file}")
        
    print("\n6. Testing ISAM2 filter creation...")
    isam_filter = RTKLibISAM2(nav)
    print("   - ISAM2 filter created successfully")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()