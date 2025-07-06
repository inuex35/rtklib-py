"""
Post-processing solution with ISAM2 filter integration
Based on original postpos.py but uses ISAM2 for filtering
"""
import numpy as np
from copy import copy, deepcopy
from rtkpos import rtkinit
from rtkpos_isam import rtkpos_isam
import __ppk_config as cfg
import rinex as rn
from pntpos import pntpos
import rtkcmn as gn
from imu_loader import ImuLoader

# Import original functions
from postpos import combres, firstpos, sqrtvar, savesol


def procpos_isam(nav, rov, base, fp_stat):
    """Process positioning with ISAM2 filter
    
    This replaces the original procpos but maintains compatibility
    """
    
    # Load IMU data if available
    if hasattr(nav, 'imu_file') and nav.imu_file:
        imu_loader = ImuLoader(nav)
        if imu_loader.timestamps is not None:
            gn.trace(3, f'Loaded IMU data: {len(imu_loader.timestamps)} measurements\n')
            # Store IMU loader in nav for access by ISAM2 filter
            nav.imu_loader = imu_loader
    
    try:
        if nav.filtertype != 'backward':
            # Run forward solution with ISAM2
            firstpos(nav, rov, base, dir=1)
            rtkpos_isam(nav, rov, base, fp_stat, dir=1) 
            sol0 = deepcopy(nav.sol) if hasattr(nav, 'sol') and nav.sol else []
            gn.trace(1, f"Forward solution contains {len(sol0)} epochs\n")
            if sol0:
                savesol(sol0,'forward_isam.pos')
            
        if nav.filtertype != 'forward':
            # Run backward solution with ISAM2
            if nav.filtertype != 'combined_noreset':
                # Reset filter states
                rb = nav.rb.copy()
                eph, geph = nav.eph.copy(), nav.geph.copy()
                glofrq = nav.glofrq.copy()
                maxepoch = nav.maxepoch
                nav = rtkinit(cfg)
                nav.rb = rb
                nav.eph, nav.geph = eph, geph
                nav.glofrq = glofrq
                nav.maxepoch = maxepoch
                
                # Reload IMU data for backward pass
                if hasattr(cfg, 'imu_file') and cfg.imu_file:
                    nav.imu_file = cfg.imu_file
                    imu_loader = ImuLoader(nav)
                    nav.imu_loader = imu_loader
                    
            firstpos(nav, rov, base, dir=-1)
            rtkpos_isam(nav, rov, base, fp_stat, dir=-1)
            sol1 = deepcopy(nav.sol)
            savesol(sol1,'backward_isam.pos')
            
        if nav.filtertype == 'combined' or nav.filtertype == 'combined_noreset':
            # Combine forward/backward solutions
            sol = combres(sol0, sol1)
        elif nav.filtertype == 'backward':
            sol = sol1
        else:
            # For forward-only, return the solution from nav.sol
            sol = nav.sol if hasattr(nav, 'sol') else sol0
            
    except Exception as e:
        gn.trace(1, f'procpos_isam error: {e}\n')
        import traceback
        traceback.print_exc()
        sol = []
        
    return sol


# Make procpos point to ISAM2 version  
procpos = procpos_isam