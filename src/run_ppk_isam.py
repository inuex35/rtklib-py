"""
RTKLib-py with ISAM2 for PPK processing with IMU integration
Maintains RTKLib structure but uses ISAM2 for filtering
"""

import sys, os, shutil

# Set run parameters
maxepoch = 1000  # Process 10 epochs for debugging
trace_level = 3  # Debug trace level
basepos = []     # Default to not specified here

# Specify input files for PPC-Dataset
datadir = '../../examples/data/PPC-Dataset/tokyo/run1'
navfile = 'base.nav'
rovfile = 'rover.obs'
basefile = 'base.obs'
imufile = 'imu.csv'

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create config file with IMU path
config_template = os.path.join(script_dir, 'config_ppc_isam.py')
shutil.copyfile(config_template, '__ppk_config.py')

# Add IMU file path to config
with open('__ppk_config.py', 'a') as f:
    f.write(f"\n# IMU data file\n")
    f.write(f"imu_file = '{os.path.join(datadir, imufile)}'\n")

# Import RTKLib files
import __ppk_config as cfg
import rinex as rn
import rtkcmn as gn
from rtkpos import rtkinit
from postpos_isam import procpos  # Use ISAM2 version
from postpos import savesol

# Generate output file names
solfile = os.path.join(datadir, rovfile[:-4] + '_isam.pos')
statfile = os.path.join(datadir, rovfile[:-4] + '_isam.pos.stat')

# Create output directory if it doesn't exist
os.makedirs(datadir, exist_ok=True)

# Setup trace output
fp_stat = open(statfile, 'w')
if trace_level > 0:
    trcfile = os.path.join(datadir, rovfile[:-4] + '_isam.trace')
    sys.stderr = open(trcfile, "w")

# Initialize solution
print(f"Initializing RTKLib with ISAM2...")
print(f"Data directory: {datadir}")
print(f"Navigation file: {navfile}")
print(f"Rover file: {rovfile}")
print(f"Base file: {basefile}")
print(f"IMU file: {imufile}")

# Change to data directory for file access
original_dir = os.getcwd()
os.chdir(datadir)

try:
    # Initialize navigation structure
    gn.tracelevel(trace_level)
    nav = rtkinit(cfg)
    nav.maxepoch = maxepoch
    
    # Add IMU file to navigation structure
    nav.imu_file = imufile
    
    # Pass fp_stat to nav for solution output
    nav.fp_stat = fp_stat
    
    # Load rover observations
    rov = rn.rnx_decode(cfg)
    print('Reading rover obs...')
    if nav.filtertype == 'backward':
        maxepoch = None  # Load all obs for backward
    rov.decode_obsfile(nav, rovfile, maxepoch)
    
    # Load base observations
    base = rn.rnx_decode(cfg)
    print('Reading base obs...')
    base.decode_obsfile(nav, basefile, None)
    if basepos != []:
        nav.rb = basepos
    elif nav.rb[0] == 0:
        nav.rb = base.pos
        
    # Load navigation data from rover obs
    print('Reading nav data...')
    rov.decode_nav(navfile, nav)
    
    # Calculate solution with ISAM2
    print('Calculating solution with ISAM2 and IMU integration...\n')
    sol = procpos(nav, rov, base, fp_stat)
    
    # Save solution to file (we're in the data directory now)
    savesol(sol, rovfile[:-4] + '_isam.pos')
    fp_stat.close()
    
    print(f"\nProcessing complete!")
    print(f"Solution saved to: {solfile}")
    print(f"Statistics saved to: {statfile}")
    
    # Print summary statistics
    if len(sol) > 0:
        fixed_count = sum(1 for s in sol if s.stat == gn.SOLQ_FIX)
        float_count = sum(1 for s in sol if s.stat == gn.SOLQ_FLOAT)
        single_count = sum(1 for s in sol if s.stat == gn.SOLQ_SINGLE)
        
        print(f"\nSolution summary:")
        print(f"  Total epochs: {len(sol)}")
        print(f"  Fixed solutions: {fixed_count} ({100*fixed_count/len(sol):.1f}%)")
        print(f"  Float solutions: {float_count} ({100*float_count/len(sol):.1f}%)")
        print(f"  Single solutions: {single_count} ({100*single_count/len(sol):.1f}%)")
        
finally:
    # Return to original directory
    os.chdir(original_dir)
    
    # Plot results
    try:
        import subprocess
        print("\nPlotting results...")
        plot_script = os.path.join(script_dir, 'plot_pos.py')
        subprocess.run([sys.executable, plot_script, solfile, '--output', datadir], check=True)
    except Exception as e:
        print(f"Failed to plot results: {e}")
        print(f"You can manually plot the results using:")
        print(f"  python plot_pos.py {solfile}")