"""
RTKLIB-Py: Top level code for post-processing solution from rinex data

Copyright (c) 2022 Tim Everett
"""

import sys, os, shutil

# set run parameters
maxepoch = 50 # max # of epochs, used for debug, None = no limit (limit for PPC-Dataset testing)
trace_level = 3  # debug trace level
basepos = []  # default to not specified here

######## specify input files ######################################

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()

# Check if we're running from examples/data directory (custom data)
if current_dir.endswith('examples/data'):
    # Use PPC-Dataset tokyo/run1 data
    datadir = 'PPC-Dataset/tokyo/run1'
    navfile = 'base.nav'
    rovfile = 'rover.obs'
    basefile = 'base.obs'
    cfgfile = os.path.join(script_dir, 'config_ppc.py')  # Use PPC-specific config
    print("Using PPC-Dataset tokyo/run1 data")
elif current_dir.endswith('PPC-Dataset/tokyo/run1'):
    # Running directly from the dataset directory
    datadir = '.'
    navfile = 'base.nav'
    rovfile = 'rover.obs'
    basefile = 'base.obs'
    cfgfile = os.path.join(script_dir, 'config_ppc.py')  # Use PPC-specific config
    print("Using PPC-Dataset tokyo/run1 data (direct)")
else:
    # Use original u-blox example data
    datadir = '../data/u-blox'
    navfile = 'rover.nav'
    rovfile = 'rover.obs'
    basefile = 'tmg23590.obs'
    cfgfile = os.path.join(script_dir, 'config_f9p.py')
    print("Using u-blox example data")

###################################################################

# Debug information
print(f"Current working directory: {current_dir}")
print(f"Script directory: {script_dir}")
print(f"Data directory: {datadir}")
print(f"Config file path: {cfgfile}")
print(f"Config file exists: {os.path.exists(cfgfile)}")
print(f"Navigation file: {os.path.join(datadir, navfile)}")
print(f"Rover file: {os.path.join(datadir, rovfile)}")
print(f"Base file: {os.path.join(datadir, basefile)}")

# Check if data files exist
nav_path = os.path.join(datadir, navfile)
rov_path = os.path.join(datadir, rovfile)
base_path = os.path.join(datadir, basefile)

print(f"Navigation file exists: {os.path.exists(nav_path)}")
print(f"Rover file exists: {os.path.exists(rov_path)}")
print(f"Base file exists: {os.path.exists(base_path)}")

# Copy config file
shutil.copyfile(cfgfile, '__ppk_config.py')

# import rtklib files
import __ppk_config as cfg
import rinex as rn
import rtkcmn as gn
from rtkpos import rtkinit
from postpos import procpos, savesol

# generate output file names
solfile = rovfile[:-4] + '.pos'
statfile = os.path.join(datadir, rovfile[:-4] + '.pos.stat')

# Create output directory if it doesn't exist
os.makedirs(datadir, exist_ok=True)

fp_stat = open(statfile, 'w')
if trace_level > 0:
    trcfile = os.path.join(datadir, rovfile[:-4] + '.trace')
    sys.stderr = open(trcfile, "w")

# init solution
os.chdir(datadir)
gn.tracelevel(trace_level)
nav = rtkinit(cfg)
nav.maxepoch = maxepoch

# load rover obs
rov = rn.rnx_decode(cfg)
print('Reading rover obs...')
if nav.filtertype == 'backward':
    maxepoch = None   # load all obs for 
rov.decode_obsfile(nav, rovfile, maxepoch)

# load base obs
base = rn.rnx_decode(cfg)
print('Reading base obs...')
base.decode_obsfile(nav, basefile, None)
if basepos != []:
    nav.rb = basepos
elif nav.rb[0] == 0:
    nav.rb = base.pos
    
# load nav data from rover obs
print('Reading nav data...')
rov.decode_nav(navfile, nav)

# calculate solution
print('Calculating solution ...\n')
sol = procpos(nav, rov, base, fp_stat)

# save solution to file
savesol(sol, solfile)
fp_stat.close()
