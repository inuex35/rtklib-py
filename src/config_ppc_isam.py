# RTKLIB-Py configuration for PPC-Dataset with ISAM2 and IMU integration
# Based on config_ppc.py but adds IMU parameters

from rtkcmn import uGNSS, rSIG

# ----------- Processing options -------------------
pmode = 'kinematic'      # static, kinematic
filtertype = 'forward'   # forward, backward, combined, combined_noreset
use_sing_pos = True      # run initial single precision sol each epoch
elmin = 15               # minimum elevation for float solution (degrees)
cnr_min = [35, 35]       # min signal strength [freq1, freq2] (dB-Hz)
maxinno = 1              # outlier threshold for phase (m)
maxcode = 10             # outlier threshold for code (m)
maxage = 30              # max age of differential
maxout = 20              # maximum outage [epoch]
thresdop = 6             # cycle slip detection by doppler method
thresslip = 0.05         # cycle slip detection by geom-free LC
interp_base = False      # interpolate base observations

# ------------  Kalman Filter Statistics ------------------------
eratio = [100, 100]      # ratio between pseudorange noise and carrier phase noise for L1, L2
efact = {uGNSS.GPS: 1.0, uGNSS.GLO: 1.5, uGNSS.GAL: 1.0, uGNSS.BDS: 2.0, uGNSS.QZS: 1.0} # relative weighting of each constellation
err = [0, 0.003, 0.003, 0.0, 0, 0, 5e-12]  # error sigmas [-, base, el, bl, snr, rcvstd, satclk]
snrmax = 52              # max signal strength for variance calc (dB-Hz)
accelh = 3               # horiz accel noise sigma (m/sec2)
accelv = 1               # vert accel noise sigma (m/sec2)
prnbias = 1e-4           # Carrier phase bias sigma (cycles)
sig_p0 = 30.0            # initial pos sigma (m)
sig_v0 = 10.0            # initial vel/acc sigma (m/sec)
sig_n0 = 30.0            # initial bias sigma (m)

#  -------------Ambiguity resolution options ----------------
armode = 0               # 0:off, 1:continuous, 3:fix-and-hold (disabled for complex data)
thresar = 3              # AR threshold
thresar1 = 0.1           # max pos variation for AR
minlock = 0              # min consecutive fix samples to include sat in AR 
glo_hwbias = 0.0         # GLONASS HW bias
elmaskar = 15            # elevation mask for AR
var_holdamb = 0.1        # Hold ambiguity variance (m)
minfix = 20              # min fix samples to set hold
minfixsats = 4           # min sat pairs to test for fix
minholdsats = 5          # min sat pairs to test for hold
mindropsats = 10         # min sat pairs to drop sats from AR

# -----------  Single precision parameters ----------------------------
sing_p0 = 100            # initial pos sigma
sing_v0 = 10             # initial vel/acc sigma
sing_elmin = 10          # minimum elevation (degrees)

# -------------Base and Rover positions ------------------
# base position, set to zeros to use rinex header pos
rb = [0, 0, 0]

# Set to zero to use standard precision computed starting position
rr_f = [0, 0, 0, 0, 0, 0]
rr_b = [0, 0, 0, 0, 0, 0]

# ----------- Configure observation signals ----------------
# Simplified to use only basic GPS signals to avoid complexity
gnss_t = [uGNSS.GPS]  # Only GPS for simplicity

# Valid signals - simplified
sig_tbl = {'1C': rSIG.L1C, '2W': rSIG.L2W, '2L': rSIG.L2L, '5Q': rSIG.L5Q}

skip_sig_tbl = {uGNSS.GPS: [],   # skip these obs
                uGNSS.GLO: [],
                uGNSS.GAL: [],
                uGNSS.BDS: [],
                uGNSS.QZS: []}

# set these from table below
freq_ix0 = {uGNSS.GPS: 0}  # L1
freq_ix1 = {uGNSS.GPS: 1}  # L2

# ---------- Frequencies currently supported-------------
freq = [1.57542e9,   # L1/E1
        1.22760e9]   # L2

dfreq_glo = [0.56250E6, 0.43750E6]  # L1, L2 

# number of frequencies, always 2 for now
nf = 2

# exclude satellites
excsats = []

# ----------- IMU Configuration (NEW) -------------------
# IMU data file path (will be set by run script)
imu_file = None

# IMU noise parameters
accel_noise_sigma = 0.01      # Accelerometer noise (m/s²)
gyro_noise_sigma = 0.001      # Gyroscope noise (rad/s)
accel_bias_rw_sigma = 0.0001  # Accelerometer bias random walk (m/s²/√s)
gyro_bias_rw_sigma = 0.00001  # Gyroscope bias random walk (rad/s/√s)

# IMU data rate
imu_rate = 100.0  # Hz

# IMU to GPS time offset (if any)
imu_time_offset = 0.0  # seconds