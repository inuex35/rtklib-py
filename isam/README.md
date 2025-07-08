# ISAM2-based PPK Processor

Object-oriented implementation of PPK (Post-Processing Kinematic) using GTSAM's ISAM2 optimizer.

## Directory Structure

```
isam/
├── __init__.py          # Main exports
├── core/                # Core components
│   ├── config.py        # Configuration management
│   ├── data_loader.py   # GNSS/IMU data loading
│   ├── optimizer.py     # ISAM2 optimizer wrapper
│   └── ppk_processor.py # Main processing logic
├── factors/             # GTSAM factor implementations
│   ├── gnss_factors.py  # GNSS measurement factors
│   ├── imu_factors.py   # IMU preintegration factors
│   └── motion_factors.py # Motion constraints
└── utils/               # Utilities
    ├── visualization.py # Result plotting
    └── evaluation.py    # Solution evaluation
```

## Usage

### Simple Example

```python
from isam import PPKProcessor, ISAMConfig

# Configure processor
config = ISAMConfig(
    datadir='path/to/data',
    rovfile='rover.obs',
    basefile='base.obs',
    navfile='base.nav',
    imufile='imu.csv',
    maxepoch=100,
    use_imu=True
)

# Create and run processor
processor = PPKProcessor(config)
solutions = processor.process()
processor.save_results('output.pos')
```

### Command Line

```bash
# Process with default settings
python run_ppk_isam_new.py

# Process with custom parameters
python run_ppk_isam_new.py --datadir /path/to/data --maxepoch 100 --trace 3

# Process without IMU
python run_ppk_isam_new.py --no-imu
```

## Key Features

- **Incremental Processing**: ISAM2 allows efficient incremental updates
- **Multi-sensor Fusion**: Tightly couples GNSS and IMU measurements
- **Motion Constraints**: Supports non-holonomic constraints for ground vehicles
- **Memory Management**: Automatic marginalization of old states
- **RTKLib Compatible**: Reads/writes RTKLib format files

## Configuration Options

Key parameters in `ISAMConfig`:
- `relinearize_threshold`: Controls when to relinearize (default: 0.1)
- `relinearize_skip`: Skip relinearization every N updates (default: 10)
- `pseudorange_sigma`: Pseudorange measurement noise (default: 3.0m)
- `use_nhc`: Enable non-holonomic constraints (default: True)

## Implementation Notes

- The processor maintains RTKLib compatibility while using GTSAM internally
- IMU preintegration is handled by GTSAM's built-in functionality
- Factor creation is abstracted through factory functions for extensibility
- The optimizer wrapper provides a clean interface to ISAM2