# RTKLib-py ISAM2 Integration

This implementation integrates ISAM2 (Incremental Smoothing and Mapping) with IMU data into RTKLib-py while maintaining the existing RTKLib structure.

## Key Features

1. **Minimal Changes to RTKLib Structure**
   - All satellite position calculations use existing RTKLib functions (satposs)
   - Observation processing uses existing RTKLib functions (zdres, ddres)
   - Only the Kalman filter is replaced with ISAM2

2. **Files Added**
   - `rtkpos_isam.py`: ISAM2 filter implementation that replaces Kalman filter
   - `postpos_isam.py`: Modified post-processing that uses ISAM2
   - `imu_loader.py`: IMU data loader compatible with RTKLib structure
   - `config_ppc_isam.py`: Configuration with IMU parameters
   - `run_ppk_isam.py`: Example run script

3. **How It Works**
   - RTKLib processes GNSS observations normally
   - Satellite positions are computed using RTKLib's satposs()
   - Double-differenced residuals are computed using RTKLib's ddres()
   - Instead of Kalman filter update, ISAM2 factors are created
   - IMU preintegration is added between GNSS epochs
   - ISAM2 optimizes the full trajectory

4. **Usage**
   ```bash
   cd rtklib-py/src
   python run_ppk_isam.py
   ```

5. **IMU Integration**
   - IMU data is loaded from CSV files (supports PPC-Dataset format)
   - IMU preintegration between GNSS epochs
   - Bias estimation and random walk modeling
   - Tight coupling with GNSS measurements

## Benefits

- Maintains compatibility with RTKLib data structures
- Uses proven RTKLib algorithms for GNSS processing
- Adds modern factor graph optimization (ISAM2)
- Enables IMU integration for improved accuracy
- Supports batch and incremental processing