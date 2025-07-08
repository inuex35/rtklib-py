#!/usr/bin/env python3
"""
Object-oriented wrapper for PPK ISAM processing
Uses existing RTKLib infrastructure with cleaner interface
"""

import sys
import os
import shutil
from dataclasses import dataclass
from typing import Optional, List
import subprocess


@dataclass
class PPKConfig:
    """Configuration for PPK processing"""
    datadir: str = '../../examples/data/PPC-Dataset/tokyo/run1'
    navfile: str = 'base.nav'
    rovfile: str = 'rover.obs'
    basefile: str = 'base.obs'
    imufile: str = 'imu.csv'
    maxepoch: int = 0
    trace_level: int = 0
    basepos: List[float] = None
    
    def __post_init__(self):
        if self.basepos is None:
            self.basepos = []


class PPKProcessor:
    """Object-oriented PPK processor using ISAM2"""
    
    def __init__(self, config: PPKConfig):
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.solutions = None
        
    def _prepare_config(self):
        """Prepare RTKLib configuration file"""
        # Copy config template
        config_template = os.path.join(self.script_dir, 'config_ppc_isam.py')
        config_file = os.path.join(self.script_dir, '__ppk_config.py')
        shutil.copyfile(config_template, config_file)
        
        # Add IMU file path and other parameters
        with open(config_file, 'a') as f:
            f.write(f"\n# IMU data file\n")
            f.write(f"imu_file = '{os.path.join(self.config.datadir, self.config.imufile)}'\n")
            if self.config.maxepoch > 0:
                f.write(f"maxepoch = {self.config.maxepoch}\n")
                
    def process(self):
        """Run PPK processing"""
        self._prepare_config()
        
        # Import required modules after changing directory
        sys.path.insert(0, self.script_dir)
        
        # Import with absolute name to avoid mangling
        ppk_config = __import__('__ppk_config')
        cfg = ppk_config
        
        import rinex as rn
        import rtkcmn as gn
        from rtkpos import rtkinit
        from postpos_isam import procpos
        from postpos import savesol
        
        # Setup output files
        solfile = os.path.join(self.config.datadir, 
                              self.config.rovfile[:-4] + '_isam_oo.pos')
        statfile = os.path.join(self.config.datadir, 
                               self.config.rovfile[:-4] + '_isam_oo.pos.stat')
        
        # Setup trace output
        fp_stat = open(statfile, 'w')
        if self.config.trace_level > 0:
            trcfile = os.path.join(self.config.datadir, 
                                  self.config.rovfile[:-4] + '_isam_oo.trace')
            sys.stderr = open(trcfile, "w")
            
        print(f"PPK ISAM2 Processing (OO)")
        print(f"========================")
        print(f"Data directory: {self.config.datadir}")
        print(f"Rover: {self.config.rovfile}")
        print(f"Base: {self.config.basefile}")
        print(f"IMU: {self.config.imufile}")
        
        # Change to data directory
        original_dir = os.getcwd()
        os.chdir(self.config.datadir)
        
        try:
            # Initialize navigation
            gn.tracelevel(self.config.trace_level)
            nav = rtkinit(cfg)
            nav.maxepoch = self.config.maxepoch
            nav.imu_file = self.config.imufile
            nav.fp_stat = fp_stat
            
            # Load rover observations
            print('Loading rover observations...')
            rov = rn.rnx_decode(cfg)
            # If maxepoch is 0, load all observations
            max_obs = None if (nav.filtertype == 'backward' or self.config.maxepoch == 0) else self.config.maxepoch
            rov.decode_obsfile(nav, self.config.rovfile, max_obs)
            
            # Load base observations
            print('Loading base observations...')
            base = rn.rnx_decode(cfg)
            base.decode_obsfile(nav, self.config.basefile, None)
            
            if self.config.basepos:
                nav.rb = self.config.basepos
            elif nav.rb[0] == 0:
                nav.rb = base.pos
                
            # Load navigation data
            print('Loading navigation data...')
            rov.decode_nav(self.config.navfile, nav)
            
            # Process with ISAM2
            print('Processing with ISAM2...')
            self.solutions = procpos(nav, rov, base, fp_stat)
            
            # Save solutions
            savesol(self.solutions, self.config.rovfile[:-4] + '_isam_oo.pos')
            fp_stat.close()
            
            print(f"\nProcessing complete!")
            print(f"Solutions: {len(self.solutions)}")
            print(f"Output: {solfile}")
            
            # Statistics
            if self.solutions:
                self._print_statistics()
                
        finally:
            os.chdir(original_dir)
            
        return self.solutions
        
    def _print_statistics(self):
        """Print solution statistics"""
        import rtkcmn as gn
        
        fixed = sum(1 for s in self.solutions if s.stat == gn.SOLQ_FIX)
        float_ = sum(1 for s in self.solutions if s.stat == gn.SOLQ_FLOAT)
        single = sum(1 for s in self.solutions if s.stat == gn.SOLQ_SINGLE)
        
        total = len(self.solutions)
        print(f"\nSolution Statistics:")
        print(f"  Fixed:  {fixed:4d} ({100*fixed/total:5.1f}%)")
        print(f"  Float:  {float_:4d} ({100*float_/total:5.1f}%)")
        print(f"  Single: {single:4d} ({100*single/total:5.1f}%)")
        
    def plot_results(self, output_dir: Optional[str] = None):
        """Plot processing results"""
        if output_dir is None:
            output_dir = self.config.datadir
            
        solfile = os.path.join(self.config.datadir,
                              self.config.rovfile[:-4] + '_isam_oo.pos')
        
        # Use existing plot script
        plot_script = os.path.join(self.script_dir, 'plot_pos.py')
        if os.path.exists(plot_script):
            try:
                subprocess.run([sys.executable, plot_script, solfile, 
                              '--output', output_dir], check=True)
                print(f"Plots saved to: {output_dir}")
            except Exception as e:
                print(f"Failed to plot: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Object-oriented PPK ISAM processor')
    parser.add_argument('--datadir', default='../../examples/data/PPC-Dataset/tokyo/run1',
                       help='Data directory')
    parser.add_argument('--maxepoch', type=int, default=0,
                       help='Maximum epochs to process (0 for all)')
    parser.add_argument('--trace', type=int, default=0,
                       help='Trace level (0-5)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results after processing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PPKConfig(
        datadir=args.datadir,
        maxepoch=args.maxepoch,
        trace_level=args.trace
    )
    
    # Process
    processor = PPKProcessor(config)
    solutions = processor.process()
    
    # Plot if requested
    if args.plot:
        processor.plot_results()
        
    return 0


if __name__ == '__main__':
    sys.exit(main())