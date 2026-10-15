#!/usr/bin/env python3
"""Unit tests for SCUA analyzer functions."""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-modules'))

from scua import (
    analyze_category,
    generate_plots,
)
import pandas as pd


class TestAnalyzeCategory(unittest.TestCase):
    """Tests for analyze_category function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            'Software': ['VASP', 'VASP', 'Python', 'Unknown'],
            'Cores': [512, 256, 1, 128],
            'Usage': [100, 50, 10, 75],
            'Energy': [50, 25, 5, 30],
            'User': ['user1', 'user2', 'user3', 'user1'],
            'ProjectID': ['proj1', 'proj1', 'proj2', 'proj1'],
            'Count': [1, 1, 1, 1]
        })
        self.allcu = 235
        self.allen = 110
        self.useunits = 'Nodeh'
    
    def test_analyze_category_basic(self):
        """Test basic category analysis."""
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_category(
                df=self.df,
                df_output=self.df,
                category_name='TestSize',
                title='Test job size',
                category_list=['VASP', 'Python'],
                category_col='Software',
                analysis_col='Cores',
                analyse_none=False,
                full_node=False,
                fullmatch=True,
                allcu=self.allcu,
                allen=self.allen,
                useunits=self.useunits,
                report_energy=False,
                savecsv=False,
                savemd=False,
                prefix='test'
            )
        
        output = f.getvalue()
        
        # Check that output contains expected sections
        self.assertIn('## Test job size distribution: weighted by usage', output)
        self.assertIn('## Test job size distribution: by number of jobs', output)
        self.assertIn('VASP', output)
        self.assertIn('Python', output)
    
    def test_analyze_category_with_none(self):
        """Test category analysis with unidentified items."""
        # Add a row with None Software value
        df_with_none = self.df.copy()
        df_with_none.loc[len(df_with_none)] = {
            'Software': None,
            'Cores': 64,
            'Usage': 50,
            'Energy': 20,
            'User': 'user4',
            'ProjectID': 'proj1',
            'Count': 1
        }
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_category(
                df=self.df,
                df_output=df_with_none,
                category_name='TestSize',
                title='Test job size',
                category_list=['VASP', 'Python'],
                category_col='Software',
                analysis_col='Cores',
                analyse_none=True,
                full_node=False,
                fullmatch=True,
                allcu=self.allcu,
                allen=self.allen,
                useunits=self.useunits,
                report_energy=False,
                savecsv=False,
                savemd=False,
                prefix='test'
            )
        
        output = f.getvalue()
        
        # Should include 'Unidentified' in output
        self.assertIn('Unidentified', output)
    
    def test_analyze_category_with_energy(self):
        """Test category analysis with energy reporting."""
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_category(
                df=self.df,
                df_output=self.df,
                category_name='TestSize',
                title='Test job size',
                category_list=['VASP'],
                category_col='Software',
                analysis_col='Cores',
                analyse_none=False,
                full_node=False,
                fullmatch=True,
                allcu=self.allcu,
                allen=self.allen,
                useunits=self.useunits,
                report_energy=True,
                savecsv=False,
                savemd=False,
                prefix='test'
            )
        
        output = f.getvalue()
        
        # Should include energy columns
        self.assertIn('kWh', output)
        self.assertIn('PercentEnergy', output)


class TestGeneratePlots(unittest.TestCase):
    """Tests for generate_plots function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use non-interactive backend for testing
        import matplotlib
        matplotlib.use('Agg')
        
        self.df = pd.DataFrame({
            'Software': ['VASP', 'VASP', 'Python', 'Unknown'],
            'Cores': [512, 256, 1, 128],
            'NGPUS': [0, 0, 0, 0],
            'Usage': [100, 50, 10, 75],
            'Energy': [50, 25, 5, 30],
            'User': ['user1', 'user2', 'user3', 'user1'],
            'ProjectID': ['proj1', 'proj1', 'proj2', 'proj1'],
            'Count': [1, 1, 1, 1]
        })
        self.allcu = 235
        self.allen = 110
        self.codelist = ['VASP', 'Python', 'Unknown']
        self.cpn = 128
        self.gpn = 4
        
        # Create args object
        self.args = type('Args', (), {
            'usegpu': False,
            'prefix': 'test'
        })()
    
    def test_generate_plots_basic(self):
        """Test basic plot generation."""
        # Create temporary directory for plots
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # This should create plot files without errors
                generate_plots(
                    df=self.df,
                    args=self.args,
                    allcu=self.allcu,
                    allen=self.allen,
                    codelist=self.codelist,
                    cpn=self.cpn,
                    gpn=self.gpn
                )
                
                # Check that plot files were created
                self.assertTrue(os.path.exists('test_codes_usage.png'))
                self.assertTrue(os.path.exists('test_top15_boxplot.png'))
            finally:
                os.chdir(old_cwd)
    
    def test_generate_plots_gpu_mode(self):
        """Test plot generation in GPU mode."""
        self.args.usegpu = True
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                generate_plots(
                    df=self.df,
                    args=self.args,
                    allcu=self.allcu,
                    allen=self.allen,
                    codelist=self.codelist,
                    cpn=self.cpn,
                    gpn=self.gpn
                )
                
                self.assertTrue(os.path.exists('test_codes_usage.png'))
                self.assertTrue(os.path.exists('test_top15_boxplot.png'))
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__':
    unittest.main()
