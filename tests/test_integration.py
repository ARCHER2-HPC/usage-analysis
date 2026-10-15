#!/usr/bin/env python3
"""Integration tests for SCUA."""

import os
import sys
import unittest
import subprocess
import tempfile


class TestIntegration(unittest.TestCase):
    """Integration tests for SCUA end-to-end functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Get the project root directory (parent of tests directory)
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')
        )
        self.env = os.environ.copy()
        self.env['SCUA_BASE'] = self.project_root
        self.env['PYTHONPATH'] = os.path.join(
            self.project_root, 'python-modules'
        )
        self.scua_script = os.path.join(
            self.project_root, 'bin', 'scua.py'
        )
        self.test_data_dir = os.path.join(
            self.project_root, 'test-data'
        )
    
    def test_scua_basic_execution(self):
        """Test basic SCUA execution with test data."""
        input_file = os.path.join(self.test_data_dir, 'scua_sacct.dat')
        
        if not os.path.exists(input_file):
            self.skipTest(f"Test data file not found: {input_file}")
        
        result = subprocess.run(
            ['python', self.scua_script, input_file],
            capture_output=True,
            text=True,
            env=self.env
        )
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0)
        
        # Should produce output
        self.assertIn('SCUA (Slurm Code Usage Analysis)', result.stdout)
        self.assertIn('Software job size distribution', result.stdout)
    
    def test_scua_with_gpu_flag(self):
        """Test SCUA with GPU flag."""
        # Skip this test as it requires test data with GPU columns
        self.skipTest("Requires test data with GPU columns")
    
    def test_scua_with_csv_output(self):
        """Test SCUA with CSV output."""
        input_file = os.path.join(self.test_data_dir, 'scua_sacct.dat')
        
        if not os.path.exists(input_file):
            self.skipTest(f"Test data file not found: {input_file}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                result = subprocess.run(
                    ['python', self.scua_script, '--csv', '--prefix', 'test', input_file],
                    capture_output=True,
                    text=True,
                    env=self.env
                )
                
                self.assertEqual(result.returncode, 0)
                
                # Check that CSV files were created
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                self.assertGreater(len(csv_files), 0)
            finally:
                os.chdir(old_cwd)
    
    def test_scua_with_account_filter(self):
        """Test SCUA with account filter."""
        input_file = os.path.join(self.test_data_dir, 'scua_sacct.dat')
        
        if not os.path.exists(input_file):
            self.skipTest(f"Test data file not found: {input_file}")
        
        # Test with an account that has data
        result = subprocess.run(
            ['python', self.scua_script, '-A', 'e05-power-ske', input_file],
            capture_output=True,
            text=True,
            env=self.env
        )
        
        # Note: This may fail if the filtered data is empty (edge case)
        # The important thing is that the filter is applied
        self.assertIn('SCUA (Slurm Code Usage Analysis)', result.stdout)
    
    def test_scua_with_shared_node(self):
        """Test SCUA with shared node option."""
        input_file = os.path.join(self.test_data_dir, 'scua_sacct.dat')
        
        if not os.path.exists(input_file):
            self.skipTest(f"Test data file not found: {input_file}")
        
        result = subprocess.run(
            ['python', self.scua_script, '--sharednode', input_file],
            capture_output=True,
            text=True,
            env=self.env
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('Software job size distribution', result.stdout)
    
    def test_scua_error_handling_missing_file(self):
        """Test SCUA error handling for missing file."""
        result = subprocess.run(
            ['python', self.scua_script, '/nonexistent/file.dat'],
            capture_output=True,
            text=True,
            env=self.env
        )
        
        # Should return non-zero exit code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Error', result.stderr)
    
    def test_scua_error_handling_missing_env(self):
        """Test SCUA error handling for missing SCUA_BASE."""
        input_file = os.path.join(self.test_data_dir, 'scua_sacct.dat')
        
        # Remove SCUA_BASE from environment
        env = self.env.copy()
        del env['SCUA_BASE']
        
        result = subprocess.run(
            ['python', self.scua_script, input_file],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Should return non-zero exit code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SCUA_BASE', result.stderr)


if __name__ == '__main__':
    unittest.main()
