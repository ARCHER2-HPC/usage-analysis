#!/usr/bin/env python3
"""Unit tests for SCUA data processing functions."""

import os
import sys
import tempfile
import unittest
from io import StringIO

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-modules'))

from scua import (
    get_overall_stats,
    get_distribution,
    get_weighted_distribution,
    compute_distribution,
    reindex_df,
    process_slurm_data,
    identify_codes,
    apply_user_split,
)
from code_def import CodeDef


class TestGetOverallStats(unittest.TestCase):
    """Tests for get_overall_stats function."""
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        df = pd.DataFrame({
            'Usage': [100, 200, 300],
            'Energy': [50, 100, 150],
            'Count': [1, 1, 1],
            'User': ['user1', 'user2', 'user1'],
            'ProjectID': ['proj1', 'proj1', 'proj2']
        })
        total_cu = 1000
        total_en = 500
        
        result = get_overall_stats(df, total_cu, total_en)
        
        self.assertEqual(result[0], 600)  # totcu
        self.assertAlmostEqual(result[1], 60.0)  # percentcu
        self.assertEqual(result[2], 300)  # toten
        self.assertAlmostEqual(result[3], 60.0)  # percenten
        self.assertEqual(result[4], 3)  # totjobs
        self.assertEqual(result[5], 2)  # totusers
        self.assertEqual(result[6], 2)  # totprojects
    
    def test_zero_total_cu(self):
        """Test handling of zero total_cu."""
        df = pd.DataFrame({
            'Usage': [100, 200],
            'Energy': [50, 100],
            'Count': [1, 1],
            'User': ['user1', 'user2'],
            'ProjectID': ['proj1', 'proj1']
        })
        total_cu = 0
        total_en = 500
        
        result = get_overall_stats(df, total_cu, total_en)
        
        self.assertEqual(result[1], 0.0)  # percentcu should be 0


class TestGetDistribution(unittest.TestCase):
    """Tests for get_distribution function."""
    
    def test_basic_distribution(self):
        """Test basic distribution calculation."""
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        min_val, max_val, q1, med, q3 = get_distribution(df, 'values')
        
        self.assertEqual(min_val, 1)
        self.assertEqual(max_val, 10)
        self.assertAlmostEqual(med, 5.5)


class TestGetWeightedDistribution(unittest.TestCase):
    """Tests for get_weighted_distribution function."""
    
    def test_basic_weighted_distribution(self):
        """Test basic weighted distribution calculation."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5],
            'weights': [10, 20, 30, 20, 10]
        })
        
        q1, med, q3 = get_weighted_distribution(df, 'values', 'weights')
        
        # Median should be around 3 (the weighted middle)
        self.assertAlmostEqual(med, 3.0, places=1)


class TestComputeDistribution(unittest.TestCase):
    """Tests for compute_distribution function."""
    
    def test_basic_compute_distribution(self):
        """Test basic distribution computation."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5],
            'Usage': [10, 20, 30, 20, 10],
            'Energy': [5, 10, 15, 10, 5],
            'User': ['a', 'b', 'c', 'd', 'e'],
            'ProjectID': ['p1', 'p1', 'p2', 'p2', 'p3'],
            'Count': [1, 1, 1, 1, 1]
        })
        total_cu = 100
        total_en = 50
        
        dist, wdist = compute_distribution(
            df, 'test', total_cu, total_en, 'values', 'Usage'
        )
        
        self.assertEqual(dist[0], 'test')
        self.assertEqual(len(dist), 13)
        self.assertEqual(len(wdist), 13)


class TestReindexDf(unittest.TestCase):
    """Tests for reindex_df function."""
    
    def test_basic_reindex(self):
        """Test basic dataframe reindexing."""
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'count': [2, 1, 3]
        })
        
        result = reindex_df(df, 'count')
        
        self.assertEqual(len(result), 6)  # 2 + 1 + 3
        self.assertEqual(result['value'].tolist(), [1, 1, 2, 3, 3, 3])


class TestProcessSlurmData(unittest.TestCase):
    """Tests for process_slurm_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args = type('Args', (), {
            'dropnan': False,
            'usegpu': False,
            'sharednode': False,
            'useunits': 'Nodeh'
        })()
        self.cpn = 128
        self.gpn = 4
        self.max_power = 850
    
    def test_basic_processing(self):
        """Test basic data processing."""
        df = pd.DataFrame({
            'JobID': ['1', '2'],
            'Energy': ['1000000', '2000000'],
            'Runtime': ['3600', '7200'],
            'Nodes': ['1', '2'],
            'NTasks': ['64', '128'],
            'Account': ['proj1-group1', 'proj2-group2']
        })
        
        result, useunits = process_slurm_data(
            df, self.args, self.cpn, self.gpn, self.max_power
        )
        
        self.assertIn('Usage', result.columns)
        self.assertIn('ProjectID', result.columns)
        self.assertIn('GroupID', result.columns)
        self.assertEqual(useunits, 'Nodeh')
    
    def test_account_without_dash(self):
        """Test account splitting when no dash present."""
        df = pd.DataFrame({
            'JobID': ['1'],
            'Energy': ['1000000'],
            'Runtime': ['3600'],
            'Nodes': ['1'],
            'NTasks': ['64'],
            'Account': ['proj1']
        })
        
        result, _ = process_slurm_data(
            df, self.args, self.cpn, self.gpn, self.max_power
        )
        
        self.assertEqual(result['ProjectID'].iloc[0], 'proj1')
    
    def test_smt_handling(self):
        """Test SMT (Simultaneous Multithreading) handling."""
        df = pd.DataFrame({
            'JobID': ['1', '2'],
            'Energy': ['1000000', '1000000'],
            'Runtime': ['3600', '3600'],
            'Nodes': ['1', '1'],
            'NTasks': ['64', '256'],  # Second job uses SMT
            'Account': ['proj1', 'proj2']
        })
        
        result, _ = process_slurm_data(
            df, self.args, self.cpn, self.gpn, self.max_power
        )
        
        # First job should have 64 cores, second should be capped at 128
        self.assertEqual(result['Cores'].iloc[0], 64)
        self.assertEqual(result['Cores'].iloc[1], 128)


class TestIdentifyCodes(unittest.TestCase):
    """Tests for identify_codes function."""
    
    def test_basic_identification(self):
        """Test basic code identification."""
        df = pd.DataFrame({
            'ExeName': ['vasp_std', 'python', 'unknown_exe'],
            'Usage': [100, 50, 25]
        })
        
        # Create mock code definitions
        vasp_code = CodeDef()
        vasp_code._CodeDef__name = 'VASP'
        vasp_code._CodeDef__regexp = 'vasp'
        vasp_code._CodeDef__pri_lang = 'Fortran'
        vasp_code._CodeDef__type = 'DFT'
        
        python_code = CodeDef()
        python_code._CodeDef__name = 'Python'
        python_code._CodeDef__regexp = 'python'
        python_code._CodeDef__pri_lang = 'Python'
        python_code._CodeDef__type = 'Scripting'
        
        codes = [vasp_code, python_code]
        
        result, motif_set, lang_set = identify_codes(df, codes)
        
        self.assertEqual(result['Software'].iloc[0], 'VASP')
        self.assertEqual(result['Software'].iloc[1], 'Python')
        self.assertEqual(result['Software'].iloc[2], 'Unknown')
    
    def test_nan_exe_handling(self):
        """Test handling of NaN executable names."""
        df = pd.DataFrame({
            'ExeName': ['vasp_std', np.nan, 'python'],
            'Usage': [100, 50, 25]
        })
        
        codes = []
        
        result, _, _ = identify_codes(df, codes)
        
        # Should have removed the NaN row
        self.assertEqual(len(result), 2)


class TestApplyUserSplit(unittest.TestCase):
    """Tests for apply_user_split function."""
    
    def test_user_split_anon(self):
        """Test user splitting with anonymization."""
        df = pd.DataFrame({
            'Software': ['Python', 'Python', 'a.out'],
            'User': ['alice', 'bob', 'alice'],
            'AnonUser': ['user0', 'user1', 'user0']
        })
        
        result = apply_user_split(df, anon=True, software_list=['Python', 'a.out'])
        
        self.assertEqual(result['Software'].iloc[0], 'Python_user0')
        self.assertEqual(result['Software'].iloc[1], 'Python_user1')
        self.assertEqual(result['Software'].iloc[2], 'a.out_user0')
    
    def test_user_split_no_anon(self):
        """Test user splitting without anonymization."""
        df = pd.DataFrame({
            'Software': ['Python', 'Python'],
            'User': ['alice', 'bob'],
            'AnonUser': ['user0', 'user1']
        })
        
        result = apply_user_split(df, anon=False, software_list=['Python'])
        
        self.assertEqual(result['Software'].iloc[0], 'Python_alice')
        self.assertEqual(result['Software'].iloc[1], 'Python_bob')


if __name__ == '__main__':
    unittest.main()
