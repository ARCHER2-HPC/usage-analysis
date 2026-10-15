#!/usr/bin/env python3
# scua.py (Slurm Code Usage Analysis)
#
# usage: scua.py [options] filename
#
#  Compute software usage data from Slurm output.
# 
# positional arguments:
#  filename         Data file containing listing of Slurm jobs
#
# optional arguments:
#   -h, --help       show list of arguments
#
# The data file is the output for Slurm sacct, produced in using a 
# command such as:
#   sacct --format JobIDRaw,JobName%30,User,Account,NNodes,NTasks,ElapsedRaw,State,ConsumedEnergyRaw,MaxRSS,AveRSS -P --delimiter , \
#        | egrep '[0-9]\.[0-9]' | egrep -v "RUNNING|PENDING|REQUEUED"
#
# The egrep extracts all subjobs and then excludes specific job states
# To get accurate usernames, this output may need to be preprocessed to 
# combine with other output from sacct. See the "scua" script for an example
# of how to do this.
#
#----------------------------------------------------------------------
# Copyright 2021, 2022 EPCC, The University of Edinburgh
#
# This file is part of usage-analysis.
#
# usage-analysis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# usage-analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with usage-analysis.  If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
"""
Slurm Code Usage Analysis (SCUA) tool.

Analyzes Slurm accounting data to produce software usage statistics,
including job size distributions, energy usage, and code identification.
"""

# Standard library imports
import argparse
import csv
import fnmatch
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# TOML support (Python 3.11+ has tomllib built-in)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import toml as tomllib
    except ImportError:
        print("Error: tomllib (Python 3.11+) or toml package is required.", file=sys.stderr)
        sys.exit(1)

# Third-party imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Local imports
from code_def import CodeDef

# =======================================================
# Constants
# =======================================================
DEFAULT_CORES_PER_NODE = 128
DEFAULT_GPUS_PER_NODE = 4
DEFAULT_MAX_POWER = 850
DEFAULT_UNKNOWN_THRESHOLD = 0.01
SECONDS_PER_HOUR = 3600
JOULES_TO_KWH = 3600000.0
MAX_ENERGY_VALUE = 1e16

pd.options.mode.chained_assignment = None


# =======================================================
# Helper Functions
# =======================================================

def get_overall_stats(
    df: pd.DataFrame,
    total_cu: float,
    total_en: float
) -> Tuple[float, float, float, float, int, int, int]:
    """Compute overall statistics for a dataframe.
    
    Args:
        df: DataFrame containing job data
        total_cu: Total compute units for percentage calculation
        total_en: Total energy for percentage calculation
        
    Returns:
        Tuple of (total_cu_used, percent_cu, total_en_used, percent_en,
                  total_jobs, total_users, total_projects)
    """
    totcu = df['Usage'].sum()
    percentcu = 100 * totcu / total_cu if total_cu > 0 else 0.0
    toten = df['Energy'].sum()
    percenten = 100 * toten / total_en if total_en > 0 else 0.0
    totjobs = df['Count'].sum()
    totusers = df['User'].nunique()
    totprojects = df['ProjectID'].nunique()
    return totcu, percentcu, toten, percenten, totjobs, totusers, totprojects


def get_distribution(df: pd.DataFrame, col: str) -> Tuple[float, float, float, float, float]:
    """Get unweighted distribution statistics for a column.
    
    Args:
        df: DataFrame containing the data
        col: Column name to analyze
        
    Returns:
        Tuple of (min, max, q1, median, q3)
    """
    min_val = df[col].min()
    max_val = df[col].max()
    q1_val = df[col].quantile(0.25)
    med_val = df[col].quantile(0.5)
    q3_val = df[col].quantile(0.75)
    return min_val, max_val, q1_val, med_val, q3_val


def get_weighted_distribution(
    df: pd.DataFrame,
    values_col: str,
    weights_col: str
) -> Tuple[float, float, float]:
    """Get weighted distribution statistics.
    
    Args:
        df: DataFrame containing the data
        values_col: Column containing values to analyze
        weights_col: Column containing weights
        
    Returns:
        Tuple of (q1, median, q3) for weighted distribution
    """
    df_sorted = df.sort_values(values_col)
    cumsum = df_sorted[weights_col].cumsum()
    total_weight = df_sorted[weights_col].sum()
    
    cutoff = total_weight * 0.5
    median = float(df_sorted[cumsum >= cutoff][values_col].iloc[0])
    
    cutoff = total_weight * 0.25
    q1 = float(df_sorted[cumsum >= cutoff][values_col].iloc[0])
    
    cutoff = total_weight * 0.75
    q3 = float(df_sorted[cumsum >= cutoff][values_col].iloc[0])
    
    return q1, median, q3


def compute_distribution(
    df: pd.DataFrame,
    label: str,
    total_cu: float,
    total_en: float,
    values_col: str,
    weights_col: str
) -> Tuple[List[Any], List[Any]]:
    """Compute distribution statistics (unweighted and weighted).
    
    Args:
        df: DataFrame containing the data
        label: Label for this distribution
        total_cu: Total compute units for percentage calculation
        total_en: Total energy for percentage calculation
        values_col: Column containing values to analyze
        weights_col: Column containing weights
        
    Returns:
        Tuple of (unweighted_stats, weighted_stats) lists
    """
    (totcu, percentcu, toten, percenten, 
     totjobs, totusers, totprojects) = get_overall_stats(df, total_cu, total_en)
    min_val, max_val, q1_val, med_val, q3_val = get_distribution(df, values_col)
    
    dist = [label, min_val, q1_val, med_val, q3_val, max_val, 
            totjobs, totcu, percentcu, toten, percenten, totusers, totprojects]
    
    wq1_val, wmed_val, wq3_val = get_weighted_distribution(df, values_col, weights_col)
    wdist = [label, min_val, wq1_val, wmed_val, wq3_val, max_val,
             totjobs, totcu, percentcu, toten, percenten, totusers, totprojects]
    
    return dist, wdist


def reindex_df(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """Expand the dataframe to prepare for resampling.
    
    Result is 1 row per count per sample.
    
    Args:
        df: DataFrame to expand
        weight_col: Column to use for repetition count
        
    Returns:
        Expanded DataFrame with repeated rows
    """
    df = df.reindex(df.index.repeat(df[weight_col]))
    df.reset_index(drop=True, inplace=True)
    return df


def load_code_definitions(code_config_dir: str) -> List[CodeDef]:
    """Load code definitions from configuration files.
    
    Args:
        code_config_dir: Directory containing .code definition files
        
    Returns:
        List of CodeDef objects
    """
    codes = []
    n_code = 0
    code_dict = {}
    code_list = []
    
    for file in os.listdir(code_config_dir):
        if fnmatch.fnmatch(file, '*.code'):
            n_code += 1
            code = CodeDef()
            code.readConfig(os.path.join(code_config_dir, file))
            codes.append(code)
            code_dict[code.name] = n_code - 1
            code_list.append(code.name)
    
    return codes, code_dict, code_list


def load_slurm_data(
    filename: str,
    use_gpu: bool
) -> pd.DataFrame:
    """Load Slurm data from CSV file.
    
    Args:
        filename: Path to the input data file
        use_gpu: Whether to include GPU columns
        
    Returns:
        DataFrame containing the loaded data
    """
    col_id = ['JobID', 'ExeName', 'User', 'Account', 'Nodes', 'NTasks', 
              'Runtime', 'State', 'Energy', 'MaxRSS', 'MeanRSS', 'CPUFreq', 'SubJobID']
    col_types = {
        'JobID': 'str',
        'ExeName': 'str',
        'User': 'str',
        'Account': 'str',
        'Nodes': 'str',
        'NTasks': 'str',
        'Runtime': 'str',
        'State': 'str',
        'Energy': 'str',
        'MaxRSS': 'str',
        'MeanRSS': 'str',
        'CPUFreq': 'str',
        'SubJobID': 'str'
    }
    
    if use_gpu:
        col_id.insert(-1, 'NGPUS')
        col_types['NGPUS'] = 'str'
        df = pd.read_csv(filename, names=col_id, dtype=col_types, sep=',', 
                         engine='python', header=0)
        df['NGPUS'] = pd.to_numeric(df['NGPUS'], errors='coerce')
    else:
        df = pd.read_csv(filename, names=col_id, dtype=col_types, sep=',', 
                         engine='python', header=0)
    
    # Count helps with number of jobs
    df['Count'] = 1
    return df


def process_slurm_data(
    df: pd.DataFrame,
    args: argparse.Namespace,
    cpn: int,
    gpn: int,
    max_power: int
) -> pd.DataFrame:
    """Process and clean Slurm data.
    
    Args:
        df: DataFrame containing raw Slurm data
        args: Command line arguments
        cpn: Cores per node
        gpn: GPUs per node
        max_power: Maximum realistic node power
        
    Returns:
        Processed DataFrame
    """
    # Convert columns to numeric type
    df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
    df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
    df['NTasks'] = pd.to_numeric(df['NTasks'], errors='coerce')
    
    # Remove unrealistic values (presumably due to counter errors)
    df.loc[df['Energy'].gt(MAX_ENERGY_VALUE), 'Energy'] = np.nan
    df['NodePower'] = df['Energy'] / (df['Runtime'] * df['Nodes'])
    
    # Convert energy to kWh
    df['Energy'] = df['Energy'] / JOULES_TO_KWH
    
    # Check if we are dropping rows without energy values
    if args.dropnan:
        df.dropna(axis=0, how='any', inplace=True)
    
    # Calculate number of used cores, handling SMT cases
    df['cpn'] = df['NTasks'] / df['Nodes']
    df['Cores'] = df['NTasks']
    df.loc[df['cpn'] > cpn, 'Cores'] = df['Nodes'] * cpn
    
    # Calculate Nodeh/Coreh usage
    useunits = args.useunits
    if useunits == 'Nodeh':
        df['Usage'] = df['Nodes'] * df['Runtime'] / SECONDS_PER_HOUR
    elif useunits == 'Coreh':
        df['Usage'] = df['Cores'] * df['Runtime'] / SECONDS_PER_HOUR
    
    if args.usegpu:
        useunits = 'GPUh'
        df['Usage'] = df['NGPUS'] * df['Runtime'] / SECONDS_PER_HOUR
    
    if not args.sharednode:
        # Adjust for partial node usage
        if args.usegpu:
            df.loc[df['NGPUS'] < gpn, 'Usage'] = (
                df['NGPUS'] * df['Runtime'] / (gpn * SECONDS_PER_HOUR)
            )
        else:
            df.loc[df['Cores'] < cpn, 'Usage'] = (
                df['Cores'] * df['Runtime'] / (cpn * SECONDS_PER_HOUR)
            )
        df.loc[df['Cores'] < cpn, 'Energy'] = df['Cores'] * df['Energy'] / cpn
        df.loc[df['Cores'] < cpn, 'NodePower'] = df['Cores'] * df['NodePower'] / cpn
    
    # Remove unrealistic values
    df.loc[df['Usage'] < 0, 'Usage'] = 0.0
    
    # Remove very unrealistic values (counter errors or short runtimes)
    df.replace(np.inf, np.nan, inplace=True)
    df.loc[df['NodePower'].gt(max_power), 'NodePower'] = np.nan
    
    # Split Account column into ProjectID and GroupID
    df['JobID'] = df['JobID'].astype(str)
    if df['Account'].str.contains('-').sum() > 0:
        df[['ProjectID', 'GroupID']] = df['Account'].str.split(pat='-', n=1, expand=True)
    else:
        df['ProjectID'] = df['Account']
        df['GroupID'] = None
    
    return df, useunits


def identify_codes(
    df: pd.DataFrame,
    codes: List[CodeDef]
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """Identify software codes using regex patterns.
    
    Args:
        df: DataFrame containing job data
        codes: List of CodeDef objects with regex patterns
        
    Returns:
        Tuple of (df with code columns, motif_set, lang_set)
    """
    df["Software"] = 'Unknown'
    df["Motif"] = 'Unknown'
    df["Language"] = 'Unknown'
    
    # Remove rows with NaN ExeName
    df.dropna(subset=['ExeName'], inplace=True)
    
    for code in codes:
        code_re = re.compile(code.regexp)
        df.loc[df.ExeName.str.contains(code_re), 
               ["Software", "Motif", "Language"]] = [code.name, code.type, code.pri_lang]
    
    # Get unique motifs and languages
    motif_set = set()
    df['Motif'].str.split().apply(motif_set.update)
    lang_set = set()
    df['Language'].str.split().apply(lang_set.update)
    
    return df, motif_set, lang_set


def apply_user_split(
    df: pd.DataFrame,
    anon: bool,
    software_list: List[str]
) -> pd.DataFrame:
    """Apply user-based splitting for specified software.
    
    Args:
        df: DataFrame containing job data
        anon: Whether to anonymize usernames
        software_list: List of software names to split
        
    Returns:
        DataFrame with split software names
    """
    user_col = 'AnonUser' if anon else 'User'
    
    for software in software_list:
        mask = df['Software'] == software
        df.loc[mask, 'Software'] = software + "_" + df.loc[mask, user_col]
    
    return df


def analyze_category(
    df: pd.DataFrame,
    df_output: pd.DataFrame,
    category_name: str,
    title: str,
    category_list: List[Any],
    category_col: str,
    analysis_col: str,
    analyse_none: bool,
    full_node: bool,
    fullmatch: bool,
    allcu: float,
    allen: float,
    useunits: str,
    report_energy: bool,
    savecsv: bool,
    savemd: bool,
    prefix: str
) -> None:
    """Analyze and output distribution for a category.
    
    Args:
        df: Full DataFrame
        df_output: Filtered DataFrame for analysis
        category_name: Name of the category
        title: Title for output
        category_list: List of categories to analyze
        category_col: Column name for categories
        analysis_col: Column to analyze
        analyse_none: Whether to analyze None/unidentified
        full_node: Whether to use full-node matching
        fullmatch: Whether to use exact matching
        allcu: Total compute units
        allen: Total energy
        useunits: Units string
        report_energy: Whether to report energy
        savecsv: Whether to save CSV output
        savemd: Whether to save markdown output
        prefix: Output file prefix
    """
    job_stats = []
    usage_stats = []
    
    for category in category_list:
        if fullmatch:
            mask = df_output[category_col].str.fullmatch(re.escape(str(category)), na=False)
        else:
            mask = df_output[category_col].str.contains(re.escape(str(category)), na=False)
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = compute_distribution(
                df_cat, category, allcu, allen, analysis_col, 'Usage'
            )
            job_stats.append(dist)
            usage_stats.append(wdist)
    
    # Get data for unidentified if needed
    if analyse_none:
        mask = df_output[category_col].values == None
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = compute_distribution(
                df_cat, 'Unidentified', allcu, allen, analysis_col, 'Usage'
            )
            job_stats.append(dist)
            usage_stats.append(wdist)
    
    # Get overall data for all jobs
    dist, wdist = compute_distribution(
        df_output, 'Overall', allcu, allen, analysis_col, 'Usage'
    )
    job_stats.append(dist)
    usage_stats.append(wdist)
    
    # Save the output
    print(f'\n## {title} distribution: weighted by usage\n')
    df_usage = pd.DataFrame(
        usage_stats, 
        columns=[category_col, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', 
                 useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects']
    )
    if not report_energy:
        df_usage.drop('kWh', axis=1, inplace=True)
        df_usage.drop('PercentEnergy', axis=1, inplace=True)
    df_usage.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_usage.to_markdown(index=False, floatfmt=".1f"))
    print()
    
    print(f'\n## {title} distribution: by number of jobs\n')
    df_jobs = pd.DataFrame(
        job_stats, 
        columns=[category_col, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', 
                 useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects']
    )
    if not report_energy:
        df_jobs.drop('kWh', axis=1, inplace=True)
        df_jobs.drop('PercentEnergy', axis=1, inplace=True)
    df_jobs.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_jobs.to_markdown(index=False, floatfmt=".1f"))
    print()
    
    # Save as CSV if required
    if savecsv:
        df_jobs.to_csv(f'{prefix}_{category_name}.csv', index=False, float_format="%.1f")
        df_usage.to_csv(f'{prefix}_{category_name}_weighted.csv', index=False, float_format="%.1f")
    if savemd:
        df_jobs.to_markdown(f'{prefix}_{category_name}.md', index=False, floatfmt=".1f")
        df_usage.to_markdown(f'{prefix}_{category_name}_weighted.md', index=False, floatfmt=".1f")


def generate_plots(
    df: pd.DataFrame,
    args: argparse.Namespace,
    allcu: float,
    allen: float,
    codelist: List[str],
    cpn: int,
    gpn: int,
    useunits: str = 'Nodeh'
) -> None:
    """Generate usage plots.
    
    Args:
        df: DataFrame containing job data
        args: Command line arguments
        allcu: Total compute units
        allen: Total energy
        codelist: List of software codes
        cpn: Cores per node
        gpn: GPUs per node
        useunits: Units string for output
    """
    df_output = df
    
    job_stats = []
    usage_stats = []
    categories = codelist
    catcol = 'Software'
    
    if args.usegpu:
        ancol = 'NGPUS'
    else:
        ancol = 'Cores'
    
    for category in categories:
        mask = df_output[catcol].str.fullmatch(re.escape(str(category)), na=False)
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = compute_distribution(
                df_cat, category, allcu, allen, ancol, 'Usage'
            )
            job_stats.append(dist)
            usage_stats.append(wdist)
    
    mask = df_output[catcol].values == None
    df_cat = df_output[mask]
    if not df_cat.empty:
        dist, wdist = compute_distribution(
            df_cat, 'Unidentified', allcu, allen, ancol, 'Usage'
        )
        job_stats.append(dist)
        usage_stats.append(wdist)
    
    # Get overall data for all jobs
    dist, wdist = compute_distribution(
        df_output, 'Overall', allcu, allen, ancol, 'Usage'
    )
    job_stats.append(dist)
    usage_stats.append(wdist)
    
    # Save the output
    df_usage = pd.DataFrame(
        usage_stats, 
        columns=[catcol, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', 
                 useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects']
    )
    df_usage.sort_values('PercentUse', inplace=True, ascending=False)
    
    # Software usage plots
    # Bar plot of software usage
    df_plot = df_usage[~df_usage['Software'].isin(['Overall', 'Unidentified'])]
    plt.figure(figsize=[8, 6])
    sns.barplot(y='Software', x='PercentUse', color='lightseagreen', data=df_plot)
    sns.despine()
    plt.xlabel('PercentUse')
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_codes_usage.png', dpi=300)
    plt.clf()
    
    # Boxplots for top 15 software by CU use
    topcodes = df_usage['Software'].head(16).to_list()[1:]
    df_topcodes = df[df['Software'].isin(topcodes)]
    plt.figure(figsize=[8, 6])
    if args.usegpu:
        xcat = 'NGPUS'
    else:
        xcat = 'Cores'
    sns.boxplot(
        y="Software",
        x=xcat,
        orient='h',
        color='lightseagreen',
        order=topcodes,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5"
        },
        data=reindex_df(df_topcodes, weight_col='Usage')
    )
    plt.xlabel(xcat)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_top15_boxplot.png', dpi=300)
    plt.clf()


def load_analysis_categories(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load analysis category definitions from TOML files.
    
    Args:
        base_dir: Base directory containing analysis-categories subdirectory
        
    Returns:
        Dictionary mapping category IDs to their definitions
    """
    categories_dir = os.path.join(base_dir, 'analysis-categories')
    all_categories = {}
    
    for filename in os.listdir(categories_dir):
        if filename.endswith('.toml'):
            filepath = os.path.join(categories_dir, filename)
            with open(filepath, 'rb') as f:
                data = tomllib.load(f)
            
            for analysis in data.get('analyses', []):
                all_categories[analysis['id']] = analysis
    
    return all_categories


def filter_categories_by_flags(
    categories: Dict[str, Dict[str, Any]],
    args: argparse.Namespace
) -> List[str]:
    """Filter analysis categories based on command-line flags.
    
    Args:
        categories: All available category definitions
        args: Parsed command-line arguments
        
    Returns:
        List of category IDs to include
    """
    outputs = []
    
    for cat_id, config in categories.items():
        required_flag = config.get('required_flag')
        
        if required_flag is None:
            outputs.append(cat_id)
        elif isinstance(required_flag, str):
            if getattr(args, required_flag, False):
                outputs.append(cat_id)
        elif isinstance(required_flag, list):
            if all(getattr(args, flag, False) for flag in required_flag):
                outputs.append(cat_id)
    
    return outputs


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Compute software usage data from Slurm output.')
    parser.add_argument('filename', type=str, nargs=1, 
                        help='Data file containing listing of Slurm jobs')
    parser.add_argument('-A', dest='account', type=str, action='store', nargs='?', 
                        default='', help='The slurm account specified for the report. Default: none')
    parser.add_argument('--anon', dest='anon', action='store_false', default=True, 
                        help='If splitting unknown use by user, anonymise usernames. Default: false')
    parser.add_argument('--cpn', dest='CPN', action='store', default=128, 
                        help='Set number of cores per node. Default: 128')
    parser.add_argument('--gpn', dest='GPN', action='store', default=4, 
                        help='Set number of GPUs per node. Only has an effect if --gpu is specified. Default: 4')
    parser.add_argument('--csv', dest='savecsv', action='store_true', default=False, 
                        help='Produce data files in CSV. Default: false')
    parser.add_argument('--cpufreq', dest='analysecpufreq', action='store_true', default=False, 
                        help='Produce CPU frequency usage distribution. Default: false')
    parser.add_argument('--dropnan', dest='dropnan', action='store_true', default=False, 
                        help='Drop all rows that contain NaN. Useful for strict comparisons between usage and energy use. Default: false')
    parser.add_argument('--energy', dest='reportenergy', action='store_true', default=False, 
                        help='Report enery use in output. Note this is known to be inaccurate when just job steps are considered - consider using the scea tool instead to report energy use from jobs. Default: false')
    parser.add_argument('--gpu', dest='usegpu', action='store_true', default=False, 
                        help='Read number of GPU used as part of input data and use nGPU and GPUh as the reporting unit')
    parser.add_argument('--lang', dest='analyselang', action='store_true', default=False, 
                        help='Produce programming language usage distribution. Default: false')
    parser.add_argument('--md', dest='savemd', action='store_true', default=False, 
                        help='Produce data files in MD. Default: false')
    parser.add_argument('--motif', dest='analysemotif', action='store_true', default=False, 
                        help='Produce algorithmic motif usage distribution. Default: false')
    parser.add_argument('--mpow', dest='MAX_POWER', action='store', default=850, 
                        help='Set maximum realistic power draw for a compute node (used to remove jobs with counter errors). Default: 850')
    parser.add_argument('--plots', dest='makeplots', action='store_true', default=False, 
                        help='Produce data plots. Default: false')
    parser.add_argument('--power', dest='analysepower', action='store_true', default=False, 
                        help='Produce node power usage distribution. Default: false')
    parser.add_argument('--prefix', dest='prefix', type=str, action='store', default='scua', 
                        help='Set the prefix to be used for output files. Default: scua')
    parser.add_argument('--projects', dest='projlist', type=str, action='store', default=None, 
                        help='The file containing a list of project IDs and associated research areas. Default: none')
    parser.add_argument('--projid', dest='analyseprojid', action='store_true', default=False, 
                        help='Produce project ID usage distribution. Default: false')
    parser.add_argument('--sharednode', dest='sharednode', action='store_true', default=False, 
                        help='Can nodes be shared by jobs? Default: false')
    parser.add_argument('--thresh', dest='unknown_thresh', type=float, action='store', 
                        default=0.01, 
                        help='Usage fraction obove which unknown exe names are printed. Default is 0.01 - 1 percent.')
    parser.add_argument('-u', dest='user', type=str, action='store', nargs='?', 
                        default='', help='The user specified for the report. Default: none')
    parser.add_argument('--units', dest='useunits', type=str, action='store', default='Nodeh', 
                        help='Units for calculating use: Nodeh or Coreh supported. Default is Nodeh')
    parser.add_argument('--usersplit', dest='usersplit', action='store_true', default=False, 
                        help='Split unknown use by user. Default: false')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for SCUA.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_arguments()
    
    # Validate environment
    scua_base = os.getenv('SCUA_BASE')
    if not scua_base:
        print("Error: SCUA_BASE environment variable is not set.", file=sys.stderr)
        return 1
    
    # Set defaults from arguments
    cpn = int(args.CPN)
    gpn = int(args.GPN)
    max_power = int(args.MAX_POWER)
    unknown_thresh = args.unknown_thresh
    
    # Read code definitions
    code_config_dir = os.path.join(scua_base, 'app-data', 'code-definitions')
    try:
        codes, code_dict, codelist = load_code_definitions(code_config_dir)
    except FileNotFoundError:
        print(f"Error: Code definitions directory not found: {code_config_dir}", 
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading code definitions: {e}", file=sys.stderr)
        return 1
    
    # Read project research areas if required
    areadict = {}
    areaset = []
    if args.projlist is not None:
        try:
            with open(args.projlist, 'r') as projfile:
                next(projfile)  # Skip header
                reader = csv.reader(projfile, skipinitialspace=True)
                areadict = dict(reader)
                areaset = list(set(areadict.values()))  # Unique area names
        except FileNotFoundError:
            print(f"Error: Project list file not found: {args.projlist}", 
                  file=sys.stderr)
            return 1
    
    # Load dataset
    try:
        df = load_slurm_data(args.filename[0], args.usegpu)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.filename[0]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    # Process data
    df, useunits = process_slurm_data(df, args, cpn, gpn, max_power)
    
    # Identify codes
    df, motif_set, lang_set = identify_codes(df, codes)
    
    # Add research areas if supplied
    if args.projlist is not None:
        df['Area'] = df['ProjectID'].map(areadict)
    
    # Set up anonymous user identifiers
    userlist = df['User'].unique()
    anonid = [f"user{i}" for i in range(len(userlist))]
    anonuid_dict = dict(zip(userlist, anonid))
    df['AnonUser'] = df['User'].map(anonuid_dict)
    
    # Get unique project IDs
    projidlist = df['ProjectID'].unique()
    
    # Restrict to specified project code if requested
    if args.account:
        df = df[df.ProjectID == args.account].copy()
        print(df)
    
    print("\n----------------------------------")
    print("# SCUA (Slurm Code Usage Analysis)")
    print()
    print("EPCC, 2021, 2022")
    print("----------------------------------\n")
    
    if not args.user is None and args.user != "":
        print("On user account " + args.user + "\n")
    if not args.account is None and args.account != "":
        print("On slurm account " + args.account + "\n")
    if args.dropnan:
        print("Dropping job steps with no energy values recorded\n")
    
    # Split out data in Python and a.out sections by user (if required)
    if args.usersplit:
        df = apply_user_split(df, args.anon, ['a.out', 'Python'])
    
    # Data quality checks
    allcu = df['Usage'].sum()
    allen = df['Energy'].sum()
    
    # Software that are unidentified but have significant use
    mask = df['Software'].values == 'Unknown'
    df_code = df[mask]
    groupf = {'Usage': 'sum', 'Count': 'sum'}
    df_group = df_code.groupby(['ExeName']).agg(groupf)
    df_group.sort_values('Usage', inplace=True, ascending=False)
    thresh = allcu * args.unknown_thresh
    print(f'\n## Unidentified executables with significant use (threashold = {args.unknown_thresh:.3f})\n')
    print(df_group.loc[df_group['Usage'] >= thresh].to_markdown(floatfmt=".1f"))
    print()
    
    if args.usersplit:
        # Make sure we gather data on these executables, broken down by user
        df_group.reset_index()
        unidentified_exe = df_group.loc[df_group['Usage'] >= thresh].index.to_list()
        # We loop over the identified executables and append the anonymised username
        if args.anon:
            for exe in unidentified_exe:
                df.loc[df['ExeName'] == exe, 'Software'] = exe
                df.loc[df['ExeName'] == exe, 'Software'] = (
                    df['Software'] + "_" + df['AnonUser']
                )
        else:
            for exe in unidentified_exe:
                df.loc[df['ExeName'] == exe, 'Software'] = exe
                df.loc[df['ExeName'] == exe, 'Software'] = (
                    df['Software'] + "_" + df['User']
                )
    
    if args.analysepower:
        # Check quality of energy data
        n_nan = df['Energy'].isna().sum()
        nan_usage = df.loc[df['Energy'].isna(), 'Usage'].sum()
        n_row = df.shape[0]
        print('\n## Energy data quality check\n')
        print(f'{"Number of subjobs =":>30s} {n_row:>10d}')
        print(f'{"Subjobs missing energy =":>30s} {n_nan:>10d} ({100*n_nan/n_row:.2f}%)')
        print(f'{"Usage missing energy =":>30s} {nan_usage:>10.1f} {useunits} ({100*nan_usage/allcu:.2f}%)\n')
        
        # Check quality of node power data
        n_small = df['Cores'].lt(cpn).sum()
        cores_usage = df.loc[df['Cores'].lt(cpn), 'Usage'].sum()
        energy_usage = df.loc[df['Cores'].lt(cpn), 'Energy'].sum()
        n_row = df.shape[0]
        print('\n## Node power data quality check\n')
        print(f'{"Number of subjobs =":>30s} {n_row:>10d}')
        print(f'{"Subjobs excluded =":>30s} {n_small:>10d} ({100*n_small/n_row:.2f}%)')
        print(f'{"Usage excluded =":>30s} {cores_usage:>10.1f} {useunits} ({100*cores_usage/allcu:.2f}%)')
        print(f'{"Energy excluded =":>30s} {energy_usage:>10.1f} kWh ({100*energy_usage/allen:.2f}%)\n')
    
    # Get the final list of software in the dataframe
    codelist = df['Software'].unique()
    codelist = list(filter(lambda item: item is not None, codelist))
    
    # Get CPUFreq set from the column (needed for CPU frequency analysis)
    cpufreq_set = set(df['CPUFreq'].unique())
    
    # Load analysis category definitions from TOML files
    try:
        all_categories = load_analysis_categories(scua_base)
    except Exception as e:
        print(f"Error loading analysis categories: {e}", file=sys.stderr)
        return 1
    
    # Filter categories based on command-line flags
    outputs = filter_categories_by_flags(all_categories, args)
    
    # Build category configuration dictionaries for analyze_category()
    title = {}
    category_list = {}
    category_col = {}
    analysis_col = {}
    analyse_none = {}
    full_node = {}
    fullmatch = {}
    
    for cat_id in outputs:
        config = all_categories[cat_id]
        title[cat_id] = config['title']
        category_col[cat_id] = config['category_column']
        
        # Handle dynamic analysis column based on GPU mode
        if 'analysis_column' in config:
            analysis_col[cat_id] = config['analysis_column']
        elif args.usegpu and 'analysis_column_gpu' in config:
            analysis_col[cat_id] = config['analysis_column_gpu']
        elif not args.usegpu and 'analysis_column_core' in config:
            analysis_col[cat_id] = config['analysis_column_core']
        
        analyse_none[cat_id] = config.get('analyse_none', False)
        full_node[cat_id] = config.get('full_node', False)
        fullmatch[cat_id] = config.get('fullmatch', True)
        
        # Set category_list based on the category type
        if config['category_column'] == 'Software':
            category_list[cat_id] = codelist
        elif config['category_column'] == 'Area':
            category_list[cat_id] = areaset
        elif config['category_column'] == 'ProjectID':
            category_list[cat_id] = projidlist
        elif config['category_column'] == 'Motif':
            category_list[cat_id] = motif_set
        elif config['category_column'] == 'Language':
            category_list[cat_id] = lang_set
        elif config['category_column'] == 'CPUFreq':
            category_list[cat_id] = cpufreq_set
    
    # Loop over the defined output categories computing distributions
    for output in outputs:
        df_output = df
        # This restricts to full nodes if needed
        if full_node[output]:
            if args.usegpu:
                mask = df['NGPUS'].ge(gpn)
            else:
                mask = df['Cores'].ge(cpn)
            df_output = df[mask]
        
        analyze_category(
            df, df_output, output, title[output],
            category_list[output], category_col[output], analysis_col[output],
            analyse_none[output], full_node[output], fullmatch[output],
            allcu, allen, useunits, args.reportenergy,
            args.savecsv, args.savemd, args.prefix
        )
    
    # Generate plots if requested
    if args.makeplots:
        generate_plots(df, args, allcu, allen, codelist, cpn, gpn, useunits)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
