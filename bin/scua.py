#!/usr//bin/env python3
# scua.py (Slurm Code Usage Analysis)
#
# usage: scua.py [options] filename
#
# Compute software usage data from Slurm output.
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
#
#

import numpy as np
import pandas as pd
import sys
import os
import re
import fnmatch
import argparse
import csv
from matplotlib import pyplot as plt
import seaborn as sns
from code_def import CodeDef

pd.options.mode.chained_assignment = None 

#=======================================================
# Functions to compute statistics
#=======================================================
# Get overall statistics in a dataframe
def getoverallstats(df, cu, en):
    totcu = df['Usage'].sum()
    percentcu = 100 * totcu / cu
    toten = df['Energy'].sum()
    percenten = 100 * toten / en
    totjobs = df['Count'].sum()
    totusers = df['User'].nunique()
    totprojects = df['ProjectID'].nunique()
    return totcu, percentcu, toten, percenten, totjobs, totusers, totprojects

# Get unweighted distribution
def getdist(df, col):
    minjob = df[col].min()
    maxjob = df[col].max()
    q1job = df[col].quantile(0.25)
    medjob = df[col].quantile(0.5)
    q3job = df[col].quantile(0.75)
    return minjob, maxjob, q1job, medjob, q3job

# Get weighted distribution
def getweighteddist(df, values, weights):
    df.sort_values(values, inplace=True)
    cumsum = df[weights].cumsum()
    cutoff = df[weights].sum() * 0.5
    median = float(df[cumsum >= cutoff][values].iloc[0])
    cutoff = df[weights].sum() * 0.25
    q1 = float(df[cumsum >= cutoff][values].iloc[0])
    cutoff = df[weights].sum() * 0.75
    q3 = float(df[cumsum >= cutoff][values].iloc[0])
    return q1, median, q3

# Get distribution statistics
#     Returns two lists, one with unweighted distribution, one with weighted distribution
def distribution(df, label, allcu, allen, values, weights):
    totcu, percentcu, toten, percenten, totjobs, totusers, totprojects = getoverallstats(df, allcu, allen)
    minval, maxval, q1val, medval, q3val = getdist(df, values)
    dist = [label, minval, q1val, medval, q3val, maxval, totjobs, totcu, percentcu, toten, percenten, totusers, totprojects]
    wq1val, wmedval, wq3val = getweighteddist(df, values, weights)
    wdist = [label, minval, wq1val, wmedval, wq3val, maxval, totjobs, totcu, percentcu, toten, percenten, totusers, totprojects]
    return dist, wdist

# Compute version of dataframe weighted by use
def reindex_df(df, weight_col):
    """expand the dataframe to prepare for resampling
    result is 1 row per count per sample"""
    df = df.reindex(df.index.repeat(df[weight_col]))
    df.reset_index(drop=True, inplace=True)
    return(df)

#=======================================================
# Main code
#=======================================================

# Defaults
CPN = 128          # Number of cores per node
GPN = 4          # Number of GPUS per node
MAX_POWER = 850    # Maximum per-node power draw to consider (in W) 

unknown_thresh = 0.01

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute software usage data from Slurm output.')
parser.add_argument('filename', type=str, nargs=1, help='Data file containing listing of Slurm jobs')
parser.add_argument('-A', dest='account', type=str, action='store', nargs='?', default='', help='The slurm account specified for the report. Default: none')
parser.add_argument('--anon', dest='anon', action='store_false', default=True, help='If splitting unknown use by user, anonymise usernames. Default: false')
parser.add_argument('--cpn', dest='CPN', action='store', default=128, help='Set number of cores per node. Default: 128')
parser.add_argument('--gpn', dest='GPN', action='store', default=4, help='Set number of GPUs per node. Only has an effect if --gpu is specified. Default: 4')
parser.add_argument('--csv', dest='savecsv', action='store_true', default=False, help='Produce data files in CSV. Default: false')
parser.add_argument('--cpufreq', dest='analysecpufreq', action='store_true', default=False, help='Produce CPU frequency usage distribution. Default: false')
parser.add_argument('--dropnan', dest='dropnan', action='store_true', default=False, help='Drop all rows that contain NaN. Useful for strict comparisons between usage and energy use. Default: false')
parser.add_argument('--energy', dest='reportenergy', action='store_true', default=False, help='Report enery use in output. Note this is known to be inaccurate when just job steps are considered - consider using the scea tool instead to report energy use from jobs. Default: false')
parser.add_argument('--gpu', dest='usegpu', action='store_true', default=False, help='Read number of GPU used as part of input data and use nGPU and GPUh as the reporting unit')
parser.add_argument('--lang', dest='analyselang', action='store_true', default=False, help='Produce programming language usage distribution. Default: false')
parser.add_argument('--md', dest='savemd', action='store_true', default=False, help='Produce data files in MD. Default: false')
parser.add_argument('--motif', dest='analysemotif', action='store_true', default=False, help='Produce algorithmic motif usage distribution. Default: false')
parser.add_argument('--mpow', dest='MAX_POWER', action='store', default=850, help='Set maximum realistic power draw for a compute node (used to remove jobs with counter errors). Default: 850')
parser.add_argument('--plots', dest='makeplots', action='store_true', default=False, help='Produce data plots. Default: false')
parser.add_argument('--power', dest='analysepower', action='store_true', default=False, help='Produce node power usage distribution. Default: false')
parser.add_argument('--prefix', dest='prefix', type=str, action='store', default='scua', help='Set the prefix to be used for output files. Default: scua')
parser.add_argument('--projects', dest='projlist', type=str, action='store', default=None, help='The file containing a list of project IDs and associated research areas. Default: none')
parser.add_argument('--projid', dest='analyseprojid', action='store_true', default=False, help='Produce project ID usage distribution. Default: false')
parser.add_argument('--sharednode', dest='sharednode', action='store_true', default=False, help='Can nodes be shared by jobs? Default: false')
parser.add_argument('--thresh', dest='unknown_thresh', type=float, action='store', default=0.01, help='Usage fraction obove which unknown exe names are printed. Default is 0.01 - 1 percent.')
parser.add_argument('-u', dest='user', type=str, action='store', nargs='?', default='', help='The user specified for the report. Default: none')
parser.add_argument('--units', dest='useunits', type=str, action='store', default='Nodeh', help='Units for calculating use: Nodeh or Coreh supported. Default is Nodeh')
parser.add_argument('--usersplit', dest='usersplit', action='store_true', default=False, help='Split unknown use by user. Default: false')
args = parser.parse_args()


# Read any code definitions
codeConfigDir = os.getenv('SCUA_BASE') + '/app-data/code-definitions'
codes = []
nCode = 0
codeDict = {}
codelist = []
for file in os.listdir(codeConfigDir):
    if fnmatch.fnmatch(file, '*.code'):
        nCode += 1
        code = CodeDef()   
        code.readConfig(codeConfigDir + '/' + file)
        codes.append(code)
        codeDict[code.name] = nCode - 1
        codelist.append(code.name)

# Read project research areas if required
areadict = {}
areaset = []
if args.projlist is not None:
    projfile = open(args.projlist, 'r')
    next(projfile)  # Skip header
    reader = csv.reader(projfile, skipinitialspace=True)
    areadict  = dict(reader)
    areaset = set(areadict.values()) # Unique area names

# Read dataset (usually saved from Slurm)
colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq','SubJobID']
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

if args.usegpu:
   colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq','NGPUS','SubJobID']
   col_types['NGPUS'] = 'str'
   df = pd.read_csv(args.filename[0], names=colid, dtype=col_types, sep=',', engine='python', header=0)
   df['NGPUS'] = pd.to_numeric(df['NGPUS'], errors='coerce')
else:
   df = pd.read_csv(args.filename[0], names=colid, dtype=col_types, sep=',', engine='python', header=0)
# Count helps with number of jobs
df['Count'] = 1

# Get the list of CPUFreq from the column (get unique words in the column)
cpufreq_set = set()
cpufreq_set = df['CPUFreq'].unique()

# Convert columns to numeric type
df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
df['NTasks'] = pd.to_numeric(df['NTasks'], errors='coerce')
# Remove unrealistic values (presumably due to counter errors)
df['Energy'].mask(df['Energy'].gt(1e16), inplace=True)
df['NodePower'] = df['Energy'] / (df['Runtime'] * df['Nodes'])
# df.replace(np.inf, np.nan, inplace=True)
# Convert the energy to kWh
df['Energy'] = df['Energy'] / 3600000.0
# Check if we are dropping rows without energy values
if args.dropnan:
    df.dropna(axis=0, how='any', inplace=True)

# This section is to get the number of used cores, we need to make sure we catch
# jobs where people are using SMT and do not count the size of these wrong
df['cpn'] = df['NTasks'] / df['Nodes']
# Default is that NTasks actually corresponds to cores
df['Cores'] = df['NTasks']
# Catch those cases where jobs are using SMT and recompute core count
df.loc[df['cpn'] > CPN, 'Cores'] = df['Nodes'] * CPN

# Calculate the number of Nodeh/Coreh
useunits = args.useunits
if useunits == 'Nodeh':
   df['Usage'] = df['Nodes'] * df['Runtime'] / 3600.0
elif useunits == 'Coreh':
   df['Usage'] = df['Cores'] * df['Runtime'] / 3600.0

if args.usegpu:
    useunits = 'GPUh'
    df['Usage'] = df['NGPUS'] * df['Runtime'] / 3600.0

if not args.sharednode:
#   If the number of cores is less than a node then we need to get a 
#   fractional node hour count and fractional energy
#   Note: energy from Slurm is taken from node-level counters so if there 
#     are multiple job steps per node they are all assigned the full node
#     energy
#   Note: the weakness here is if people are using less cores than a full
#     node but are still using SMT. We will overcount the time for this case.
   if args.usegpu:
      df.loc[df['NGPUS'] < GPN, 'Usage'] = df['NGPUS'] * df['Runtime'] / (GPN * 3600.0)
   else:
      df.loc[df['Cores'] < CPN, 'Usage'] = df['Cores'] * df['Runtime'] / (CPN * 3600.0)
   df.loc[df['Cores'] < CPN, 'Energy'] = df['Cores'] * df['Energy'] / CPN 
   df.loc[df['Cores'] < CPN, 'NodePower'] = df['Cores'] * df['NodePower'] / CPN

# Remove unrealistic values
df.loc[df['Usage'] < 0, 'Usage'] = 0.0


# Remove very unrealistic values (presumably due to counter errors or very short runtimes)
df.replace(np.inf, np.nan, inplace=True)
df['NodePower'].mask(df['NodePower'].gt(MAX_POWER), inplace=True)

# Split Account column into ProjectID and GroupID
#    Need to be careful, if there are no subbudgets we cannot split
df['JobID'] = df['JobID'].astype(str)
if df['Account'].str.contains('-').sum() > 0:
   df[['ProjectID','GroupID']] = df['Account'].str.split(pat='-', n=1, expand=True)
else:
   df['ProjectID'] = df['Account']
   df['GroupID'] = None
projidlist = df['ProjectID'].unique()

# Identify the codes using regex from the code definitions
df["Software"] = 'Unknown'
df["Motif"] = 'Unknown'
df["Language"] = 'Unknown'
# Rarely, an exe name is NaN so we need to remove those
df.dropna(subset = ['ExeName'], inplace=True)
for code in codes:
    codere = re.compile(code.regexp)
    df.loc[df.ExeName.str.contains(codere), ["Software", "Motif", "Language"]] = [code.name, code.type, code.pri_lang]

# Get the list of motifs fnd languages from the column (get unique words in the column)
motif_set = set()
df['Motif'].str.split().apply(motif_set.update)
lang_set = set()
df['Language'].str.split().apply(lang_set.update)

# Add research areas if supplied
if args.projlist is not None:
    df['Area'] = df['ProjectID'].map(areadict)

# Set up anonymous user identifiers
userlist = df['User'].unique()
anonid = [f"user{i}" for i in range(len(userlist))]
zipobj = zip(userlist, anonid)
# Create a dictionary from zip object
anonuid_dict = dict(zipobj)
df['AnonUser'] = df['User'].map(anonuid_dict)

# Restrict to specified project code if requested
if args.account:
    df.drop(df[df.ProjectID != args.account].index, inplace=True)
    print(df)

print("\n----------------------------------")
print("# SCUA (Slurm Code Usage Analysis)")
print()
print("EPCC, 2021, 2022")
print("----------------------------------\n")

if not args.user is None and args.user != "":
    print("On user account " + args.user + "\n");
if not args.account is None and args.account != "":
    print("On slurm account " + args.account + "\n");
if args.dropnan:
    print("Dropping job steps with no energy values recorded\n");

######################################################################
# Split out data in Python and a.out sections by user (if required)
#
if args.usersplit:
    if args.anon:
        df.loc[df['Software'] == 'a.out', 'Software'] = df['Software'] + "_" + df['AnonUser']
        df.loc[df['Software'] == 'Python', 'Software'] = df['Software'] + "_" + df['AnonUser']
    else:
        df.loc[df['Software'] == 'a.out', 'Software'] = df['Software'] + "_" + df['User']
        df.loc[df['Software'] == 'Python', 'Software'] = df['Software'] + "_" + df['User']

######################################################################
# Data quality checks
#
allcu = df['Usage'].sum()
allen = df['Energy'].sum()

# Software that are unidentified but have significant use
mask = df['Software'].values == 'Unknown'
df_code = df[mask]
groupf = {'Usage':'sum', 'Count':'sum'}
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
            df.loc[df['ExeName'] == exe, 'Software'] = df['Software'] + "_" + df['AnonUser']
    else:
        for exe in unidentified_exe:
            df.loc[df['ExeName'] == exe, 'Software'] = exe
            df.loc[df['ExeName'] == exe, 'Software'] = df['Software'] + "_" + df['User']

if args.analysepower:
   # Check quality of energy data
   #
   nNaN = df['Energy'].isna().sum()
   NaNUsage = df.loc[df['Energy'].isna(), 'Usage'].sum()
   nRow = df.shape[0]
   print('\n## Energy data quality check\n')
   print(f'{"Number of subjobs =":>30s} {nRow:>10d}')
   print(f'{"Subjobs missing energy =":>30s} {nNaN:>10d} ({100*nNaN/nRow:.2f}%)')
   print(f'{"Usage missing energy =":>30s} {NaNUsage:>10.1f} {useunits} ({100*NaNUsage/allcu:.2f}%)\n')

   # Check quality of node power data
   #
   nsmall = df['Cores'].lt(CPN).sum()
   coresusage = df.loc[df['Cores'].lt(CPN), 'Usage'].sum()
   energyusage = df.loc[df['Cores'].lt(CPN), 'Energy'].sum()
   nRow = df.shape[0]
   print('\n## Node power data quality check\n')
   print(f'{"Number of subjobs =":>30s} {nRow:>10d}')
   print(f'{"Subjobs excluded =":>30s} {nsmall:>10d} ({100*nsmall/nRow:.2f}%)')
   print(f'{"Usage excluded =":>30s} {coresusage:>10.1f} {useunits} ({100*coresusage/allcu:.2f}%)')
   print(f'{"Energy excluded =":>30s} {energyusage:>10.1f} kWh ({100*energyusage/allen:.2f}%)\n')

######################################################################
# Get the final list of software in the dataframe
#
# Get the list of codes from the software column and remove "None"
codelist = df['Software'].unique()
codelist = list(filter(lambda item: item is not None, codelist))

######################################################################
# Define the distribution analyses to perform
#
outputs = []
title = {}
category_list = {}
category_col = {}
analysis_col = {}
analyse_none = {}
full_node = {}
fullmatch = {}

# Job size distribution for software is always computed
category = 'CodeSize'
outputs.append(category)
title[category] = 'Software job size'
category_list[category] = codelist
category_col[category] = 'Software'
if args.usegpu:
   analysis_col[category] = 'NGPUS'
else:
   analysis_col[category] = 'Cores'
analyse_none[category] = True
full_node[category] = False
fullmatch[category] = True
# Optional categories
if args.analysepower:
    category = 'CodePower'
    title[category] = 'Software node power use'
    outputs.append(category)
    category_list[category] = codelist
    category_col[category] = 'Software'
    analysis_col[category] = 'NodePower'
    analyse_none[category] = True
    full_node[category] = True
    fullmatch[category] = True
if args.projlist is not None:
    category = 'AreaSize'
    outputs.append(category)
    title[category] = 'Research area job size'
    category_list[category] = areaset
    category_col[category] = 'Area'
    if args.usegpu:
       analysis_col[category] = 'NGPUS'
    else:
       analysis_col[category] = 'Cores'
    analyse_none[category] = False
    full_node[category] = False
    fullmatch[category] = True
if args.analyseprojid:
    category = 'ProjSize'
    outputs.append(category)
    title[category] = 'Project ID job size'
    category_list[category] = projidlist
    category_col[category] = 'ProjectID'
    if args.usegpu:
       analysis_col[category] = 'NGPUS'
    else:
       analysis_col[category] = 'Cores'
    analyse_none[category] = True
    full_node[category] = False
    fullmatch[category] = True
if args.analysepower and args.projlist is not None:
    category = 'AreaPower'
    title[category] = 'Research area node power use'
    outputs.append(category)
    category_list[category] = areaset
    category_col[category] = 'Area'
    analysis_col[category] = 'NodePower'
    analyse_none[category] = False
    full_node[category] = True
    fullmatch[category] = True
if args.analysemotif:
    category = 'MotifSize'
    title[category] = 'Algorithmic motifs job size'
    outputs.append(category)
    category_list[category] = motif_set
    category_col[category] = 'Motif'
    if args.usegpu:
       analysis_col[category] = 'NGPUS'
    else:
       analysis_col[category] = 'Cores'
    analyse_none[category] = False
    full_node[category] = False
    fullmatch[category] = False
if args.analysemotif and args.analysepower:
    category = 'MotifPower'
    title[category] = 'Algorithmic motifs node power use'
    outputs.append(category)
    category_list[category] = motif_set
    category_col[category] = 'Motif'
    analysis_col[category] = 'NodePower'
    analyse_none[category] = False
    full_node[category] = True
    fullmatch[category] = False
if args.analyselang:
    category = 'LangSize'
    title[category] = 'Programming language job size'
    outputs.append(category)
    category_list[category] = lang_set
    category_col[category] = 'Language'
    if args.usegpu:
       analysis_col[category] = 'NGPUS'
    else:
       analysis_col[category] = 'Cores'
    analyse_none[category] = False
    full_node[category] = False
    fullmatch[category] = True
if args.analyselang and args.analysepower:
    category = 'LangPower'
    title[category] = 'Programming language node power use'
    outputs.append(category)
    category_list[category] = lang_set
    category_col[category] = 'Language'
    analysis_col[category] = 'NodePower'
    analyse_none[category] = False
    full_node[category] = True
    fullmatch[category] = True
if args.analysecpufreq:
    category = 'CPUFreqSize'
    title[category] = 'CPU frequency job size'
    outputs.append(category)
    category_list[category] = cpufreq_set
    category_col[category] = 'CPUFreq'
    if args.usegpu:
       analysis_col[category] = 'NGPUS'
    else:
       analysis_col[category] = 'Cores'
    analyse_none[category] = True
    full_node[category] = False
    fullmatch[category] = True
if args.analysecpufreq and args.analysepower:
    category = 'CPUFreqPower'
    title[category] = 'CPU frequency node power use'
    outputs.append(category)
    category_list[category] = cpufreq_set
    category_col[category] = 'CPUFreq'
    analysis_col[category] = 'NodePower'
    analyse_none[category] = True
    full_node[category] = True
    fullmatch[category] = True

######################################################################
# Loop over the defined output categories computing distributions
# and outputting the results
#
for output in outputs:
    df_output = df
    # This restricts to full nodes if needed
    if full_node[output]:
        if args.usegpu:
           mask = df['GPUS'].ge(GPN)
        else:
           mask = df['Cores'].ge(CPN)
        df_output = df[mask]

    job_stats = []
    usage_stats = []
    categories = category_list[output]
    # The category column and analysis columns
    catcol = category_col[output]
    ancol = analysis_col[output]
    for category in categories:
        if fullmatch[output]:
            mask = df_output[catcol].str.fullmatch(re.escape(str(category)), na=False)
        else:
            mask = df_output[catcol].str.contains(re.escape(str(category)), na=False)
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = distribution(df_cat, category, allcu, allen, ancol, 'Usage')
            job_stats.append(dist)
            usage_stats.append(wdist)

    # Get the data for unidentified (if needed)
    if analyse_none[output]:
        mask = df_output[catcol].values == None
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = distribution(df_cat, 'Unidentified', allcu, allen, ancol, 'Usage')
            job_stats.append(dist)
            usage_stats.append(wdist)

    # Get overall data for all jobs
    dist, wdist = distribution(df_output, 'Overall', allcu, allen, ancol, 'Usage')
    job_stats.append(dist)
    usage_stats.append(wdist)

    # Save the output
    print(f'\n## {title[output]} distribution: weighted by usage\n')
    df_usage = pd.DataFrame(usage_stats, columns=[catcol, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects'])
    if not args.reportenergy:
        df_usage.drop('kWh', axis=1, inplace=True)
        df_usage.drop('PercentEnergy', axis=1, inplace=True)
    df_usage.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_usage.to_markdown(index=False, floatfmt=".1f"))
    print()
    print(f'\n## {title[output]} distribution: by number of jobs\n')
    df_jobs = pd.DataFrame(job_stats, columns=[catcol, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects'])
    if not args.reportenergy:
        df_jobs.drop('kWh', axis=1, inplace=True)
        df_jobs.drop('PercentEnergy', axis=1, inplace=True)
    df_jobs.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_jobs.to_markdown(index=False, floatfmt=".1f"))
    print()
    # Save as CSV if required
    if args.savecsv:
        df_jobs.to_csv(f'{args.prefix}_{output}.csv', index=False, float_format="%.1f")
        df_usage.to_csv(f'{args.prefix}_{output}_weighted.csv', index=False, float_format="%.1f")
    if args.savemd:
        df_jobs.to_markdown(f'{args.prefix}_{output}.md', index=False, floatfmt=".1f")
        df_usage.to_markdown(f'{args.prefix}_{output}_weighted.md', index=False, floatfmt=".1f")


######################################################################
# Plot any data that depends on derived statistics
#   This section is a hack and should be replaced by a proper approach!
#
if args.makeplots:
    df_output = df

    job_stats = []
    usage_stats = []
    categories = codelist
    # The category column and analysis columns
    catcol = 'Software'
    if args.usegpu:
       ancol = 'NGPUS'
    else:
       ancol = 'Cores'
    for category in categories:
        if fullmatch[output]:
            mask = df_output[catcol].str.fullmatch(re.escape(str(category)), na=False)
        else:
            mask = df_output[catcol].str.contains(re.escape(str(category)), na=False)
        df_cat = df_output[mask]
        if not df_cat.empty:
            dist, wdist = distribution(df_cat, category, allcu, allen, ancol, 'Usage')
            job_stats.append(dist)
            usage_stats.append(wdist)

    mask = df_output[catcol].values == None
    df_cat = df_output[mask]
    if not df_cat.empty:
        dist, wdist = distribution(df_cat, 'Unidentified', allcu, allen, ancol, 'Usage')
        job_stats.append(dist)
        usage_stats.append(wdist)

    # Get overall data for all jobs
    dist, wdist = distribution(df_output, 'Overall', allcu, allen, ancol, 'Usage')
    job_stats.append(dist)
    usage_stats.append(wdist)

    # Save the output
    df_usage = pd.DataFrame(usage_stats, columns=[catcol, 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', useunits, 'PercentUse', 'kWh', 'PercentEnergy', 'Users', 'Projects'])
    df_usage.sort_values('PercentUse', inplace=True, ascending=False)

    # Software usage plots
    # Bar plot of software usage
    df_plot = df_usage[~df_usage['Software'].isin(['Overall','Unidentified'])]
    plt.figure(figsize=[8,6])
    sns.barplot(y='Software', x='PercentUse', color='lightseagreen', data=df_plot)
    sns.despine()
    plt.xlabel('PercentUse')
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_codes_usage.png', dpi=300)
    plt.clf()
    # Boxplots for top 15 software by CU use
    topcodes = df_usage['Software'].head(16).to_list()[1:]
    df_topcodes = df[df['Software'].isin(topcodes)]
    plt.figure(figsize=[8,6])
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
            "marker":"o",
            "markerfacecolor":"white",
            "markeredgecolor":"black",
            "markersize":"5"
            },
        data=reindex_df(df_topcodes, weight_col='Usage')
        )
    plt.xlabel(xcat)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_top15_boxplot.png', dpi=300)
    plt.clf()
