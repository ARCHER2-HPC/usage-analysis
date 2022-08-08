#!/usr//bin/env python
# scua.py (Slurm Code Usage Analysis)
#
# usage: scua.py [-h] [--plots] [--prefix=PREFIX] filename
#
# Compute software usage data from Slurm output.
# 
# positional arguments:
#  filename         Data file containing listing of Slurm jobs
#
# optional arguments:
#   -h, --help       show this help message and exit
#   --anon           output anonymised CSV raw job data
#   --plots          Produce data plots
#   --prefix=PREFIX  Set the prefix to be used for output files
#
# The data file is the output for Slurm sacct, produced in using a 
# command such as:
#   sacct --format JobIDRaw,JobName%30,Account,NNodes,NTasks,ElapsedRaw,State -P --delimiter :: \
#        | egrep '[0-9]\.[0-9]' | egrep -v "RUNNING|PENDING|REQUEUED"
#
# The egrep extracts all subjobs and then excludes specific job states
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
from matplotlib import pyplot as plt
import seaborn as sns
from code_def import CodeDef

pd.options.mode.chained_assignment = None 

#=======================================================
# Functions to compute statistics
#=======================================================
# Get statistics based on numbers of jobs
def getjobstats(df, cu, en):
    totcu = df['Nodeh'].sum()
    percentcu = 100 * totcu / cu
    toten = df['Energy'].sum()
    percenten = 100 * toten / en
    totjobs = df['Count'].sum()
    minjob = df['Cores'].min()
    maxjob = df['Cores'].max()
    q1job = df['Cores'].quantile(0.25)
    medjob = df['Cores'].quantile(0.5)
    q3job = df['Cores'].quantile(0.75)
    return totcu, percentcu, toten, percenten, totjobs, minjob, maxjob, q1job, medjob, q3job

# Get statistics weighted by CU (nodeh)
def getweightedstats(df):
    df.sort_values('Cores', inplace=True)
    cumsum = df.Nodeh.cumsum()
    cutoff = df.Nodeh.sum() * 0.5
    meduse = float(df.Cores[cumsum >= cutoff].iloc[0])
    cutoff = df.Nodeh.sum() * 0.25
    q1use = float(df.Cores[cumsum >= cutoff].iloc[0])
    cutoff = df.Nodeh.sum() * 0.75
    q3use = float(df.Cores[cumsum >= cutoff].iloc[0])
    return meduse, q1use, q3use

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

# Constants
CPN = 128

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute software usage data from Slurm output.')
parser.add_argument('filename', type=str, nargs=1, help='Data file containing listing of Slurm jobs')
parser.add_argument('--plots', dest='makeplots', action='store_true', default=False, help='Produce data plots')
parser.add_argument('--web', dest='webdata', action='store_true', default=False, help='Produce web data')
parser.add_argument('--prefix', dest='prefix', type=str, action='store', default='scua', help='Set the prefix to be used for output files')
parser.add_argument('-S', dest='startdate', type=str, action='store', nargs='?', default='', help='The start date specified for the report')
parser.add_argument('-E', dest='enddate', type=str, action='store', nargs='?',default='', help='The end date specified for the report')
parser.add_argument('-A', dest='account', type=str, action='store', nargs='?', default='', help='The slurm account specified for the report')
parser.add_argument('-u', dest='user', type=str, action='store', nargs='?', default='', help='The user specified for the report')
args = parser.parse_args()

# Read any code definitions
codeConfigDir = os.getenv('SCUA_BASE') + '/app-data/code-definitions'
codes = []
nCode = 0
# Create a dictionary of codes
codeDict = {}
for file in os.listdir(codeConfigDir):
    if fnmatch.fnmatch(file, '*.code'):
        nCode += 1
        code = CodeDef()   
        code.readConfig(codeConfigDir + '/' + file)
        codes.append(code)
        codeDict[code.name] = nCode - 1

# Read dataset (usually saved from Slurm)
colid = ['JobID','ExeName','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS']
df = pd.read_csv(args.filename[0], names=colid, sep='::', engine='python')
# Count helps with number of jobs
df['Count'] = 1

# Convert energy to numeric type
df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
# Remove unrealistic values (presumably due to counter errors)
df['Energy'].mask(df['Energy'].gt(1e16), inplace=True)
# Convert the energy to kWh
df['Energy'] = df['Energy'] / 3600000.0

# This section is to get the number of used cores, we need to make sure we catch
# jobs where people are using SMT and do not count the size of these wrong
df['cpn'] = df['NTasks'] / df['Nodes']
# Default is that NTasks actually corresponds to cores
df['Cores'] = df['NTasks']
# Catch those cases where jobs are using SMT and recompute core count
df.loc[df['cpn'] > CPN, 'Cores'] = df['Nodes'] * CPN

# Calculate the number of Nodeh
#   If the number of cores is less than a node then we need to get a 
#   fractional node hour count and fractional energy
#   Note: energy from Slurm is taken from node-level counters so if there 
#     are multiple job steps per node they are all assigned the full node
#     energy
#   Note: the weakness here is if people are using less cores than a full
#     node but are still using SMT. We will overcount the time for this case.
df['Nodeh'] = df['Nodes'] * df['Runtime'] / 3600.0
df.loc[df['Cores'] < CPN, 'Nodeh'] = df['Cores'] * df['Runtime'] / (CPN * 3600.0)
df.loc[df['Cores'] < CPN, 'Energy'] = df['Cores'] * df['Energy'] / CPN 

# Split JobID column into JobID and subjob ID
df['JobID'] = df['JobID'].astype(str)
df[['JobID','SubJobID']] = df['JobID'].str.split('.', 1, expand=True)

# Split Account column into ProjectID and GroupID
df['JobID'] = df['JobID'].astype(str)
df[['ProjectID','GroupID']] = df['Account'].str.split('-', 1, expand=True)

# Identify the codes using regex from the code definitions
df["Software"] = None
for code in codes:
    codere = re.compile(code.regexp)
    df.loc[df.ExeName.str.contains(codere), "Software"] = code.name

if args.makeplots:
    plt.figure(figsize=[6,2])
    sns.boxplot(
        x="Cores",
        orient='h',
        color='lightseagreen',
        showmeans=True,
        width=0.25,
        meanprops={
            "marker":"o",
            "markerfacecolor":"white",
            "markeredgecolor":"black",
            "markersize":"5"
            },
        data=reindex_df(df, weight_col='Nodeh')
        )
    plt.xlabel('Cores')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_overall_boxplot.png', dpi=300)
    plt.clf()

# Loop over codes getting the statistics
allcu = df['Nodeh'].sum()
allen = df['Energy'].sum()
job_stats = []
usage_stats = []
for code in codes:
    mask = df['Software'].values == code.name
    df_code = df[mask]
    if not df_code.empty:
        # Job number stats
        totcu, percentcu, toten, percenten, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df_code, allcu, allen)
        job_stats.append([code.name, minjob, q1job, medjob, q3job,maxjob, totjobs,totcu, percentcu, toten, percenten])
        # Job stats weighed by CU (Nodeh) use
        meduse, q1use, q3use = getweightedstats(df_code)
        usage_stats.append([code.name, minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu, toten, percenten])

# Get the data for unidentified executables
mask = df['Software'].values == None
df_code = df[mask]
if not df_code.empty:
    # Job number stats
    totcu, percentcu, toten, percenten, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df_code, allcu, allen)
    job_stats.append(['Unidentified', minjob, q1job, medjob, q3job,maxjob, totjobs, totcu, percentcu, toten, percenten])
    # Usage stats
    meduse, q1use, q3use = getweightedstats(df_code)
    usage_stats.append(['Unidentified', minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu, toten, percenten])
# Get overall data for all jobs
# Job size statistics from job numbers
totcu, percentcu, toten, percenten, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df, allcu, allen)
job_stats.append(['Overall', minjob, q1job, medjob, q3job,maxjob, totjobs, totcu, percentcu, toten, percenten])
# Job size statistics weighted by usage
meduse, q1use, q3use = getweightedstats(df)
usage_stats.append(['Overall', minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu, toten, percenten])

# Output data
print("\n----------------------------------")
print("# SCUA (Slurm Code Usage Analysis)")
print()
print("EPCC, 2021, 2022")
print("----------------------------------\n")

print("Time period " + args.startdate + " - " + args.enddate + " \n");
if not args.user is None and args.user != "":
    print("On user account " + args.user + "\n");
if not args.account is None and args.account != "":
    print("On slurm account " + args.account + "\n");

# Print out final stats tables
# Weighted by CU use
print('\n## Job size (in cores) by software: weighted by usage\n')
df_usage = pd.DataFrame(usage_stats, columns=['Software', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', 'Nodeh', 'PercentUse', 'kWh', 'PercentEnergy'])
if args.webdata:
    df_usage.drop('Nodeh', axis=1, inplace=True)
    df_usage.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_usage.to_markdown(index=False, floatfmt=".1f"))
    df_usage.to_markdown(f'{args.prefix}_stats_by_uasge.md', index=False, floatfmt=".1f")
    df_usage.to_csv(f'{args.prefix}_stats_by_uasge.csv', index=False, float_format="%.1f")
else:
    df_usage.sort_values('Nodeh', inplace=True, ascending=False)
    print(df_usage.to_markdown(index=False, floatfmt=".1f"))
    df_usage.to_markdown(f'{args.prefix}_stats_by_uasge.md', index=False, floatfmt=".1f")
    df_usage.to_csv(f'{args.prefix}_stats_by_uasge.csv', index=False, float_format="%.1f")

if args.makeplots:
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
    sns.boxplot(
        y="Software",
        x="Cores",
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
        data=reindex_df(df_topcodes, weight_col='Nodeh')
        )
    plt.xlabel('Cores')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{args.prefix}_top15_boxplot.png', dpi=300)
    plt.clf()

# No weighting
print('\n## Job size (in cores) by software: based on job numbers\n')
df_job = pd.DataFrame(job_stats, columns=['Software', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Jobs', 'Nodeh', 'PercentUse', 'kWh', 'PercentEnergy'])
if args.webdata:
    df_job.drop('Nodeh', axis=1, inplace=True)
    df_job.sort_values('PercentUse', inplace=True, ascending=False)
    print(df_job.to_markdown(index=False, floatfmt=".1f"))
    df_job.to_markdown(f'{args.prefix}_stats_by_jobs.md', index=False, floatfmt=".1f")
    df_job.to_csv(f'{args.prefix}_stats_by_jobs.csv', index=False, float_format="%.1f")
else:
    df_job.sort_values('Nodeh', inplace=True, ascending=False)
    print(df_job.to_markdown(index=False, floatfmt=".1f"))
    df_job.to_markdown(f'{args.prefix}_stats_by_jobs.md', index=False, floatfmt=".1f")
    df_job.to_csv(f'{args.prefix}_stats_by_jobs.csv', index=False, float_format="%.1f")
print()

# Codes that are unidentified but have more than 1% of total use
mask = df['Software'].values == None
df_code = df[mask]
groupf = {'Nodeh':'sum', 'Count':'sum'}
df_group = df_code.groupby(['ExeName']).agg(groupf)
df_group.sort_values('Nodeh', inplace=True, ascending=False)
thresh = totcu * 0.01
print('\n## Unidentified executables with significant use\n')
print(df_group.loc[df_group['Nodeh'] >= thresh].to_markdown(floatfmt=".1f"))
print()

# Check quality of energy data
nNaN = df['Energy'].isna().sum()
nRow = df.shape[0]
print('\n## Energy data quality check\n')
print(f'{"Number of subjobs =":>30s} {nRow:>10d}')
print(f'{"Subjobs missing energy =":>30s} {nNaN:>10d} ({100*nNaN/nRow:.2f}%)\n')


