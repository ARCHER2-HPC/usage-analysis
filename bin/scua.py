#!/usr//bin/env python
# scua.py (Slurm Code Usage Analysis)
#
# Python program to take output from sacct and analyse code usage
#
# Usage:
#   scua.py <sacct data file>
#
# The data file is the output for Slurm sacct, produced in using a 
# command such as:
#   sacct --format JobIDRaw,JobName%30,Account,NNodes,NTasks,ElapsedRaw,State -P --delimiter , \
#        | egrep '[0-9]\.[0-9]' | grep COMPLETED
#
# The egrep extracts all subjobs and then grep captures the completed jobs.
# This was done as the -s CD option to sacct was not found to work correctly.
#
#----------------------------------------------------------------------
# Copyright 2021 EPCC, The University of Edinburgh
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
from code_def import CodeDef

#=======================================================
# Functions to compute statistics
#=======================================================
# Get statistics based on numbers of jobs
def getjobstats(df, cu):
    totcu = df['Nodeh'].sum()
    percentcu = 100 * totcu / cu
    totjobs = df['Count'].sum()
    minjob = df['NTasks'].min()
    maxjob = df['NTasks'].max()
    q1job = df['NTasks'].quantile(0.25)
    medjob = df['NTasks'].quantile(0.5)
    q3job = df['NTasks'].quantile(0.75)
    return totcu, percentcu, totjobs, minjob, maxjob, q1job, medjob, q3job

# Get statistics weighted by CU (nodeh)
def getweightedstats(df):
    df.sort_values('NTasks', inplace=True)
    cumsum = df.Nodeh.cumsum()
    cutoff = df.Nodeh.sum() * 0.5
    meduse = float(df.NTasks[cumsum >= cutoff].iloc[0])
    cutoff = df.Nodeh.sum() * 0.25
    q1use = float(df.NTasks[cumsum >= cutoff].iloc[0])
    cutoff = df.Nodeh.sum() * 0.75
    q3use = float(df.NTasks[cumsum >= cutoff].iloc[0])
    return meduse, q1use, q3use

#=======================================================
# Read any code definitions
#=======================================================
codeConfigDir = os.getenv('SCUA_BASE') + '/data/code-definitions'
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

colid = ['JobID','ExeName','Account','Nodes','NTasks','Runtime','State']
df = pd.read_csv(sys.argv[1], names=colid)
df['Count'] = 1
df['Nodeh'] = df['Nodes'] * df['Runtime'] / 3600.0
# Split JobID column into JobID and subjob ID
df['JobID'] = df['JobID'].astype(str)
df[['JobID','SubJobID']] = df['JobID'].str.split('.', 1, expand=True)

# Identify the codes using regex from the code definitions
df["Code"] = None
for code in codes:
    codere = re.compile(code.regexp)
    df.loc[df.ExeName.str.contains(codere), "Code"] = code.name

# Loop over codes getting the statistics
allcu = df['Nodeh'].sum()
job_stats = []
usage_stats = []
for code in codes:
    mask = df['Code'].values == code.name
    df_code = df[mask]
    if not df_code.empty:
        # Job number stats
        totcu, percentcu, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df_code, allcu)
        job_stats.append([code.name, minjob, q1job, medjob, q3job,maxjob,totjobs,totcu, percentcu])
        # Usage stats
        meduse, q1use, q3use = getweightedstats(df_code)
        usage_stats.append([code.name, minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu])

# Get the data from unidentified executables
mask = df['Code'].values == None
df_code = df[mask]
if not df_code.empty:
    # Job number stats
    totcu, percentcu, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df_code, allcu)
    job_stats.append(['Unidentified', minjob, q1job, medjob, q3job,maxjob, totjobs, totcu, percentcu])
    # Usage stats
    meduse, q1use, q3use = getweightedstats(df_code)
    usage_stats.append(['Unidentified', minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu])
# Get overall data
# Job size statistics from job numbers
totcu, percentcu, totjobs, minjob, maxjob, q1job, medjob, q3job = getjobstats(df, allcu)
job_stats.append(['Overall', minjob, q1job, medjob, q3job,maxjob, totjobs, totcu, percentcu])
# Job size statistics weighted by usage
meduse, q1use, q3use = getweightedstats(df)
usage_stats.append(['Overall', minjob, q1use, meduse, q3use, maxjob, totjobs, totcu, percentcu])

# Print out title
print("\n----------------------------------")
print("# SCUA (Slurm Code Usage Analysis)")
print()
print("EPCC, 2021")
print("----------------------------------\n")

# Print out final stats tables
print('\n## Job size (in cores) by code: weighted by usage\n')
df_usage = pd.DataFrame(usage_stats, columns=['Code', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'TotJobs', 'TotCU', 'PercentCU'])
df_usage.sort_values('TotCU', inplace=True, ascending=False)
print(df_usage.to_markdown(index=False, floatfmt=".1f"))
df_usage.to_csv(path_or_buf='stats_by_uasge.csv', index=False, float_format=".1f"))
print('\n## Job size (in cores) by code: based on job numbers\n')
df_job = pd.DataFrame(job_stats, columns=['Code', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'TotJobs', 'TotCU', 'PercentCU'])
df_job.sort_values('TotCU', inplace=True, ascending=False)
print(df_job.to_markdown(index=False, floatfmt=".1f"))
df_job.to_csv(path_or_buf='stats_by_jobs.csv', index=False, float_format=".1f"))
print()

# Codes that are unidentified but have more than 1% of total use
mask = df['Code'].values == None
df_code = df[mask]
groupf = {'Nodeh':'sum', 'Count':'sum'}
df_group = df_code.groupby(['ExeName']).agg(groupf)
df_group.sort_values('Nodeh', inplace=True, ascending=False)
thresh = totcu * 0.01
print('\n## Unidentified executables with significant use\n')
print(df_group.loc[df_group['Nodeh'] >= thresh].to_markdown(floatfmt=".1f"))
print()

