#!/usr//bin/env python
# scua.py (Slurm Code Usage Analysis)
#
# Python program to take output from sacct and analyse code usage
#

import numpy as np
import pandas as pd
import sys
import os
import re
import fnmatch
from code_def import CodeDef

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
job_stats = []
usage_stats = []
for code in codes:
    mask = df['Code'].values == code.name
    df_code = df[mask]
    if not df_code.empty:
        # Job number stats
        totcu = df_code['Nodeh'].sum()
        totjobs = df_code['Count'].sum()
        minjob = df_code['NTasks'].min()
        maxjob = df_code['NTasks'].max()
        q1job = df_code['NTasks'].quantile(0.25)
        medjob = df_code['NTasks'].quantile(0.5)
        q3job = df_code['NTasks'].quantile(0.75)
        job_stats.append([code.name, minjob, q1job, medjob, q3job,maxjob,totjobs,totcu])
        # Usage stats
        df_code.sort_values('NTasks', inplace=True)
        cumsum = df_code.Nodeh.cumsum()
        cutoff = df_code.Nodeh.sum() * 0.5
        meduse = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
        cutoff = df_code.Nodeh.sum() * 0.25
        q1use = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
        cutoff = df_code.Nodeh.sum() * 0.75
        q3use = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
        usage_stats.append([code.name, minjob, q1use, meduse, q3use, maxjob, totjobs, totcu])

# Get the data from unidentified executables
mask = df['Code'].values == None
df_code = df[mask]
if not df_code.empty:
    # Job number stats
    totcu = df_code['Nodeh'].sum()
    totjobs = df_code['Count'].sum()
    minjob = df_code['NTasks'].min()
    maxjob = df_code['NTasks'].max()
    q1job = df_code['NTasks'].quantile(0.25)
    medjob = df_code['NTasks'].quantile(0.5)
    q3job = df_code['NTasks'].quantile(0.75)
    job_stats.append(['Unidentified', minjob, q1job, medjob, q3job,maxjob, totjobs, totcu])
    # Usage stats
    df_code.sort_values('Nodes', inplace=True)
    cumsum = df_code.Nodeh.cumsum()
    cutoff = df_code.Nodeh.sum() * 0.5
    meduse = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
    cutoff = df_code.Nodeh.sum() * 0.25
    q1use = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
    cutoff = df_code.Nodeh.sum() * 0.75
    q3use = float(df_code.NTasks[cumsum >= cutoff].iloc[0])
    usage_stats.append(['Unidentified', minjob, q1use, meduse, q3use, maxjob, totjobs, totcu])
# Get overall data
# Job size statistics from job numbers
totcu = df['Nodeh'].sum()
totjobs = df['Count'].sum()
minjob = df['NTasks'].min()
maxjob = df['NTasks'].max()
q1job = df['NTasks'].quantile(0.25)
medjob = df['NTasks'].quantile(0.5)
q3job = df['NTasks'].quantile(0.75)
job_stats.append(['Overall', minjob, q1job, medjob, q3job,maxjob, totjobs,totcu])
# Job size statistics weighted by usage
df.sort_values('Nodes', inplace=True)
cumsum = df.Nodeh.cumsum()
cutoff = df.Nodeh.sum() * 0.5
meduse = float(df.NTasks[cumsum >= cutoff].iloc[0])
cutoff = df.Nodeh.sum() * 0.25
q1use = float(df.NTasks[cumsum >= cutoff].iloc[0])
cutoff = df.Nodeh.sum() * 0.75
q3use = float(df.NTasks[cumsum >= cutoff].iloc[0])
usage_stats.append(['Overall', minjob, q1use, meduse, q3use, maxjob,  totjobs,totcu])

# Print out title
print("\n---------------------------------")
print("# SCUA (Slurm Code Usage Analysis")
print()
print("EPCC, 2021")
print("---------------------------------\n")

# Print out final stats tables
print('\n## Job size by code: weighted by usage\n')
df_usage = pd.DataFrame(usage_stats, columns=['Code', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'TotJobs', 'TotCU'])
df_usage.sort_values('TotCU', inplace=True, ascending=False)
print(df_usage.to_markdown(index=False, floatfmt=".1f"))
print('\n## Job size by code: based on job numbers\n')
df_job = pd.DataFrame(job_stats, columns=['Code', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'TotJobs', 'TotCU'])
df_job.sort_values('TotCU', inplace=True, ascending=False)
print(df_job.to_markdown(index=False, floatfmt=".1f"))
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

