#!/usr//bin/env python3
#
# This short script combines two outputs from sacct 
# to add username details to Slurm job steps (the username
# is typically only available from the top level job
#
# Usage:
#    add_userid.py job-step-datafile job-datafile output-datafile
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

import numpy as np
import pandas as pd
import sys
import csv

def getngpu(x):
    tokens = x.split('=')
    return tokens[1]

###################################
# Add number of GPUs to job steps
colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq']
df_step = pd.read_csv(sys.argv[1], names=colid, sep='::', engine='python')
colid = ['JobID','State','AllocCPU','AllocGPU','AllocMem','AllocNode']
df_gstep = pd.read_csv(sys.argv[2], names=colid, sep=',', engine='python')
df_gstep.drop(['State','AllocCPU','AllocMem','AllocNode'], axis=1, inplace=True)

df_gstep['AllocGPU'] = df_gstep['AllocGPU'].astype(str)
df_gstep['NGPUS'] = df_gstep['AllocGPU'].map(getngpu)

df_step.sort_values('JobID', inplace=True)
df_gstep.sort_values('JobID', inplace=True)

df_step['NGPUS'] = df_gstep['NGPUS']

###################################
# Add number of GPUs to jobs
colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq']
df_job = pd.read_csv(sys.argv[3], names=colid, sep='::', engine='python')
colid = ['JobID','State','Billing','AllocCPU','Energy','AllocGPU','AllocMem','AllocNode']
df_gjob = pd.read_csv(sys.argv[4], names=colid, sep=',', engine='python')
df_gjob.drop(['State','Billing','AllocCPU','Energy','AllocMem','AllocNode'], axis=1, inplace=True)
df_gjob = df_gjob.replace(to_replace='None', value=np.nan).dropna()
df_gjob = df_gjob.loc[df_gjob['AllocGPU'].str.contains('gpu')]
df_gjob['AllocGPU'] = df_gjob['AllocGPU'].astype(str)
df_gjob['NGPUS'] = df_gjob['AllocGPU'].map(getngpu)

df_job.sort_values('JobID', inplace=True)
df_gjob.sort_values('JobID', inplace=True)

df_job['NGPUS'] = df_gjob['NGPUS']

##################################
# Extra settings for jobs
df_job['ExeName'] = 'no_srun'
df_job['SubJobID'] = 0

df_step['JobID'] = df_step['JobID'].astype(str)
df_step[['JobID','SubJobID']] = df_step['JobID'].str.split(pat='.', n=1, expand=True)

# Remove just the last duplicated job ID
#  If there are duplicates, the last one corresponds to the top level job
df = pd.concat([df_step, df_job], ignore_index=True)
df['JobID'] = df['JobID'].astype(int)
m1 = df['JobID'].duplicated(keep='last')
m2 = ~df.duplicated(['JobID'], keep=False)
df = df[m1|m2]

# Read user IDs associated with jobs
userdict = {}
userfile = open(sys.argv[3], 'r')
next(userfile)  # Skip header
reader = csv.reader(userfile, skipinitialspace=True)
for row in reader:
    if (len(row) > 1):
        userdict[int(row[0])] = row[1]

# Map the username by matching job id
df['User'] = df['JobID'].map(userdict)

df.to_csv(sys.argv[6], index=False)

