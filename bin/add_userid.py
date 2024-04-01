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

colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq']
df_step = pd.read_csv(sys.argv[1], names=colid, sep='::', engine='python')
colid = ['JobID','ExeName','User','Account','Nodes','NTasks','Runtime','State','Energy','MaxRSS','MeanRSS','CPUFreq']
df_job = pd.read_csv(sys.argv[2], names=colid, sep='::', engine='python')
df_job['ExeName'] = 'no_srun'
df_job['SubJobID'] = 0

df_step['JobID'] = df_step['JobID'].astype(str)
df_step[['JobID','SubJobID']] = df_step['JobID'].str.split(pat='.', n=1, expand=True)

# Remove just the last duplicated job ID
#  If there are duplicates, the last one corresponds to the top level job
df = pd.concat([df_step, df_job])
m1 = df.duplicated(['JobID'], keep='last')
m2 = ~df.duplicated(['JobID'], keep=False)
df = df[m1|m2]

# Read user IDs associated with jobs
userdict = {}
userfile = open(sys.argv[3], 'r')
next(userfile)  # Skip header
reader = csv.reader(userfile, skipinitialspace=True)
for row in reader:
    if (len(row) > 1):
        userdict[row[0]] = row[1]

# Map the username by matching job id
df['User'] = df['JobID'].map(userdict)

df.to_csv(sys.argv[4], index=False)
