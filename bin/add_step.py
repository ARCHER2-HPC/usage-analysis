#!/usr//bin/env python
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
df = pd.read_csv(sys.argv[1], names=colid, sep='::', engine='python')

# df['JobID'] = df['JobID'].astype(str)
# df[['JobID','SubJobID']] = df['JobID'].str.split('.', 1, expand=True)

# Read steps associated with jobs
stepdict = {}
stepfile = open(sys.argv[2], 'r')
next(stepfile)  # Skip header
reader = csv.reader(stepfile, skipinitialspace=True)
for row in reader:
    if (len(row) > 1):
        tokens = row[0].split('.')
        stepdict[int(tokens[0])] = row[1]

# Map the exe name by matching job id
df['ExeName'] = df['JobID'].map(stepdict)

df.to_csv(sys.argv[3], index=False)

