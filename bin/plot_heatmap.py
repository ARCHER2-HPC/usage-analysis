#!/usr/bin/env python
#
# This script plots heatmaps of scheduling coefficients and usage using matplotlib
# Input to the sript is from the SAFE Scehduling Coefficient report
#
import sys
import re
import numpy as np

prefix = sys.argv[1].strip()
csvfile = open(sys.argv[2], "r")


# Hardcoded job sizes are horrible but we would have to parse the file twice to get them
size = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512",
        "513-1024", "1025-2048","2049-4096","4097-8192"]
jobtime = []
eff = []
wait = []
rjobs = []
sjobs = []
usage = []
values = []
rjvalues = []
sjvalues = []
uvalues = []
wvalues = []
submit = []

########################################################
# Read the data from CSV file
########################################################
intable = False
for line in csvfile:

   if re.match("Runtime:", line):
      sizeline = line.rstrip()
      tokens = sizeline.split()
      joblen = tokens[1]
      jobtime.append(joblen)
      intable = True
      values = [None] * len(size)
      rjvalues = [0] * len(size)
      sjvalues = [0] * len(size)
      uvalues = [0.0] * len(size)
      wvalues = ["0:0:0"] * len(size)
      continue
      # End of if start block match

   # If we are in the time range then accumulate data
   if intable:
               
      # Skip the end indicator
      if re.match(",", line):
         eff.append(values)
         rjobs.append(rjvalues)
         sjobs.append(sjvalues)
         usage.append(uvalues)
         wait.append(wvalues)
         intable = False
         continue
      elif re.match("Nodes", line):
         continue
      dataline = line.rstrip()
      tokens = dataline.split(",")
      ind = size.index(tokens[0])
      sjvalues[ind] = int(tokens[1])
      rjvalues[ind] = int(tokens[2])
      uvalues[ind] = float(tokens[3])
      wvalues[ind] = tokens[5]
      values[ind] = float(tokens[6])

csvfile.close()

for i in range(len(jobtime)):
   for j in range(len(size)):
      if (eff[i][j] is None) or (rjobs[i][j] == 0):
         eff[i][j] = -1
         usage[i][j] = -1.0

xeff = np.array(eff)
xusage = np.array(usage)

########################################################
# Plot heatmaps
########################################################
import matplotlib

# Plot to image file without need for X server
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.use("Agg")

# Import the required functions
from matplotlib import pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax1 = plt.subplot(1, 1, 1)
masked_array = np.ma.masked_where(xeff==-1, xeff)
cmap = cm.RdYlGn
cmap.set_bad('w', 1.0)
cax1 = ax1.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
plt.xticks(np.arange(len(size)), size, rotation=45)
ax1.set_yticks(np.arange(len(jobtime)))
ax1.set_yticklabels(jobtime)
ax1.set_xlabel("Job Size / nodes")
ax1.set_ylabel("Run Time / h")
ax1.set_title("Scheduling Coefficient Matrix (numbers are numbers of jobs)")
cbar1 = plt.colorbar(cax1, orientation='vertical', shrink=0.45, aspect=10)
cbar1.set_label('Scheduling Coefficient', rotation=270, labelpad=15)
for i in range(len(jobtime)):
   for j in range(len(size)):
      ax1.text(j, i, str(rjobs[i][j]), horizontalalignment='center', verticalalignment='center', fontsize=7)

fig.savefig(prefix + "_sc_heatmap.png", bbox_inches='tight', dpi=300)

fig.clf()
ax3 = plt.subplot(1, 1, 1)
masked_array = np.ma.masked_where(xusage==-1.0, xusage)
maxu = masked_array.max()
cmap = cm.hot_r
cmap.set_bad('w', 1.0)
cax3 = ax3.imshow(masked_array, interpolation='nearest', cmap=cmap)
plt.xticks(np.arange(len(size)), size, rotation=45)
ax3.set_yticks(np.arange(len(jobtime)))
ax3.set_yticklabels(jobtime)
ax3.set_xlabel("Job Size / nodes")
ax3.set_ylabel("Run Time / h")
ax3.set_title("Usage Matrix\n(Colours indicate CU usage and boxes contain number of jobs)")
cbar3 = plt.colorbar(cax3, orientation='vertical', shrink=0.45, aspect=10)
cbar3.set_label('Usage / CU', rotation=270, labelpad=15)
for i in range(len(jobtime)):
   for j in range(len(size)):
      if xusage[i][j]/maxu > 0.9:
         ax3.text(j, i, str(rjobs[i][j]), horizontalalignment='center', verticalalignment='center', fontsize=7, color='white')
      else:
         ax3.text(j, i, str(rjobs[i][j]), horizontalalignment='center', verticalalignment='center', fontsize=7)

fig.savefig(prefix + "_usage_heatmap.png", bbox_inches='tight', dpi=300)

