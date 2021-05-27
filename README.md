# Usage Analysis Tools

This repository contains usage analysis tools used on the
[UK National Supercomputing Service, ARCHER2](https://www.archer2.ac.uk) along
with historical data from the analysis of service usage. Information on the 
individual tools is provided below.

## Slurm Code Usage Analysis, SCUA

The SCUA tool queries the Slurm accounting database, extracts data on subjobs 
and uses this data to match the executable names to a known set of applications.
Once it has done this, it analyses the use of each application and outputs
statistics on overall use by particular applications and the job sizes associated
with different applications.

### Usage

```
Usage: scua [options]
Options:
 -A account      Limit to specified account code, e.g. z01
 -E date/time    End date/time as YYYY-MM-DDTHH:MM, e.g. 2021-02-01T00:00
 -k              Keeps the intermediate output from sacct in: scua_sacct.csv
 -g              Produce graphs of usage data (as png)
 -h              Show this help
 -p prefix       Prefix for output file names
 -S date/time    Start date/time as YYYY-MM-DDTHH:MM, e.g. 2021-02-01T00:00
 -u user         Limit to specific user
```

### Output

SCUA prints the usage statistics to Markdown-formatted tables on STDOUT and produces
two CSV files with the statistics:

- `${prefix}_stats_by_usage.csv`: Code usage statistics weighted by CU usage per job
- `${prefix}stats_by_jobs.csv`: Code usage statistics based on per job size

If you specify the `-g` option (to produce graphs) you will also obtain three additional
image files:

- `${prefix}_codes_usage.png`: Bar chart of CU usage broken down by code
- `${prefix}_overall_boxplot.png`: Boxplot representing overall job size statistics
  weighted by CU usage per job
- `${prefix}_top15_boxplot.png`: Boxplot representing job size statistics for the top 15
  codes by usage weighted by CU usage per job
