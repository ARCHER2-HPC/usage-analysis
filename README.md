# Usage Analysis Tools

This repository contains usage analysis tools used on the
[UK National Supercomputing Service, ARCHER2](https://www.archer2.ac.uk) along
with historical data from the analysis of service usage. Information on the 
individual tools is provided below.

## Slurm Code Usage Analysis, SCUA

The SCUA tool queries the Slurm accounting database, extracts data on job steps 
and uses this data to match the executable names to a known set of applications.
Once it has done this, it analyses the data and can produce output on various
facets of usage.

You can use the `scua` command on a system with Slurm available to extract the 
data from the Slurm database and run the analyses or you can use the `scua.py`
command to analyse a data file that contains a dump from the Slurm database in
the correct format - this option is useful if you want to run multiple, separate
analyses as it runs much quicker (extracting data from the Slurm database is
typically the most time consuming step) and you can move the data dump to a different
system to perform the analysis.

Note: at the moment, if graphical graphical plots are requested, they are only
produced for analysis broken down by software use. Tables of data (as CSV and/or
markdown) are available for all analysis breakdowns.

### Requirements

`scua` is a bash script and so has no requirements other than it must be run on 
the system where the Slurm `sacct` command is available.

`scua.py` uses Pandas, numpy, matplotlib and seaborn on top of a standard Python
3 installation.

### Analyses available

The type of analysis you wish to run on the data is specified by command line 
options (see *Usage*, below). The following analysis types are available:

| Analysis | `scua` Argument Combination | `scua.py` Argument Combination | Description |
|----------|-----------------------------|--------------------------------|-------------|
| Software + Size | (default, always performed) | (default, always performed) | Analyses usage by parallel job step size and software package |
| Software + Node Power | `-w` | `--power` | Analyses job step mean node power draw and software package. |
| Motif + Size | `-t` | `--motif` | Analyses usage by parallel job step size and computational motif. |
| Motif + Power | `-t -w` | `--motif --power` | Analyses job step mean node power draw and computational motif. |
| Area + Size | `-a <project list CSV>` | `--projects=<project list CSV>` | Analyses usage by parallel job step size and research area. Requires a CSV file linking account codes to research areas. |
| Area + Power | `-a <project list CSV> -w` | `--projects=<project list CSV> --power` | Analyses job step mean node power draw and research area. Requires a CSV file linking account codes to research areas. |

### `scua` Usage

```
Usage: scua [options]
Options:
 -a account_csv  Perform analysis by research area. account_csv is a CSV file with a mapping of account codes to research areas
 -A account      Limit to specified account code, e.g. z01
 -c              Save tables of data (as csv)
 -E date/time    End date/time as YYYY-MM-DDTHH:MM, e.g. 2021-02-01T00:00
 -k              Keeps the intermediate output from sacct in `scua_sacct.csv`
 -g              Produce graphs of usage data (as png)
 -h              Show this help
 -m              Save tables of data (as markdown)
 -p prefix       Prefix for output file names
 -S date/time    Start date/time as YYYY-MM-DDTHH:MM, e.g. 2021-02-01T00:00
 -t              Perform analysis by computational motif
 -u user         Limit to specific user
 -w              Perform analysis of mean node power draw
```

### `scua.py` Usage

```
Usage: scua.py [options] input
Options:
 -projects=account_csv  Perform analysis by research area. account_csv is a CSV file with a mapping of account codes to research areas
 -A account             Limit to specified account code, e.g. z01
 --csv                  Save tables of data (as csv)
 --plots                Produce graphs of usage data (as png)
 --help                 Show this help
 --md                   Save tables of data (as markdown)
 --prefix=prefix        Prefix for output file names
 --motif                Perform analysis by computational motif
 --power                Perform analysis of mean node power draw
 --dropnan              Drop all rows that contain NaN. Useful for strict comparisons between usage and energy use as some job steps may be missing energy use data
```

### Output

SCUA prints the usage statistics to Markdown-formatted tables on STDOUT. If you specify
the `-c` option, it will save the same tables as CSV files and if you specify the `-m` option
it will save the same tables as markdown files. All files will be prefixed with the specified
prefix or `scua` if no prefix is supplied.

If you specify the `-g` option (to produce graphs) you will also obtain three additional
image files:

- `${prefix}_codes_usage.png`: Bar chart of CU usage broken down by code
- `${prefix}_overall_boxplot.png`: Boxplot representing overall job size statistics
  weighted by CU usage per job
- `${prefix}_top15_boxplot.png`: Boxplot representing job size statistics for the top 15
  codes by usage weighted by CU usage per job
- `${prefix}_node_power_distribution.png`: Histogram of mean node power draw
