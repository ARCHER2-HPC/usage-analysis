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
 -k              Keeps the intermediate output from sacct
 -h              Show this help
 -S date/time    Start date/time as YYYY-MM-DDTHH:MM, e.g. 2021-02-01T00:00
 -u user         Limit to specific user
```

### Output

SCUA prints the usage statistics to Markdown-formatted tables on STDOUT and produces
two CSV files with the statistics:

- `stats_by_usage.csv`: Code usage statistics weighted by CU usage per job
- `stats_by_jobs.csv`: Code usage statistics based on per job size

The repository also contains output from SCUA on a monthly basis for ARCHER2, see:

- [Historical code use data from ARCHER2](historical-data/archer2-code-use)
