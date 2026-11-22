# Derived Pro targets

`pro.json` contains 731 records: `instance_id`, `n_resolved` (successes), `n_valid` (True/False), `n_errors`, `n_missing`, and `n_reports`. The target success rate is `n_resolved / n_valid`. There are 8,887 valid outcomes and three unknown outcomes excluded from denominators.

These are the study's aggregates over 13 source reports, not duplicated raw reports. Source URLs/hashes and the four unresolved upstream mappings are in `../sources.json`. `scripts/aggregate_pro_targets.py` reconstructs the same targets when all original reports are available.

`pro-complete-traceable.json` uses the same fields for the 381 tasks with valid
outcomes in all 13 reports, but aggregates only the nine reports with recovered
upstream configurations. Every row has nine valid outcomes. This holds task
IDs and Judge predictions fixed while changing the report population. The task
subset is defined by `pro.json`; it is not selected again using nine-report
coverage. Rebuild this file with `aggregate_pro_targets.py --traceable-only`
and the nine downloaded, hash-matched reports.
