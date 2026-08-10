# reports/comparisons

One JSON per declared comparison. Required: `claim`, `control_run`,
`candidate_runs`. Optional: `matching_strategy`, and `permitted_differences`
with a justification for each — `bin/check_no_rescue.sh` reads those and fails on
any config difference between the runs that is not declared there.

**The metric is not declared here.** It is read from the `primary_metric_key` of
the claim packet named by `claim`, whose commit time `bin/check_prereg.sh`
already compares against the run's `started_utc`. A comparison file may echo it
as `primary_metric`, and the echo is checked: a declaration naming a different
metric than its packet is refused rather than reconciled, because §7.4's
"predeclared primary comparison" only means something if the metric cannot be
chosen after the numbers are in.

`architecture_mechanics.metrics.statistics.primary_comparison` takes a path to
one of these files and has no `metric` parameter at all. First written by
prompt 12.
