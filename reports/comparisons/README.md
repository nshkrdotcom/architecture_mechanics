# reports/comparisons

Two kinds of file, and the difference matters.

**`planned/<name>-<rung>-<strategy>.json`** — the declared comparison, committed
*before* its runs exist. It names the control and candidate architectures, the
frozen configuration of every arm in full, the seed set, the matching strategy,
`permitted_differences`, and the claim whose `primary_metric_key` will decide it.
Its arms are named `planned:…` rather than by run ID, because §8.3 derives a run
ID from a digest of the config *and the source tree*, so the ID of a run that has
not happened is not yet defined. Written by
`experiments/comparison.py --declare`, regenerable, and held to the source tree
by `tests/experiments/test_comparison_harness.py`.

**`<name>-<rung>-<strategy>-<cell>-s<seed>.json`** — the resolved declaration,
written after the runs by the same module. This is the file
`bin/check_no_rescue.sh` reads: one control run, one candidate run, at the same
seed and the same cell, with every configuration difference between them either
absent or declared with a justification. Plans are in a subdirectory precisely
because the gate globs `*.json` here non-recursively and must find only files it
can check.

Required in a resolved declaration: `claim`, `control_run`, `candidate_runs`,
`matching_strategy`, `permitted_differences`. Also carried: the parameter
accounting for both §7.2 strategies, the compute ledger, and the checks the
resolver ran.

## The pair is the unit

A declaration names **one** control run and the candidate run matched to it. A
five-seed comparison is five declarations sharing a seed set, not one
declaration naming ten runs: control and candidates take the *same seed set*
rather than the same number of seeds, so the §7.4 test over the arm is the paired
one prompt 08 calibrated — and the gate would rightly call a control at one seed
read against a candidate at another an undeclared difference.

## The metric is not declared here

It is read from the `primary_metric_key` of the claim packet named by `claim`,
whose commit time `bin/check_prereg.sh` already compares against the run's
`started_utc`. A plan may echo it, and the echo is checked: a declaration naming
a different metric than its packet is refused rather than reconciled, because
§7.4's "predeclared primary comparison" only means something if the metric cannot
be chosen after the numbers are in.

## Running one

```bash
uv run python -m architecture_mechanics.experiments.comparison --declare a0_vs_a1
uv run python -m architecture_mechanics.experiments.comparison --check
uv run python -m architecture_mechanics.experiments.runner \
  --comparison a0_vs_a1 --ladder R3 --dry-run
uv run python -m architecture_mechanics.experiments.runner \
  --comparison a0_vs_a1 --ladder R3 --emit-bundle
```

`--comparison` takes the architectures, seeds, width, budget, task and claim from
the plan and refuses a command line that also names any of them: a second place
to set a §7.2 frozen variable is what this object exists to remove. An
undeclared difference between the arms is refused before the first model is
built, because a comparison that only fails at the gate has already spent the
GPU time.

`architecture_mechanics.metrics.statistics.primary_comparison` takes a path to a
resolved declaration and has no `metric` parameter at all. Format first written
by prompt 08, produced by prompt 12.
