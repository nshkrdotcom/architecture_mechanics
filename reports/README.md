# reports

Tables, figures, and the matched-comparison declarations in
`comparisons/*.json` that `bin/check_no_rescue.sh` reads. Generated only from
recorded artifacts under `runs/`, never from a live model.

One exception, and it is the stricter kind: `*_calibration.json` records how a
measuring instrument scores against known answers — the generator plus the
oracle, chance, and marginal reference predictors. It has no parent run because
no model exists in it at all. Regenerate with
`make metrics-selftest` / `python -m architecture_mechanics.metrics.capability
--calibrate --json <path>`.
