# reports

Tables, figures, and the matched-comparison declarations in
`comparisons/*.json` that `bin/check_no_rescue.sh` reads. Generated only from
recorded artifacts under `runs/`, never from a live model.

`figures/` holds the north star 10.2 figures with, beside each PNG, its caption
and an `INDEX.json` recording its sha256 and every input parameter. Regenerate
with `make figures`; the PNG comes back byte-identical, so a diff in `figures/`
means the evidence changed and not the renderer. Figure 1 is the exception in
the same way `*_calibration.json` is: it draws the benchmark itself, from one
example the generator produces on demand, and so has no parent run.

One exception, and it is the stricter kind: `*_calibration.json` records how a
measuring instrument scores against known answers — the generator plus the
oracle, chance, and marginal reference predictors. It has no parent run because
no model exists in it at all. Regenerate with
`make metrics-selftest` / `python -m architecture_mechanics.metrics.capability
--calibrate --json <path>`.
