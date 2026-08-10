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
measuring instrument scores against known answers, and has no parent run because
no model exists in it at all.

- `t0_capability_calibration.json` — the §6.1 metrics against the generator plus
  the oracle, chance, and marginal reference predictors. Regenerate with
  `make metrics-selftest` / `python -m architecture_mechanics.metrics.capability
  --calibrate --json <path>`.
- `statistics_calibration.json` — the §7.4 estimators against two known truths:
  a null with no effect at all, and an injected effect of declared size. Records
  the empirical false-positive rate of every estimator at three, five and ten
  seeds under three noise shapes, the power curve, and the minimum detectable
  effect at five seeds — the number that tells every later mission whether a null
  result means "no effect" or "underpowered". Regenerate with
  `make statistics-calibration`.
- `geometry_calibration.json` — the §6.2 measures against five *constructed*
  representations whose geometry is known before it is measured: an orthogonal
  basis, a known superposition, a random rotation of the first, a degenerate
  collapse, and pure noise. The noise column is the one that decides which
  measures can carry a claim alone. Regenerate with `make geometry-table`.

`geometry_across_seeds.json` is an ordinary derived report: how far A0's
representation geometry moves between initialisation seeds on identical data,
which is the reference every later "architecture X differs from architecture Y"
statement has to be read against. Regenerate with `make geometry-across-seeds`.
