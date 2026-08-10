.PHONY: test lint gpu-check selftest metrics-selftest geometry-selftest \
        statistics-selftest statistics-calibration t0 gates \
        r0 r1 r2 index figure1 figures geometry-table geometry-across-seeds

# The pre-registration every ladder run below is a child of. A recorded run
# must have one: bin/check_prereg.sh refuses a manifest without it, and the
# runner refuses to write a run directory without it. Named here explicitly;
# omitting --claim resolves the same packet from its committed covers: block.
CLAIM ?= claims/a0-baseline-solves-t0.yml

test:
	uv run pytest tests -q

lint:
	uv run ruff check src tests

gpu-check:
	uv run python -m architecture_mechanics.device

# The data gate: generates all six §4.4 control conditions and asserts the
# invariants that make every downstream measurement meaningful. Run it before
# interpreting anything.
selftest:
	uv run python -m architecture_mechanics.data.feature_program --selftest

# The metric gate: calibrates every §6.1 ruler against the oracle, chance, the
# training marginal, and the frequency ceiling, then re-checks every retirement
# decision recorded in METRIC_SPECS. A retired metric that starts passing fails
# this gate just as loudly as a retained one that stops passing.
metrics-selftest:
	uv run python -m architecture_mechanics.metrics.capability --selftest

# The geometry gate: measures every §6.2 ruler against five constructed
# representations whose answer is known in advance — orthogonal, superposed,
# rotated, collapsed, and pure noise — and re-derives which of them can carry a
# claim alone. A measure whose noise null moves fails this gate.
geometry-selftest:
	uv run python -m architecture_mechanics.metrics.geometry --selftest

# The statistics gate: re-runs both §7.4 calibrations at reduced replicate counts
# and fails if any estimator's false-positive rate has left its recorded
# tolerance — including if one recorded as unusable starts behaving. Twenty
# seconds of CPU; no GPU, no model, no data.
statistics-selftest:
	uv run python -m architecture_mechanics.metrics.statistics --selftest

# The recorded calibration at full replicate counts: 2000 null replicates per
# estimator per noise shape per seed count, the power sweep, and the minimum
# detectable effect. About four minutes of CPU. Its numbers are what
# state/08_statistics.md quotes and what tests/metrics/test_statistics_selftest_gate.py
# checks the register against.
statistics-calibration:
	uv run python -m architecture_mechanics.metrics.statistics --calibrate \
	  --json reports/statistics_calibration.json

# The recorded expected-versus-measured table, regenerated into reports/.
geometry-table:
	uv run python -m architecture_mechanics.metrics.geometry --table \
	  --json reports/geometry_calibration.json

# How much A0 differs from itself: the reference every later "architecture X
# differs from architecture Y" claim has to be read against. Reads the recorded
# summaries and npz files; runs no model.
geometry-across-seeds:
	uv run python -m architecture_mechanics.metrics.geometry \
	  --across-runs $(wildcard runs/R1-softmax-positive_control-*) \
	  --json reports/geometry_across_seeds.json

# T0 end to end with no model anywhere: generate, apply each reference
# predictor, compute every metric, print the table.
t0:
	uv run python -m architecture_mechanics.metrics.capability --t0

gates: selftest metrics-selftest geometry-selftest statistics-selftest

# The section 7.3 run ladder. R0 builds the model and checks the section 8.5
# invariants without training; R1 is the known-easy positive control and exits
# non-zero if A0 does not solve it; R2 is the capacity-stressed kill screen.
#
# --emit-bundle refuses to finish having written an incomplete §8.4 bundle.
r0:
	uv run python -m architecture_mechanics.experiments.runner --ladder R0 \
	  --claim $(CLAIM) --emit-bundle --assert-pass

r1:
	uv run python -m architecture_mechanics.experiments.runner --ladder R1 \
	  --claim $(CLAIM) --emit-bundle --assert-pass

r2:
	uv run python -m architecture_mechanics.experiments.runner --ladder R2 \
	  --claim $(CLAIM) --emit-bundle

# Every run in the laboratory against its claim packet, in one screen. The
# partial defence against selection over experiments; prompt 29 is told to look.
index:
	uv run python -m architecture_mechanics.experiments.index

# The section 10.2 figures, from recorded artifacts only. --verify-deterministic
# deletes the PNG, regenerates it, and exits non-zero unless the bytes match:
# a figure that moves when nothing moved makes review impossible.
figure1:
	uv run python -m architecture_mechanics.reporting.figures --figure 1 \
	  --verify-deterministic

figures: figure1
