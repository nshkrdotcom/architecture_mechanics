.PHONY: test lint gpu-check selftest metrics-selftest t0 gates r0 r1 r2 index

# The pre-registration every ladder run below is a child of. A recorded run
# must name one: bin/check_prereg.sh refuses a manifest without it, and the
# runner refuses to write a run directory without it.
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

# T0 end to end with no model anywhere: generate, apply each reference
# predictor, compute every metric, print the table.
t0:
	uv run python -m architecture_mechanics.metrics.capability --t0

gates: selftest metrics-selftest

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
