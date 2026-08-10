.PHONY: test lint gpu-check selftest metrics-selftest t0 gates

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
