.PHONY: test lint gpu-check selftest

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
