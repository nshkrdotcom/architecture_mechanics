"""A test suite nobody has seen fail is not known to work.

This file was first committed with `_EXPECTED_TO_MATCH = False`, so the
assertion below was deliberately false; `uv run pytest tests -q` was run once to
confirm the harness collected it, executed it, and reported a failure with a
readable diff. Prompt 01's artifact records that output. It is now correct.
"""

from __future__ import annotations

from architecture_mechanics.seeding import seed_everything

_EXPECTED_TO_MATCH = True


def test_harness_detects_a_false_assertion():
    record = seed_everything(20260809)
    assert (record.seed == 20260809) is _EXPECTED_TO_MATCH
