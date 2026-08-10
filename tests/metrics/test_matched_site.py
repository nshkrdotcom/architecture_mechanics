"""The matched-site baseline: §6.4's seventh family, §7.4's standing requirement.

"Reporting probe AUC without matched-site baselines" is on §7.4's list of things
to avoid, and the reason is that a mechanism-specific variable being decodable
is almost never the interesting fact — an ordinary residual stream of the same
width at the same depth usually is too. The comparison only means something if
the baseline differs in exactly one respect, so this file checks that the utility
matches depth and dimension, records what it had to adjust, refuses what it
cannot match, and produces a record that has no way to omit the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from architecture_mechanics.metrics.geometry import (
    GeometryError,
    matched_site_baseline,
    matched_site_comparison,
    probe_split,
    site_depth,
)


@pytest.fixture
def states():
    rng = np.random.default_rng(20260810)
    rows = 512
    return {
        "embed": rng.standard_normal((rows, 24)),
        "layers.0.mix.readout": rng.standard_normal((rows, 24)),
        "layers.0.mix.memory": rng.standard_normal((rows, 40)),
        "layers.0.resid_mid": rng.standard_normal((rows, 24)),
        "layers.0.resid_out": rng.standard_normal((rows, 24)),
        "layers.1.mix.readout": rng.standard_normal((rows, 24)),
        "layers.1.resid_mid": rng.standard_normal((rows, 24)),
        "final_norm": rng.standard_normal((rows, 24)),
    }


def test_site_depth_reads_the_block_index():
    assert site_depth("layers.0.mix.readout") == 0
    assert site_depth("layers.11.resid_out") == 11
    assert site_depth("embed") is None
    assert site_depth("final_norm") is None
    assert site_depth("layers.x.resid_out") is None


def test_baseline_is_the_ordinary_stream_at_the_same_depth(states):
    matched = matched_site_baseline(states, "layers.1.mix.readout")
    assert matched.baseline_site == "layers.1.resid_mid"
    assert matched.depth == 1
    assert matched.dim == 24
    assert matched.adjustments == ()
    assert matched.projection_seed is None
    assert matched.candidate.shape == matched.baseline.shape


def test_a_wider_mechanism_site_is_reduced_by_a_recorded_random_projection(states):
    """A2's memory will be wider than the stream. PCA would hand one side its own
    best subspace; a seeded random projection is neutral by being uninformed."""
    matched = matched_site_baseline(states, "layers.0.mix.memory", projection_seed=7)
    assert matched.dim == 24
    assert matched.candidate.shape == (512, 24)
    assert matched.baseline.shape == (512, 24)
    assert matched.projection_seed == 7
    assert any("randomly projected 40 -> 24" in note for note in matched.adjustments)

    again = matched_site_baseline(states, "layers.0.mix.memory", projection_seed=7)
    assert again.candidate == pytest.approx(matched.candidate)
    different = matched_site_baseline(states, "layers.0.mix.memory", projection_seed=8)
    assert not np.allclose(different.candidate, matched.candidate)


def test_a_site_outside_the_block_stack_has_no_depth_to_match(states):
    for site in ("embed", "final_norm"):
        with pytest.raises(GeometryError, match="no depth to match"):
            matched_site_baseline(states, site)


def test_an_explicit_baseline_at_another_depth_is_refused(states):
    with pytest.raises(GeometryError, match="match on depth"):
        matched_site_baseline(
            states, "layers.1.mix.readout", baseline_site="layers.0.resid_mid"
        )


def test_a_depth_with_no_ordinary_state_captured_is_refused(states):
    trimmed = {k: v for k, v in states.items() if k != "layers.1.resid_mid"}
    with pytest.raises(GeometryError, match="no ordinary hidden state"):
        matched_site_baseline(trimmed, "layers.1.mix.readout")


def test_an_uncaptured_site_is_refused(states):
    with pytest.raises(GeometryError, match="no captured site"):
        matched_site_baseline(states, "layers.3.mix.readout")
    with pytest.raises(GeometryError, match="no captured site"):
        matched_site_baseline(states, "layers.1.mix.readout", baseline_site="layers.1.nope")


def test_rows_must_describe_the_same_positions(states):
    broken = dict(states)
    broken["layers.1.mix.readout"] = np.zeros((128, 24))
    with pytest.raises(GeometryError, match="same positions"):
        matched_site_baseline(broken, "layers.1.mix.readout")


def test_the_comparison_record_cannot_report_a_candidate_without_its_baseline(states):
    rng = np.random.default_rng(3)
    features = np.where(rng.random((512, 6)) < 0.3, rng.random((512, 6)), 0.0)
    split = probe_split(np.repeat(np.arange(64), 8), seed=4)

    comparison = matched_site_comparison(
        matched_site_baseline(states, "layers.1.mix.readout"), features, split
    )
    record = comparison.as_dict()
    assert set(record) == {"sites", "candidate", "baseline", "difference", "representation_similarity"}
    assert record["candidate"]["site"] == "layers.1.mix.readout"
    assert record["baseline"]["site"] == "layers.1.resid_mid"
    assert record["difference"]["probe_macro_r2"] == pytest.approx(
        record["candidate"]["probe_macro_r2"] - record["baseline"]["probe_macro_r2"]
    )
    # No accessor returns one side alone.
    assert not any(
        name for name in dir(comparison)
        if name.startswith("candidate_") and not name.startswith("_")
    )


def test_the_same_protocol_runs_on_both_sides(states):
    """Both halves are measured with one split and one estimator, so a difference
    cannot come from a difference in how they were measured."""
    rng = np.random.default_rng(5)
    features = np.where(rng.random((512, 6)) < 0.3, rng.random((512, 6)), 0.0)
    split = probe_split(np.repeat(np.arange(64), 8), seed=6)
    comparison = matched_site_comparison(
        matched_site_baseline(states, "layers.0.mix.readout"), features, split
    )
    assert comparison.candidate.split is comparison.baseline.split
    assert comparison.candidate.n_rows == comparison.baseline.n_rows
    assert comparison.candidate.d_model == comparison.baseline.d_model
    # Two independent random matrices: neither decodes the features, and the two
    # are unrelated to each other.
    assert comparison.candidate.probe.macro_r2 < 0.1
    assert comparison.baseline.probe.macro_r2 < 0.1
    assert comparison.similarity < 0.1
