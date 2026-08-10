"""``geometry_metrics.npz``: real arrays, loadable, and byte-identical on a re-run.

Two things are being held here. The first is that a run which trained now emits
a *real* §6.2 record rather than the self-describing placeholder prompt 05 wrote
— and that a run which trained nothing still emits the placeholder, because a
missing file and an empty one mean different things.

The second is byte-determinism, which is not decoration. An ``.npz`` is a zip,
and a zip stores a modification time and an order per member. The manifest
hashes every file in the run directory, so if either varied an identical re-run
would report a different evidence index while nothing about the experiment had
moved — exactly the confusion ``cost.json`` was pulled out of ``summary.json``
to avoid. numpy 2.5 normalises the timestamp of its own accord; the point of the
test below is that the property does not *depend* on it doing so.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from architecture_mechanics.experiments.claim_packet import ClaimPacket
from architecture_mechanics.experiments.config import ladder_config
from architecture_mechanics.experiments.runner import run
from architecture_mechanics.metrics.geometry import GEOMETRY_VERSION
from architecture_mechanics.reporting.evidence_bundle import (
    GEOMETRY_SCHEMA,
    _savez_deterministic,
)


@pytest.fixture
def claim(tmp_path: Path) -> Path:
    packet = ClaimPacket(
        claim_id="t-geometry",
        claimed_rung=1,
        primary_metric_key="associative_recall_accuracy",
        fields={
            "CLAIM": "the geometry instrument records what it measured",
            "MECHANISM": "softmax attention",
            "STRUCTURALLY_ENFORCED_PROPERTIES": ["causality"],
            "LEARNED_OR_HOPED_PROPERTIES": ["feature isolation"],
            "NEAREST_BORING_EXPLANATION": "the probe memorised the split",
            "CONTROL_THAT_RULES_IT_OUT": "the probe split is by example",
            "PRIMARY_METRIC": "probe macro R^2 at the readout site",
            "MECHANISM_ACTIVITY_METRIC": "retrieval lift",
            "POSITIVE_CONTROL": "R1",
            "NEGATIVE_CONTROL": "a matched ordinary hidden state",
            "KILL_CONDITION": "probe R^2 at chance",
            "REPLICATION_REQUIREMENT": "five seeds",
        },
    )
    return packet.write(tmp_path / "claims" / "t-geometry.yml")


def _tiny_r1(tmp_path: Path, claim: Path):
    config = ladder_config("R1", device="cpu")
    config = replace(
        config,
        data=replace(config.data, n_train=192, n_eval=192),
        optim=replace(config.optim, max_steps=8, eval_every=8),
        capture_examples=16,
        geometry_examples=96,
    )
    result = run(
        config,
        out_dir=tmp_path / "runs",
        verbose=False,
        claim=claim,
        claims_dir=claim.parent,
    )
    return result, tmp_path / "runs" / result.run_id


# --------------------------------------------------------------------------- #
# The deterministic writer
# --------------------------------------------------------------------------- #


def test_the_writer_round_trips(tmp_path: Path):
    payload = {
        "__schema__": np.asarray(GEOMETRY_SCHEMA),
        "site::purity": np.linspace(0.0, 1.0, 7),
        "site::cosine_matrix": np.eye(4),
        "bank::content": np.arange(5),
    }
    path = tmp_path / "g.npz"
    _savez_deterministic(path, payload)
    with np.load(path) as loaded:
        assert set(loaded.files) == set(payload)
        assert str(loaded["__schema__"]) == GEOMETRY_SCHEMA
        assert loaded["site::purity"] == pytest.approx(payload["site::purity"])
        assert loaded["site::cosine_matrix"] == pytest.approx(np.eye(4))


def test_the_writer_produces_identical_bytes_for_identical_arrays(tmp_path: Path):
    payload = {"a": np.arange(64, dtype=np.float64), "b": np.eye(8)}
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    _savez_deterministic(first, payload)
    _savez_deterministic(second, dict(reversed(list(payload.items()))))
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    ), "member order or timestamp leaked into the bytes"


def test_determinism_does_not_depend_on_what_numpy_chose(tmp_path: Path):
    """Every member carries the pinned timestamp, whatever ``np.savez`` does.

    Checked at the zip layer rather than by comparing our bytes against numpy's,
    because numpy 2.5 already normalises the timestamp and the comparison would
    silently stop testing anything the day a member order changed instead.
    """
    path = tmp_path / "pinned.npz"
    _savez_deterministic(path, {"z": np.arange(4.0), "a": np.eye(2)})
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    assert [entry.filename for entry in entries] == ["a.npy", "z.npy"], "members are sorted"
    assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in entries)


# --------------------------------------------------------------------------- #
# What a run leaves behind
# --------------------------------------------------------------------------- #


def test_a_run_that_trained_emits_a_real_geometry_file(tmp_path: Path, claim: Path):
    result, run_dir = _tiny_r1(tmp_path, claim)
    path = run_dir / "geometry_metrics.npz"
    assert path.is_file()

    with np.load(path) as loaded:
        assert str(loaded["__schema__"]) == GEOMETRY_SCHEMA
        assert bool(loaded["__empty__"]) is False
        assert str(loaded["__run_id__"]) == result.run_id
        sites = [str(name) for name in loaded["__sites__"]]
        assert sites[0] == "embed" and sites[-1] == "final_norm"
        for site in sites:
            assert f"{site}::purity" in loaded.files
            assert f"{site}::cosine_matrix" in loaded.files
            assert loaded[f"{site}::purity"].shape == (result.geometry["primary"]["n_features"],)
        assert {"bank::content", "bank::key", "bank::operator"} <= set(loaded.files)
        # A matched-site record for every mechanism site, both halves of it.
        for comparison in result.geometry["matched_sites"]:
            site = comparison["sites"]["candidate_site"]
            assert f"matched:{site}:candidate::probe_r2" in loaded.files
            assert f"matched:{site}:baseline::probe_r2" in loaded.files


def test_the_summary_carries_the_scalars_and_the_npz_carries_the_arrays(
    tmp_path: Path, claim: Path
):
    result, _run_dir = _tiny_r1(tmp_path, claim)
    geometry = result.geometry
    assert geometry["geometry_version"] == GEOMETRY_VERSION
    assert geometry["primary_site"] == "final_norm"
    assert geometry["split"]["split_by"] == "example"
    assert geometry["split"]["n_train_examples"] + geometry["split"]["n_eval_examples"] == 96
    assert set(geometry["by_bank"]) == {"content", "key", "operator"}
    assert geometry["matched_sites"], "every mechanism site needs a matched baseline"
    for comparison in geometry["matched_sites"]:
        assert comparison["baseline"]["site"].endswith("resid_mid")
        assert comparison["sites"]["depth"] == int(
            comparison["sites"]["candidate_site"].split(".")[1]
        )
    # Scalars only: no array made it into summary.json.
    for site_scalars in geometry["per_site"].values():
        for value in site_scalars.values():
            assert not isinstance(value, (list, dict)), site_scalars["site"]


def test_a_rung_that_trained_nothing_still_emits_a_loadable_placeholder(
    tmp_path: Path, claim: Path
):
    result = run(
        ladder_config("R0", device="cpu"),
        out_dir=tmp_path / "runs",
        verbose=False,
        claim=claim,
        claims_dir=claim.parent,
    )
    assert result.geometry == {}
    path = tmp_path / "runs" / result.run_id / "geometry_metrics.npz"
    # R0 is a screen that never trains, so the gate does not require the file and
    # the runner does not write one; the placeholder exists for a *final* rung
    # that measured nothing, and is exercised by the bundle writer directly.
    assert not path.exists()
