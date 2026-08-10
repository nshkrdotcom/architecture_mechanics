"""Figure 2: the marks that stop a one-seed map being read as a result.

Everything here is driven by a synthetic ``phase_report``-shaped record rather
than by the recorded sweep, so each rule is exercised against inputs chosen to
sit on the wrong side of it. A test that only ever saw the real sweep would pass
for a figure that drew no marks at all, because the real sweep might happen to
contain no saturated cell and no disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from architecture_mechanics.reporting import figure_style as style
from architecture_mechanics.reporting import figures

WINDOW = {"floor": 0.05, "ceiling": 0.95}


def _arm(recall: float, purity: float, *, active: bool = True) -> dict:
    return {
        "run_id": f"R2-fake-{recall:.4f}-{purity:.4f}",
        "arch": "softmax",
        "d_model": 32,
        "parameters": 1,
        "recall": recall,
        "feature_f1": 0.5,
        "reconstruction_loss": 0.1,
        "content_purity": purity,
        "content_probe_r2": 0.1,
        "content_features": 64,
        "all_bank_purity": purity,
        "interference_fraction": 0.13,
        "probe_macro_r2": 0.09,
        "effective_rank": 40.0,
        "participation_ratio": 20.0,
        "mechanism_active": active,
        "mechanism_reasons": [],
        "kill_fired": [],
        "recall_skill": 0.4,
        "final_step_gain": 0.01,
        "still_rising": True,
    }


def _saturation(recall: float) -> str:
    if recall <= WINDOW["floor"]:
        return "at_chance"
    if recall >= WINDOW["ceiling"]:
        return "at_ceiling"
    return "interior"


def _point(*, f_content, d_model, activation_prob, a0, a1, p0, p1, active=True) -> dict:
    control, candidate = _arm(a0, p0, active=active), _arm(a1, p1, active=active)
    return {
        "comparison": f"phase_T32_d{d_model}",
        "ladder": "R2",
        "cell": f"phase-F{f_content}-T32-p{activation_prob:.2f}".replace(".", ""),
        "seed": 20260809,
        "condition": "capacity_stressed",
        "d_model": d_model,
        "f_content": f_content,
        "f_total": f_content + 28,
        "seq_len": 32,
        "activation_prob": activation_prob,
        "expected_active_content_features": round(16 * activation_prob, 3),
        "ratio_content": f_content / d_model,
        "ratio_total": (f_content + 28) / d_model,
        "control": control,
        "candidate": candidate,
        "difference": {
            "recall": a0 - a1,
            "content_purity": p0 - p1,
            "all_bank_purity": p0 - p1,
            "interference_fraction": 0.0,
            "probe_macro_r2": 0.0,
        },
        "saturation": {"control": _saturation(a0), "candidate": _saturation(a1)},
        "both_alive": min(a0, a1) > WINDOW["floor"],
        "both_mechanisms_active": active,
        "compute_ledger": {},
        "checks": {},
        "permitted_differences": {},
    }


@pytest.fixture
def report() -> dict:
    """Four (F, d) points by two sparsities, built to contain every mark.

    - ``(32, 64)`` ratio 0.5, sparsest: A0 at the ceiling.
    - ``(64, 64)`` and ``(32, 32)`` both ratio 1: the duplicate the figure draws
      as adjacent rows.
    - ``(64, 16)`` ratio 4, densest: both arms at chance, and one inert.
    - one cell where the recall difference is below what five seeds resolve while
      the purity difference is above it — the disagreement the figure outlines.
    """
    points = [
        _point(f_content=32, d_model=64, activation_prob=0.06, a0=0.97, a1=0.80,
               p0=0.24, p1=0.23),
        _point(f_content=32, d_model=64, activation_prob=0.40, a0=0.60, a1=0.55,
               p0=0.22, p1=0.10),  # disagreement: |dr| < 0.128, |dp| > 0.049
        _point(f_content=64, d_model=64, activation_prob=0.06, a0=0.70, a1=0.40,
               p0=0.20, p1=0.18),
        _point(f_content=64, d_model=64, activation_prob=0.40, a0=0.30, a1=0.10,
               p0=0.18, p1=0.16),
        _point(f_content=32, d_model=32, activation_prob=0.06, a0=0.50, a1=0.30,
               p0=0.19, p1=0.17),
        _point(f_content=32, d_model=32, activation_prob=0.40, a0=0.20, a1=0.24,
               p0=0.17, p1=0.16),  # A1 ahead on recall
        _point(f_content=64, d_model=16, activation_prob=0.06, a0=0.10, a1=0.02,
               p0=0.15, p1=0.14),
        _point(f_content=64, d_model=16, activation_prob=0.40, a0=0.01, a1=0.00,
               p0=0.14, p1=0.13, active=False),
    ]
    return {
        "schema": "am.phase_map.v1",
        "ladder": "R2",
        "main_seq_len": 32,
        "sparsities": [0.06, 0.40],
        "width_feature_points": [[32, 64], [32, 32], [64, 64], [64, 16]],
        "content_group_size": 16,
        "window": dict(WINDOW),
        "resolution": {"n_seeds": 1, "smallest_resolvable_recall_difference": 0.128},
        "cuts": [{"cut": "sequence length", "kept": "T=32", "cost_if_kept": "x",
                  "why_this_axis": "y"}],
        "cost_model": {},
        "controls": {
            "phase_r1": [
                _point(f_content=64, d_model=48, activation_prob=0.20, a0=0.9055,
                       a1=0.8954, p0=0.30, p1=0.29)
            ],
            "phase_negative_control_d32": [
                _point(f_content=64, d_model=32, activation_prob=0.12, a0=0.0,
                       a1=0.0, p0=0.20, p1=0.19)
            ],
            "phase_length_d32": [
                _point(f_content=64, d_model=32, activation_prob=0.12, a0=0.4,
                       a1=0.2, p0=0.2, p1=0.18)
            ],
        },
        "points": points,
    }


def test_rows_are_ordered_by_bottleneck_ratio_with_the_duplicates_adjacent(report):
    rows, columns = figures._grid_axes(report)
    ratios = [row["ratio_content"] for row in rows]
    assert ratios == sorted(ratios)
    assert ratios == [0.5, 1.0, 1.0, 4.0]
    # Within one ratio the wider model comes first, so the pair that tests
    # "does the ratio alone decide it" is two neighbouring rows.
    assert [row["d_model"] for row in rows] == [64, 64, 32, 16]
    assert columns == [0.06, 0.40]


def test_the_disagreement_rule_is_both_conditions_and_not_either(report):
    rows, columns = figures._grid_axes(report)
    marks = figures._surface(report, rows, columns)["marks"]
    assert len(marks["disagreement"]) == 1
    row, column = marks["disagreement"][0]
    assert rows[row]["ratio_content"] == 0.5
    assert columns[column] == 0.40

    # A cell with a large recall difference and a large purity difference is not
    # a disagreement, and neither is a tied cell whose purity is also tied.
    outlined = {
        (rows[i]["f_content"], rows[i]["d_model"], columns[j])
        for i, j in marks["disagreement"]
    }
    for point in report["points"]:
        difference = point["difference"]
        tied = abs(difference["recall"]) < figures.RECALL_FIVE_SEED_MDE
        split = abs(difference["content_purity"]) > figures.PURITY_FIVE_SEED_MDE
        key = (point["f_content"], point["d_model"], point["activation_prob"])
        assert (tied and split) == (key in outlined), point["cell"]


def test_saturation_is_marked_per_arm_and_shaded_on_the_difference(report):
    rows, columns = figures._grid_axes(report)
    marks = figures._surface(report, rows, columns)["marks"]
    assert len(marks["control_at_ceiling"]) == 1
    assert marks["candidate_at_ceiling"] == []
    assert len(marks["control_at_chance"]) == 1
    assert len(marks["candidate_at_chance"]) == 2
    # Every cell where either arm is saturated is shaded on the difference
    # panels, because a difference between two saturated numbers is not a
    # difference between two architectures.
    assert len(marks["either_saturated"]) == 3


def test_a_cell_where_the_candidate_leads_is_marked_with_a_sign(report):
    rows, columns = figures._grid_axes(report)
    marks = figures._surface(report, rows, columns)["marks"]
    assert len(marks["candidate_ahead_recall"]) == 1
    assert marks["candidate_ahead_purity"] == []


def test_an_inert_mechanism_is_marked_rather_than_dropped(report):
    rows, columns = figures._grid_axes(report)
    marks = figures._surface(report, rows, columns)["marks"]
    assert len(marks["inert"]) == 1


def test_a_cell_nobody_ran_is_not_a_zero(report):
    """Two sparsities times four points is eight cells; drop one and the matrix
    must carry ``nan`` there, which the figure draws as its own colour."""
    report["points"] = report["points"][:-1]
    rows, columns = figures._grid_axes(report)
    matrices = figures._surface(report, rows, columns)["matrices"]
    assert np.isnan(matrices["control_recall"]).sum() == 1
    assert np.nanmin(matrices["control_recall"]) > 0.0


def test_the_caption_says_one_seed_and_screening_in_its_first_two_sentences(report):
    caption = figures.figure2_caption(figures.figure2_params(report))
    opening = ". ".join(caption.split(". ")[:2]).upper()
    assert "SCREENING EVIDENCE" in opening
    assert "ONE SEED PER CELL" in opening


def test_the_caption_carries_what_a_reader_needs_to_disbelieve_it(report):
    params = figures.figure2_params(report)
    caption = figures.figure2_caption(params)
    for fragment in (
        "0.128",
        "0.049",
        "final_norm",
        "F/d = 1",
        "claims/phase-map-a0-a1-sparsity-bottleneck.yml",
        "--figure 2",
        str(params["seed"][0]),
        "0.9055",  # the positive control's known answer
    ):
        assert fragment in caption, fragment


def test_the_caption_numbers_are_not_frozen_literals(report):
    """Move the data and the caption must move with it."""
    before = figures.figure2_caption(figures.figure2_params(report))
    for point in report["points"]:
        point["difference"]["content_purity"] = 0.0
        point["difference"]["recall"] = 0.0
    after = figures.figure2_caption(figures.figure2_params(report))
    assert before != after
    assert "none of the cells met both conditions" in after


def test_the_figure_is_page_width_and_drawn_in_ink_only(report, tmp_path):
    """Greyscale safety as a property rather than a claim: every pixel neutral."""
    fig = figures.draw_figure2(report)
    path = tmp_path / "fig2.png"
    style.save_png(fig, path)
    from matplotlib import image as mpimg

    pixels = mpimg.imread(path)
    height, width = pixels.shape[:2]
    assert width == round(style.PAGE_WIDTH_IN * style.SAVE_DPI)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    assert np.array_equal(red, green) and np.array_equal(green, blue)
    assert height > 0


def test_building_figure_two_without_a_recorded_sweep_refuses_rather_than_invents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(figures, "phase_report", lambda: {"points": [], "controls": {}})
    with pytest.raises(ValueError, match="no recorded phase-sweep runs"):
        figures.build_figure2(tmp_path)
