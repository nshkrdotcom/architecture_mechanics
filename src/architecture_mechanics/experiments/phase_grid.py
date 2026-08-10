"""The §10.2 figure 2 sweep grid: architecture × sparsity × bottleneck ratio.

§4.5 asks for a map rather than a score — "the first useful result is a map
across difficulty, not a single best score" — over ``F`` in {32, 64, 128}, ``d``
in {16, 32, 64}, sequence length in {32, 64, 128}, a sparsity range, both
architectures, at one screening seed. This module is that grid, as data, and
nothing else: it declares cells and it declares which comparisons cover them.
It runs nothing and reads nothing.

Three definitions this grid rests on, each of which could reasonably have been
made differently and so is written down rather than implied.

**``F`` is the content bank.** A T1 example carries three banks — content, key
and operator — and only the content bank is what the model must reconstruct and
transport. The key bank (24) and the operator bank (4) are addressing machinery
whose width is a property of the *task*, not of the bottleneck, and scaling them
with ``F`` would change ``key_bits``, ``n_keys`` and therefore the task itself at
every row of the map. So ``F`` here moves ``n_content_features`` over §4.5's
three values and the other 28 features are a fixed overhead in every cell. Both
ratios are recorded per cell: :data:`PHASE_CELLS` carries ``f_content`` and
``f_total``, and the figure marks the transition on the content ratio while the
caption gives the total one, because a reader who assumed the other convention
must be able to see which was used.

**Sparsity is held comparable across ``F``.** ``activation_prob`` is a
per-feature probability *within the group a position draws from*, so at a fixed
group count a wider content bank means wider groups and a denser position — and
the sparsity axis would then mean something different at every row of the map,
which is fatal for a diagram whose two axes are sparsity and the bottleneck
ratio. :data:`PHASE_GROUP_SIZE` fixes the group at 16 content features and
``n_content_groups`` scales with ``F`` instead, so the expected number of active
features at a position is ``16 * activation_prob`` in every cell and the answer
set a query must reproduce has the same size distribution everywhere. What
changes with ``F`` is how many distinct features are superposed into ``d``,
which is the thing the map is about.

**The grid was cut before it was run, and the cut is here.** See
:data:`PHASE_CUTS`.
"""

from __future__ import annotations

from architecture_mechanics.experiments.t1_ladder import POSITIVE_CONTROL_CELL, Cell

__all__ = [
    "PHASE_CELLS",
    "PHASE_COMPARISONS",
    "PHASE_CONTENT_FEATURES",
    "PHASE_CUTS",
    "PHASE_FIXED_FEATURES",
    "PHASE_GROUP_SIZE",
    "PHASE_LADDER",
    "PHASE_LENGTH_CELLS",
    "PHASE_MAIN_CELLS",
    "PHASE_MAIN_SEQ_LEN",
    "PHASE_SEQ_LENS",
    "PHASE_SPARSITIES",
    "PHASE_WIDTHS",
    "PHASE_WIDTH_FEATURE_POINTS",
    "cell_axes",
    "phase_cells",
    "phase_cost_model",
]

PHASE_LADDER = "R2"
"""§7.3's kill screen, unmodified. The sweep is screening depth by declaration —
the mission that owns this map is explicitly not a replication — so it takes R2's
committed preset (``capacity_stressed``, 16 384 training examples, 4 096 held
out, 2 000 steps, ``eval_every`` 500) rather than inventing a cheaper rung. The
only things this grid moves are the four generator fields below and ``d_model``,
which the comparison declares."""

PHASE_CONTENT_FEATURES: tuple[int, ...] = (32, 64, 128)
"""§4.5's ``F``, applied to the content bank. See the module docstring."""

PHASE_WIDTHS: tuple[int, ...] = (16, 32, 64)
"""§4.5's ``d``. Declared by the comparison rather than by the cell, because the
model width is an architecture variable and a cell is a dataset."""

PHASE_SPARSITIES: tuple[float, ...] = (0.06, 0.12, 0.24, 0.40)
"""§4.3's sparsity axis at the four levels ``t1_ladder.DIFFICULTY_AXES`` already
declares, so this map and prompts 09 and 13 speak about the same knob at the same
values. At :data:`PHASE_GROUP_SIZE` = 16 the expected active content features at
a position are 0.96, 1.92, 3.84 and 6.40; ``min_active_per_position = 1`` clamps
the first slightly, which is a property of the generator recorded here rather
than a surprise in the figure."""

PHASE_SEQ_LENS: tuple[int, ...] = (32, 64, 128)
"""§4.5's three sequence lengths. Only :data:`PHASE_MAIN_SEQ_LEN` is crossed with
the rest of the grid; the other two appear in :data:`PHASE_LENGTH_CELLS` as a
ribbon at one point of it. See :data:`PHASE_CUTS`."""

PHASE_MAIN_SEQ_LEN = 32
"""The sequence length the map is drawn at.

The cheapest of §4.5's three, and that is why it was chosen: measured cost per
run at 2 000 steps is 0.56× the same cell at ``T = 64``, which is what buys the
map its width in ``F``, ``d`` and sparsity. The cost of the choice is that the
map's transport distances (``distance_buckets`` 5–9 and 10–16, inherited
unchanged from ``capacity_stressed``) occupy a larger fraction of a shorter
sequence than they do in prompt 13's pilots at ``T = 48``."""

PHASE_GROUP_SIZE = 16
"""Content features per group, fixed. See the module docstring."""

PHASE_FIXED_FEATURES = 28
"""The key bank (24) and the operator bank (4), unchanged in every cell.
``f_total = f_content + PHASE_FIXED_FEATURES`` and the figure says so."""

PHASE_WIDTH_FEATURE_POINTS: tuple[tuple[int, int], ...] = (
    (32, 64),
    (32, 32),
    (32, 16),
    (64, 64),
    (64, 32),
    (64, 16),
    (128, 32),
    (128, 16),
)
"""The ``(F, d)`` points of the map, and therefore its bottleneck ratios.

Eight of §4.5's nine, ordered by ``F`` then by decreasing ``d``. The content
ratio ``F/d`` of each is 0.5, 1, 2, 1, 2, 4, 4, 8 — so the map spans a factor of
sixteen and crosses 1 twice, and **three ratios are realised by two different
``(F, d)`` pairs**: ratio 1 at (32, 32) and (64, 64), ratio 2 at (32, 16) and
(64, 32), ratio 4 at (64, 16) and (128, 32). Those duplicates are the point of
keeping the grid two-dimensional rather than collapsing it onto the ratio: they
are what says whether the bottleneck ratio is the controlling variable or whether
the absolute width matters on its own, and a map drawn against ratio alone could
not ask that question.

``(128, 64)`` — ratio 2, a third time — is the point that was dropped, and it was
dropped because it is the most expensive cell in the grid and the ratio it adds
is the one already measured twice."""

PHASE_CUTS: tuple[dict, ...] = (
    {
        "cut": "sequence length is not crossed with the rest of the grid",
        "kept": "T = 32 for the whole map, plus a two-point ribbon at T = 64 and 128",
        "cost_if_kept": (
            "crossing all three of §4.5's lengths with 8 (F, d) points, 4 sparsities and 2 "
            "architectures is 192 runs; at the measured 38-83 s per run at T = 32, 68-150 s "
            "at T = 64 and 130-250 s at T = 128 that is about four hours of GPU, against a "
            "declared budget of about one"
        ),
        "why_this_axis": (
            "§10.2 names the figure's axes as architecture x sparsity x bottleneck ratio. "
            "Sequence length is in §4.5's starting scale but is not an axis of the figure, "
            "so it is the axis whose loss costs the deliverable least. It is sampled rather "
            "than dropped: PHASE_LENGTH_CELLS runs T = 64 and 128 at one interior point of "
            "the map, which turns 'not measured' into a three-point curve at one cell."
        ),
    },
    {
        "cut": "the (F=128, d=64) point",
        "kept": "the other eight of §4.5's nine (F, d) points",
        "cost_if_kept": "4 more cells, 8 more runs, about 10 minutes",
        "why_this_axis": (
            "its bottleneck ratio of 2 is already realised twice in the grid, at (32, 16) and "
            "(64, 32), and it is the most expensive cell of the nine: 150 s per run at T = 64 "
            "for A1 against 67 s for the cheapest. Dropping the most expensive of three "
            "duplicates costs the map no ratio and no F value and no d value."
        ),
    },
    {
        "cut": "more than one seed",
        "kept": "seed 20260809, §7.2's first, for every cell",
        "cost_if_kept": "five seeds is five times the grid, so about five hours",
        "why_this_axis": (
            "§4.5 asks for one screening seed and §7.3 R2 is a one-seed rung. This is "
            "deliberate rather than a budget cut: a phase diagram is a map and is labelled "
            "as one, in the caption and not only in the artifact. Prompt 15 owns the "
            "replication, and the last section of state/14_figure2.md names the cells it "
            "should replicate at."
        ),
    },
)
"""What this grid is not, priced, with the reason each was the right thing to
lose. Written before the sweep ran and quoted verbatim in
``state/14_figure2.md``."""


def _sparsity_label(value: float) -> str:
    """``0.06`` -> ``p006``. The label ``t1_ladder`` already uses for this axis."""
    return f"p{value:.2f}".replace(".", "")


def _cell(
    *,
    f_content: int,
    seq_len: int,
    activation_prob: float,
    axis: str,
    condition: str = "capacity_stressed",
    prefix: str = "phase",
) -> Cell:
    """One grid cell, as a generator override on ``capacity_stressed``.

    Four fields move together and that is the whole of what a cell is. They are
    declared as one override dict rather than as four axes because they are not
    four independent knobs here: ``n_content_groups`` is a *function* of
    ``n_content_features``, chosen so that sparsity means the same thing at every
    ``F`` (see the module docstring), and a cell that could set one without the
    other would silently change what its own x-axis meant.
    """
    if f_content % PHASE_GROUP_SIZE:
        raise ValueError(
            f"F = {f_content} is not a multiple of the fixed group size "
            f"{PHASE_GROUP_SIZE}; a partial group would make the sparsity axis mean "
            "something different in this cell than in its neighbours"
        )
    return Cell(
        name=f"{prefix}-F{f_content}-T{seq_len}-{_sparsity_label(activation_prob)}",
        axis=axis,
        level={
            "f_content": f_content,
            "f_total": f_content + PHASE_FIXED_FEATURES,
            "seq_len": seq_len,
            "activation_prob": activation_prob,
            "expected_active_content_features": round(
                PHASE_GROUP_SIZE * activation_prob, 3
            ),
            "condition": condition,
        },
        overrides={
            "n_content_features": f_content,
            "n_content_groups": f_content // PHASE_GROUP_SIZE,
            "seq_len": seq_len,
            "activation_prob": activation_prob,
        },
        condition=condition,
    )


PHASE_MAIN_CELLS: tuple[Cell, ...] = tuple(
    _cell(
        f_content=f_content,
        seq_len=PHASE_MAIN_SEQ_LEN,
        activation_prob=activation_prob,
        axis="phase",
    )
    for f_content in PHASE_CONTENT_FEATURES
    for activation_prob in PHASE_SPARSITIES
)
"""The map's twelve datasets: three feature-bank widths by four sparsities, at
one sequence length. Each is run at every ``d`` the grid pairs it with, so the
number of *cells* is twelve and the number of ``(F, d, sparsity)`` points is
thirty-two."""

PHASE_LENGTH_RIBBON_F = 64
PHASE_LENGTH_RIBBON_D = 32
PHASE_LENGTH_RIBBON_SPARSITY = 0.12
"""The interior point the sequence-length ribbon is measured at: bottleneck ratio
2, the middle of the map, at the base condition's own sparsity. Chosen before the
sweep ran, on the grid's geometry rather than on any result."""

PHASE_LENGTH_CELLS: tuple[Cell, ...] = tuple(
    _cell(
        f_content=PHASE_LENGTH_RIBBON_F,
        seq_len=seq_len,
        activation_prob=PHASE_LENGTH_RIBBON_SPARSITY,
        axis="phase_length",
    )
    for seq_len in PHASE_SEQ_LENS
    if seq_len != PHASE_MAIN_SEQ_LEN
)
"""§4.5's other two sequence lengths at one point of the map. Together with the
main grid's cell at the same ``(F, d, sparsity)`` this is a three-point curve in
sequence length, which is what the axis was reduced to."""

PHASE_NEGATIVE_CONTROL_CELL: Cell = _cell(
    f_content=PHASE_LENGTH_RIBBON_F,
    seq_len=PHASE_MAIN_SEQ_LEN,
    activation_prob=PHASE_LENGTH_RIBBON_SPARSITY,
    axis="phase_negative_control",
    condition="negative_control",
    prefix="phasenull",
)
"""§4.4's information-destroyed condition at the map's middle point, for both
architectures.

The map's own shapes are new — a feature bank, a group structure and a sequence
length no recorded run has used — so prompt 13's negative control does not cover
them, and "the task does not leak" is not a property that transfers across a
change of generator configuration merely because the generator is the same. The
source is removed, so the answer exists nowhere in the input and mutual
information with the target is zero by construction; either arm above 0.05 exact
recall or 0.05 normalized skill is a hard stop for the whole laboratory and not
merely for this claim. One cell rather than thirty-two because the leak it looks
for would be a property of the generator's construction, which is shared by every
cell of the map, and two runs is what that costs."""

PHASE_CELLS: tuple[Cell, ...] = (
    PHASE_MAIN_CELLS + PHASE_LENGTH_CELLS + (PHASE_NEGATIVE_CONTROL_CELL,)
)


def phase_cells() -> tuple[Cell, ...]:
    """Every cell this grid declares, in a stable order."""
    return PHASE_CELLS


def cell_axes(name: str) -> dict:
    """The ``(F, T, sparsity)`` a cell name stands for, from the registry.

    Read out of :data:`PHASE_CELLS` rather than parsed back out of the string,
    because a name is a label and the level is the datum; a reader that recovered
    numbers by splitting on hyphens would keep working after the numbers and the
    name stopped agreeing.
    """
    for cell in PHASE_CELLS:
        if cell.name == name:
            return dict(cell.level)
    raise KeyError(f"{name!r} is not a cell of the phase grid")


def _cells_at(width: int, seq_len: int) -> tuple[str, ...]:
    """The main-grid cells this width is measured at, in grid order."""
    return tuple(
        cell.name
        for cell in PHASE_MAIN_CELLS
        if cell.level["seq_len"] == seq_len
        and (cell.level["f_content"], width) in PHASE_WIDTH_FEATURE_POINTS
    )


_GRID_NOTE = (
    "One panel of §10.2's figure 2, at one model width. The comparison is per width "
    "because d_model is declared by the comparison and not by the cell — a cell is a "
    "dataset — so the map is assembled from one plan per width rather than from one plan "
    "with a width axis, and each plan is therefore a §7.2 matched comparison in its own "
    "right with its own committed configs. "
    "Screening depth by declaration: R2's committed preset, one seed, both arms. At one "
    "seed per cell nothing here resolves a capability difference — prompt 09 measured that "
    "five seeds cannot resolve a T1 recall difference below 0.128 and a single pair is "
    "weaker still — so no cell of this sweep is evidence for a capability or a "
    "representation difference at any rung. What it is evidence about is where each "
    "architecture is off its floor, below its ceiling, and running an active mechanism, all "
    "three of which are comparisons against a fixed threshold rather than against another "
    "architecture and are therefore decidable from one run. §7.5 rungs 2 and 3 are not "
    "available from this mission and are not claimed."
)

_PHASE_R1: dict[str, dict] = {
    "phase_r1": {
        "claim_id": "phase-map-a0-a1-sparsity-bottleneck",
        "primary_metric": "associative_recall_accuracy",
        "control_arch": "softmax",
        "candidate_archs": ("linear",),
        "task": "T1",
        "cells": (POSITIVE_CONTROL_CELL.name,),
        "d_model": None,
        "owner_prompt": "14",
        "rungs": {"R1": 1},
        "notes": (
            "This sweep's own §7.3 R1, run before any cell of the map. §7.3's 'never skip R1' "
            "applies to a mission, not only to an architecture: sixty-eight screening runs "
            "produced by a broken instrument are sixty-eight measurements of the bug. The "
            "known-easy positive control at the frozen R1 preset (d = d_recommended = 48, "
            "distance one to two, no distractors, one association, 32 768 examples, 4 000 "
            "steps) has a known answer — prompt 04 recorded A0 at 0.9055, prompt 11 recorded "
            "A1 at 0.8954 — so this pair must come out null, and a harness that reports a gap "
            "here is measuring itself. It is declared under this mission's own claim packet "
            "rather than re-run under prompt 13's, so that this claim's rung-0 and rung-1 "
            "evidence is its own and prompt 13's gates file is not enlarged by a mission that "
            "is not its."
        ),
    },
}

_PHASE_MAIN_COMPARISONS: dict[str, dict] = {
    f"phase_T{PHASE_MAIN_SEQ_LEN}_d{width}": {
        "claim_id": "phase-map-a0-a1-sparsity-bottleneck",
        "primary_metric": "associative_recall_accuracy",
        "control_arch": "softmax",
        "candidate_archs": ("linear",),
        "task": "T1",
        "cells": _cells_at(width, PHASE_MAIN_SEQ_LEN),
        "d_model": width,
        "owner_prompt": "14",
        "rungs": {PHASE_LADDER: 1},
        "notes": (
            f"d = {width}. Bottleneck ratios at this width: "
            + ", ".join(
                f"F={f_content} -> {f_content / width:g}"
                for f_content, w in PHASE_WIDTH_FEATURE_POINTS
                if w == width
            )
            + ". "
            + _GRID_NOTE
        ),
    }
    for width in PHASE_WIDTHS
}

_PHASE_LENGTH_COMPARISON: dict[str, dict] = {
    f"phase_length_d{PHASE_LENGTH_RIBBON_D}": {
        "claim_id": "phase-map-a0-a1-sparsity-bottleneck",
        "primary_metric": "associative_recall_accuracy",
        "control_arch": "softmax",
        "candidate_archs": ("linear",),
        "task": "T1",
        "cells": tuple(cell.name for cell in PHASE_LENGTH_CELLS),
        "d_model": PHASE_LENGTH_RIBBON_D,
        "owner_prompt": "14",
        "rungs": {PHASE_LADDER: 1},
        "notes": (
            "The sequence-length ribbon: §4.5's other two lengths at one interior point of "
            f"the map (F = {PHASE_LENGTH_RIBBON_F}, d = {PHASE_LENGTH_RIBBON_D}, bottleneck "
            f"ratio {PHASE_LENGTH_RIBBON_F / PHASE_LENGTH_RIBBON_D:g}, activation "
            f"probability {PHASE_LENGTH_RIBBON_SPARSITY}). Read together with the main "
            f"grid's cell at the same point, which is at T = {PHASE_MAIN_SEQ_LEN}, it is a "
            "three-point curve in sequence length. This is what the sequence-length axis was "
            "reduced to and PHASE_CUTS says what that cost. "
            "The transport distances are NOT scaled with the sequence: distance_buckets stay "
            "at capacity_stressed's 5-9 and 10-16 at every length, so what this ribbon varies "
            "is how much other material sits in the prefix, not how far the value has to "
            "travel. That is deliberate — a ribbon that moved both would not say which one "
            "mattered — and it is a limit on what the ribbon can be read as saying. "
            + _GRID_NOTE
        ),
    }
}

_PHASE_NEGATIVE_CONTROL_COMPARISON: dict[str, dict] = {
    f"phase_negative_control_d{PHASE_LENGTH_RIBBON_D}": {
        "claim_id": "phase-map-a0-a1-sparsity-bottleneck",
        "primary_metric": "associative_recall_accuracy",
        "control_arch": "softmax",
        "candidate_archs": ("linear",),
        "task": "T1",
        "cells": (PHASE_NEGATIVE_CONTROL_CELL.name,),
        "d_model": PHASE_LENGTH_RIBBON_D,
        "owner_prompt": "14",
        "rungs": {PHASE_LADDER: 1},
        "notes": (
            "The map's own §4.4 information-destroyed control, at the same interior point as "
            f"the length ribbon (F = {PHASE_LENGTH_RIBBON_F}, d = {PHASE_LENGTH_RIBBON_D}, "
            f"T = {PHASE_MAIN_SEQ_LEN}, activation probability "
            f"{PHASE_LENGTH_RIBBON_SPARSITY}) and therefore matched to a cell of the map in "
            "every §7.2 variable except the one thing that is removed. Because the program "
            "oracle scores at or below an input-blind predictor on this condition, the "
            "comparison that decides it is against the frequency ceiling fitted on the split "
            "being scored, per prompt 03's committed CEILING_DOMINATED_METRICS. "
            + _GRID_NOTE
        ),
    }
}

PHASE_COMPARISONS: dict[str, dict] = (
    _PHASE_R1
    | _PHASE_MAIN_COMPARISONS
    | _PHASE_LENGTH_COMPARISON
    | _PHASE_NEGATIVE_CONTROL_COMPARISON
)
"""This mission's own R1, then one §7.2 matched comparison per model width, then
the sequence-length ribbon — in the order they are run, which is also the order
§7.3 requires: the positive control is not a step that can be taken later.

Merged into ``comparison.DECLARED_COMPARISONS`` there rather than here, so that
the laboratory still has exactly one registry of comparisons and ``make
comparisons`` still regenerates all of them."""


def phase_cost_model() -> dict:
    """The measured per-run cost this grid was priced against, before it ran.

    Six probe runs at 100 steps with the full R2 data budget, extrapolated to
    2 000 steps by the measured per-step training time. Recorded here because the
    grid was cut on these numbers and a cut whose arithmetic is not in the
    repository is a preference.
    """
    return {
        "method": (
            "generate the full R2 split, run 100 steps with one complete evaluation, "
            "mechanism and geometry pass, then add the measured per-step training time for "
            "the remaining 1 900 steps"
        ),
        "device": "RTX 5060 Ti",
        "probes": (
            {"arch": "softmax", "f_content": 32, "d": 16, "seq_len": 64,
             "generate_s": 22.9, "wall_at_100_s": 43.0, "train_at_100_s": 1.4,
             "peak_mib": 866.4, "estimated_full_run_s": 69.9},
            {"arch": "linear", "f_content": 32, "d": 16, "seq_len": 64,
             "generate_s": 23.2, "wall_at_100_s": 44.2, "train_at_100_s": 1.2,
             "peak_mib": 954.4, "estimated_full_run_s": 66.9},
            {"arch": "linear", "f_content": 128, "d": 64, "seq_len": 64,
             "generate_s": 25.5, "wall_at_100_s": 93.4, "train_at_100_s": 3.0,
             "peak_mib": 4206.3, "estimated_full_run_s": 150.4},
            {"arch": "softmax", "f_content": 128, "d": 64, "seq_len": 64,
             "generate_s": 29.9, "wall_at_100_s": 88.2, "train_at_100_s": 1.3,
             "peak_mib": 2126.3, "estimated_full_run_s": 112.6},
            {"arch": "linear", "f_content": 128, "d": 64, "seq_len": 128,
             "generate_s": 47.3, "wall_at_100_s": 158.5, "train_at_100_s": 4.7,
             "peak_mib": 8346.6, "estimated_full_run_s": 247.6},
            {"arch": "linear", "f_content": 128, "d": 64, "seq_len": 32,
             "generate_s": 13.4, "wall_at_100_s": 44.8, "train_at_100_s": 2.0,
             "peak_mib": 2136.7, "estimated_full_run_s": 83.2},
        ),
        "what_dominates": (
            "not training. At the cheapest probe the 2 000 training steps are 28 s of a 70 s "
            "run; the other 42 s are data generation (23 s for 16 384 + 4 096 examples) and "
            "the evaluation, mechanism and geometry passes (19 s). Both scale with F and with "
            "sequence length and neither scales with the step budget, which is why a shorter "
            "budget would not have bought a wider grid and why the sequence length was the "
            "axis that had to give."
        ),
        # Two arms per cell of every declared comparison, controls included. The
        # two §7.2 matching strategies resolve to the same runs here, so they
        # cost nothing extra and are not counted twice.
        "planned_runs": 2 * sum(len(spec["cells"]) for spec in PHASE_COMPARISONS.values()),
        "estimated_total_minutes": 72,
    }
