"""§6.3 mechanism activity: was the special branch used, and for what.

An architecture that trains while ignoring its special branch is a failed
mechanism experiment even when the task loss is good. §7.3's R1 says so
directly: the model must solve the tiny task *and* the mechanism must become
active. This module is the second half.

Prompt 04 builds only what A0 and R1 need — generic distribution statistics for
any mixing mechanism whose weights form a distribution over source positions,
and one program-grounded measure that asks whether the weight went to the
position the ground-truth program actually reads from. Prompt 13 extends this
with the rest of the §6.3 list (gate variance, routing entropy, write/erase
norms, per-stream rank, gradient flow, ablation effect size) once there are
mechanisms that have those things.

The distinction that matters here: entropy and off-diagonal mass say the
mechanism is *doing something*. Retrieval mass says it is doing *the task*. A
uniform prefix average has low self-mass and looks busy; only the second measure
separates it from retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch

from architecture_mechanics.data.feature_program import ProgramRecord

__all__ = [
    "ACTIVITY_GATES",
    "MECHANISM_VERSION",
    "RetrievalReport",
    "attention_retrieval",
    "mechanism_is_active",
]

MECHANISM_VERSION = "mech-1.0.0"

ACTIVITY_GATES = {
    "min_off_diagonal_mass": 0.05,
    "max_entropy_ratio": 0.95,
    "min_retrieval_lift": 2.0,
}
"""What counts as non-degenerate on the R1 positive control.

Not a scientific threshold — R1 is instrument validation, and nothing is claimed
at any rung from these numbers. They are the three ways a mixing mechanism can
be trivially inert while the loss still falls:

``min_off_diagonal_mass``  all the weight on the query's own position, so no
                           token ever moves and the model is a position-wise MLP;
``max_entropy_ratio``      weight spread evenly over the causal window, so the
                           mechanism is a running average that selects nothing;
``min_retrieval_lift``     weight goes somewhere, but not preferentially to the
                           position the program reads from — the task is being
                           solved by something other than retrieval.

A lift of 2.0 means the true source receives at least twice the weight a flat
distribution over the causal prefix would give it. On the positive control, where
the source sits one or two positions before the query and the prefix is up to
eleven long, that is a low bar deliberately: it is a floor under "inert", not a
measure of quality.
"""


@dataclass(frozen=True)
class RetrievalReport:
    """Where a layer's attention went, relative to the ground-truth program."""

    layer: str
    n_steps: int
    source_mass: float
    """Mean over required operations of the weight the query position places on
    the true source position, averaged over heads."""

    best_head_source_mass: float
    """The same quantity for the single head with the highest average. Heads
    specialise; averaging over all of them understates a mechanism in which one
    head does the retrieval. The head is chosen on the aggregate over every
    step, never per example, so this is a summary and not a selection."""

    chance_mass: float
    """What a flat distribution over the causal prefix would have given the
    source. This is the comparison that makes ``source_mass`` mean anything: on
    the positive control the query is at position 11, so chance is about 0.083,
    and a mass of 0.3 is a real signal while on a length-2 prefix it would be
    below chance."""

    lift: float
    argmax_hit_rate: float
    """Fraction of operations where the head-averaged weight is *largest* at the
    true source."""

    def as_dict(self) -> dict:
        return asdict(self)


def attention_retrieval(
    weights: torch.Tensor,
    programs: Sequence[ProgramRecord],
    *,
    layer: str = "",
) -> RetrievalReport | None:
    """Score one layer's attention against the program's source positions.

    Args:
        weights: ``(B, H, T, T)`` attention distributions, rows summing to one
            over the causal prefix.
        programs: the program records for the same ``B`` examples, in order.
        layer: label carried into the report.

    Returns ``None`` when no example in the batch has a step with a source
    position — T0 reconstruction, or the information-destroyed control, where
    there is nothing to retrieve and a retrieval number would be a fiction.
    """
    if weights.dim() != 4:
        raise ValueError(f"expected (B, H, T, T) attention weights, got {tuple(weights.shape)}")
    batch, heads, seq_len, _ = weights.shape
    if len(programs) < batch:
        raise ValueError(f"{len(programs)} program records for {batch} rows of weights")

    array = weights.detach().to(torch.float64).cpu().numpy()
    per_head: list[np.ndarray] = []
    averaged: list[float] = []
    chance: list[float] = []
    hits: list[float] = []

    for row in range(batch):
        for step in programs[row].steps:
            if step.source is None or step.dest >= seq_len or step.source > step.dest:
                continue
            column = array[row, :, step.dest, step.source]
            per_head.append(column)
            mean_over_heads = array[row, :, step.dest, :].mean(axis=0)
            averaged.append(float(mean_over_heads[step.source]))
            chance.append(1.0 / (step.dest + 1))
            hits.append(float(int(np.argmax(mean_over_heads[: step.dest + 1])) == step.source))

    if not averaged:
        return None

    stacked = np.stack(per_head, axis=0)  # (n_steps, H)
    mean_mass = float(np.mean(averaged))
    mean_chance = float(np.mean(chance))
    return RetrievalReport(
        layer=layer,
        n_steps=len(averaged),
        source_mass=mean_mass,
        best_head_source_mass=float(stacked.mean(axis=0).max()) if heads else float("nan"),
        chance_mass=mean_chance,
        lift=float(mean_mass / mean_chance) if mean_chance > 0 else float("inf"),
        argmax_hit_rate=float(np.mean(hits)),
    )


def mechanism_is_active(
    distribution_stats: Mapping[str, float],
    retrieval: Mapping[str, RetrievalReport | None],
) -> dict:
    """Apply :data:`ACTIVITY_GATES` and say which of the three, if any, failed.

    ``distribution_stats`` is the flat per-layer dictionary a model's
    ``mechanism_activity`` returns (``layers.0.entropy_ratio`` and friends);
    ``retrieval`` maps a layer label to its :class:`RetrievalReport`.

    The verdict is taken over the *best* layer for each measure, because a
    two-layer model is free to do its transport in either one. A model that
    moves nothing in layer 0 and everything in layer 1 has an active mechanism.
    """
    off_diagonal = _best(distribution_stats, "off_diagonal_mass", how="max")
    entropy_ratio = _best(distribution_stats, "entropy_ratio", how="min")
    lifts = [report.lift for report in retrieval.values() if report is not None]
    lift = max(lifts) if lifts else None

    reasons: list[str] = []
    if off_diagonal is None or off_diagonal < ACTIVITY_GATES["min_off_diagonal_mass"]:
        reasons.append(f"off_diagonal_mass={off_diagonal} below {ACTIVITY_GATES['min_off_diagonal_mass']}")
    if entropy_ratio is None or entropy_ratio > ACTIVITY_GATES["max_entropy_ratio"]:
        reasons.append(f"entropy_ratio={entropy_ratio} above {ACTIVITY_GATES['max_entropy_ratio']}")
    if lift is not None and lift < ACTIVITY_GATES["min_retrieval_lift"]:
        reasons.append(f"retrieval lift={lift:.3f} below {ACTIVITY_GATES['min_retrieval_lift']}")

    return {
        "active": not reasons,
        "reasons": reasons,
        "best_off_diagonal_mass": off_diagonal,
        "best_entropy_ratio": entropy_ratio,
        "best_retrieval_lift": lift,
        "retrieval_measurable": bool(lifts),
        "gates": dict(ACTIVITY_GATES),
        "mechanism_version": MECHANISM_VERSION,
    }


def _best(stats: Mapping[str, float], suffix: str, *, how: str) -> float | None:
    values = [value for key, value in stats.items() if key.endswith("." + suffix)]
    if not values:
        return None
    return max(values) if how == "max" else min(values)
