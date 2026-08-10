"""Paper figures, generated only from recorded artifacts.

§8.5's last required test is a property of this module: *report generated only
from recorded artifacts*. It is enforced two ways. The narrow way is
:data:`ARTIFACT_READ_ROOTS` — while a figure is being built, nothing inside this
laboratory may be read except ``runs/`` and ``reports/`` and the output
directory the figure is writing into, and
``tests/reporting/test_figure_provenance.py`` audits every file the process
opens to prove it. The wide way is that there is no code here that draws a
number: every mark on every figure comes from a tensor the generator produced or
a value a recorded run wrote down. A schematic that illustrates what the
benchmark *would* look like is worth less than no figure at all, because it is
indistinguishable from one that is true.

Figure 1 has no parent run: it is the benchmark itself, drawn from one example
the generator produces on demand. Its inputs are therefore its dataset
configuration, and the caption carries all of them.

Figure 2 (prompt 14) is the first figure here with parent runs. It draws north
star 10.2's capability-and-geometry phase diagram from
:func:`architecture_mechanics.reporting.tables.phase_report`, which reads
``runs/`` and ``reports/comparisons/`` and computes nothing that is not already
on disk. It is *screening* evidence — one seed per cell — and saying so is part
of the artifact and not only of the surrounding prose: the caption's first two
sentences say it, a line inside the image says it, and
``tests/reporting/test_figure2.py`` fails if either stops saying it.

Figures 3 and 4 arrive at prompts 22 and 23. All four share
:mod:`architecture_mechanics.reporting.figure_style`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from ..data.feature_program import (
    FeatureProgramDataset,
    condition_config,
    generate_dataset,
)
from ..experiments.manifest import lab_root
from . import figure_style as style

# ``tables`` is imported at module scope on purpose.
# tests/reporting/test_figure_provenance.py audits every file the *build* opens,
# and a deferred import would open this module's own source inside the audited
# window — a read of the source tree, which is precisely what the audit exists to
# catch. Importing here puts it before the hook, where it belongs.
from .tables import PURITY_FIVE_SEED_MDE, phase_report

FIGURE_VERSION = "am-fig-1.1.0"

ARTIFACT_READ_ROOTS: tuple[str, ...] = ("runs", "reports")
"""The only directories inside the laboratory a figure may take a *number* from.

``runs/`` holds recorded evidence and ``reports/`` holds artifacts derived from
it. Everything else — ``configs/``, ``claims/``, the source tree, a scratch file
in the working directory — is off limits, so a figure cannot quietly acquire a
number that no run produced. Datasets are not read at all: they are regenerated
in process from a configuration recorded in the caption, which is stronger than
reading a file, because it cannot drift from the generator.

The output directory is the one other place a figure touches, and it is not an
exception to the rule: everything the figure reads there it wrote itself on a
previous line (:func:`write_index` merges the index it maintains). No datum
enters a figure through its own output.
"""

DEFAULT_OUT_DIR = "paper/figures"
"""Where the paper's figures live, because that is where the paper reads them.

``reports/`` is for artifacts derived from runs; a figure is a page element, and
prompt 27 assembles the paper from this directory. The index and the caption
sidecars live beside the PNGs so that a figure and the record of how it was made
cannot be moved apart.
"""

FIGURE_STEMS: dict[int, str] = {
    1: "fig1_benchmark_schematic",
    2: "fig2_phase_diagram",
    3: "fig3_mechanism_intervention",
    4: "fig4_trajectory",
}
"""The paper's filenames for all four of north star 10.2's figures.

Fixed here for figures that do not exist yet on purpose: the name is what prose,
build rules and the reader's citation refer to, and renaming a figure after it
has been referenced is how a paper acquires a broken cross-reference.
"""


# --------------------------------------------------------------------------- #
# Figure 1 — benchmark schematic
# --------------------------------------------------------------------------- #

FIGURE1_CONDITION = "capacity_stressed"
FIGURE1_SPLIT = "train"
FIGURE1_EXAMPLE = 0
"""The example hand-checked line by line in ``state/02_generator.md``. Fixed
rather than chosen: an example picked after looking at several is a selected
example, and the first one is the only choice that carries no selection."""


@dataclass(frozen=True)
class FigureResult:
    """One figure and everything needed to regenerate it."""

    number: int
    path: Path
    sha256: str
    caption: str
    params: dict

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "path": str(self.path),
            "sha256": self.sha256,
            "caption": self.caption,
            "params": self.params,
        }


def figure1_dataset() -> FeatureProgramDataset:
    """The dataset figure 1 is drawn from: the capacity-stressed T1 condition.

    Unmodified :func:`condition_config`, so the example drawn here is bit for
    bit the example every capacity-stressed run trains on — not a
    figure-friendly variant of it.
    """
    return generate_dataset(condition_config(FIGURE1_CONDITION, split=FIGURE1_SPLIT))


def figure1_params(dataset: FeatureProgramDataset) -> dict:
    """Every input needed to regenerate figure 1, read off the real dataset."""
    cfg = dataset.config
    record = dataset.programs[FIGURE1_EXAMPLE]
    step = record.steps[0]
    active = dataset.active_mask[FIGURE1_EXAMPLE].sum(dim=-1).float()
    return {
        "generator_version": dataset.generator_version,
        "figure_version": FIGURE_VERSION,
        "condition": cfg.condition,
        "family": cfg.family,
        "split": cfg.split,
        "seed": cfg.seed,
        "example_index": FIGURE1_EXAMPLE,
        "n_examples": cfg.n_examples,
        "seq_len": cfg.seq_len,
        "n_features": dataset.n_features,
        "n_content_features": dataset.banks.n_content,
        "n_key_features": dataset.banks.n_key,
        "n_op_features": dataset.banks.n_op,
        "d_recommended": cfg.d_recommended,
        "f_over_d": round(dataset.n_features / cfg.d_recommended, 3),
        "activation_prob": cfg.activation_prob,
        "example_mean_active_features_per_position": round(float(active.mean().item()), 3),
        "example_density": round(float(active.mean().item()) / dataset.n_features, 4),
        "dataset_mean_active_features_per_position": round(
            float(dataset.active_mask.sum(dim=-1).float().mean().item()), 3
        ),
        "n_distractors": cfg.n_distractors,
        "n_associations": cfg.n_associations,
        "template_id": record.template_id,
        "operation": step.op,
        "source": step.source,
        "dest": step.dest,
        "key_id": step.key_id,
        "distance": step.distance,
        "distractor_positions": list(step.distractors),
        "answer_features": list(step.answer_features),
        "dataset_content_hash": dataset.content_hash,
        **style.style_provenance(),
    }


def figure1_caption(params: dict) -> str:
    """The caption. Everything in it is a value read off the real dataset.

    A figure whose exact inputs are not recoverable from its caption is a
    decoration, so the regeneration command and the dataset's content hash are
    part of the caption rather than a footnote to it.
    """
    answer = "{" + ", ".join(str(f) for f in sorted(params["answer_features"])) + "}"
    return (
        "**Figure 1. The ground-truth benchmark.** One generated example of task "
        f"family {params['family']} (associative recall) under the "
        f"`{params['condition']}` condition. Each of the {params['seq_len']} "
        f"positions carries a sparse subset of {params['n_features']} known "
        f"latent features, partitioned into a content bank "
        f"({params['n_content_features']}), a key bank ({params['n_key_features']}) "
        f"and an op bank ({params['n_op_features']}); cell darkness is the "
        "feature's magnitude, white is exactly inactive. The op bank's first "
        "row is active at every content-carrying position, which is why it "
        "reads as a solid rule rather than a border. Position "
        f"{params['source']} binds key {params['key_id']} to a value; the query "
        f"at position {params['dest']} carries the same key bits and no content, "
        f"and the required output there is that value — a transport of "
        f"{params['distance']} positions past {params['n_distractors']} "
        f"distractors, with {params['n_associations']} competing bindings "
        f"elsewhere in the sequence. Outlined columns are the source and the "
        "query, the heavier boxes are their (identical) key bits, shaded "
        "columns are the distractors, and the dashed arrows are the answer "
        f"features {answer} leaving the source for the "
        "supervised target. Nothing in the input marks a distractor or a "
        "source: those come from the ground-truth program record, which is what "
        f"makes the benchmark checkable. The model must carry the value through "
        f"a width of d = {params['d_recommended']} < F = {params['n_features']} "
        f"(F/d = {params['f_over_d']}), so the features are in forced "
        "superposition. Regenerate with `python -m "
        "architecture_mechanics.reporting.figures --figure 1`; generator "
        f"{params['generator_version']}, figure {params['figure_version']}, style "
        f"{params['figure_style_version']}, matplotlib "
        f"{params['matplotlib_version']}, seed {params['seed']}, split "
        f"{params['split']}, example {params['example_index']} of "
        f"{params['n_examples']}, per-feature activation probability "
        f"{params['activation_prob']} within a position's content group "
        f"(realised: {params['dataset_mean_active_features_per_position']} "
        "active features per position across the dataset, "
        f"{params['example_mean_active_features_per_position']} in this "
        f"example, a density of {params['example_density']}), dataset content "
        f"hash `{params['dataset_content_hash']}`."
    )


def _footer_lines(params: dict) -> tuple[str, str]:
    """Two short provenance lines drawn inside the image itself.

    A PNG separated from its caption is a common accident; these keep the file
    self-describing when it happens.
    """
    return (
        (
            f"{params['generator_version']}  {params['condition']}/{params['split']}"
            f"  seed={params['seed']}  example={params['example_index']}"
            f"  F={params['n_features']}  d={params['d_recommended']}"
            f"  T={params['seq_len']}  p_act={params['activation_prob']}"
        ),
        (
            f"distractors={params['n_distractors']}"
            f"  associations={params['n_associations']}"
            f"  dataset={params['dataset_content_hash'][:12]}"
            f"  fig={params['figure_version']}  style={params['figure_style_version']}"
        ),
    )


def draw_figure1(dataset: FeatureProgramDataset) -> Figure:
    """Draw the benchmark schematic from one real example.

    Every mark is a value from the dataset. Reading top to bottom: the required
    operation as an arc over the sequence; a strip saying what each position
    *is*; the feature matrix itself with the required output beside it; a key;
    and the bottleneck drawn to scale, because ``d < F`` is the premise the
    whole benchmark rests on and is otherwise invisible in a picture of data.

    Laid out in inches rather than by ``tight_layout`` so that a figure claiming
    to be column-width is column-width, and so the bytes do not move when
    matplotlib's layout heuristics change.
    """
    style.apply_style()

    index = FIGURE1_EXAMPLE
    record = dataset.programs[index]
    step = record.steps[0]
    banks = dataset.banks
    x = dataset.inputs[index].numpy()
    target = dataset.targets[index, step.dest].numpy()
    seq_len, n_features = x.shape
    n_content, n_key = banks.n_content, banks.n_key
    source, dest = int(step.source), int(step.dest)
    distractors = sorted(int(p) for p in step.distractors)
    answer = sorted(int(f) for f in step.answer_features)
    bindings = [p.index for p in record.positions if p.op_code == "BIND"]
    d_model = dataset.config.d_recommended

    width = style.COLUMN_WIDTH_IN
    height = 4.05
    fig = Figure(figsize=(width, height), dpi=style.SAVE_DPI)
    fig.patch.set_facecolor(style.PAPER)

    def rect(x0_in: float, y0_in: float, w_in: float, h_in: float):
        """Axes placed in inches from the bottom-left, so the layout is exact."""
        return fig.add_axes([x0_in / width, y0_in / height, w_in / width, h_in / height])

    left, heat_w = 0.64, 2.24
    target_x, target_w = left + heat_w + 0.10, 0.11
    # White rows above and below the matrix. The op bank is 4 rows of 124 and
    # would otherwise sit flush against the axis frame, where it reads as part
    # of the border rather than as data.
    pad_rows = 2.5
    y_top, y_bottom = -0.5 - pad_rows, n_features - 0.5 + pad_rows
    heat_y, heat_h = 1.20, 2.05
    role_y, role_h = heat_y + heat_h + 0.02, 0.10
    arc_y, arc_h = role_y + role_h + 0.01, 0.58
    key_y, key_h = 0.56, 0.42
    bottle_y, bottle_h = 0.20, 0.30

    ax_heat = rect(left, heat_y, heat_w, heat_h)
    ax_target = rect(target_x, heat_y, target_w, heat_h)
    ax_role = rect(left, role_y, heat_w, role_h)
    ax_arc = rect(left, arc_y, heat_w, arc_h)
    ax_key = rect(0.08, key_y, width - 0.16, key_h)
    ax_bottle = rect(0.08, bottle_y, width - 0.16, bottle_h)

    cmap = style.magnitude_colormap()
    tiny = style.FONT_SIZE_TINY

    def bare(axes, xlim, ylim):
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.patch.set_alpha(0.0)
        for spine in axes.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------ #
    # The feature matrix. F rows by T columns, in true proportion: the op
    # bank really is 4 rows of 124, and pretending otherwise would overstate
    # how much of an input the operator markers occupy.
    # ------------------------------------------------------------------ #
    # Distractor columns, shaded behind the data. They are ordinary content
    # positions — nothing in the input marks them — so the shading comes from
    # the program record, which is the only place that knows.
    for position in distractors:
        ax_heat.axvspan(
            position - 0.5,
            position + 0.5,
            facecolor=style.INK_FAINT,
            edgecolor="none",
            zorder=0,
        )
    ax_heat.imshow(
        x.T,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, seq_len - 0.5, n_features - 0.5, -0.5),
        zorder=2,
    )
    ax_heat.set_xlim(-0.5, seq_len - 0.5)
    ax_heat.set_ylim(y_bottom, y_top)
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(style.INK_STRONG)
    for boundary in (n_content - 0.5, n_content + n_key - 0.5):
        ax_heat.axhline(boundary, color=style.INK_LIGHT, linewidth=0.45, dashes=(2.5, 1.5), zorder=3)

    ax_heat.set_xticks([0, 12, 24, source, dest])
    ax_heat.set_xticklabels(["0", "12", "24", str(source), str(dest)], fontsize=tiny)
    ax_heat.set_yticks([0, n_content, n_content + n_key])
    ax_heat.set_yticklabels(["0", str(n_content), str(n_content + n_key)], fontsize=tiny)
    ax_heat.tick_params(length=1.5, pad=1.2)
    ax_heat.set_xlabel("sequence position $t$", labelpad=1.5, fontsize=style.FONT_SIZE_SMALL)

    for name, lo, hi in (
        ("content", 0, n_content),
        ("key", n_content, n_content + n_key),
        ("op", n_content + n_key, n_features),
    ):
        ax_heat.annotate(
            name,
            xy=(-0.5, (lo + hi - 1) / 2.0),
            xycoords="data",
            xytext=(-20.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
            fontsize=tiny,
            color=style.INK_STRONG,
        )
    ax_heat.set_ylabel("ground-truth feature", labelpad=23.0, fontsize=style.FONT_SIZE_SMALL)

    # The two columns the task is about, outlined so they are findable in a
    # matrix that is deliberately hard to read at a glance.
    for position in (source, dest):
        ax_heat.add_patch(
            Rectangle(
                (position - 0.5, -0.5),
                1.0,
                n_features,
                fill=False,
                edgecolor=style.INK_STRONG,
                linewidth=0.5,
                zorder=5,
            )
        )
        ax_heat.add_patch(
            Rectangle(
                (position - 0.5, n_content - 0.5),
                1.0,
                n_key,
                fill=False,
                edgecolor=style.INK,
                linewidth=0.8,
                zorder=6,
            )
        )
    ax_heat.plot(
        [source, dest],
        [n_content + n_key / 2.0, n_content + n_key / 2.0],
        color=style.INK,
        linewidth=0.5,
        dashes=(1.5, 1.2),
        zorder=6,
    )
    ax_heat.annotate(
        "same key",
        xy=((source + dest) / 2.0, n_content + n_key / 2.0),
        ha="center",
        va="center",
        fontsize=4.8,
        color=style.INK,
        bbox={"facecolor": style.PAPER, "edgecolor": "none", "pad": 0.5},
        zorder=7,
    )

    # ------------------------------------------------------------------ #
    # The required output at the query position, on the same feature axis.
    # ------------------------------------------------------------------ #
    ax_target.imshow(
        target.reshape(-1, 1),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, 0.5, n_features - 0.5, -0.5),
    )
    ax_target.set_xlim(-0.5, 0.5)
    ax_target.set_ylim(y_bottom, y_top)
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    for spine in ax_target.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(style.INK)
    ax_target.text(
        0.0,
        y_top,
        f"required\noutput\nat $t={dest}$",
        ha="center",
        va="bottom",
        fontsize=tiny,
        color=style.INK,
        linespacing=1.1,
    )

    # The transport, drawn in feature space: the answer features leave the
    # source column and arrive as the supervised target. Two lines, because
    # this example's answer is two features; nothing here is drawn by hand.
    for feature in answer:
        fig.add_artist(
            ConnectionPatch(
                xyA=(source + 0.5, feature),
                coordsA=ax_heat.transData,
                xyB=(-0.5, feature),
                coordsB=ax_target.transData,
                arrowstyle="-|>",
                mutation_scale=4.0,
                linewidth=0.6,
                color=style.INK,
                shrinkA=0.4,
                shrinkB=0.4,
                linestyle=(0, (2.2, 1.6)),
                zorder=8,
            )
        )

    # ------------------------------------------------------------------ #
    # What each position is. This strip is the op bank, drawn legibly: at
    # true proportion it is 4 rows out of 124 and cannot be read.
    # ------------------------------------------------------------------ #
    op_of = {p.index: p.op_code for p in record.positions}
    op_fill = {
        "CONTENT": style.INK_FAINT,
        "BIND": style.INK_MID,
        "QUERY": style.INK,
        "NOOP": style.INK_LIGHT,
    }
    for position in range(seq_len):
        ax_role.add_patch(
            Rectangle(
                (position - 0.5, 0.0),
                1.0,
                1.0,
                facecolor=op_fill[op_of[position]],
                edgecolor=style.PAPER,
                linewidth=0.15,
            )
        )
    for position in distractors:
        ax_role.add_patch(
            Rectangle(
                (position - 0.5, 0.0),
                1.0,
                1.0,
                facecolor=style.INK_FAINT,
                edgecolor=style.INK_STRONG,
                linewidth=0.3,
                hatch="////",
            )
        )
    # The source is a binding like the other five; only the program record says
    # which one the query asks for, so it is outlined rather than recoloured.
    ax_role.add_patch(
        Rectangle(
            (source - 0.5, 0.0),
            1.0,
            1.0,
            fill=False,
            edgecolor=style.INK,
            linewidth=0.6,
            zorder=3,
        )
    )
    bare(ax_role, (-0.5, seq_len - 0.5), (0.0, 1.0))

    # ------------------------------------------------------------------ #
    # The required operation, as an arc over the sequence.
    # ------------------------------------------------------------------ #
    bare(ax_arc, (-0.5, seq_len - 0.5), (0.0, 1.0))
    ax_arc.add_patch(
        FancyArrowPatch(
            (source, 0.06),
            (dest, 0.06),
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="-|>",
            mutation_scale=5.0,
            linewidth=0.9,
            color=style.INK,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=4,
        )
    )
    ax_arc.text(
        0.0,
        0.98,
        f"task {record.family} ({step.op}):\n"
        f"at the query, emit the value bound to key {step.key_id}",
        ha="left",
        va="top",
        fontsize=style.FONT_SIZE_SMALL,
        color=style.INK,
        linespacing=1.2,
    )
    for position, label in ((source, "source"), (dest, "query")):
        ax_arc.annotate(
            f"{label}\n$t={position}$",
            xy=(position, 0.05),
            xytext=(position + 0.6, 0.30),
            ha="right",
            va="bottom",
            fontsize=tiny,
            color=style.INK,
            linespacing=1.1,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": style.INK,
                "shrinkA": 0.5,
                "shrinkB": 0.0,
            },
        )

    # ------------------------------------------------------------------ #
    # The key. Counts come from the record, so they cannot drift from the
    # example being drawn.
    # ------------------------------------------------------------------ #
    bare(ax_key, (0.0, 1.0), (0.0, 1.0))
    swatch_w, swatch_h = 0.030, 0.19

    def entry(col: float, row: float, facecolor: str, text: str, hatch: str | None = None) -> None:
        ax_key.add_patch(
            Rectangle(
                (col, row),
                swatch_w,
                swatch_h,
                facecolor=facecolor,
                edgecolor=style.INK_STRONG,
                linewidth=0.3,
                hatch=hatch,
            )
        )
        ax_key.text(
            col + swatch_w + 0.014,
            row + swatch_h / 2.0,
            text,
            ha="left",
            va="center",
            fontsize=tiny,
            color=style.INK,
        )

    entry(0.0, 0.74, style.INK_FAINT, "content position")
    entry(0.0, 0.42, style.INK_MID, f"key$\\rightarrow$value binding ($\\times${len(bindings)})")
    entry(0.52, 0.74, style.INK, "query (the supervised position)")
    entry(0.52, 0.42, style.INK_FAINT, f"distractor ($\\times${len(distractors)})", hatch="////")

    # A tenth of the bar is exactly zero, so the jump the colormap makes at the
    # first active feature is visible rather than merely described.
    ramp = np.concatenate([np.zeros(26), np.linspace(1e-3, 1.0, 230)]).reshape(1, -1)
    ax_key.imshow(
        ramp,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 0.16, 0.03, 0.22),
        zorder=3,
    )
    ax_key.add_patch(
        Rectangle((0.0, 0.03), 0.16, 0.19, fill=False, edgecolor=style.INK_STRONG, linewidth=0.3)
    )
    ax_key.text(
        0.174,
        0.125,
        "cell darkness = feature magnitude; white = inactive",
        ha="left",
        va="center",
        fontsize=tiny,
        color=style.INK,
    )

    # ------------------------------------------------------------------ #
    # The bottleneck, to scale. F and d are drawn on one axis so that the
    # ratio is read off the picture rather than off the label.
    # ------------------------------------------------------------------ #
    bare(ax_bottle, (0.0, n_features), (0.0, 1.0))
    ax_bottle.add_patch(
        Rectangle(
            (0, 0.60),
            n_features,
            0.32,
            facecolor=style.INK_FAINT,
            edgecolor=style.INK_STRONG,
            linewidth=0.5,
        )
    )
    ax_bottle.add_patch(
        Rectangle(
            (0, 0.06), d_model, 0.32, facecolor=style.INK, edgecolor=style.INK, linewidth=0.5
        )
    )
    ax_bottle.annotate(
        "",
        xy=(d_model / 2.0, 0.40),
        xytext=(d_model / 2.0, 0.58),
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.6, "color": style.INK_STRONG},
    )
    ax_bottle.text(
        n_features / 2.0,
        0.76,
        f"$F = {n_features}$ ground-truth features",
        ha="center",
        va="center",
        fontsize=tiny,
        color=style.INK,
    )
    ax_bottle.text(
        d_model + n_features * 0.018,
        0.22,
        f"$d = {d_model}$ model width  ($F/d = {n_features / d_model:.2f}$: forced superposition)",
        ha="left",
        va="center",
        fontsize=tiny,
        color=style.INK,
    )

    # ------------------------------------------------------------------ #
    # Provenance, inside the image, for the day the PNG is separated from
    # its caption.
    # ------------------------------------------------------------------ #
    for offset, text in zip((0.095, 0.030), _footer_lines(figure1_params(dataset))):
        fig.text(
            0.08 / width,
            offset / height,
            text,
            ha="left",
            va="bottom",
            fontsize=4.2,
            family="monospace",
            color=style.INK_MID,
        )

    return fig

# --------------------------------------------------------------------------- #
# Figure 2 — capability and geometry phase diagram
# --------------------------------------------------------------------------- #

FIGURE2_CAPABILITY = "recall"
FIGURE2_GEOMETRY = "content_purity"
"""The two surfaces §10.2 asks to be drawn over one pair of axes: exact
answer-set recall, prompt 03's frozen primary metric, and the content bank's
mean feature purity, the §6.2 measure declared in this sweep's claim packet.
Both come from ``reporting.tables.phase_report``, which reads recorded runs and
computes nothing that is not already on disk."""

RECALL_FIVE_SEED_MDE = 0.128
"""Prompt 09's five-seed minimum detectable effect for T1 exact recall. Used
here for one thing only: deciding which cells of the map are *capability-tied* —
below what five pairs could resolve — so that the disagreement region can be
outlined. A single pair resolves nothing at all, so this is a lower bound on
what one pair would need and the caption says so."""


def _grid_axes(report: dict) -> tuple[list[dict], list[float]]:
    """The map's rows and columns, derived from the points the sweep recorded.

    Rows are ``(F, d)`` points ordered by bottleneck ratio and, within a ratio,
    by decreasing width — so the two cells that share a ratio are adjacent and
    the "does the ratio alone decide it" question is answerable by looking at two
    neighbouring rows rather than by hunting across the figure.
    """
    seen: dict[tuple[int, int], dict] = {}
    for point in report["points"]:
        key = (point["f_content"], point["d_model"])
        seen.setdefault(
            key,
            {
                "f_content": point["f_content"],
                "f_total": point["f_total"],
                "d_model": point["d_model"],
                "ratio_content": point["ratio_content"],
                "ratio_total": point["ratio_total"],
            },
        )
    rows = sorted(seen.values(), key=lambda row: (row["ratio_content"], -row["d_model"]))
    columns = sorted({point["activation_prob"] for point in report["points"]})
    return rows, columns


def _surface(report: dict, rows: list[dict], columns: list[float]) -> dict:
    """Every panel's matrix, plus the per-cell marks, in one pass over the points.

    ``numpy.nan`` means "no run recorded here", which the figure draws as a
    distinct colour rather than as zero.
    """
    shape = (len(rows), len(columns))
    index = {(row["f_content"], row["d_model"]): i for i, row in enumerate(rows)}
    column_index = {value: j for j, value in enumerate(columns)}

    def blank():
        return np.full(shape, np.nan)

    out = {
        "control_recall": blank(),
        "candidate_recall": blank(),
        "control_purity": blank(),
        "candidate_purity": blank(),
        "recall_difference": blank(),
        "purity_difference": blank(),
    }
    marks: dict[str, list[tuple[int, int]]] = {
        "control_at_chance": [],
        "control_at_ceiling": [],
        "candidate_at_chance": [],
        "candidate_at_ceiling": [],
        "inert": [],
        "either_saturated": [],
        "candidate_ahead_recall": [],
        "candidate_ahead_purity": [],
        "disagreement": [],
    }

    for point in report["points"]:
        i = index[(point["f_content"], point["d_model"])]
        j = column_index[point["activation_prob"]]
        control, candidate = point["control"], point["candidate"]
        out["control_recall"][i, j] = _number(control["recall"])
        out["candidate_recall"][i, j] = _number(candidate["recall"])
        out["control_purity"][i, j] = _number(control["content_purity"])
        out["candidate_purity"][i, j] = _number(candidate["content_purity"])
        out["recall_difference"][i, j] = _number(point["difference"]["recall"])
        out["purity_difference"][i, j] = _number(point["difference"]["content_purity"])

        for side, arm in (("control", "control"), ("candidate", "candidate")):
            state = point["saturation"][arm]
            if state == "at_chance":
                marks[f"{side}_at_chance"].append((i, j))
            elif state == "at_ceiling":
                marks[f"{side}_at_ceiling"].append((i, j))
        if not point["both_mechanisms_active"]:
            marks["inert"].append((i, j))
        if set(point["saturation"].values()) - {"interior"}:
            marks["either_saturated"].append((i, j))

        recall_difference = point["difference"]["recall"]
        purity_difference = point["difference"]["content_purity"]
        if isinstance(recall_difference, (int, float)) and recall_difference < 0:
            marks["candidate_ahead_recall"].append((i, j))
        if isinstance(purity_difference, (int, float)) and purity_difference < 0:
            marks["candidate_ahead_purity"].append((i, j))
        if (
            isinstance(recall_difference, (int, float))
            and isinstance(purity_difference, (int, float))
            and abs(recall_difference) < RECALL_FIVE_SEED_MDE
            and abs(purity_difference) > PURITY_FIVE_SEED_MDE
        ):
            marks["disagreement"].append((i, j))

    return {"matrices": out, "marks": marks}


def _number(value):
    return float(value) if isinstance(value, (int, float)) else np.nan


def _finite_range(*arrays) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    return (low, high) if high > low else (low, low + 1.0)


def figure2_params(report: dict) -> dict:
    """Every input needed to regenerate figure 2, read off the recorded sweep."""
    rows, columns = _grid_axes(report)
    surface = _surface(report, rows, columns)
    marks = surface["marks"]
    matrices = surface["matrices"]
    points = report["points"]

    def cell_label(index: tuple[int, int]) -> str:
        row, column = rows[index[0]], columns[index[1]]
        return f"F{row['f_content']}/d{row['d_model']}@p{column:.2f}"

    controls = {}
    for name, entries in sorted(report["controls"].items()):
        controls[name] = [
            {
                "cell": entry["cell"],
                "d_model": entry["d_model"],
                "seq_len": entry["seq_len"],
                "control_recall": entry["control"]["recall"],
                "candidate_recall": entry["candidate"]["recall"],
                "control_skill": entry["control"]["recall_skill"],
                "candidate_skill": entry["candidate"]["recall_skill"],
                "control_purity": entry["control"]["content_purity"],
                "candidate_purity": entry["candidate"]["content_purity"],
                "both_mechanisms_active": entry["both_mechanisms_active"],
            }
            for entry in entries
        ]

    purity_low, purity_high = _finite_range(
        matrices["control_purity"], matrices["candidate_purity"]
    )
    return {
        "figure_version": FIGURE_VERSION,
        "schema": report["schema"],
        "ladder": report["ladder"],
        "seed": sorted({point["seed"] for point in points}),
        "n_points": len(points),
        "n_runs": 2 * (len(points) + sum(len(v) for v in report["controls"].values())),
        "main_seq_len": report["main_seq_len"],
        "content_group_size": report["content_group_size"],
        "rows": rows,
        "sparsities": columns,
        "window": report["window"],
        "recall_five_seed_mde": RECALL_FIVE_SEED_MDE,
        "purity_five_seed_mde": PURITY_FIVE_SEED_MDE,
        "purity_scale": [round(purity_low, 4), round(purity_high, 4)],
        "max_abs_recall_difference": round(
            float(np.nanmax(np.abs(matrices["recall_difference"]))), 4
        ),
        "max_abs_purity_difference": round(
            float(np.nanmax(np.abs(matrices["purity_difference"]))), 4
        ),
        "n_interior": sum(1 for point in points if point["both_alive"]),
        "n_control_at_chance": len(marks["control_at_chance"]),
        "n_control_at_ceiling": len(marks["control_at_ceiling"]),
        "n_candidate_at_chance": len(marks["candidate_at_chance"]),
        "n_candidate_at_ceiling": len(marks["candidate_at_ceiling"]),
        "n_inert": len(marks["inert"]),
        "n_candidate_ahead_recall": len(marks["candidate_ahead_recall"]),
        "disagreement_cells": sorted(cell_label(index) for index in marks["disagreement"]),
        "controls": controls,
        "cuts": [entry["cut"] for entry in report["cuts"]],
        "resolution": report["resolution"],
        **style.style_provenance(),
    }


def figure2_caption(params: dict) -> str:
    """The caption. One seed and screening depth are its first two sentences.

    Not a stylistic choice: a figure outlives its caption's context far more
    often than it outlives its caption, and a phase diagram read as a replicated
    result is the specific misreading this mission was told to make impossible.
    """
    rows = params["rows"]
    ratios = sorted({row["ratio_content"] for row in rows})
    disagreement = params["disagreement_cells"]
    controls = params["controls"]
    null_pair = (controls.get("phase_negative_control_d32") or [{}])[0]
    r1_pair = (controls.get("phase_r1") or [{}])[0]
    ribbon = controls.get("phase_length_d32") or []
    ribbon_text = "; ".join(
        f"T={entry['seq_len']}: A0 {_caption_number(entry['control_recall'])}, "
        f"A1 {_caption_number(entry['candidate_recall'])}"
        for entry in sorted(ribbon, key=lambda entry: entry["seq_len"] or 0)
    )
    disagreement_text = (
        "none of the cells met both conditions"
        if not disagreement
        else ", ".join(f"`{name}`" for name in disagreement)
    )
    return (
        "**Figure 2. Capability and geometry over the same axes — SCREENING "
        "EVIDENCE, ONE SEED PER CELL.** Every cell of this map is a single "
        f"training run per architecture at seed {params['seed'][0]} and §7.3's "
        f"{params['ladder']} screening budget; nothing in it is replicated and "
        "nothing in it is claimed at claim-ladder rung 2 or 3. Prompt 09 "
        "measured that *five* paired seeds at this scale cannot resolve an exact "
        f"recall difference below {params['recall_five_seed_mde']} or a mean "
        f"purity difference below {params['purity_five_seed_mde']}, and a single "
        "pair has no interval at all, so the panels are a map of where to look "
        "and not a measurement of how much. "
        "Top row: exact answer-set recall on held-out programs. Bottom row: mean "
        "feature purity of the content bank at each run's primary site "
        "(`final_norm`). Columns: A0 (causal softmax attention), A1 (kernelized "
        "linear attention, phi = elu + 1, no erasure), and the paired difference "
        "A0 − A1 with magnitude as darkness — cells where A1 leads carry a minus "
        "sign. "
        f"Rows are the {len(rows)} (F, d) points of §4.5's grid ordered by "
        f"bottleneck ratio F/d ({', '.join(f'{ratio:g}' for ratio in ratios)}); "
        "three of those ratios are realised by two different (F, d) pairs, drawn "
        "as adjacent rows, so a reader can see directly whether the ratio alone "
        "decides the result. The heavy rule marks F/d = 1, where superposition "
        "of the content bank becomes forced; F here is the content bank and a "
        "further 28 addressing features (24 key, 4 operator) sit in every cell, "
        "so the whole-bank ratio is (F+28)/d and crosses 1 within the same band. "
        f"Columns are the per-group activation probability at a fixed group of "
        f"{params['content_group_size']} content features, so a position carries "
        f"{params['content_group_size']}·p active content features in expectation "
        "at every F and the sparsity axis means the same thing on every row. "
        f"Sequence length is {params['main_seq_len']} throughout the map"
        + (f"; the ribbon at F=64/d=32/p=0.12 reads {ribbon_text}. " if ribbon_text else ". ")
        + "Open circles mark cells at or below the pre-registered chance floor "
        f"({params['window']['floor']}), filled triangles cells at or above the "
        f"ceiling ({params['window']['ceiling']}); saturated cells cannot carry a "
        "difference in either direction and the difference panels shade them. "
        f"{params['n_interior']} of {params['n_points']} cells have both "
        "architectures strictly inside that window. Heavy outlines mark the "
        "cells where the two surfaces disagree — the paired recall difference "
        "below what five seeds could resolve while the paired purity difference "
        f"is above it: {disagreement_text}. "
        + (
            "Every mechanism was active at every cell by §6.3's three gates. "
            if params["n_inert"] == 0
            else f"{params['n_inert']} cell(s) had an inactive mechanism and are marked; "
            "no difference at those cells separates two mechanisms. "
        )
        + "Controls: the known-easy positive control returns A0 "
        f"{_caption_number(r1_pair.get('control_recall'))} against A1 "
        f"{_caption_number(r1_pair.get('candidate_recall'))}, a comparison whose "
        "answer is known in advance to be null; §4.4's information-destroyed "
        "condition at the map's middle point returns A0 "
        f"{_caption_number(null_pair.get('control_recall'))} and A1 "
        f"{_caption_number(null_pair.get('candidate_recall'))}. "
        f"{params['n_runs']} runs in total, every §7.2 variable shared between "
        "the arms of every pair, `permitted_differences` empty. Regenerate with "
        "`python -m architecture_mechanics.reporting.figures --figure 2`; "
        f"figure {params['figure_version']}, style "
        f"{params['figure_style_version']}, matplotlib "
        f"{params['matplotlib_version']}, grid and cuts in "
        "`architecture_mechanics.experiments.phase_grid`, pre-registration "
        "`claims/phase-map-a0-a1-sparsity-bottleneck.yml`."
    )


def _caption_number(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "not recorded"


def draw_figure2(report: dict) -> Figure:
    """Draw the phase diagram: two surfaces, three columns, one pair of axes.

    Laid out in inches for the same reason figure 1 is — a figure claiming to be
    page width must be page width, and the bytes must not move when matplotlib's
    layout heuristics change.
    """
    style.apply_style()
    rows, columns = _grid_axes(report)
    surface = _surface(report, rows, columns)
    matrices, marks = surface["matrices"], surface["marks"]
    params = figure2_params(report)

    width = style.PAGE_WIDTH_IN
    height = 5.0
    fig = Figure(figsize=(width, height), dpi=style.SAVE_DPI)
    fig.patch.set_facecolor(style.PAPER)

    def rect(x0: float, y0: float, w: float, h: float):
        return fig.add_axes([x0 / width, y0 / height, w / width, h / height])

    left, panel_w, gap = 0.95, 1.72, 0.20
    xs = [left + index * (panel_w + gap) for index in range(3)]
    row_y = [3.18, 1.44]
    panel_h = 1.34
    cmap = style.sequential_colormap()

    purity_low, purity_high = params["purity_scale"]
    recall_difference_max = max(params["max_abs_recall_difference"], 1e-6)
    purity_difference_max = max(params["max_abs_purity_difference"], 1e-6)

    # The heavy rule sits above the first row whose bottleneck ratio reaches 1:
    # every row below it superposes more content features than the residual
    # stream has dimensions.
    transition = next(
        (index for index, row in enumerate(rows) if row["ratio_content"] >= 1.0), None
    )

    panels = (
        ("A0  softmax attention", matrices["control_recall"], 0.0, 1.0, "control"),
        ("A1  linear attention", matrices["candidate_recall"], 0.0, 1.0, "candidate"),
        ("A0 $-$ A1", np.abs(matrices["recall_difference"]), 0.0, recall_difference_max, "diff"),
        (None, matrices["control_purity"], purity_low, purity_high, "control"),
        (None, matrices["candidate_purity"], purity_low, purity_high, "candidate"),
        (
            None,
            np.abs(matrices["purity_difference"]),
            0.0,
            purity_difference_max,
            "diff",
        ),
    )

    for index, (header, matrix, vmin, vmax, kind) in enumerate(panels):
        band, column = divmod(index, 3)
        axes = rect(xs[column], row_y[band], panel_w, panel_h)
        _draw_panel(
            axes,
            matrix,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rows=rows,
            columns=columns,
            show_row_labels=column == 0,
            transition=transition,
        )
        if header:
            axes.set_title(header, fontsize=style.FONT_SIZE_SMALL, pad=3.0)
        if column == 0:
            axes.set_ylabel(
                "exact recall" if band == 0 else "content-feature purity",
                fontsize=style.FONT_SIZE_SMALL,
                labelpad=20.0,
            )
        _mark_panel(
            axes,
            matrix,
            vmin=vmin,
            vmax=vmax,
            marks=marks,
            kind=kind,
            band=band,
        )

    fig.text(
        (xs[1] + panel_w / 2.0) / width,
        (row_y[1] - 0.30) / height,
        "per-group activation probability  (denser $\\rightarrow$)",
        ha="center",
        va="center",
        fontsize=style.FONT_SIZE_SMALL,
    )
    fig.text(
        (xs[0] - 0.72) / width,
        (row_y[0] + panel_h + 0.30) / height,
        "bottleneck ratio $F/d$   (tighter $\\downarrow$)",
        ha="left",
        va="center",
        fontsize=style.FONT_SIZE_SMALL,
    )

    _draw_colourbars(
        fig,
        width=width,
        height=height,
        cmap=cmap,
        scales=(
            ("exact recall", 0.0, 1.0),
            ("|A0 $-$ A1| recall", 0.0, recall_difference_max),
            ("content purity", purity_low, purity_high),
            ("|A0 $-$ A1| purity", 0.0, purity_difference_max),
        ),
    )
    _draw_figure2_legend(fig, width=width, height=height, params=params)
    return fig


def _draw_panel(
    axes,
    matrix,
    *,
    vmin: float,
    vmax: float,
    cmap,
    rows: list[dict],
    columns: list[float],
    show_row_labels: bool,
    transition: int | None,
) -> None:
    n_rows, n_columns = matrix.shape
    axes.set_facecolor(style.MISSING)
    axes.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, n_columns - 0.5, n_rows - 0.5, -0.5),
    )
    axes.set_xlim(-0.5, n_columns - 0.5)
    axes.set_ylim(n_rows - 0.5, -0.5)
    axes.set_xticks(range(n_columns))
    axes.set_xticklabels([f"{value:g}" for value in columns], fontsize=tiny_size())
    axes.set_yticks(range(n_rows))
    axes.set_yticklabels(
        [f"{row['ratio_content']:g}  {row['f_content']}/{row['d_model']}" for row in rows]
        if show_row_labels
        else [""] * n_rows,
        fontsize=tiny_size(),
    )
    axes.tick_params(length=1.5, pad=1.5)
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(style.INK_STRONG)
    # Cell borders, so a saturated mark reads as belonging to one cell.
    for row in range(1, n_rows):
        axes.axhline(row - 0.5, color=style.PAPER, linewidth=0.4)
    for column in range(1, n_columns):
        axes.axvline(column - 0.5, color=style.PAPER, linewidth=0.4)
    if transition:
        axes.axhline(
            transition - 0.5, color=style.INK, linewidth=1.1, solid_capstyle="butt"
        )


def tiny_size() -> float:
    return style.FONT_SIZE_TINY


def _mark_panel(axes, matrix, *, vmin, vmax, marks, kind, band) -> None:
    """Saturation, inert mechanisms, sign and the disagreement region.

    Marks rather than colours, because this programme's figures carry no hue and
    a second greyscale ramp on top of the first would be unreadable. Each mark's
    ink is chosen against the cell it sits on, so the darkest and lightest cells
    — which are exactly the saturated ones — stay legible.

    ``kind`` is ``control``, ``candidate`` or ``diff``: an arm panel marks its
    own arm's saturation, and a difference panel marks the cells where *either*
    arm is saturated, because a difference between two saturated numbers is not
    a difference between two architectures.
    """
    span = (vmax - vmin) or 1.0

    def ink_at(index: tuple[int, int]) -> str:
        value = matrix[index[0], index[1]]
        if not np.isfinite(value):
            return style.INK
        return style.contrast_ink((float(value) - vmin) / span)

    if kind in ("control", "candidate"):
        side = kind
        for index in marks[f"{side}_at_chance"]:
            axes.plot(
                index[1], index[0], marker="o", markersize=1.9, markerfacecolor="none",
                markeredgecolor=ink_at(index), markeredgewidth=0.45, linestyle="none",
            )
        for index in marks[f"{side}_at_ceiling"]:
            axes.plot(
                index[1], index[0], marker="^", markersize=2.1,
                markerfacecolor=ink_at(index), markeredgecolor=ink_at(index),
                markeredgewidth=0.0, linestyle="none",
            )
        for index in marks["inert"]:
            axes.plot(
                index[1], index[0], marker="x", markersize=2.4,
                markeredgecolor=ink_at(index), markeredgewidth=0.6, linestyle="none",
            )
        return

    for index in marks["either_saturated"]:
        axes.add_patch(
            Rectangle(
                (index[1] - 0.5, index[0] - 0.5),
                1.0,
                1.0,
                fill=False,
                hatch="////",
                edgecolor=ink_at(index),
                linewidth=0.0,
            )
        )
    ahead = marks["candidate_ahead_recall"] if band == 0 else marks["candidate_ahead_purity"]
    for index in ahead:
        axes.text(
            index[1], index[0], "$-$", ha="center", va="center",
            fontsize=style.FONT_SIZE_TINY, color=ink_at(index),
        )
    for index in marks["disagreement"]:
        axes.add_patch(
            Rectangle(
                (index[1] - 0.5, index[0] - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=ink_at(index),
                linewidth=1.0,
                zorder=6,
            )
        )


def _draw_colourbars(fig, *, width, height, cmap, scales) -> None:
    bar_w, bar_h, bar_y = 1.20, 0.075, 0.86
    xs = [0.52 + index * 1.62 for index in range(len(scales))]
    ramp = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    for x0, (label, low, high) in zip(xs, scales, strict=True):
        axes = fig.add_axes([x0 / width, bar_y / height, bar_w / width, bar_h / height])
        axes.imshow(ramp, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        axes.set_yticks([])
        axes.set_xticks([0, 255])
        axes.set_xticklabels([f"{low:g}", f"{high:.3g}"], fontsize=style.FONT_SIZE_TINY)
        axes.tick_params(length=1.2, pad=1.0)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color(style.INK_STRONG)
        axes.set_title(label, fontsize=style.FONT_SIZE_TINY, pad=1.8)


def _draw_figure2_legend(fig, *, width, height, params) -> None:
    """The marks, then the two lines that stop this being read as a result."""
    axes = fig.add_axes([0.30 / width, 0.40 / height, (width - 0.60) / width, 0.30 / height])
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.patch.set_alpha(0.0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    entries = (
        ("o", f"at chance ($\\leq$ {params['window']['floor']:.2f})", False),
        ("^", f"at ceiling ($\\geq$ {params['window']['ceiling']:.2f})", True),
        ("x", "mechanism inert", False),
    )
    x = 0.0
    for marker, label, filled in entries:
        axes.plot(
            [x], [0.72], marker=marker, markersize=2.2,
            markerfacecolor=style.INK if filled else "none",
            markeredgecolor=style.INK, markeredgewidth=0.5, linestyle="none",
            clip_on=False,
        )
        axes.text(x + 0.016, 0.72, label, ha="left", va="center", fontsize=style.FONT_SIZE_TINY)
        x += 0.20
    axes.add_patch(
        Rectangle((x, 0.60), 0.013, 0.24, fill=False, edgecolor=style.INK, linewidth=1.0,
                  clip_on=False)
    )
    axes.text(
        x + 0.020,
        0.72,
        "capability tied, geometry not",
        ha="left",
        va="center",
        fontsize=style.FONT_SIZE_TINY,
    )
    x += 0.29
    axes.add_patch(
        Rectangle((x, 0.60), 0.013, 0.24, fill=False, hatch="////", edgecolor=style.INK_STRONG,
                  linewidth=0.0, clip_on=False)
    )
    axes.text(
        x + 0.020,
        0.72,
        "an arm is saturated",
        ha="left",
        va="center",
        fontsize=style.FONT_SIZE_TINY,
    )

    axes.text(
        0.0,
        0.16,
        "ONE SEED PER CELL, R2 screening budget — a map of where to look, not a "
        "measurement of how much. Five paired seeds could not resolve a recall "
        f"difference below {params['recall_five_seed_mde']:.3f} or a purity "
        f"difference below {params['purity_five_seed_mde']:.3f}.",
        ha="left",
        va="center",
        fontsize=style.FONT_SIZE_TINY,
        color=style.INK,
    )

    controls = params["controls"]
    ribbon = controls.get("phase_length_d32") or []
    null_pair = (controls.get("phase_negative_control_d32") or [{}])[0]
    r1_pair = (controls.get("phase_r1") or [{}])[0]
    lines = (
        (
            "R1 positive control (known-easy, answer known to be null): "
            f"A0 {_caption_number(r1_pair.get('control_recall'))}  "
            f"A1 {_caption_number(r1_pair.get('candidate_recall'))}   |   "
            "information-destroyed control at F64/d32/p0.12: "
            f"A0 {_caption_number(null_pair.get('control_recall'))}  "
            f"A1 {_caption_number(null_pair.get('candidate_recall'))}"
        ),
        "sequence-length ribbon at F64/d32/p0.12: "
        + "  ".join(
            f"T={entry['seq_len']} A0 {_caption_number(entry['control_recall'])} "
            f"A1 {_caption_number(entry['candidate_recall'])}"
            for entry in sorted(ribbon, key=lambda entry: entry["seq_len"] or 0)
        )
        + f"   |   map drawn at T={params['main_seq_len']}",
    )
    for offset, text in zip((0.22, 0.10), lines, strict=True):
        fig.text(
            0.30 / width,
            offset / height,
            text,
            ha="left",
            va="bottom",
            fontsize=4.2,
            family="monospace",
            color=style.INK_MID,
        )


# --------------------------------------------------------------------------- #
# Build, save, record
# --------------------------------------------------------------------------- #


def build_figure2(out_dir: Path) -> FigureResult:
    """Read the recorded sweep, draw it, save it, and write its caption."""
    report = phase_report()
    if not report["points"]:
        raise ValueError(
            "no recorded phase-sweep runs to draw: reports/comparisons holds no resolved "
            "declaration for any phase_T* comparison. Run `make phase-sweep` first; this "
            "figure reads recorded artifacts and never re-runs anything."
        )
    params = figure2_params(report)
    fig = draw_figure2(report)
    stem = FIGURE_STEMS[2]
    path = Path(out_dir) / f"{stem}.png"
    digest = style.save_png(fig, path)
    caption = figure2_caption(params)
    Path(out_dir).joinpath(f"{stem}.caption.md").write_text(caption + "\n")
    return FigureResult(number=2, path=path, sha256=digest, caption=caption, params=params)


def build_figure1(out_dir: Path) -> FigureResult:
    """Generate the source example, draw it, save it, and write its caption."""
    dataset = figure1_dataset()
    params = figure1_params(dataset)
    fig = draw_figure1(dataset)
    stem = FIGURE_STEMS[1]
    path = Path(out_dir) / f"{stem}.png"
    digest = style.save_png(fig, path)
    caption = figure1_caption(params)
    Path(out_dir).joinpath(f"{stem}.caption.md").write_text(caption + "\n")
    return FigureResult(number=1, path=path, sha256=digest, caption=caption, params=params)


BUILDERS = {1: build_figure1, 2: build_figure2}


def build_figure(number: int, out_dir: Path) -> FigureResult:
    if number not in BUILDERS:
        planned = {3: "prompt 22", 4: "prompt 23"}
        if number in planned:
            raise NotImplementedError(
                f"figure {number} (north star 10.2) is owned by {planned[number]}; "
                "there are no results to draw it from yet"
            )
        raise ValueError(f"no such figure: {number}")
    return BUILDERS[number](Path(out_dir))


def write_index(out_dir: Path, results: list[FigureResult]) -> Path:
    """An index of every figure with its hash, so a stale PNG is detectable.

    Carries no timestamp: the point of the file is that it changes when, and
    only when, a figure changes.
    """
    path = Path(out_dir) / "INDEX.json"
    existing: dict[str, dict] = {}
    if path.exists():
        existing = {entry["number"]: entry for entry in json.loads(path.read_text())["figures"]}
        existing = {str(k): v for k, v in existing.items()}
    for result in results:
        existing[str(result.number)] = result.as_dict()
    ordered = [existing[key] for key in sorted(existing, key=int)]
    path.write_text(
        json.dumps({"schema": "am.figures_index.v1", "figures": ordered}, indent=2, sort_keys=True)
        + "\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a paper figure from recorded artifacts.",
    )
    parser.add_argument("--figure", type=int, required=True, help="figure number (north star 10.2)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"output directory (default: <lab>/{DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--verify-deterministic",
        action="store_true",
        help="delete the PNG, regenerate it, and fail unless the bytes are identical",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else lab_root() / DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_figure(args.figure, out_dir)
    index_path = write_index(out_dir, [result])

    second: str | None = None
    if args.verify_deterministic:
        result.path.unlink()
        second = build_figure(args.figure, out_dir).sha256

    if not args.quiet:
        print(f"figure {result.number}: {result.path}")
        print(f"  sha256   {result.sha256}")
        if second is not None:
            print(f"  regenerated after delete: {second}")
            print(f"  byte-identical: {second == result.sha256}")
        print(f"  index    {index_path}")
        print(f"  caption  {result.caption}")

    if second is not None and second != result.sha256:
        print("FAIL: regeneration is not byte-identical", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
