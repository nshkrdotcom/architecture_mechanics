"""Paper figures, generated only from recorded artifacts.

§8.5's last required test is a property of this module: *report generated only
from recorded artifacts*. It is enforced two ways. The narrow way is
:data:`ARTIFACT_READ_ROOTS` — while a figure is being built, nothing inside this
laboratory may be opened except ``runs/`` and ``reports/``, and
``tests/reporting/test_figure_provenance.py`` audits every file the process
opens to prove it. The wide way is that there is no code here that draws a
number: every mark on every figure comes from a tensor the generator produced or
a value a recorded run wrote down. A schematic that illustrates what the
benchmark *would* look like is worth less than no figure at all, because it is
indistinguishable from one that is true.

Figure 1 has no parent run: it is the benchmark itself, drawn from one example
the generator produces on demand. Its inputs are therefore its dataset
configuration, and the caption carries all of them.

Figures 2-4 (north star 10.2) arrive at prompts 14, 22 and 23 and will read
recorded run outputs under ``runs/``. They share
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

FIGURE_VERSION = "am-fig-1.0.0"

ARTIFACT_READ_ROOTS: tuple[str, ...] = ("runs", "reports")
"""The only directories inside the laboratory a figure may read.

``runs/`` holds recorded evidence and ``reports/`` holds derived artifacts and
this module's own output. Everything else — ``configs/``, ``claims/``,
``paper/``, a scratch file in the working directory — is off limits, so a figure
cannot quietly acquire a number that no run produced. Datasets are not read at
all: they are regenerated in process from a configuration recorded in the
caption, which is stronger than reading a file, because it cannot drift from
the generator.
"""

DEFAULT_OUT_DIR = "reports/figures"


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
# Build, save, record
# --------------------------------------------------------------------------- #


def build_figure1(out_dir: Path) -> FigureResult:
    """Generate the source example, draw it, save it, and write its caption."""
    dataset = figure1_dataset()
    params = figure1_params(dataset)
    fig = draw_figure1(dataset)
    path = Path(out_dir) / "figure1_benchmark.png"
    digest = style.save_png(fig, path)
    caption = figure1_caption(params)
    Path(out_dir).joinpath("figure1_benchmark.caption.md").write_text(caption + "\n")
    return FigureResult(number=1, path=path, sha256=digest, caption=caption, params=params)


BUILDERS = {1: build_figure1}


def build_figure(number: int, out_dir: Path) -> FigureResult:
    if number not in BUILDERS:
        planned = {2: "prompt 14", 3: "prompt 22", 4: "prompt 23"}
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
