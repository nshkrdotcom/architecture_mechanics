"""Shared figure style: sizes, ink, a greyscale-safe palette, deterministic save.

Four figures are planned (north star 10.2) and they arrive at prompts 06, 14,
22 and 23 — months apart, in different sessions. A style decided once and
imported is the difference between a paper and a scrapbook, so every rule this
programme's figures obey lives here and nowhere else.

Two properties are enforced rather than asserted:

**Greyscale safety.** A palette is greyscale-safe when its entries differ in
*luminance*, not merely in hue: printed in black and white, or read by someone
with deuteranopia, hue carries nothing. :data:`SERIES` is ordered by increasing
relative luminance with a minimum separation, and a test recomputes those
luminances rather than trusting this docstring. Every series also carries a
distinct linestyle and marker, so the encoding is redundant three times over.

**Byte-identical regeneration.** A figure that changes when nothing changed
makes review impossible: the reviewer cannot tell a re-render from a new
result. Matplotlib does not give this for free, so :func:`apply_style` resets
every rcParam to matplotlib's own defaults before applying ours (a machine with
a ``matplotlibrc`` would otherwise render differently), pins the font family to
the one matplotlib bundles, and forces the Agg backend; :func:`save_png` then
strips the ``Software`` metadata chunk, which is the only non-content bytes
matplotlib writes into a PNG. What remains depends on the matplotlib version
and this module, both of which are recorded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

FIGURE_STYLE_VERSION = "am-figstyle-1.0.0"

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

COLUMN_WIDTH_IN = 3.4
"""One column of a two-column paper. The default: a figure that survives here
survives anywhere, and the constraint forces the drawing to be simple."""

PAGE_WIDTH_IN = 7.0
"""Full text width, for a figure that genuinely cannot be read at one column."""

SAVE_DPI = 400
"""Fixed, because it is part of the bytes. High enough that a single feature
cell in figure 1 is several pixels tall."""

# --------------------------------------------------------------------------- #
# Type
# --------------------------------------------------------------------------- #

FONT_SIZE = 7.0
"""Body size. 7pt at column width is about 9pt at page width once a reader
scales the figure, which is the smallest comfortable size in print."""

FONT_SIZE_SMALL = 6.0
FONT_SIZE_TINY = 5.0
"""Provenance footers only. Legible when zoomed, invisible when not, which is
the correct priority for a string nobody reads until they want to rerun it."""

FONT_SIZE_TITLE = 8.0

# --------------------------------------------------------------------------- #
# Ink
# --------------------------------------------------------------------------- #

INK = "#000000"
INK_STRONG = "#333333"
INK_MID = "#777777"
INK_LIGHT = "#b4b4b4"
INK_FAINT = "#dcdcdc"
PAPER = "#ffffff"
"""A neutral ramp. Figure 1 is drawn entirely from these: a schematic that is
already black and white cannot fail a greyscale check."""


@dataclass(frozen=True)
class SeriesStyle:
    """One plotted series: colour, dash pattern and marker, all three distinct."""

    name: str
    color: str
    linestyle: str | tuple
    marker: str


SERIES: tuple[SeriesStyle, ...] = (
    SeriesStyle("s0", "#000000", "-", "o"),
    SeriesStyle("s1", "#35618f", (0, (4, 1.6)), "s"),
    SeriesStyle("s2", "#d1652e", (0, (1.4, 1.2)), "^"),
    SeriesStyle("s3", "#6fb3b3", (0, (5, 1.4, 1.2, 1.4)), "D"),
    SeriesStyle("s4", "#e0c34a", (0, (3, 1.2, 1.2, 1.2, 1.2, 1.2)), "v"),
)
"""Up to five architectures on one axis. Ordered by increasing luminance with a
minimum pairwise separation of :data:`MIN_LUMINANCE_SEPARATION`; the ordering is
the point, because a reader converting to greyscale gets a legible ranking
rather than five indistinguishable mid-greys."""

MIN_LUMINANCE_SEPARATION = 0.10
"""Enforced by ``tests/reporting/test_figure_style.py``. 0.10 in relative
luminance is roughly the smallest difference that survives a laser printer."""


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of an ``#rrggbb`` colour, in ``[0, 1]``."""
    hexed = color.lstrip("#")
    if len(hexed) != 6:
        raise ValueError(f"expected #rrggbb, got {color!r}")
    channels = [int(hexed[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

PNG_METADATA: dict[str, None] = {"Software": None}
"""Matplotlib stamps its own version into a PNG ``tEXt`` chunk by default. It
carries no information about the experiment and it changes the bytes on every
matplotlib upgrade, so it is dropped; ``None`` removes the entry rather than
blanking it. Matplotlib writes no timestamp chunk, which a test verifies
instead of assuming."""

_STYLE_RCPARAMS: dict[str, object] = {
    # Type. DejaVu ships inside matplotlib, so the glyphs are pinned by the
    # matplotlib version rather than by whatever fonts this machine has.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.monospace": ["DejaVu Sans Mono"],
    "mathtext.fontset": "dejavusans",
    "text.usetex": False,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE_TITLE,
    "xtick.labelsize": FONT_SIZE_SMALL,
    "ytick.labelsize": FONT_SIZE_SMALL,
    "legend.fontsize": FONT_SIZE_SMALL,
    # Ink.
    "figure.facecolor": PAPER,
    "figure.edgecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.edgecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": INK_STRONG,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_STRONG,
    "ytick.color": INK_STRONG,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "grid.color": INK_FAINT,
    "grid.linewidth": 0.4,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    # Rasterisation. Fixed because they are part of the bytes.
    "figure.dpi": SAVE_DPI,
    "savefig.dpi": SAVE_DPI,
    "image.interpolation": "nearest",
    "image.resample": False,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "path.simplify": True,
    "path.simplify_threshold": 0.111111111111,
    "agg.path.chunksize": 0,
    "svg.hashsalt": "architecture-mechanics",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Reset matplotlib to its defaults, then apply this programme's style.

    The reset matters more than the style: without it a machine carrying a
    ``matplotlibrc`` renders different bytes from the same code, and the
    delete-and-regenerate check would pass locally while failing for a
    replicator. Idempotent, and safe to call from every figure.
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(_STYLE_RCPARAMS)


MAGNITUDE_FLOOR = 0.35
"""Ink given to a feature whose magnitude is barely above zero.

The generator draws active magnitudes ``Uniform(0, 1)``, so a linear
white-to-black ramp renders a real feature at 0.03 as white — visually
identical to an inactive one, which is the one distinction the figure exists to
show. The ramp therefore jumps to this ink level the instant a feature is
active and is linear from there. Disclosed in the colourbar, whose axis is
still the true magnitude.
"""


def magnitude_colormap() -> LinearSegmentedColormap:
    """Transparent at exactly zero, then a linear ink ramp from :data:`MAGNITUDE_FLOOR`.

    Use with ``vmin=0, vmax=1``. The discontinuity is deliberate and is the
    honest way to draw "inactive" and "active but small" as different things.

    Zero is transparent rather than white so that a shaded band behind the
    matrix — marking distractor positions, say — shows through the inactive
    cells instead of being painted over by them. On this programme's white
    figures the two render identically everywhere else.
    """
    knee = 1e-6
    floor = 1.0 - MAGNITUDE_FLOOR
    return LinearSegmentedColormap.from_list(
        "am_magnitude",
        [
            (0.0, (1.0, 1.0, 1.0, 0.0)),
            (knee, (floor, floor, floor, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 1.0)),
        ],
        N=1024,
    )


def sha256_file(path: str | Path) -> str:
    """Hex digest of a file's bytes. The unit a figure's reproducibility is in."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_png(fig, path: str | Path) -> str:
    """Save deterministically and return the sha256 of the bytes written.

    No ``bbox_inches="tight"``: it silently changes the figure's width, and a
    figure claiming to be column-width must actually be column-width.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=SAVE_DPI, metadata=PNG_METADATA)
    return sha256_file(out)


def style_provenance() -> dict:
    """What a reader needs to know about how the pixels were made."""
    return {
        "figure_style_version": FIGURE_STYLE_VERSION,
        "matplotlib_version": matplotlib.__version__,
        "dpi": SAVE_DPI,
        "column_width_in": COLUMN_WIDTH_IN,
        "font_size_pt": FONT_SIZE,
        "magnitude_floor": MAGNITUDE_FLOOR,
    }
