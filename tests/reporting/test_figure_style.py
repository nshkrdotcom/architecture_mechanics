"""The shared figure style, checked rather than described.

Two of this module's promises are the kind that quietly stop being true: a
palette said to be greyscale-safe, and a style said to be deterministic. Both
are recomputed here from the values themselves, so a later prompt that adds a
sixth series or a prettier colour finds out immediately.
"""

from __future__ import annotations

import matplotlib
import pytest

from architecture_mechanics.reporting import figure_style as style


def test_series_palette_is_ordered_by_luminance():
    luminances = [style.relative_luminance(s.color) for s in style.SERIES]
    assert luminances == sorted(luminances), (
        "series must be ordered light-to-dark so a greyscale reader gets a "
        f"legible ranking; got {luminances}"
    )


def test_series_palette_survives_conversion_to_greyscale():
    luminances = [style.relative_luminance(s.color) for s in style.SERIES]
    gaps = [
        (a.name, b.name, abs(la - lb))
        for a, la in zip(style.SERIES, luminances)
        for b, lb in zip(style.SERIES, luminances)
        if a.name < b.name
    ]
    worst = min(gaps, key=lambda item: item[2])
    assert worst[2] >= style.MIN_LUMINANCE_SEPARATION, (
        f"{worst[0]} and {worst[1]} differ by only {worst[2]:.4f} in luminance; "
        "printed in black and white they would be the same line"
    )


def test_series_encoding_is_redundant_three_ways():
    """Colour alone fails for a colourblind reader; dashes alone fail on a
    dense plot. Every series differs in all three channels."""
    for attribute in ("color", "linestyle", "marker"):
        values = [getattr(s, attribute) for s in style.SERIES]
        assert len(set(map(str, values))) == len(values), f"duplicate {attribute}"


def test_relative_luminance_matches_known_anchors():
    assert style.relative_luminance("#000000") == pytest.approx(0.0)
    assert style.relative_luminance("#ffffff") == pytest.approx(1.0)
    assert style.relative_luminance("#808080") == pytest.approx(0.2159, abs=1e-3)


def test_relative_luminance_rejects_a_short_colour():
    with pytest.raises(ValueError):
        style.relative_luminance("#abc")


def test_magnitude_colormap_separates_inactive_from_barely_active():
    """The whole point of the floor: a feature at 0.01 must not render as white.

    The generator draws magnitudes ``Uniform(0, 1)``, so without this the
    figure would show "inactive" and "active but small" identically — the one
    distinction the benchmark is about.
    """
    cmap = style.magnitude_colormap()
    inactive = cmap(0.0)
    barely = cmap(0.01)
    assert inactive[3] == 0.0, "exactly zero must be transparent"
    assert barely[3] == 1.0, "an active feature must be opaque"
    assert barely[0] <= 1.0 - style.MAGNITUDE_FLOOR + 1e-6


def test_magnitude_colormap_is_monotone_in_ink():
    cmap = style.magnitude_colormap()
    levels = [cmap(v)[0] for v in (0.01, 0.25, 0.5, 0.75, 1.0)]
    assert levels == sorted(levels, reverse=True), "larger magnitude must be darker"
    assert cmap(1.0)[0] == pytest.approx(0.0, abs=1e-3)


def test_apply_style_overrides_a_hostile_rcparam():
    """A machine with its own ``matplotlibrc`` must not change the bytes."""
    matplotlib.rcParams["font.size"] = 31.0
    matplotlib.rcParams["figure.facecolor"] = "#ff00ff"
    style.apply_style()
    assert matplotlib.rcParams["font.size"] == style.FONT_SIZE
    assert matplotlib.colors.to_hex(matplotlib.rcParams["figure.facecolor"]) == style.PAPER


def test_apply_style_is_idempotent():
    style.apply_style()
    first = dict(matplotlib.rcParams)
    style.apply_style()
    changed = {k for k in first if str(first[k]) != str(matplotlib.rcParams[k])}
    assert not changed, f"applying the style twice changed {sorted(changed)}"


def test_apply_style_forces_a_headless_backend():
    style.apply_style()
    assert matplotlib.get_backend().lower() == "agg"


def test_style_provenance_names_the_versions_a_reader_needs():
    provenance = style.style_provenance()
    assert provenance["figure_style_version"] == style.FIGURE_STYLE_VERSION
    assert provenance["matplotlib_version"] == matplotlib.__version__
    assert provenance["dpi"] == style.SAVE_DPI
