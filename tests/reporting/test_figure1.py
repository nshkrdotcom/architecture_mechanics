"""Figure 1 draws the hand-checked example, and says so accurately.

The failure this guards against is not a crash. It is a figure that still
renders beautifully after the generator's semantics have moved underneath it,
or a caption whose numbers were true once. So the assertions here are of two
kinds: the drawn example must be the one hand-checked line by line in
``state/02_generator.md``, and every number in the caption must be recomputed
from the dataset rather than compared to a literal in this file.

The one exception is :func:`test_figure1_draws_the_hand_checked_example`, which
*is* literals — the values a human verified by reading the sequence. If the
generator changes them, the figure is no longer the picture that was checked
and someone has to look again.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from architecture_mechanics.reporting import figure_style as style
from architecture_mechanics.reporting import figures


@pytest.fixture(scope="module")
def dataset():
    return figures.figure1_dataset()


@pytest.fixture(scope="module")
def built(tmp_path_factory, dataset):
    out = tmp_path_factory.mktemp("figure1")
    return figures.build_figure1(out)


def png_chunks(path: Path) -> list[tuple[str, int]]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, offset = [], 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8].decode("ascii")
        chunks.append((kind, length))
        offset += 12 + length
        if kind == "IEND":
            break
    return chunks


# --------------------------------------------------------------------------- #
# The example
# --------------------------------------------------------------------------- #


def test_figure1_draws_the_hand_checked_example(dataset):
    """The worked example of ``state/02_generator.md``, value by value."""
    record = dataset.programs[figures.FIGURE1_EXAMPLE]
    step = record.steps[0]
    assert dataset.config.condition == "capacity_stressed"
    assert dataset.config.family == "T1"
    assert dataset.config.seed == 20260809
    assert dataset.n_features == 124
    assert dataset.config.seq_len == 48
    assert dataset.config.d_recommended == 16
    assert record.template_id == "338cb786edd88a5d"
    assert (step.op, step.source, step.dest, step.key_id) == ("recall_by_key", 36, 47, 3)
    assert step.distance == 11
    assert step.distractors == (41, 42, 44)
    assert step.answer_features == (1, 20)
    assert step.information_destroyed is False


def test_figure1_shows_all_five_required_elements(dataset):
    """North star 10.2 figure 1: features, source, pointer/operator,
    destination, distractors — plus the bottleneck the benchmark rests on.

    Checked against the data the drawing reads, so an element cannot be listed
    here and missing from the example."""
    record = dataset.programs[figures.FIGURE1_EXAMPLE]
    step = record.steps[0]
    x = dataset.inputs[figures.FIGURE1_EXAMPLE]

    assert x.shape == (48, 124), "the ground-truth feature matrix across positions"
    assert x[step.source, : dataset.banks.n_content].count_nonzero() > 0, "source carries content"
    op_bank = list(dataset.op_indices)
    assert x[step.dest, op_bank].count_nonzero() == 1, "the query position carries an operator"
    assert x[step.dest, : dataset.banks.n_content].count_nonzero() == 0, "the query carries no content"
    assert len(step.distractors) == 3 and all(
        step.source < p < step.dest for p in step.distractors
    ), "distractors sit between source and destination"
    assert dataset.config.d_recommended < dataset.n_features, "d < F"


def test_the_query_and_the_source_carry_the_same_key_bits(dataset):
    """The figure draws a box round both and calls them the same key."""
    x = dataset.inputs[figures.FIGURE1_EXAMPLE]
    record = dataset.programs[figures.FIGURE1_EXAMPLE]
    step = record.steps[0]
    key_bank = list(dataset.key_indices)
    source_bits = (x[step.source, key_bank] != 0).tolist()
    query_bits = (x[step.dest, key_bank] != 0).tolist()
    assert source_bits == query_bits and any(source_bits)


def test_the_required_output_is_the_source_content(dataset):
    """What the two dashed arrows in the figure assert."""
    index = figures.FIGURE1_EXAMPLE
    step = dataset.programs[index].steps[0]
    target = dataset.targets[index, step.dest]
    nonzero = sorted(int(f) for f in target.nonzero().flatten().tolist())
    assert nonzero == sorted(step.answer_features)
    content = list(dataset.content_indices)
    assert np.allclose(
        target[content].numpy(), dataset.inputs[index, step.source][content].numpy()
    )


# --------------------------------------------------------------------------- #
# The caption
# --------------------------------------------------------------------------- #


def test_caption_carries_every_parameter_needed_to_regenerate(built, dataset):
    """§ the mission's own bar: a figure whose inputs are not recoverable from
    its caption is a decoration. Every value is recomputed from the dataset."""
    caption = built.caption
    cfg = dataset.config
    required = {
        "generator version": dataset.generator_version,
        "seed": str(cfg.seed),
        "F": f"F = {dataset.n_features}",
        "d": f"d = {cfg.d_recommended}",
        "sequence length": f"{cfg.seq_len} ",
        "sparsity": str(cfg.activation_prob),
        "distractor count": f"{cfg.n_distractors} distractors",
        "condition": cfg.condition,
        "split": cfg.split,
        "example index": f"example {figures.FIGURE1_EXAMPLE}",
        "dataset content hash": dataset.content_hash,
        "regeneration command": "--figure 1",
        "figure style version": style.FIGURE_STYLE_VERSION,
    }
    missing = {name: value for name, value in required.items() if value not in caption}
    assert not missing, f"caption is missing {missing}"


def test_caption_is_written_beside_the_png(built):
    sidecar = built.path.with_name(f"{figures.FIGURE_STEMS[1]}.caption.md")
    assert sidecar.read_text().strip() == built.caption.strip()


def test_the_figure_is_written_where_the_paper_looks_for_it(built):
    """The one property of this module that other files depend on by name.

    ``paper/figures/fig1_benchmark_schematic.png`` is what prompt 27's prose,
    the Makefile and this programme's build checks all refer to. Renaming it is
    allowed; renaming it silently is not, so the name is pinned here rather than
    left implicit in a path expression."""
    assert figures.DEFAULT_OUT_DIR == "paper/figures"
    assert figures.FIGURE_STEMS[1] == "fig1_benchmark_schematic"
    assert built.path.name == "fig1_benchmark_schematic.png"


def test_caption_numbers_are_not_frozen_literals(dataset):
    """Change the dataset and the caption must follow it."""
    altered = dataclasses.replace(
        dataset, config=dataclasses.replace(dataset.config, d_recommended=8)
    )
    caption = figures.figure1_caption(figures.figure1_params(altered))
    assert "d = 8" in caption and "F/d = 15.5" in caption


# --------------------------------------------------------------------------- #
# The pixels
# --------------------------------------------------------------------------- #


def test_figure1_is_column_width(built):
    image = mpimg.imread(built.path)
    assert image.shape[1] == round(style.COLUMN_WIDTH_IN * style.SAVE_DPI), (
        "the figure must actually be the width it claims; bbox_inches='tight' "
        "silently changes it"
    )
    assert image.shape[0] < image.shape[1] * 1.6, "taller than this stops fitting a column"


def test_figure1_is_legible_in_greyscale_because_it_has_no_colour(built):
    """The strongest available form of the greyscale requirement: every pixel
    is a grey, so there is no hue to lose."""
    image = mpimg.imread(built.path)[..., :3]
    spread = image.max(axis=-1) - image.min(axis=-1)
    assert float(spread.max()) == 0.0, "figure 1 must be drawn in ink only"


def test_figure1_uses_the_full_ink_range(built):
    """A figure that is all mid-grey is greyscale-safe and unreadable."""
    grey = mpimg.imread(built.path)[..., 0]
    assert grey.min() < 0.05, "nothing is black"
    assert grey.max() > 0.95, "nothing is white"


def test_figure1_pixels_depend_on_the_data(built, dataset, tmp_path):
    """The test that separates a figure from an illustration: perturb one
    feature of the drawn example and the bytes must move."""
    perturbed_inputs = dataset.inputs.clone()
    step = dataset.programs[figures.FIGURE1_EXAMPLE].steps[0]
    perturbed_inputs[figures.FIGURE1_EXAMPLE, step.source, step.answer_features[0]] = 0.0
    perturbed = dataclasses.replace(dataset, inputs=perturbed_inputs)
    other = style.save_png(figures.draw_figure1(perturbed), tmp_path / "perturbed.png")
    assert other != built.sha256


def test_png_carries_no_timestamp_and_no_software_stamp(built):
    """The two chunks that would make an unchanged figure look changed."""
    kinds = {kind for kind, _ in png_chunks(built.path)}
    assert "tIME" not in kinds
    assert not kinds & {"tEXt", "iTXt", "zTXt"}, "metadata chunks carry a matplotlib version"


def test_delete_and_regenerate_is_byte_identical(built, tmp_path):
    first = figures.build_figure1(tmp_path)
    digest = first.sha256
    first.path.unlink()
    assert not first.path.exists()
    second = figures.build_figure1(tmp_path)
    assert second.sha256 == digest


# --------------------------------------------------------------------------- #
# The rest of the figure programme
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("number", "prompt"), [(3, "prompt 22"), (4, "prompt 23")])
def test_unbuilt_figures_name_the_prompt_that_owns_them(number, prompt, tmp_path):
    with pytest.raises(NotImplementedError) as excinfo:
        figures.build_figure(number, tmp_path)
    assert prompt in str(excinfo.value)


def test_figure_two_is_built_rather_than_planned():
    """Prompt 14 delivered it. Its own tests are in ``test_figure2.py``; what is
    checked here is that the programme's figure registry knows it exists, so the
    "not yet written" branch above cannot go on claiming it."""
    assert sorted(figures.BUILDERS) == [1, 2]
    assert figures.FIGURE_STEMS[2] == "fig2_phase_diagram"


def test_an_unknown_figure_is_refused(tmp_path):
    with pytest.raises(ValueError):
        figures.build_figure(9, tmp_path)


def test_index_records_the_hash_and_merges_without_dropping_a_figure(built, tmp_path):
    import json

    path = figures.write_index(tmp_path, [built])
    entry = json.loads(path.read_text())["figures"][0]
    assert entry["sha256"] == built.sha256
    assert entry["number"] == 1
    assert entry["params"]["dataset_content_hash"]

    fake = dataclasses.replace(built, number=3)
    figures.write_index(tmp_path, [fake])
    numbers = [item["number"] for item in json.loads(path.read_text())["figures"]]
    assert numbers == [1, 3]
