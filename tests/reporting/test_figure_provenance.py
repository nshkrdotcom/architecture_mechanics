"""§8.5's last required test: a report generated only from recorded artifacts.

Reading it as a code-review rule ("don't load random files") makes it
unfalsifiable. Read as a property of the process, it is checkable: install an
audit hook, build the figure, and look at every file the interpreter opened.
A figure that quietly reads a scratch file of hand-tuned numbers cannot hide
from that, because the read happens whatever the code looks like.

Two things make this a check rather than a ritual:

- the **control**. A second subprocess does exactly the same build and then
  reads a file it has no business reading. If the audit does not flag it, the
  audit proves nothing about the first case, and the test fails.
- the **warm-up**. The first build in each subprocess is thrown away
  unaudited, so module imports and matplotlib's font cache — which really do
  open files inside the tree — are not confused with reading data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from architecture_mechanics.experiments.manifest import lab_root
from architecture_mechanics.reporting import figures

AUDIT_SCRIPT = r'''
import json, sys, tempfile
from pathlib import Path

from architecture_mechanics.reporting import figures

mode, out, number = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])

# Unaudited warm-up: lazy imports and matplotlib's font cache happen once here
# so that the audited build below sees data reads and nothing else.
with tempfile.TemporaryDirectory() as warm:
    figures.build_figure(number, Path(warm))

opened = []
recording = False


def hook(event, args):
    if recording and event == "open":
        opened.append({"path": str(args[0]), "mode": str(args[1])})


sys.addaudithook(hook)
recording = True
figures.build_figure(number, out)
if mode == "violating":
    # What a figure that made its numbers up would look like from outside.
    (Path(figures.lab_root()) / "pyproject.toml").read_text()
recording = False

sys.stdout.write(json.dumps(opened))
'''


def _audited_build(mode: str, out: Path, number: int = 1) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-c", AUDIT_SCRIPT, mode, str(out), str(number)],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _permitted_roots(out_dir: Path) -> list[Path]:
    """Where a figure may legitimately touch the disk.

    The laboratory's recorded artifacts, this build's own output, and the
    Python environment. Everything else — this repository's ``configs/``,
    ``claims/`` and source tree, a sibling project, a file in ``$HOME`` — is a
    violation, because none of it is evidence.
    """
    root = lab_root()
    permitted = [out_dir.resolve(), root / ".venv"]
    permitted += [root / name for name in figures.ARTIFACT_READ_ROOTS]
    permitted += [
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sysconfig.get_paths()["purelib"]),
        Path(sysconfig.get_paths()["platlib"]),
        Path.home() / ".cache" / "matplotlib",
        Path.home() / ".config" / "matplotlib",
        Path("/usr"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
    ]
    return [path.resolve() for path in permitted]


def _violations(opened: list[dict], out_dir: Path) -> list[str]:
    permitted = _permitted_roots(out_dir)
    bad = []
    for entry in opened:
        path = Path(entry["path"])
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - a path that cannot be resolved is suspect
            bad.append(entry["path"])
            continue
        if not any(resolved == root or root in resolved.parents for root in permitted):
            bad.append(str(resolved))
    return bad


@pytest.fixture(scope="module")
def clean_build(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit_clean")
    return out, _audited_build("clean", out)


def test_the_audit_sees_the_figure_being_written(clean_build):
    """If the hook observed nothing, every assertion below would be vacuous."""
    out, opened = clean_build
    written = [e for e in opened if "w" in e["mode"] and e["path"].endswith(".png")]
    assert written, f"the audit hook recorded nothing about the PNG: {opened}"
    assert (out / f"{figures.FIGURE_STEMS[1]}.png").exists()


def test_building_a_figure_reads_nothing_outside_recorded_artifacts(clean_build):
    out, opened = clean_build
    assert _violations(opened, out) == []


def test_the_figure_reads_no_dataset_from_disk(clean_build):
    """Figure 1's example is regenerated in process from a configuration the
    caption records, which cannot drift from the generator the way a cached
    tensor file can."""
    _, opened = clean_build
    suspicious = [
        e["path"]
        for e in opened
        if Path(e["path"]).suffix in {".pt", ".npz", ".npy", ".safetensors", ".pkl"}
    ]
    assert suspicious == []


def test_the_audit_catches_a_figure_that_reads_something_it_should_not(tmp_path):
    """The control. Same build, one extra read of a file under the laboratory
    that is neither ``runs/`` nor ``reports/``."""
    out = tmp_path / "audit_violating"
    out.mkdir()
    opened = _audited_build("violating", out)
    bad = _violations(opened, out)
    assert bad, "the audit failed to notice a read it was supposed to catch"
    assert any(path.endswith("pyproject.toml") for path in bad)


def test_declared_read_roots_are_the_ones_enforced():
    """The docstring rule and the constant the test reads are the same rule."""
    assert figures.ARTIFACT_READ_ROOTS == ("runs", "reports")
    for name in figures.ARTIFACT_READ_ROOTS:
        assert (lab_root() / name).is_dir()


REGENERATE_SCRIPT = r"""
import sys
from pathlib import Path
from architecture_mechanics.reporting import figures

print(figures.build_figure(int(sys.argv[2]), Path(sys.argv[1])).sha256)
"""


def _regenerate(out: Path, env: dict, number: int = 1) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", REGENERATE_SCRIPT, str(out), str(number)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("number", [1, 2])
def test_regeneration_is_byte_identical_across_processes(tmp_path, number):
    """Same-process determinism is nearly free and nearly worthless. A
    reviewer regenerating this figure next month has a fresh interpreter, a
    different hash seed, and a different working directory."""
    first = _regenerate(tmp_path / "a", {**os.environ, "PYTHONHASHSEED": "1"}, number)
    second = _regenerate(tmp_path / "b", {**os.environ, "PYTHONHASHSEED": "997"}, number)
    assert first == second


# --------------------------------------------------------------------------- #
# Figure 2 — the first figure with parent runs, and therefore the first whose
# read audit is about data rather than about the absence of it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def clean_build_figure2(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit_clean_fig2")
    return out, _audited_build("clean", out, 2)


def test_figure_two_reads_nothing_outside_recorded_artifacts(clean_build_figure2):
    out, opened = clean_build_figure2
    assert _violations(opened, out) == []


def test_figure_two_actually_reads_the_recorded_runs(clean_build_figure2):
    """The complement of the test above, and the one that makes it non-vacuous.

    Figure 1 reads no data at all, so "read nothing forbidden" is satisfied by a
    figure that reads nothing. Figure 2 must read the runs, and it must read a
    lot of them: one ``summary.json`` per arm of every cell.
    """
    _, opened = clean_build_figure2
    root = lab_root()
    summaries = {
        entry["path"]
        for entry in opened
        if entry["path"].endswith("summary.json")
        and str(root / "runs") in entry["path"]
    }
    declarations = {
        entry["path"]
        for entry in opened
        if str(root / "reports" / "comparisons") in entry["path"]
    }
    assert len(summaries) >= 64, f"only {len(summaries)} recorded runs were read"
    assert declarations, "the resolved declarations were never opened"


def test_figure_two_carries_its_own_warning_into_the_file(tmp_path):
    """The caption is a sidecar and sidecars get separated from PNGs."""
    result = figures.build_figure(2, tmp_path)
    assert result.params["n_points"] >= 32
    assert result.params["resolution"]["n_seeds"] == 1
    written = (tmp_path / f"{figures.FIGURE_STEMS[2]}.caption.md").read_text()
    assert "ONE SEED PER CELL" in written


def test_the_cli_verifies_determinism_and_reports_the_two_hashes(tmp_path, capsys):
    code = figures.main(["--figure", "1", "--out-dir", str(tmp_path), "--verify-deterministic"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "byte-identical: True" in printed
    digests = [line.split()[-1] for line in printed.splitlines() if "sha256" in line]
    assert digests and len(digests[0]) == 64
