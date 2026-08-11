"""Run identity and the §8.3 provenance manifest.

Every run directory carries a ``manifest.json`` that answers, without reference
to anyone's memory, the question a reviewer actually asks: *what produced this,
and could I produce it again?* §8.3 lists fourteen things that answer requires,
and ``bin/check_evidence.sh`` refuses a run whose manifest is missing any of the
thirteen it can check for presence.

Two design points are load-bearing.

**Run IDs are content-derived.** §8.3: "run IDs should derive from stable config
and source identity rather than timestamps alone." The digest here is over the
full config, the generator version, a hash of every source file, and the seed.
A timestamp would make every re-run a new directory, so a laboratory that
re-ran the same thing eleven times while debugging would end up with eleven
directories that look like eleven experiments. Here it ends up with one, which
is the truth. The converse matters just as much: change one line of a model and
the ID changes, because the run really is a different run.

**"Dirty" is scoped to the run's inputs.** ``git status`` counts an uncommitted
run directory as a dirty tree, and the previous run's output is always
uncommitted while the next one starts; so is the claim gates file this run is
about to update. Scoring those as dirt would make ``dirty_tree`` true for every
run after the first and mean nothing. So :func:`git_facts` asks about ``src``,
``tests``, ``configs``, ``pyproject.toml`` and ``uv.lock`` — source, contract,
and dependencies — plus this run's own claim packet, which is an input. Anything
modified there means this run cannot be recovered from history, which is what
``dirty_tree`` is for. The paths responsible are recorded, so the judgement is
auditable rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "MACHINE_SIDE_FILES",
    "MANIFEST_SCHEMA",
    "PROVENANCE_FIELDS",
    "RunManifest",
    "build_manifest",
    "dependency_lock_hash",
    "evidence_index",
    "file_digest",
    "git_facts",
    "lab_root",
    "run_id_for",
    "source_tree_hash",
    "utc_now",
]

MANIFEST_SCHEMA = "am.manifest.v1"

MACHINE_SIDE_FILES: tuple[str, ...] = ("cost.json",)
"""Files in a run directory that describe *this machine at that instant* rather
than the experiment: wall clock, peak and reserved VRAM, free VRAM at start.

They are gitignored for the reason recorded in ``runs/README.md`` — committing
them would make an identical re-run dirty the tree — and an agreeing repeat
rewrites them while leaving every scientific artifact alone. Indexing them was
therefore a promise the laboratory could not keep: prompt 10 found six recorded
manifests whose ``evidence_index`` names a ``cost.json`` digest that no longer
matches the file, all six produced by exactly that repeat path. They are
excluded from the index and skipped by the index verification, so that every
remaining entry is a digest of something the repository actually carries and can
be held to.
"""

PROVENANCE_FIELDS: tuple[str, ...] = (
    "git_commit",
    "dirty_tree",
    "config",
    "architecture_id",
    "parameter_count",
    "dataset_generator_version",
    "split_hashes",
    "seed",
    "device",
    "precision",
    "dependency_lock_hash",
    "parent_claim_packet",
    "started_utc",
    "evidence_index",
)
"""The fields the external ``ml.evidence`` contract requires to be non-empty.

The order matters: the gate is compared element by element, not as a set.
``evidence_index`` was added to the gate after prompt 14's runs were recorded,
and the laboratory learned of it from its own drift test rather than from a red
gate — which is what that test exists for. It costs nothing here because
:meth:`RunManifest.write` computes the index before it checks, so a run that
emitted nothing to index is the only thing the new field can refuse, and a run
that emitted nothing has no evidence.

Mirrored here rather than imported because the packet-owned contract must stay
independent of the laboratory it verifies.
``tests/provenance/test_gate_agreement.py`` reads the contract and fails if this
tuple ever drifts from it."""

SOURCE_PATHS: tuple[str, ...] = ("src", "tests", "configs", "pyproject.toml", "uv.lock")
"""What ``dirty_tree`` is computed over, before the run's own claim packet is
added to it. See the module docstring.

``claims/`` as a whole is deliberately absent: the gates file this run is about
to write lives there, so including the directory would mean every run after the
first reported a dirty tree because of its own predecessor's output. The claim
*packet* — the input — is added per run by :func:`build_manifest`."""


def utc_now() -> str:
    """An ISO-8601 UTC timestamp, to the second."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def lab_root() -> Path:
    """The laboratory this source tree belongs to.

    Derived from this file's location, not from the working directory, so that
    a reproduction driven from ``/tmp`` against an exported source tree records
    the exported tree's identity and not the identity of wherever it was
    launched.
    """
    return Path(__file__).resolve().parents[3]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_hash(root: Path | None = None) -> str:
    """A digest of every ``.py`` file under ``src/``, path included.

    Path-included so that moving a file changes the hash: two source trees that
    differ only in where a function lives are not the same source tree, and a
    run produced by one is not reproducible from the other.
    """
    root = Path(root or lab_root())
    source = root / "src"
    digest = hashlib.sha256()
    if not source.is_dir():
        return "unavailable"
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(file_digest(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_lock_hash(root: Path | None = None) -> str:
    """sha256 of ``uv.lock`` — the exact resolved dependency set, §8.3."""
    lock = Path(root or lab_root()) / "uv.lock"
    return file_digest(lock) if lock.is_file() else "unavailable"


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_facts(root: Path | None = None, *, paths: tuple[str, ...] = SOURCE_PATHS) -> dict:
    """Commit, branch, and whether the source that produced this run is in history.

    Three ways this resolves, in order:

    1. the source tree is inside a git work tree — real answers;
    2. it is not, but ``AM_SOURCE_COMMIT`` is set — a pinned export, which is
       what ``reproduce.sh`` produces with ``git archive``, so the commit is
       known and the tree is clean by construction;
    3. neither — commit unknown and ``dirty_tree`` *true*, because an
       unidentifiable source tree is exactly as unreproducible as a modified
       one and should not be recorded as if it were fine.
    """
    root = Path(root or lab_root())
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside == "true":
        status = _git(root, "status", "--porcelain", "--", *paths)
        dirty_paths = [line[3:] for line in (status or "").splitlines() if line.strip()]
        return {
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_source": "work_tree",
            "dirty_tree": bool(dirty_paths) or status is None,
            "dirty_paths": dirty_paths,
        }

    pinned = os.environ.get("AM_SOURCE_COMMIT", "").strip()
    if pinned:
        return {
            "git_commit": pinned,
            "git_branch": None,
            "git_source": "pinned_export",
            "dirty_tree": False,
            "dirty_paths": [],
        }

    return {
        "git_commit": None,
        "git_branch": None,
        "git_source": "unavailable",
        "dirty_tree": True,
        "dirty_paths": [],
    }


def run_id_for(
    config_dict: dict,
    *,
    generator_version: str,
    source_hash: str,
    seed: int,
    ladder: str,
    arch: str,
    condition: str,
) -> str:
    """``<rung>-<arch>-<condition>-s<seed>-<digest>``, digest over content only.

    ``seed`` and ``generator_version`` are already inside ``config_dict``; they
    are hashed again by name because §8.3 names them, and a later refactor that
    moves one out of the config must not silently drop it out of the identity.
    """
    payload = json.dumps(
        {
            "config": config_dict,
            "generator_version": generator_version,
            "source_tree_hash": source_hash,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{ladder}-{arch}-{condition}-s{seed}-{digest}"


def evidence_index(
    run_dir: Path, *, exclude: tuple[str, ...] = ("manifest.json", *MACHINE_SIDE_FILES)
) -> list[dict]:
    """§8.3's "generated evidence index": every file this run emitted, hashed.

    ``manifest.json`` excludes itself — it is written last and cannot contain
    its own digest. Checkpoints appear here as well as in ``checkpoint_hashes``;
    they are gitignored, so the index is the only committed record that a
    checkpoint existed and what it was.

    :data:`MACHINE_SIDE_FILES` are excluded for the opposite reason: they are
    rewritten by an agreeing repeat, so an index entry for them goes stale
    against a run whose science never moved. An index that is wrong for a
    legitimate reason cannot be enforced, and an index nobody enforces is not
    provenance.
    """
    run_dir = Path(run_dir)
    entries: list[dict] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in exclude:
            continue
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)}
        )
    return entries


@dataclass
class RunManifest:
    """One run's §8.3 provenance record.

    Field order is the order §8.3 lists them, so a reader comparing the two can
    do it line by line.
    """

    schema: str
    run_id: str
    ladder_rung: str

    git_commit: str | None
    git_branch: str | None
    git_source: str
    dirty_tree: bool
    dirty_paths: list[str]

    config: dict
    architecture_id: str
    parameter_count: int
    parameter_report: dict
    operation_state_summary: dict

    dataset_generator_version: str
    split_hashes: dict
    seed: int

    device: dict
    precision: str
    numerics: dict

    dependency_lock_hash: str
    source_tree_hash: str
    environment: dict

    parent_claim_packet: str
    claimed_rung: int | None
    primary_metric: str

    started_utc: str
    finished_utc: str | None = None
    checkpoint_hashes: dict = field(default_factory=dict)
    evidence_index: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def missing_provenance(self) -> list[str]:
        """The §8.3 fields this manifest cannot satisfy, by the gate's own rule.

        The gate treats ``None``, ``""``, ``[]`` and ``{}`` as absent.
        Reproducing that rule here means the runner refuses to write an
        incomplete manifest at the point of writing, where the fix is obvious,
        rather than at gate time three commits later.

        ``dirty_tree`` is the one field the rule cannot police: ``False`` is its
        most common legitimate value and is indistinguishable from unset by any
        emptiness test. Only ``None`` — a manifest that never asked — is caught.
        """
        record = self.as_dict()
        return [name for name in PROVENANCE_FIELDS if record.get(name) in (None, "", [], {})]

    def write(self, run_dir: Path) -> Path:
        """Write ``manifest.json``, after indexing everything else in the directory.

        Written last on purpose: the evidence index is a hash of the run's other
        outputs, so a manifest that existed before them could only describe a
        directory that did not yet exist.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_index = evidence_index(run_dir)
        missing = self.missing_provenance()
        if missing:
            raise ValueError(
                f"manifest for {self.run_id} is missing §8.3 provenance fields: {missing}"
            )
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False) + "\n")
        return path


def build_manifest(
    *,
    config,
    model,
    train_dataset,
    eval_dataset,
    device_record,
    parent_claim_packet: str,
    claimed_rung: int | None,
    primary_metric: str,
    started_utc: str,
    root: Path | None = None,
) -> RunManifest:
    """Assemble the §8.3 record for a run that is about to start.

    Everything here is known before the first gradient step, which is the point:
    the manifest is a description of *what is being run*, and a description
    written after the fact could be a description of what one wishes had been
    run. Only the evidence index, the checkpoint hashes, and ``finished_utc``
    are filled in afterwards, and each of those is a hash of something the run
    produced rather than a statement about it.
    """
    from architecture_mechanics.data.feature_program import GENERATOR_VERSION
    from architecture_mechanics.models.common import parameter_report

    root = Path(root or lab_root())
    # The claim packet is an input to this run, so a modified one means a dirty
    # tree. check_prereg.sh catches an *uncommitted* packet; it cannot catch a
    # committed packet edited since, because it only sees the last commit that
    # touched the file. This closes that half.
    facts = git_facts(
        root, paths=(*SOURCE_PATHS, parent_claim_packet) if parent_claim_packet else SOURCE_PATHS
    )
    source_hash = source_tree_hash(root)
    config_dict = config.as_dict()
    model_config = model.config

    device = dict(device_record.as_dict())
    # Free VRAM at that instant is a fact about the machine, not the run; it
    # travels with cost.json, which is not committed.
    device.pop("free_memory_bytes", None)

    architecture_id = (
        f"{model_config.arch}-L{model_config.n_layers}H{model_config.n_heads}"
        f"d{model_config.d_model}-{model_config.residual_write}-{model_config.positional}"
        f"@{model_config.as_dict()['model_version']}"
    )
    report = parameter_report(model)

    return RunManifest(
        schema=MANIFEST_SCHEMA,
        run_id=run_id_for(
            config_dict,
            generator_version=GENERATOR_VERSION,
            source_hash=source_hash,
            seed=config.seed,
            ladder=config.ladder,
            arch=config.arch.arch,
            condition=config.data.condition,
        ),
        ladder_rung=config.ladder,
        git_commit=facts["git_commit"],
        git_branch=facts["git_branch"],
        git_source=facts["git_source"],
        dirty_tree=facts["dirty_tree"],
        dirty_paths=facts["dirty_paths"],
        config=config_dict,
        architecture_id=architecture_id,
        parameter_count=int(report["total"]),
        parameter_report=report,
        operation_state_summary=model.operation_state_summary(),
        dataset_generator_version=GENERATOR_VERSION,
        split_hashes={
            "train": train_dataset.content_hash,
            "eval": eval_dataset.content_hash,
            "train_n_examples": train_dataset.n_examples,
            "eval_n_examples": eval_dataset.n_examples,
        },
        seed=int(config.seed),
        device=device,
        precision=config.optim.precision,
        numerics={
            "precision": config.optim.precision,
            "float32_matmul_precision": config.optim.float32_matmul_precision,
            "torch_version": _torch_version(),
        },
        dependency_lock_hash=dependency_lock_hash(root),
        source_tree_hash=source_hash,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "lab_root": str(root),
        },
        parent_claim_packet=parent_claim_packet,
        claimed_rung=claimed_rung,
        primary_metric=primary_metric,
        started_utc=started_utc,
    )


def _torch_version() -> str:
    import torch

    return torch.__version__
