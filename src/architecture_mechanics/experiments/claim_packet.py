"""Reading, validating, and binding §7.1 claim packets to runs.

Two objects live here, and the distinction between them is the whole point.

A :class:`ClaimPacket` is a *prediction*: twelve fields written before the run,
committed to git, and never edited afterwards without a new commit that also
predates the next run. ``bin/check_prereg.sh`` compares that commit time against
the run's ``started_utc``, which is the only mechanically checkable difference
between a prediction and a story.

A :class:`ClaimGates` file is a *measurement*: which rungs of §7.5's ladder the
evidence actually supports. It is written only by :func:`evaluate_rungs`, which
takes a finished :class:`~architecture_mechanics.experiments.runner.RunResult`
and reads numbers out of it. There is no path from a config file, a CLI flag, or
a YAML field to a passed rung, and :func:`load_gates` refuses a file that claims
one without evidence or without its predecessor — so a hand-edited gates file
does not quietly survive, it fails on the next read.

The asymmetry is deliberate. A researcher may write any claim they like; what
they may not do is mark it supported.

A packet may also declare, in its ``covers:`` block, which runs it governs —
rung, architecture, and condition. That is how a run finds its parent claim when
the command line does not name one, and it is deliberately the *packet's*
statement rather than the runner's: a scope committed in advance cannot adopt a
run it did not predict, whereas a rule living in the runner could be widened
after the fact by whoever did not like the result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "COVERAGE_AXES",
    "GATES_SCHEMA",
    "PACKET_SCHEMA",
    "REQUIRED_FIELDS",
    "RUNGS",
    "ClaimGates",
    "ClaimPacket",
    "ClaimPacketError",
    "RungEvaluation",
    "evaluate_rungs",
    "load_gates",
    "load_packet",
    "packets_covering",
    "update_gates_from_run",
]

PACKET_SCHEMA = "am.claim_packet.v1"
GATES_SCHEMA = "am.claim_gates.v1"

REQUIRED_FIELDS: tuple[str, ...] = (
    "CLAIM",
    "MECHANISM",
    "STRUCTURALLY_ENFORCED_PROPERTIES",
    "LEARNED_OR_HOPED_PROPERTIES",
    "NEAREST_BORING_EXPLANATION",
    "CONTROL_THAT_RULES_IT_OUT",
    "PRIMARY_METRIC",
    "MECHANISM_ACTIVITY_METRIC",
    "POSITIVE_CONTROL",
    "NEGATIVE_CONTROL",
    "KILL_CONDITION",
    "REPLICATION_REQUIREMENT",
)
"""§7.1's twelve, in §7.1's order."""

COVERAGE_AXES: tuple[str, ...] = ("ladder", "arch", "condition")
"""The axes a packet's ``covers:`` block must name to claim a run.

A pre-registration declares, in advance and in git, which runs it governs. All
three axes are required: a packet saying only ``arch: [softmax]`` would silently
adopt every softmax run this laboratory ever produces, including rungs it never
predicted anything about. Widening a scope therefore costs a commit — which is
the same discipline the packet itself is under, applied to its own reach."""

RUNGS: tuple[str, ...] = (
    "implementation_survives",
    "mechanism_is_active",
    "capability_difference_replicates",
    "representation_difference_replicates",
    "controlled_mechanism_signal",
    "causal_mechanism_evidence",
    "cross_architecture_principle",
)
"""§7.5's seven, in §7.5's order. Index is the rung number."""

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ClaimPacketError(ValueError):
    """A pre-registration that would not mean what it says."""


# --------------------------------------------------------------------------- #
# The prediction
# --------------------------------------------------------------------------- #


@dataclass
class ClaimPacket:
    """One §7.1 pre-registration packet.

    ``fields`` holds the twelve under their §7.1 names. They are kept in a
    mapping rather than as twelve attributes because they are a *contract with
    a document*, and a typo in an attribute name would be a silently missing
    field where a typo in a key is a validation failure.
    """

    claim_id: str
    claimed_rung: int
    fields: dict[str, Any]
    primary_metric_key: str | None = None
    """Optional machine name of the primary metric — e.g.
    ``associative_recall_accuracy``. ``PRIMARY_METRIC`` is prose for a human;
    this is what ``reproduce.sh`` compares. Validated if present."""

    extra: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # -- validation -------------------------------------------------------- #

    def validate(self) -> None:
        """Refuse anything that is not a usable pre-registration.

        A blank ``KILL_CONDITION`` is not a kill condition, and a packet that
        parses is not therefore a packet that promises anything. Empty strings,
        empty lists, lists of empty strings, and placeholder text all fail here
        rather than at the gate, because the gate runs after the run.
        """
        problems: list[str] = []

        if not isinstance(self.claim_id, str) or not _ID_PATTERN.match(self.claim_id or ""):
            problems.append(
                f"claim_id {self.claim_id!r} must be lowercase letters, digits, '.', '-', '_'"
            )
        if self.path is not None and self.claim_id != self.path.stem:
            problems.append(
                f"claim_id {self.claim_id!r} does not match filename {self.path.name!r}"
            )

        if isinstance(self.claimed_rung, bool) or not isinstance(self.claimed_rung, int):
            problems.append(f"claimed_rung {self.claimed_rung!r} must be an integer")
        elif not 0 <= self.claimed_rung < len(RUNGS):
            problems.append(f"claimed_rung {self.claimed_rung} is outside 0..{len(RUNGS) - 1}")

        for name in REQUIRED_FIELDS:
            if name not in self.fields:
                problems.append(f"missing §7.1 field {name}")
                continue
            problems.extend(_field_problems(name, self.fields[name]))

        unknown = sorted(set(self.fields) - set(REQUIRED_FIELDS))
        if unknown:
            problems.append(f"unrecognised §7.1 fields {unknown}")

        if self.primary_metric_key is not None and not str(self.primary_metric_key).strip():
            problems.append("primary_metric_key is present but empty")

        problems.extend(_coverage_problems(self.extra.get("covers")))

        if problems:
            where = f" in {self.path}" if self.path else ""
            raise ClaimPacketError(
                f"claim packet {self.claim_id!r}{where} is not a pre-registration:\n  "
                + "\n  ".join(problems)
            )

    # -- declared scope ------------------------------------------------------ #

    @property
    def coverage(self) -> dict[str, tuple[str, ...]] | None:
        """The ``covers:`` block, or ``None`` if this packet claims no run.

        A packet with no ``covers:`` is still a valid pre-registration — it is
        simply one that must be named explicitly, by ``--claim``, to be attached
        to anything.
        """
        raw = self.extra.get("covers")
        if not isinstance(raw, dict):
            return None
        return {axis: tuple(raw.get(axis) or ()) for axis in COVERAGE_AXES}

    def covers_run(self, *, ladder: str, arch: str, condition: str) -> bool:
        """Does this packet's declared scope contain this run?

        Exact membership on all three axes. No wildcards and no prefixes: a
        scope that can be read two ways is not a scope, and the cost of being
        wrong here is a run attributed to a prediction that was never made
        about it.
        """
        covers = self.coverage
        if covers is None:
            return False
        run = {"ladder": ladder, "arch": arch, "condition": condition}
        return all(run[axis] in covers[axis] for axis in COVERAGE_AXES)

    # -- serialisation ----------------------------------------------------- #

    def as_dict(self) -> dict:
        record: dict[str, Any] = {"claim_id": self.claim_id, "schema": PACKET_SCHEMA}
        record.update(self.extra)
        for name in REQUIRED_FIELDS:
            record[name] = self.fields.get(name)
        if self.primary_metric_key is not None:
            record["primary_metric_key"] = self.primary_metric_key
        record["claimed_rung"] = self.claimed_rung
        return record

    def write(self, path: Path | str) -> Path:
        """Validate, then write. An invalid packet never reaches disk."""
        self.path = Path(path)
        self.validate()
        text = yaml.safe_dump(self.as_dict(), sort_keys=False, width=88, allow_unicode=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text)
        return self.path


def _field_problems(name: str, value: Any) -> list[str]:
    if isinstance(value, str):
        return [] if value.strip() else [f"§7.1 field {name} is empty"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"§7.1 field {name} is an empty list"]
        return [
            f"§7.1 field {name} entry {index} is empty or not text"
            for index, item in enumerate(value)
            if not isinstance(item, str) or not item.strip()
        ]
    if value is None:
        return [f"§7.1 field {name} is missing"]
    return [f"§7.1 field {name} must be text or a list of text, got {type(value).__name__}"]


def _coverage_problems(raw: Any) -> list[str]:
    """Validate an optional ``covers:`` block. Absent is fine; malformed is not."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"covers must be a mapping of {list(COVERAGE_AXES)} to lists, got {type(raw).__name__}"]

    problems: list[str] = []
    unknown = sorted(set(raw) - set(COVERAGE_AXES))
    if unknown:
        problems.append(f"covers has unrecognised axes {unknown}; expected {list(COVERAGE_AXES)}")
    for axis in COVERAGE_AXES:
        if axis not in raw:
            problems.append(
                f"covers is missing axis {axis}: a scope that leaves an axis open adopts "
                "every future run on it"
            )
            continue
        values = raw[axis]
        if not isinstance(values, (list, tuple)) or not values:
            problems.append(f"covers.{axis} must be a non-empty list, got {values!r}")
            continue
        problems.extend(
            f"covers.{axis} entry {index} is empty or not text"
            for index, value in enumerate(values)
            if not isinstance(value, str) or not value.strip()
        )
    return problems


def load_packet(path: Path | str) -> ClaimPacket:
    """Read and validate a packet. Raises rather than returning a broken one."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:  # pragma: no cover - exercised by hand, not by suite
        raise ClaimPacketError(f"{path} does not parse: {error}") from error
    if not isinstance(raw, dict):
        raise ClaimPacketError(f"{path} is not a mapping")

    fields_present = {name: raw[name] for name in REQUIRED_FIELDS if name in raw}
    extra = {
        key: value
        for key, value in raw.items()
        if key not in REQUIRED_FIELDS
        and key not in {"claim_id", "claimed_rung", "schema", "primary_metric_key"}
    }
    packet = ClaimPacket(
        claim_id=raw.get("claim_id", ""),
        claimed_rung=raw.get("claimed_rung"),
        fields=fields_present,
        primary_metric_key=raw.get("primary_metric_key"),
        extra=extra,
        path=path,
    )
    packet.validate()
    return packet


def packets_covering(
    claims_dir: Path | str, *, ladder: str, arch: str, condition: str
) -> tuple[list[ClaimPacket], list[tuple[Path, str]]]:
    """Every packet in ``claims_dir`` whose declared scope contains this run.

    Returns the matches together with the files that could not be read at all,
    because a packet too broken to parse must not be able to make itself
    invisible: if nothing matches, the caller reports the unreadable ones as
    part of saying so.

    Nothing here decides anything. The caller refuses on zero matches and on
    more than one — a run that two pre-registrations both claim is a run whose
    prediction is ambiguous, and picking one would be picking for the
    researcher.
    """
    claims_dir = Path(claims_dir)
    matches: list[ClaimPacket] = []
    unreadable: list[tuple[Path, str]] = []
    if not claims_dir.is_dir():
        return matches, unreadable

    for path in sorted([*claims_dir.glob("*.yml"), *claims_dir.glob("*.yaml")]):
        try:
            packet = load_packet(path)
        except (ClaimPacketError, OSError) as error:
            unreadable.append((path, str(error).splitlines()[0]))
            continue
        if packet.covers_run(ladder=ladder, arch=arch, condition=condition):
            matches.append(packet)
    return matches, unreadable


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RungEvaluation:
    """One rung's verdict, together with what was measured to reach it.

    Produced only by :func:`evaluate_rungs`. Nothing else in the laboratory
    constructs one, and :class:`ClaimGates` accepts nothing else.
    """

    rung: int
    passed: bool
    evidence: tuple[str, ...]
    measured: dict
    evaluated_by: str

    @property
    def name(self) -> str:
        return RUNGS[self.rung]

    @property
    def key(self) -> str:
        return f"{self.rung}_{self.name}"


def evaluate_rungs(result, *, run_dir: str) -> list[RungEvaluation]:
    """Read a finished run and say which of §7.5's rungs it supports.

    Only the first two are decidable from a single run, and only the first two
    are returned. Rung 2 upward are statements about a *difference between
    architectures* replicating across seeds; one run of one architecture cannot
    support them, and a function that returned "not passed" for them would be
    saying something weaker than "not evaluated". Absent means not evaluated,
    and the gate reads absent as not passed, which is the correct reading.

    ``rung 0 implementation_survives``  every §8.5 invariant held, training did
        not produce a non-finite loss, and the run reached its final evaluation.

    ``rung 1 mechanism_is_active``  §6.3's activity gates all passed, measured
        by :func:`~architecture_mechanics.metrics.mechanism.mechanism_is_active`
        on captured attention, not asserted.
    """
    evaluations: list[RungEvaluation] = []
    evidence = (run_dir,)

    failed_checks = sorted(name for name, record in result.checks.items() if not record["ok"])
    numerical_failure = any(
        entry.get("numerical_failure") for entry in result.history
    ) or any(
        entry.get("train_loss") is not None
        and not _finite(entry.get("train_loss"))
        for entry in result.history
    )
    trained = result.config.get("optim", {}).get("max_steps", 0) > 0
    reached_final = bool(result.final) or not trained

    survives = not failed_checks and not numerical_failure and reached_final
    evaluations.append(
        RungEvaluation(
            rung=0,
            passed=survives,
            evidence=evidence if survives else (),
            measured={
                "invariant_checks": len(result.checks),
                "invariant_failures": failed_checks,
                "numerical_failure": numerical_failure,
                "reached_final_evaluation": reached_final,
                "run_id": result.run_id,
            },
            evaluated_by="evaluate_rungs/r0_invariants",
        )
    )

    verdict = (result.mechanism or {}).get("verdict")
    if verdict is not None:
        active = bool(verdict.get("active")) and survives
        evaluations.append(
            RungEvaluation(
                rung=1,
                passed=active,
                evidence=evidence if active else (),
                measured={
                    "active": bool(verdict.get("active")),
                    "best_off_diagonal_mass": verdict.get("best_off_diagonal_mass"),
                    "best_entropy_ratio": verdict.get("best_entropy_ratio"),
                    "best_retrieval_lift": verdict.get("best_retrieval_lift"),
                    "reasons": verdict.get("reasons"),
                    "mechanism_version": (result.mechanism or {}).get("mechanism_version"),
                    "run_id": result.run_id,
                },
                evaluated_by="evaluate_rungs/mechanism_is_active",
            )
        )
    return evaluations


def _finite(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class ClaimGates:
    """``claims/<claim_id>.gates.json`` — which rungs the evidence supports.

    ``highest_supported_rung`` is never stored as input. It is recomputed from
    the rungs on every serialisation, so raising it by hand survives exactly
    until the next write, and :func:`load_gates` refuses the file in the
    meantime.
    """

    claim_id: str
    rungs: dict[str, dict] = field(default_factory=dict)

    def record(self, evaluation: RungEvaluation, *, source: str) -> None:
        """Take one measured verdict. The only way a rung is ever marked passed.

        Every evaluation is kept, passing or not. A rung is supported if some
        run supported it, and a run that failed it is a fact about this claim
        that belongs in the same file — dropping it would leave a gates file
        that says "passed" where the honest statement is "passed once, failed
        twice". Re-evaluating the same run replaces its earlier entry rather
        than appending, so a re-run does not look like a replication.
        """
        if not isinstance(evaluation, RungEvaluation):
            raise ClaimPacketError(
                "claim gates accept only a RungEvaluation produced by evaluate_rungs(); "
                f"got {type(evaluation).__name__}. A rung is not something a config can assert."
            )
        if evaluation.passed and not evaluation.evidence:
            raise ClaimPacketError(
                f"rung {evaluation.rung} ({evaluation.name}) passed with no evidence"
            )
        entry = self.rungs.setdefault(evaluation.key, {})
        history = [record for record in entry.get("evaluations", []) if record.get("source") != source]
        history.append(
            {
                "source": source,
                "passed": bool(evaluation.passed),
                "measured": evaluation.measured,
                "evaluated_by": evaluation.evaluated_by,
                "evaluated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            }
        )
        history.sort(key=lambda record: record["source"])
        evidence = sorted({record["source"] for record in history if record["passed"]})
        self.rungs[evaluation.key] = {
            "passed": bool(evidence),
            "evidence": evidence,
            "evaluations": history,
        }

    @property
    def highest_supported_rung(self) -> int | None:
        """The largest ``n`` with rungs ``0..n`` all passed with evidence.

        §7.5: "each step requires all previous steps". A rung 4 sitting above an
        unpassed rung 3 supports nothing, so the ladder stops at the gap.
        """
        highest: int | None = None
        for index, name in enumerate(RUNGS):
            entry = self.rungs.get(f"{index}_{name}") or {}
            if entry.get("passed") and entry.get("evidence"):
                highest = index
            else:
                break
        return highest

    def as_dict(self) -> dict:
        ordered = {
            f"{index}_{name}": self.rungs[f"{index}_{name}"]
            for index, name in enumerate(RUNGS)
            if f"{index}_{name}" in self.rungs
        }
        return {
            "schema": GATES_SCHEMA,
            "claim_id": self.claim_id,
            "rungs": ordered,
            "highest_supported_rung": self.highest_supported_rung,
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def load_gates(path: Path | str, *, claim_id: str | None = None) -> ClaimGates:
    """Read a gates file, refusing one that has been edited into a lie.

    A rung marked passed with no evidence, or passed above an unpassed
    predecessor, fails here. Neither is producible by :meth:`ClaimGates.record`,
    so either means the file was written by something other than the code that
    evaluated the evidence — which is the one thing this file is not allowed to
    be.
    """
    path = Path(path)
    if not path.is_file():
        return ClaimGates(claim_id=claim_id or path.name.split(".")[0])
    raw = json.loads(path.read_text())
    rungs = raw.get("rungs") or {}
    gates = ClaimGates(claim_id=raw.get("claim_id") or claim_id or path.name.split(".")[0], rungs=rungs)

    seen_gap = False
    for index, name in enumerate(RUNGS):
        entry = rungs.get(f"{index}_{name}") or {}
        passed = bool(entry.get("passed"))
        if passed and not entry.get("evidence"):
            raise ClaimPacketError(
                f"{path}: rung {index} ({name}) is marked passed with no evidence"
            )
        if passed and seen_gap:
            raise ClaimPacketError(
                f"{path}: rung {index} ({name}) is marked passed above an unpassed rung"
            )
        if not passed:
            seen_gap = True
    return gates


def update_gates_from_run(
    result,
    *,
    run_dir: str,
    claims_dir: Path | str,
    claim_id: str,
) -> tuple[ClaimGates, Path]:
    """Evaluate a finished run and fold the verdicts into that claim's gates file.

    The single writer. Everything it records came out of ``result``, which came
    out of a model that was actually run.
    """
    claims_dir = Path(claims_dir)
    path = claims_dir / f"{claim_id}.gates.json"
    gates = load_gates(path, claim_id=claim_id)
    for evaluation in evaluate_rungs(result, run_dir=run_dir):
        gates.record(evaluation, source=run_dir)
    gates.write(path)
    return gates, path
