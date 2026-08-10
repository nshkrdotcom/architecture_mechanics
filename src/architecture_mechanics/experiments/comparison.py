"""The declared comparison object, and the construction that refuses an unfair one.

§7.4 lists "silently tuning the candidate more than the control" among the things
to avoid. It is silent because it happens in small increments across days, each
individually defensible: the candidate would not train, so its learning rate
moved; it was slower, so it got fewer steps and then more steps; a width was
retyped at a call site. Nobody decides to run an unfair comparison. The fix is
therefore not discipline — it is a declared object whose config diff a script
checks, and a construction path that refuses to produce the unfair version at
all.

``bin/check_no_rescue.sh`` defines the format and is the outer gate: it reads
every ``reports/comparisons/*.json``, diffs the manifests of the runs it names,
and fails on any difference not listed in ``permitted_differences`` with a
justification. This module is the inner half. Three things it adds that the gate
structurally cannot:

**It fails before the GPU.** The gate can only speak about runs that exist, so
by the time it objects the compute has been spent. A comparison here is *built*
from one rung preset before anything runs: both arms come from the same
:func:`~architecture_mechanics.experiments.config.ladder_config` call with only
the architecture changed, the pairwise diff is computed on the configs, and an
undeclared difference raises :class:`ComparisonError` at construction.

**It declares the comparison before the runs exist.** A plan under
``reports/comparisons/planned/`` names the arms, the frozen configs, the seed
set and both matching strategies, and is committed before the first run — so the
comparison is a prediction rather than a description. It is deliberately not in
the gate's own directory: it names planned arms rather than run IDs, which are
content digests and cannot be known in advance. The gate's directory holds only
resolved declarations, and the resolver puts them there.

**It checks the work, not only the configuration.** Equal configs are necessary
and not sufficient for §7.2's compute parity: a screen that stopped early did
less work than its matched partner while recording the same ``max_steps``. The
resolver reads each run's measured step count, examples processed and wall clock
into a ledger, refuses a pair in which the *candidate* did more of the work, and
records — with the direction named — a pair in which it did less, because that is
what a kill screen looks like and refusing it would discard the collapse it
describes.

Both §7.2 matching strategies are named constructions here, and
:func:`matched_plans` produces both by default. Producing one is possible and
costs an explicit written justification, because "we reported the flattering
one" is the failure §7.2's two-comparison rule exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from architecture_mechanics.data.task_families import get_family
from architecture_mechanics.experiments.config import (
    LADDERS,
    SEED_FAMILY,
    RunConfig,
    RunConfigError,
    config_fingerprint,
    seed_family,
)
from architecture_mechanics.experiments.manifest import lab_root, utc_now
from architecture_mechanics.experiments.t1_ladder import (
    BASE_CELL,
    NEGATIVE_CONTROL_CELL,
    Cell,
    cell_config,
    cells,
)
from architecture_mechanics.models.common import (
    ModelConfig,
    parameter_matched_config,
    parameters_for,
    primitive_names,
)

__all__ = [
    "ALWAYS_PERMITTED",
    "COMPARISONS_DIR",
    "COMPUTE_BUDGET_KEYS",
    "DECLARATION_SCHEMA",
    "DECLARED_COMPARISONS",
    "FROZEN_VARIABLES",
    "MATCHING_STRATEGIES",
    "PLANNED_DIR",
    "PLAN_SCHEMA",
    "ComparisonError",
    "ComparisonPlan",
    "MatchedPair",
    "compute_ledger",
    "config_differences",
    "declare",
    "flatten_config",
    "load_plan",
    "main",
    "matched_plans",
    "plans_for",
    "resolve",
    "run_comparison",
]

PLAN_SCHEMA = "am.comparison_plan.v1"
DECLARATION_SCHEMA = "am.comparison_declaration.v1"

DECLARATION_FIELDS: tuple[str, ...] = (
    "claim",
    "primary_metric",
    "control_run",
    "candidate_runs",
    "matching_strategy",
    "permitted_differences",
)
"""What every resolved declaration must carry, and :func:`resolve` checks before
writing one.

Three of these are read by ``bin/check_no_rescue.sh`` and two by
``metrics.statistics.load_comparison``; between them they are the whole of what a
comparison *is*. ``permitted_differences`` may legitimately be empty and is
therefore checked for presence rather than for content — an absent key and an
empty map say different things, and only the second is a claim that there was
nothing to declare. ``tests/provenance/test_gate_agreement.py`` checks this tuple
against the keys the gate actually reads."""

COMPARISONS_DIR = Path("reports") / "comparisons"
"""Where resolved declarations live: the directory ``bin/check_no_rescue.sh``
globs, non-recursively, for ``*.json``."""

PLANNED_DIR = COMPARISONS_DIR / "planned"
"""Where plans live — inside the comparisons directory for a reader, outside the
gate's non-recursive glob for the gate.

A plan names *planned* arms, not run IDs. §8.3 derives a run ID from a digest of
the config and the source tree, so the ID of a run that has not happened is
unknowable, and a plan that guessed one would be wrong the moment a source file
changed. Putting plans where the gate would read them would therefore mean
committing a file the gate must fail on, which trains everyone to ignore it."""

MATCHING_STRATEGIES: tuple[str, ...] = ("width_matched", "parameter_matched")
"""§7.2: "When exact parameter matching is impossible, report both: width-matched
comparison; parameter- or compute-matched comparison."

**width_matched** — every arm at the same ``d_model`` and the same depth.
Parameter counts may differ, and the difference is recorded rather than removed.
This is the comparison that asks "at this width, does the mechanism change the
result", and it is the one whose independent variable is cleanly the mechanism.

**parameter_matched** — the candidate keeps the declared width and the
*control's* width is retuned until the two parameter counts agree. This is the
comparison that asks "at this parameter budget, does the mechanism change the
result", and it exists because a candidate that adds capacity would otherwise be
credited for parameters as well as for mechanism. The control moves, not the
candidate, so the mechanism under study is never re-sized to make a number work.

Both are produced by default. When the counts cannot be equalised exactly the
residual and its direction are recorded, along with the bracketing widths, so a
reader can see which way the remaining error leans."""

ALWAYS_PERMITTED: frozenset[str] = frozenset(
    {"architecture", "architecture_id", "arch", "model.architecture"}
)
"""Config keys whose difference needs no declaration: the independent variable.

Mirrored from ``bin/check_no_rescue.sh``, where the same set is spelled out, and
held to it by ``tests/provenance/test_gate_agreement.py``. A key matches if the
whole dotted key or its last component is in the set, which is the gate's rule;
in this laboratory that is exactly ``arch.arch``, the mixing primitive."""

COMPUTE_BUDGET_KEYS: tuple[str, ...] = (
    "optim.max_steps",
    "optim.batch_size",
    "data.n_train",
    "data.n_eval",
    "optim.stopping_rule",
)
"""The keys that decide how much training compute an arm receives.

§7.2: "Do not allow a candidate to receive more training compute merely because
it is slower unless the claim explicitly concerns fixed wall-clock performance."
A justification is not enough for these — a plan that moves one must also set
``wall_clock_claim``, which is a statement about what the *claim* is and not
about what the candidate needed. That makes the one legitimate reason to differ
here impossible to reach by accident, which is the whole difference between a
declared exception and a rescue."""


class ComparisonError(ValueError):
    """A comparison that would not mean what it says."""


# --------------------------------------------------------------------------- #
# §7.2's frozen variables, and where each is enforced
# --------------------------------------------------------------------------- #

FROZEN_VARIABLES: tuple[dict, ...] = (
    {
        "variable": "dataset generator version and seed family",
        "keys": ("generator_version", "metric_version", "model_version", "seed"),
        "enforced_by": (
            "RunConfig.as_dict stamps all three versions; run_config_from_dict refuses a "
            "config whose stamps disagree with the source tree. Seeds come from "
            "seed_family(), a prefix of the one frozen SEED_FAMILY, and both arms are built "
            "from the same list."
        ),
    },
    {
        "variable": "train/test program split",
        "keys": ("data.condition", "data.data_seed", "data.generator_overrides"),
        "enforced_by": (
            "Both arms are built by cell_config() from one Cell, so the condition and its "
            "single override are the same object. The resolver additionally requires the two "
            "runs' recorded split_hashes to be identical, which checks the data and not only "
            "the request for it."
        ),
    },
    {
        "variable": "token/feature budget",
        "keys": ("data.n_train", "data.n_eval"),
        "enforced_by": (
            "From the rung preset in config.LADDERS, shared by both arms; a difference is a "
            "COMPUTE_BUDGET_KEYS difference and needs wall_clock_claim as well as a "
            "justification."
        ),
    },
    {
        "variable": "tokenizer or synthetic encoding",
        "keys": ("generator_version", "data.condition"),
        "enforced_by": (
            "There is no tokenizer here; the encoding is the feature program's, so the "
            "generator version and the condition are the whole of it. Both arms share both."
        ),
    },
    {
        "variable": "model width and depth where possible",
        "keys": ("arch.d_model", "arch.n_layers", "arch.n_heads", "arch.mlp_ratio"),
        "enforced_by": (
            "The rung preset owns the width; ArchSpec owns depth, heads and MLP ratio, and "
            "the candidate is built by replacing arch.arch alone. width_matched leaves all "
            "four equal. parameter_matched moves arch.d_model on the *control* only and "
            "declares it with the accounting attached."
        ),
    },
    {
        "variable": "optimizer and learning-rate schedule",
        "keys": (
            "optim.optimizer",
            "optim.learning_rate",
            "optim.beta1",
            "optim.beta2",
            "optim.eps",
            "optim.weight_decay",
            "optim.decay_matrices_only",
            "optim.grad_clip",
            "optim.schedule",
            "optim.warmup_fraction",
            "optim.min_lr_fraction",
        ),
        "enforced_by": (
            "OptimizationConfig defaults, one place, no literal at any training call site. "
            "Both arms receive the same instance unless a plan overrides it in writing."
        ),
    },
    {
        "variable": "batch size or total processed examples",
        "keys": ("optim.batch_size", "optim.max_steps"),
        "enforced_by": (
            "COMPUTE_BUDGET_KEYS: equal by construction from the rung preset, and the "
            "resolver re-checks the *measured* step count and examples processed from each "
            "run's own history, because a run that stopped early did less work than its "
            "partner while recording the same max_steps."
        ),
    },
    {
        "variable": "initialization policy",
        "keys": (
            "arch.init_std",
            "arch.scale_residual_projections",
            "arch.bias",
            "arch.residual_write",
            "arch.positional",
        ),
        "enforced_by": (
            "ArchSpec defaults, shared; the seed makes the draw itself identical up to the "
            "shapes the two mechanisms ask for."
        ),
    },
    {
        "variable": "evaluation cadence",
        "keys": ("optim.eval_every", "capture_examples", "geometry_examples"),
        "enforced_by": (
            "The rung preset owns eval_every; the two capture sizes are RunConfig fields "
            "rather than call-site arguments, so a candidate cannot be measured on more rows "
            "than its control — which would read as a difference in geometry."
        ),
    },
    {
        "variable": "stopping rule",
        "keys": ("optim.stopping_rule",),
        "enforced_by": (
            "fixed_steps, and the reported number is the final evaluation rather than the "
            "best one. A COMPUTE_BUDGET_KEYS member: changing it changes how much training a "
            "run gets."
        ),
    },
    {
        "variable": "precision",
        "keys": ("optim.precision", "optim.float32_matmul_precision"),
        "enforced_by": (
            "OptimizationConfig, shared; recorded again in each manifest's numerics block "
            "with the torch version."
        ),
    },
    {
        "variable": "seed set",
        "keys": ("seed",),
        "enforced_by": (
            "ComparisonPlan.seeds is a prefix of SEED_FAMILY and is refused otherwise. Each "
            "pair is one seed, so control and candidate share the *set* and not merely its "
            "size, and the §7.4 test over the arm is paired — which is what prompt 08's "
            "power calibration assumed."
        ),
    },
    {
        "variable": "the loss being optimised",
        "keys": ("optim.loss", "optim.value_loss_weight", "optim.activity_loss_weight"),
        "enforced_by": (
            "Not in §7.2's list because §7.2 assumes it; named here because a candidate "
            "trained against a different objective is not being compared at all."
        ),
    },
    {
        "variable": "the rung and the machine",
        "keys": ("ladder", "device"),
        "enforced_by": (
            "One rung per comparison — a control screened while a candidate is piloted is "
            "not a comparison — and one device, recorded in both manifests."
        ),
    },
)
"""Every §7.2 frozen variable, the config keys that carry it, and what stops it
from drifting. ``tests/experiments/test_comparison_harness.py`` asserts that
these keys cover every field of a :class:`RunConfig` except ``arch.arch``, so a
config field added later cannot be silently unfrozen."""


# --------------------------------------------------------------------------- #
# The gate's diff, in Python
# --------------------------------------------------------------------------- #


def flatten_config(config: Mapping, prefix: str = "") -> dict:
    """Dotted-key flattening, identical to ``bin/check_no_rescue.sh``'s.

    The gate's diff is over flattened keys, and ``permitted_differences`` is
    keyed by them, so this laboratory has to speak the same dialect: a key the
    gate would call ``optim.learning_rate`` cannot be declared here as
    ``learning_rate``. Held to the gate by ``tests/controls`` running the real
    script over a fixture this module produced.
    """
    out: dict = {}
    for key, value in (config or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten_config(value, name + "."))
        else:
            out[name] = value
    return out


def _identity_key(key: str) -> bool:
    return key in ALWAYS_PERMITTED or key.split(".")[-1] in ALWAYS_PERMITTED


def config_differences(control: Mapping, candidate: Mapping) -> dict[str, tuple]:
    """Every flattened key on which two configs disagree, the identity aside.

    Returns ``{key: (control_value, candidate_value)}``, missing keys included as
    ``None`` — a key present in one config and absent from the other is a
    difference and the gate treats it as one.
    """
    left = flatten_config(control)
    right = flatten_config(candidate)
    out: dict[str, tuple] = {}
    for key in sorted(set(left) | set(right)):
        if _identity_key(key):
            continue
        if left.get(key) != right.get(key):
            out[key] = (left.get(key), right.get(key))
    return out


def _undeclared(
    differences: Mapping[str, tuple], permitted: Mapping[str, str]
) -> list[str]:
    """The gate's own verdict on one pair's diff, phrased for a raise."""
    problems = []
    for key, (control_value, candidate_value) in differences.items():
        if key in permitted:
            if not str(permitted[key]).strip():
                problems.append(f"{key}: declared with an empty justification")
            continue
        problems.append(
            f"{key}: control={control_value!r} candidate={candidate_value!r} — undeclared"
        )
    return problems


# --------------------------------------------------------------------------- #
# Arm construction
# --------------------------------------------------------------------------- #

_SECTIONS = ("arch", "data", "optim")


def _apply_overrides(config: RunConfig, overrides: Mapping[str, object]) -> RunConfig:
    """Apply dotted-key overrides to a :class:`RunConfig`.

    Dotted keys rather than nested dicts because that is the language
    ``permitted_differences`` and the gate already speak: the plan's override and
    the declaration that permits it are then literally the same string, and a
    plan cannot move ``optim.learning_rate`` while declaring ``learning_rate``.

    ``arch.arch`` is refused: that is the arm's identity, set by the plan's
    ``control_arch`` and ``candidate_archs``, and an override reaching it would
    let one arm quietly become the other.
    """
    if not overrides:
        return config
    flat = flatten_config(config.as_dict())
    sections: dict[str, dict] = {name: {} for name in _SECTIONS}
    top: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in flat:
            raise ComparisonError(
                f"override {key!r} names no configuration field; expected one of the "
                f"{len(flat)} keys of RunConfig.as_dict()"
            )
        if _identity_key(key):
            raise ComparisonError(
                f"override {key!r} is the architecture identity, which is what the arms "
                "already declare; a comparison whose arms could rename themselves is not a "
                "comparison"
            )
        head, _, tail = key.partition(".")
        if head in sections:
            if "." in tail:
                raise ComparisonError(f"override {key!r} is nested deeper than a config field")
            sections[head][tail] = value
        elif tail:
            raise ComparisonError(f"override {key!r} names no configuration section")
        else:
            top[head] = value
    updated = config
    for name, block in sections.items():
        if block:
            updated = replace(updated, **{name: replace(getattr(updated, name), **block)})
    return replace(updated, **top) if top else updated


def _model_config(config: RunConfig) -> ModelConfig:
    """The concrete model a run config would build, without generating data.

    The feature bank width and the sequence length are properties of the
    generator's configuration, so the family can be asked for them directly. An
    architecture comparison should not have to draw 16 384 examples to find out
    how many parameters it is about to compare.
    """
    generator = config.data.generator_config()
    banks = get_family(generator.family).banks(generator)
    model_config, _ = config.arch.bind(
        n_features=banks.n_features,
        seq_len=generator.seq_len,
        d_recommended=generator.d_recommended,
    )
    return model_config


def _parameters(model_config: ModelConfig) -> int:
    """Parameter count of a config, by building it and counting.

    Wrapped rather than called directly so a test can make an arm artificially
    larger and exercise the parameter-matching path: A0 and A1 have identical
    counts at every width in this laboratory, so nothing real here would ever
    move the control's width, and an unexercised construction is an unchecked
    one.
    """
    return parameters_for(model_config)


def _cell_by_name(name: str) -> Cell:
    known = {cell.name: cell for cell in cells()}
    known[NEGATIVE_CONTROL_CELL.name] = NEGATIVE_CONTROL_CELL
    if name not in known:
        raise ComparisonError(
            f"unknown cell {name!r}; the R3 matrix declares {sorted(known)}"
        )
    return known[name]


@dataclass(frozen=True)
class MatchedPair:
    """One control run and one candidate run, at one cell and one seed.

    The pair is the unit everywhere in this module, because it is the unit the
    gate checks and the unit §7.4's paired test consumes. A comparison over five
    seeds is five pairs sharing a seed set, not one comparison with ten runs in
    it: a control at seed A read against a candidate at seed B is unpaired, and
    ``bin/check_no_rescue.sh`` would correctly call the seed an undeclared
    difference.
    """

    comparison: str
    ladder: str
    strategy: str
    cell: str
    seed: int
    control: RunConfig
    candidate_arch: str
    candidate: RunConfig
    parameters: dict
    permitted_differences: dict
    differences: dict

    @property
    def name(self) -> str:
        return f"{self.comparison}-{self.ladder}-{self.strategy}-{self.cell}-s{self.seed}"

    def planned_label(self, config: RunConfig) -> str:
        """What a plan calls an arm before its run ID exists.

        Prefixed ``planned:`` so it cannot be mistaken for a run ID: §8.3 derives
        those from a digest of the config *and the source tree*, so the ID of a
        run that has not happened yet is not merely unknown but not yet defined.
        """
        return (
            f"planned:{self.ladder}-{config.arch.arch}-{config.data.condition}"
            f"-s{config.seed}-{self.cell}-{self.strategy}"
        )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "cell": self.cell,
            "control_run": self.planned_label(self.control),
            "candidate_runs": [self.planned_label(self.candidate)],
            "declaration": f"{self.name}.json",
        }


@dataclass(frozen=True)
class ComparisonPlan:
    """A declared comparison, before its runs exist.

    Everything §7.2 freezes is either inherited from the rung preset — one place,
    shared by both arms — or written here in a form the gate can read. What the
    plan adds beyond the preset is the *pairing*: which architecture is the
    control, which are the candidates, which matching strategy, and which seeds,
    with control and candidate taking the same seed set rather than the same
    number of seeds.
    """

    name: str
    claim_id: str
    ladder: str
    matching_strategy: str
    control_arch: str = "softmax"
    candidate_archs: tuple[str, ...] = ("linear",)
    seeds: tuple[int, ...] = (SEED_FAMILY[0],)
    cell_names: tuple[str, ...] = (BASE_CELL,)
    d_model: int | None = None
    device: str = "cuda"
    task: str = "T1"
    primary_metric: str | None = None
    permitted_differences: Mapping[str, str] = field(default_factory=dict)
    candidate_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    wall_clock_claim: bool = False
    single_strategy_justification: str = ""
    notes: str = ""
    owner_prompt: str = "12"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_archs", tuple(self.candidate_archs))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "cell_names", tuple(self.cell_names))
        object.__setattr__(self, "permitted_differences", dict(self.permitted_differences))
        object.__setattr__(
            self,
            "candidate_overrides",
            {arch: dict(block) for arch, block in dict(self.candidate_overrides).items()},
        )

        if not self.name or "/" in self.name:
            raise ComparisonError(f"bad comparison name {self.name!r}")
        if not self.claim_id:
            raise ComparisonError(
                f"comparison {self.name!r} names no claim. §7.4 asks for *predeclared* "
                "primary comparisons, and the prediction lives in a committed claim packet "
                "whose commit time bin/check_prereg.sh compares against the run."
            )
        if self.ladder not in LADDERS:
            raise ComparisonError(f"unknown rung {self.ladder!r}; expected {sorted(LADDERS)}")
        if self.matching_strategy not in MATCHING_STRATEGIES:
            raise ComparisonError(
                f"unknown matching strategy {self.matching_strategy!r}; §7.2 names "
                f"{list(MATCHING_STRATEGIES)}"
            )
        known_archs = set(primitive_names())
        for arch in (self.control_arch, *self.candidate_archs):
            if arch not in known_archs:
                raise ComparisonError(
                    f"unknown architecture {arch!r}; this laboratory has {sorted(known_archs)}"
                )
        if not self.candidate_archs:
            raise ComparisonError("a comparison needs at least one candidate architecture")
        if self.control_arch in self.candidate_archs:
            raise ComparisonError(
                f"{self.control_arch!r} is both the control and a candidate; an architecture "
                "compared against itself is a seed-variance measurement, which prompt 09 "
                "already recorded"
            )
        # The seed set, not the seed count. A prefix of the one frozen family, so
        # two arms replicated at five seeds share their five runs' seeds with
        # every other arm in the laboratory and the §7.4 test is paired.
        if tuple(self.seeds) != seed_family(len(self.seeds)):
            raise ComparisonError(
                f"seeds {list(self.seeds)} are not a prefix of §7.2's frozen family "
                f"{list(SEED_FAMILY[: len(self.seeds)])}. Both arms must run the *same seed "
                "set* — the same count is not the same set, and an unpaired comparison at "
                "five seeds is materially weaker than the paired one prompt 08 calibrated."
            )
        for name in self.cell_names:
            _cell_by_name(name)
        unknown = sorted(set(self.candidate_overrides) - set(self.candidate_archs))
        if unknown:
            raise ComparisonError(f"candidate_overrides names no such candidate: {unknown}")
        for key, justification in self.permitted_differences.items():
            if not str(justification).strip():
                raise ComparisonError(
                    f"permitted difference {key!r} has an empty justification; the point of "
                    "declaring it is the sentence"
                )
            if key in COMPUTE_BUDGET_KEYS and not self.wall_clock_claim:
                raise ComparisonError(
                    f"{key!r} decides how much training compute an arm receives. §7.2: a "
                    "candidate may not get more compute merely because it is slower unless "
                    "the claim explicitly concerns wall clock. Set wall_clock_claim if that "
                    "is what this claim is about; otherwise match the budget."
                )
        # A task family with no ladder preset cannot be compared, and saying so
        # here is cheaper than discovering it after the arms are built.
        from architecture_mechanics.experiments.runner import TASK_FAMILIES

        if self.task not in set(TASK_FAMILIES.values()):
            raise ComparisonError(
                f"task family {self.task!r} has no declared §7.3 ladder in this laboratory; "
                f"runner.TASK_FAMILIES offers {sorted(set(TASK_FAMILIES.values()))}. Every "
                "§4.4 condition is built from T1, so a T0 comparison needs a T0 operating "
                "point declared in config.LADDERS first — that is a rung, not a relaxation, "
                "and it is not this module's to invent."
            )
        for name in self.cell_names:
            cell = _cell_by_name(name)
            family = cell_config(cell, ladder=self.ladder).data.generator_config().family
            if family != self.task:
                raise ComparisonError(
                    f"cell {name!r} is built from family {family}, not the declared "
                    f"{self.task}"
                )

    # ---------------------------------------------------------------- arms ---

    def _base(self, cell: Cell, seed: int, arch: str) -> RunConfig:
        return cell_config(
            cell,
            ladder=self.ladder,
            seed=seed,
            arch=arch,
            d_model=self.d_model,
            device=self.device,
        )

    def pairs(self) -> tuple[MatchedPair, ...]:
        """Every matched pair this plan declares, refusing an unfair one.

        This is the construction-time gate. Both arms start from one
        :func:`cell_config` call, so every §7.2 variable is shared by
        construction; the strategy and the plan's declared overrides are the only
        things that may move, and anything that moved without being declared
        raises here — before a model is built, let alone trained.
        """
        out: list[MatchedPair] = []
        problems: list[str] = []
        for cell_name in self.cell_names:
            cell = _cell_by_name(cell_name)
            for seed in self.seeds:
                for arch in self.candidate_archs:
                    control = self._base(cell, seed, self.control_arch)
                    candidate = _apply_overrides(
                        self._base(cell, seed, arch), self.candidate_overrides.get(arch, {})
                    )
                    permitted = dict(self.permitted_differences)
                    if self.matching_strategy == "parameter_matched":
                        control, extra, accounting = self._parameter_match(control, candidate)
                        permitted.update(extra)
                    else:
                        accounting = self._parameter_accounting(control, candidate)
                    differences = config_differences(control.as_dict(), candidate.as_dict())
                    undeclared = _undeclared(differences, permitted)
                    if undeclared:
                        problems.append(
                            f"{self.name} {self.ladder} {self.matching_strategy} "
                            f"{cell_name} s{seed} {self.control_arch} vs {arch}:\n"
                            + "\n".join(f"         {line}" for line in undeclared)
                        )
                    out.append(
                        MatchedPair(
                            comparison=self.name,
                            ladder=self.ladder,
                            strategy=self.matching_strategy,
                            cell=cell_name,
                            seed=seed,
                            control=control,
                            candidate_arch=arch,
                            candidate=candidate,
                            parameters=accounting,
                            permitted_differences=permitted,
                            differences={
                                key: {"control": value[0], "candidate": value[1]}
                                for key, value in differences.items()
                            },
                        )
                    )
        if problems:
            raise ComparisonError(
                "this comparison would not be matched:\n"
                + "\n".join(f"       {line}" for line in problems)
                + "\n       Declare each difference in permitted_differences with a "
                "justification, or match the configs. §7.4: silently tuning the candidate "
                "more than the control is the failure this object exists to prevent."
            )
        return tuple(out)

    # ------------------------------------------------------- matching ---

    def _parameter_accounting(self, control: RunConfig, candidate: RunConfig) -> dict:
        control_model = _model_config(control)
        candidate_model = _model_config(candidate)
        control_parameters = _parameters(control_model)
        candidate_parameters = _parameters(candidate_model)
        return {
            "strategy": self.matching_strategy,
            "control": {
                "arch": control.arch.arch,
                "d_model": control_model.d_model,
                "n_layers": control_model.n_layers,
                "parameters": control_parameters,
            },
            "candidate": {
                "arch": candidate.arch.arch,
                "d_model": candidate_model.d_model,
                "n_layers": candidate_model.n_layers,
                "parameters": candidate_parameters,
            },
            "parameter_difference": candidate_parameters - control_parameters,
            "relative_difference": (
                (candidate_parameters - control_parameters) / control_parameters
                if control_parameters
                else None
            ),
        }

    def _parameter_match(self, control: RunConfig, candidate: RunConfig) -> tuple[RunConfig, dict, dict]:
        """Retune the control's width to the candidate's parameter count.

        The control moves and the candidate does not, on purpose: the candidate
        carries the mechanism under study, and a mechanism whose width is chosen
        to make a comparison come out is not being measured. When the two counts
        already agree — which is the case for every A0/A1 pair in this
        laboratory, at every width — nothing moves, no difference is declared,
        and the accounting says the two strategies coincide.
        """
        target = _parameters(_model_config(candidate))
        control_model = _model_config(control)
        if _parameters(control_model) == target:
            accounting = self._parameter_accounting(control, candidate)
            accounting["adjustment"] = {
                "adjusted": False,
                "reason": (
                    "the control already has exactly the candidate's parameter count at this "
                    "width, so the width-matched and parameter-matched comparisons are the "
                    "same comparison"
                ),
                "coincides_with_width_matched": True,
            }
            return control, {}, accounting

        matched_model, report = parameter_matched_config(control_model, target)
        adjusted = replace(control, arch=replace(control.arch, d_model=matched_model.d_model))
        accounting = self._parameter_accounting(adjusted, candidate)
        accounting["adjustment"] = {
            "adjusted": True,
            "control_d_model_before": control_model.d_model,
            "control_d_model_after": matched_model.d_model,
            "target_parameters": report["target_parameters"],
            "matched_parameters": report["matched_parameters"],
            "relative_error": report["relative_error"],
            "narrower": report["narrower"],
            "wider": report["wider"],
            "searched_widths": report["searched_widths"],
            "coincides_with_width_matched": False,
            "exact": report["matched_parameters"] == report["target_parameters"],
        }
        justification = (
            f"§7.2 parameter matching: the control's width was moved from "
            f"{control_model.d_model} to {matched_model.d_model} to bring its parameter count "
            f"({report['matched_parameters']}) to the candidate's ({report['target_parameters']}), "
            f"a relative error of {report['relative_error']:+.4f}. The candidate is unchanged. "
            "The width-matched comparison is reported alongside this one, per §7.2."
        )
        return adjusted, {"arch.d_model": justification}, accounting

    # --------------------------------------------------------------- record ---

    def arms(self) -> list[dict]:
        """The frozen configuration of every arm, per cell, at the first seed.

        Committed in full because that is what makes the plan a pre-registration
        rather than an intention: the exact configuration each arm will run is in
        git before it runs, and the resolver refuses a run whose recorded config
        does not match it. Only ``seed`` varies across the pairs of one cell, and
        the seed set is declared separately.
        """
        out = []
        for cell_name in self.cell_names:
            by_cell = {"cell": cell_name}
            for pair in self.pairs():
                if pair.cell != cell_name or pair.seed != self.seeds[0]:
                    continue
                by_cell.setdefault(
                    "control",
                    {
                        "arch": pair.control.arch.arch,
                        "config": pair.control.as_dict(),
                        "parameters": pair.parameters["control"]["parameters"],
                    },
                )
                by_cell.setdefault("candidates", []).append(
                    {
                        "arch": pair.candidate_arch,
                        "config": pair.candidate.as_dict(),
                        "parameters": pair.parameters["candidate"]["parameters"],
                        "parameter_accounting": pair.parameters,
                        "permitted_differences": pair.permitted_differences,
                        "differences_from_control": pair.differences,
                    }
                )
            out.append(by_cell)
        return out

    def as_dict(self) -> dict:
        pairs = self.pairs()
        return {
            "schema": PLAN_SCHEMA,
            "status": "planned",
            "name": self.name,
            "comparison": self.name,
            "owner_prompt": self.owner_prompt,
            "claim": self.claim_id,
            "primary_metric": self.primary_metric,
            "primary_metric_source": (
                f"claims/{self.claim_id}.yml#primary_metric_key — read from the packet at "
                "resolution; the echo above is checked against it and a disagreement is "
                "refused, not reconciled"
            ),
            "ladder": self.ladder,
            "task": self.task,
            "matching_strategy": self.matching_strategy,
            "single_strategy_justification": self.single_strategy_justification,
            "control_arch": self.control_arch,
            "candidate_archs": list(self.candidate_archs),
            "seed_set": list(self.seeds),
            "cells": list(self.cell_names),
            "d_model": self.d_model,
            "device": self.device,
            "permitted_differences": dict(self.permitted_differences),
            "candidate_overrides": {
                arch: dict(block) for arch, block in self.candidate_overrides.items()
            },
            "wall_clock_claim": self.wall_clock_claim,
            # The variables and the keys that carry them, not the prose about
            # where each is enforced: that is one paragraph per variable, it is
            # identical in every plan, and it lives in
            # experiments/comparison.py#FROZEN_VARIABLES where a test holds it to
            # the shape of RunConfig. What has to be *in* the plan is the list a
            # reader checks the two configs against.
            "frozen_variables": {
                "source": "experiments/comparison.py#FROZEN_VARIABLES",
                "variables": [
                    {"variable": entry["variable"], "keys": list(entry["keys"])}
                    for entry in FROZEN_VARIABLES
                ],
            },
            "arms": self.arms(),
            "pairs": [pair.as_dict() for pair in pairs],
            "notes": self.notes,
        }

    @property
    def filename(self) -> str:
        return f"{self.name}-{self.ladder}-{self.matching_strategy}.json"

    def write(self, lab: Path | None = None) -> Path:
        path = Path(lab or lab_root()) / PLANNED_DIR / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False) + "\n")
        return path


def load_plan(path: Path | str) -> ComparisonPlan:
    """Rebuild a plan from its committed JSON, refusing one that has drifted.

    The plan on disk carries both the declaration and the arms it implies. Only
    the declaration is read back; the arms are rebuilt from it and compared, so a
    plan whose committed configs no longer match what this source tree would
    produce is refused rather than run. That is the same rule
    :func:`~architecture_mechanics.experiments.config.run_config_from_dict`
    applies to a manifest: re-running it would be a different experiment under
    the same name.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("schema") != PLAN_SCHEMA:
        raise ComparisonError(
            f"{path} has schema {raw.get('schema')!r}, expected {PLAN_SCHEMA!r}"
        )
    plan = ComparisonPlan(
        name=raw["name"],
        claim_id=raw["claim"],
        ladder=raw["ladder"],
        matching_strategy=raw["matching_strategy"],
        control_arch=raw.get("control_arch", "softmax"),
        candidate_archs=tuple(raw.get("candidate_archs") or ()),
        seeds=tuple(raw.get("seed_set") or ()),
        cell_names=tuple(raw.get("cells") or (BASE_CELL,)),
        d_model=raw.get("d_model"),
        device=raw.get("device", "cuda"),
        task=raw.get("task", "T1"),
        primary_metric=raw.get("primary_metric"),
        permitted_differences=raw.get("permitted_differences") or {},
        candidate_overrides=raw.get("candidate_overrides") or {},
        wall_clock_claim=bool(raw.get("wall_clock_claim")),
        single_strategy_justification=raw.get("single_strategy_justification") or "",
        notes=raw.get("notes") or "",
        owner_prompt=raw.get("owner_prompt") or "",
    )
    rebuilt = plan.as_dict()
    for key in ("arms", "pairs"):
        if raw.get(key) != rebuilt[key]:
            raise ComparisonError(
                f"{path} records {key} that this source tree no longer produces. The plan is "
                "a pre-registration of exact configurations; running it now would be a "
                "different experiment under the same name. Re-declare it deliberately."
            )
    return plan


def matched_plans(
    *,
    strategies: Sequence[str] = MATCHING_STRATEGIES,
    single_strategy_justification: str = "",
    **plan_fields,
) -> tuple[ComparisonPlan, ...]:
    """Both §7.2 matching strategies, which is the default and usually the answer.

    §7.2 asks for the width-matched *and* the parameter-matched comparison. The
    reason it asks for both is that either one alone can be the flattering one,
    and which one flatters is knowable in advance. So producing both is free
    here and producing one costs a written justification that goes into the plan
    file and travels with the result.
    """
    unknown = sorted(set(strategies) - set(MATCHING_STRATEGIES))
    if unknown:
        raise ComparisonError(f"unknown matching strategies {unknown}")
    if set(strategies) != set(MATCHING_STRATEGIES) and not single_strategy_justification.strip():
        raise ComparisonError(
            "§7.2 requires both the width-matched and the parameter-matched comparison. "
            "Producing one is allowed and needs a sentence: pass "
            "single_strategy_justification, which is written into the plan and travels with "
            "the result, so a reader can see that one was chosen and why."
        )
    return tuple(
        ComparisonPlan(
            matching_strategy=strategy,
            single_strategy_justification=single_strategy_justification,
            **plan_fields,
        )
        for strategy in MATCHING_STRATEGIES
        if strategy in set(strategies)
    )


def plans_for(name: str, ladder: str, *, lab: Path | None = None) -> tuple[ComparisonPlan, ...]:
    """Every committed plan for one comparison at one rung, both strategies.

    Refuses a rung declared under only one strategy unless that plan says why.
    """
    root = Path(lab or lab_root())
    found = []
    for strategy in MATCHING_STRATEGIES:
        path = root / PLANNED_DIR / f"{name}-{ladder}-{strategy}.json"
        if path.is_file():
            found.append(load_plan(path))
    if not found:
        raise ComparisonError(
            f"no comparison {name!r} declared at rung {ladder}. Expected "
            + " or ".join(
                str(PLANNED_DIR / f"{name}-{ladder}-{strategy}.json")
                for strategy in MATCHING_STRATEGIES
            )
            + ". A comparison is declared and committed before it is run — that is what "
            "makes it a prediction; see reports/comparisons/README.md."
        )
    if len(found) < len(MATCHING_STRATEGIES) and not any(
        plan.single_strategy_justification.strip() for plan in found
    ):
        missing = sorted(set(MATCHING_STRATEGIES) - {plan.matching_strategy for plan in found})
        raise ComparisonError(
            f"comparison {name!r} at rung {ladder} declares only "
            f"{[plan.matching_strategy for plan in found]}; §7.2 asks for both and the "
            f"missing one is {missing}. Declare it, or record "
            "single_strategy_justification in the plan that exists."
        )
    return tuple(found)


# --------------------------------------------------------------------------- #
# Compute ledger
# --------------------------------------------------------------------------- #


def compute_ledger(run_dir: Path) -> dict:
    """What one run actually cost, from its own recorded artifacts.

    §7.4's compute-parity rule is enforced upstream by config equality, which is
    necessary and not sufficient: two runs can declare the same ``max_steps`` and
    do different amounts of work, because a screen may stop early. So the
    measured step count comes from the run's own evaluation history rather than
    from its config, and the examples processed are computed from it.

    ``cost.json`` is gitignored — it is a measurement of this machine at that
    instant — so its numbers are *copied into* the declaration, which is
    committed. Otherwise the ledger would evaporate the moment anyone cloned the
    laboratory. When it is absent the ledger says so instead of reporting zero.
    """
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    config = flatten_config(summary.get("config") or {})
    history = summary.get("history") or []
    declared_steps = int(config.get("optim.max_steps") or 0)
    measured_steps = int(history[-1]["step"]) if history else 0
    batch = int(config.get("optim.batch_size") or 0)

    ledger = {
        "run_id": summary.get("run_id"),
        "declared_steps": declared_steps,
        "measured_steps": measured_steps,
        "stopped_early": measured_steps < declared_steps,
        "batch_size": batch,
        "examples_processed": measured_steps * batch,
        "train_examples": (summary.get("model") or {}).get("train_examples"),
        "evaluations": len(history),
    }
    cost_path = run_dir / "cost.json"
    if cost_path.is_file():
        cost = json.loads(cost_path.read_text())
        ledger.update(
            {
                "wall_clock_seconds": cost.get("wall_clock_seconds"),
                "train_seconds": cost.get("train_seconds"),
                "peak_allocated_mib": cost.get("peak_allocated_mib"),
                "cost_source": "cost.json",
            }
        )
    else:
        ledger.update(
            {
                "wall_clock_seconds": None,
                "train_seconds": None,
                "peak_allocated_mib": None,
                "cost_source": (
                    "absent — cost.json is gitignored, so a clone records the copy in this "
                    "declaration and not the file"
                ),
            }
        )
    return ledger


def _ledger_parity(control: dict, candidate: dict, *, wall_clock_claim: bool) -> dict:
    """Whether two matched runs did the same amount of work, measured.

    Wall clock is *reported* and never enforced: A1 being slower per step than A0
    at the same step count is a property of the mechanism and is exactly what a
    compute-parity rule is meant to leave alone. What must be equal is the work —
    steps taken and examples seen.

    The asymmetry in ``candidate_did_more_work`` is §7.4's own. "Do not allow a
    candidate to receive more training compute merely because it is slower" is a
    rule about favouring the candidate, and it is enforced as one: a candidate
    that did *more* work than its control is refused, while a candidate that did
    *less* — because a kill screen stopped it, which is a result and not a rescue —
    is recorded with the disparity named and the direction stated. Refusing that
    second case would throw away the evidence of the collapse it describes.
    """
    equal_steps = control["measured_steps"] == candidate["measured_steps"]
    equal_examples = control["examples_processed"] == candidate["examples_processed"]
    candidate_did_more = (
        candidate["measured_steps"] > control["measured_steps"]
        or candidate["examples_processed"] > control["examples_processed"]
    )
    seconds = [
        value
        for value in (control.get("train_seconds"), candidate.get("train_seconds"))
        if isinstance(value, (int, float)) and value > 0
    ]
    if equal_steps and equal_examples:
        verdict = "matched"
    elif candidate_did_more:
        verdict = (
            "UNEQUAL WORK favouring the candidate: it took "
            f"{candidate['measured_steps']} steps against the control's "
            f"{control['measured_steps']}"
        )
    else:
        verdict = (
            "unequal work favouring the control: the candidate took "
            f"{candidate['measured_steps']} steps against the control's "
            f"{control['measured_steps']}, which is what an early-stopped screen looks like"
        )
    return {
        "measured_steps_equal": equal_steps,
        "examples_processed_equal": equal_examples,
        "candidate_did_more_work": candidate_did_more,
        "wall_clock_ratio_candidate_over_control": (
            round(candidate["train_seconds"] / control["train_seconds"], 4)
            if len(seconds) == 2
            else None
        ),
        "wall_clock_is_declared_as_the_claim": wall_clock_claim,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Resolution: from plan plus runs to the gate's own object
# --------------------------------------------------------------------------- #


def _manifest(lab: Path, run_id: str) -> dict:
    path = lab / "runs" / run_id / "manifest.json"
    if not path.is_file():
        raise ComparisonError(f"run {run_id} has no manifest.json at {path}")
    return json.loads(path.read_text())


def _primary_metric(plan: ComparisonPlan, lab: Path) -> tuple[str, str]:
    """The metric, from the committed packet, checked against the plan's echo."""
    from architecture_mechanics.experiments.claim_packet import load_packet

    packet_path = lab / "claims" / f"{plan.claim_id}.yml"
    if not packet_path.is_file():
        raise ComparisonError(
            f"comparison {plan.name!r} names claim {plan.claim_id!r}, which is not in "
            f"{lab / 'claims'}. The prediction must exist, and be committed, before the "
            "comparison that tests it runs — bin/check_prereg.sh compares its commit time "
            "against the run's started_utc. Write claims/"
            f"{plan.claim_id}.yml with the twelve §7.1 fields and "
            f"primary_metric_key: {plan.primary_metric or '<the metric>'}, commit it, then run."
        )
    packet = load_packet(packet_path)
    metric = packet.primary_metric_key
    if not metric:
        raise ComparisonError(
            f"claim packet {packet_path} has no primary_metric_key; prose cannot be compared "
            "against a number"
        )
    if plan.primary_metric and plan.primary_metric != metric:
        raise ComparisonError(
            f"plan {plan.filename} was committed echoing primary_metric "
            f"{plan.primary_metric!r} and packet {plan.claim_id!r} declares {metric!r}. The "
            "echo is a commitment made when the plan was committed; a metric that changed "
            "between the declaration and the packet is choosing an outcome."
        )
    return str(metric), f"claims/{plan.claim_id}.yml#primary_metric_key"


def resolve(
    plan: ComparisonPlan,
    run_ids: Mapping[str, str],
    *,
    lab: Path | None = None,
    write: bool = True,
) -> list[dict]:
    """Turn a plan plus the runs it produced into ``reports/comparisons/*.json``.

    ``run_ids`` maps a config fingerprint to the run ID that config produced, so
    a run is matched to its arm by *what it was*, not by the order it happened
    in or by a name someone typed.

    Every refusal here is a refusal to *record* a comparison, which is the last
    place it can be caught: an undeclared difference between the two recorded
    configs, a run whose recorded config is not the one the plan declared, split
    hashes that disagree, or measured work that does not match. The gate would
    catch the first of those; the rest it cannot see.
    """
    lab = Path(lab or lab_root())
    metric, metric_source = _primary_metric(plan, lab)
    out: list[dict] = []

    for pair in plan.pairs():
        arms = {"control": pair.control, "candidate": pair.candidate}
        ids = {}
        for role, config in arms.items():
            fingerprint = config_fingerprint(config)
            run_id = run_ids.get(fingerprint)
            if not run_id:
                raise ComparisonError(
                    f"{pair.name}: the {role} arm ({config.arch.arch}, seed {config.seed}) has "
                    "no recorded run. A declaration names runs; run the comparison with "
                    f"--comparison {plan.name} --ladder {plan.ladder} first."
                )
            ids[role] = run_id

        manifests = {role: _manifest(lab, run_id) for role, run_id in ids.items()}
        checks: dict[str, object] = {}

        # 1. Each run really is the arm the plan declared.
        for role, config in arms.items():
            recorded = flatten_config(manifests[role].get("config") or {})
            declared = flatten_config(config.as_dict())
            drift = {
                key: {"declared": declared.get(key), "recorded": recorded.get(key)}
                for key in sorted(set(declared) | set(recorded))
                if declared.get(key) != recorded.get(key)
            }
            if drift:
                raise ComparisonError(
                    f"{pair.name}: recorded {role} run {ids[role]} does not match the "
                    f"configuration the plan declared: {json.dumps(drift, indent=2)}"
                )
        checks["runs_match_the_declared_configs"] = True

        # 2. The gate's own diff, on the recorded configs rather than on ours.
        differences = config_differences(
            manifests["control"].get("config") or {}, manifests["candidate"].get("config") or {}
        )
        undeclared = _undeclared(differences, pair.permitted_differences)
        if undeclared:
            raise ComparisonError(
                f"{pair.name}: the recorded runs differ outside permitted_differences:\n"
                + "\n".join(f"       {line}" for line in undeclared)
            )
        checks["config_diff_within_permitted_differences"] = True

        # 3. The data, not only the request for it.
        hashes = {
            role: {
                key: (manifests[role].get("split_hashes") or {}).get(key)
                for key in ("train", "eval")
            }
            for role in arms
        }
        if hashes["control"] != hashes["candidate"]:
            raise ComparisonError(
                f"{pair.name}: the two arms trained on different data. "
                f"split_hashes {json.dumps(hashes, indent=2)}"
            )
        checks["split_hashes_identical"] = True
        checks["seed_is_shared_by_both_arms"] = (
            manifests["control"].get("seed") == manifests["candidate"].get("seed") == pair.seed
        )

        # 4. Measured work, not declared work.
        ledger = {
            role: compute_ledger(lab / "runs" / run_id) for role, run_id in ids.items()
        }
        parity = _ledger_parity(
            ledger["control"], ledger["candidate"], wall_clock_claim=plan.wall_clock_claim
        )
        matched_work = parity["measured_steps_equal"] and parity["examples_processed_equal"]
        checks["measured_work_matched"] = matched_work
        if not matched_work:
            work = json.dumps(
                {
                    role: {
                        "measured_steps": entry["measured_steps"],
                        "examples_processed": entry["examples_processed"],
                    }
                    for role, entry in ledger.items()
                }
            )
            if parity["candidate_did_more_work"] and not plan.wall_clock_claim:
                raise ComparisonError(
                    f"{pair.name}: the configs matched and the candidate did more of the work "
                    f"— {work}. §7.4 forbids a candidate receiving more training compute than "
                    "its control merely because it is slower. Re-run the control to the same "
                    "budget, or declare wall clock as the claim."
                )
            if parity["candidate_did_more_work"]:
                checks["unequal_work_declared_as_the_claim"] = True
            else:
                # The candidate did less: a kill screen stopped it, or it failed.
                # That is a result and is recorded as one — refusing it here would
                # discard the evidence of the collapse — but it is named, because a
                # capability number read off a truncated run is not comparable.
                checks["unequal_work_favours_the_control"] = True

        record = {
            "schema": DECLARATION_SCHEMA,
            "name": pair.name,
            "comparison": plan.name,
            "claim": plan.claim_id,
            "primary_metric": metric,
            "primary_metric_source": metric_source,
            "control_run": ids["control"],
            "candidate_runs": [ids["candidate"]],
            "matching_strategy": plan.matching_strategy,
            "permitted_differences": dict(pair.permitted_differences),
            "ladder": plan.ladder,
            "task": plan.task,
            "cell": pair.cell,
            "seed": pair.seed,
            "seed_set": list(plan.seeds),
            "control_arch": plan.control_arch,
            "candidate_arch": pair.candidate_arch,
            "plan": str(PLANNED_DIR / plan.filename),
            "parameter_accounting": pair.parameters,
            "config_differences": pair.differences,
            "compute_ledger": {**ledger, "parity": parity},
            "checks": checks,
            "resolved_utc": utc_now(),
        }
        absent = [key for key in DECLARATION_FIELDS if key not in record]
        empty = [
            key
            for key in DECLARATION_FIELDS
            if key != "permitted_differences" and not record.get(key)
        ]
        if absent or empty:
            raise ComparisonError(
                f"{pair.name}: the declaration would be missing {sorted(set(absent + empty))}, "
                "which is what a comparison is made of"
            )
        if write:
            path = lab / COMPARISONS_DIR / f"{pair.name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
            _verify_readable(path)
        out.append(record)
    return out


def _verify_readable(path: Path) -> None:
    """Prompt 08's reader must accept what we just wrote.

    ``metrics.statistics.load_comparison`` and ``primary_metric_for`` are what
    turn a declaration into a §7.4 comparison record. A file this module emits
    that they refuse would be discovered by whoever came to compute the interval,
    which is after the runs. Checking it here costs a file read.
    """
    from architecture_mechanics.metrics.statistics import load_comparison

    declaration = load_comparison(path)
    if not declaration.control_run or not declaration.candidate_runs:
        raise ComparisonError(f"{path} was written without the runs it names")


# --------------------------------------------------------------------------- #
# Running a declared comparison
# --------------------------------------------------------------------------- #


def run_comparison(
    name: str,
    *,
    ladder: str,
    lab: Path | None = None,
    out_dir: Path | None = None,
    emit_bundle: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    dry_run: bool = False,
    assert_pass: bool = False,
) -> int:
    """Run every arm of a declared comparison, then record the declarations.

    Deliberately thin. §13.3 forbids building an orchestration layer while the
    science is unfinished, and there is nothing here but: read the committed
    plans, refuse anything unmatched *before* the first model is built, run each
    distinct arm configuration once, and hand the results to :func:`resolve`.

    "Each distinct configuration once" matters more than it looks. Under both
    matching strategies the control is often the same configuration — for A0
    against A1 it always is, since their parameter counts agree at every width —
    and a run ID is a digest of the configuration, so the second strategy costs
    no GPU time at all. Two comparisons, one set of runs, and the runs cannot
    have been tuned differently for the two because they are literally the same
    runs.
    """
    from architecture_mechanics.experiments.runner import run as run_one

    lab = Path(lab or lab_root())
    plans = plans_for(name, ladder, lab=lab)
    if out_dir is None and not dry_run:
        raise ComparisonError(
            "a comparison must be recorded: --out none writes no run directories, so there "
            "would be no manifests to declare a comparison over and nothing for "
            "bin/check_no_rescue.sh to check. Use --dry-run to check the plan without running."
        )

    # Everything that can be refused without a GPU, refused before the GPU.
    pairs_by_plan = [(plan, plan.pairs()) for plan in plans]
    # A dry run is a check and not a run, so it reports every problem it found
    # rather than stopping at the first: an operator asking "is this comparison
    # ready" wants the whole answer, including the matched arms it would run.
    claim_problem = None
    metric, metric_source = plans[0].primary_metric, "not yet read from the packet"
    try:
        metric, metric_source = _primary_metric(plans[0], lab)
        for plan in plans[1:]:
            _primary_metric(plan, lab)
    except ComparisonError as error:
        if not dry_run:
            raise
        claim_problem = str(error)

    ordered: dict[str, RunConfig] = {}
    for _, pairs in pairs_by_plan:
        for pair in pairs:
            for config in (pair.control, pair.candidate):
                ordered.setdefault(config_fingerprint(config), config)

    if verbose:
        print(f"comparison {name} at rung {ladder}")
        print(f"  claim          {plans[0].claim_id}")
        print(f"  primary metric {metric}  ({metric_source})")
        print(f"  strategies     {[plan.matching_strategy for plan in plans]}")
        print(f"  seed set       {list(plans[0].seeds)}")
        print(f"  cells          {list(plans[0].cell_names)}")
        for _, pairs in pairs_by_plan:
            for pair in pairs:
                accounting = pair.parameters
                print(
                    f"  {pair.name}: {accounting['control']['arch']}"
                    f"(d={accounting['control']['d_model']}, "
                    f"{accounting['control']['parameters']} params) vs "
                    f"{accounting['candidate']['arch']}"
                    f"(d={accounting['candidate']['d_model']}, "
                    f"{accounting['candidate']['parameters']} params)"
                )
        print(f"  {len(ordered)} distinct arm configurations to run")

    if dry_run:
        if verbose:
            print("  --dry-run: nothing was run and no declaration was written")
            print(
                "  every pair above is matched: no configuration difference outside "
                "permitted_differences"
            )
            if claim_problem:
                print(f"\nNOT READY TO RUN\n  {claim_problem}")
        return 1 if claim_problem else 0

    run_ids: dict[str, str] = {}
    failures: list[str] = []
    claim = f"claims/{plans[0].claim_id}.yml"
    for index, (fingerprint, config) in enumerate(ordered.items(), start=1):
        if verbose:
            print(f"\n[{index}/{len(ordered)}] {config.arch.arch} seed {config.seed}")
        result = run_one(
            config,
            out_dir=out_dir,
            verbose=verbose,
            claim=claim,
            emit_bundle=emit_bundle,
            overwrite=overwrite,
        )
        run_ids[fingerprint] = result.run_id
        if not result.passed:
            failures.append(f"{result.run_id}: {result.verdict}")
            if assert_pass:
                print(f"FAILED: {result.run_id}: {result.verdict}", file=sys.stderr)
                return 1

    written = []
    for plan in plans:
        written += resolve(plan, run_ids, lab=lab, write=True)
    if verbose:
        print(f"\n{len(written)} declarations written to {COMPARISONS_DIR}:")
        for record in written:
            print(f"  {record['name']}.json  {record['control_run']} vs {record['candidate_runs'][0]}")
        if failures:
            print("\nruns that did not pass their rung's own check:")
            for line in failures:
                print(f"  {line}")
    return 0


# --------------------------------------------------------------------------- #
# The declared comparisons of this laboratory
# --------------------------------------------------------------------------- #

DECLARED_COMPARISONS: dict[str, dict] = {
    "a0_vs_a1": {
        "claim_id": "a1-vs-a0-t1-capability-gap",
        "primary_metric": "associative_recall_accuracy",
        "control_arch": "softmax",
        "candidate_archs": ("linear",),
        "task": "T1",
        "cells": (BASE_CELL,),
        "rungs": {"R2": 1, "R3": 1, "R4": 5},
        "notes": (
            "A0 (causal softmax attention) against A1 (kernelized linear attention) on T1, "
            "declared by prompt 12 before prompt 13 ran anything. Every §7.2 variable comes "
            "from the rung preset in experiments/config.py and is therefore shared by both "
            "arms; the mixing primitive is the only difference, and permitted_differences is "
            "empty because there is nothing else to declare. The two matching strategies "
            "coincide exactly here: A0 and A1 have identical parameter counts at every width "
            "in this laboratory (13576 at d=16, 39192 at 32, 77096 at 48, 127288 at 64 on "
            "capacity_stressed), because linear attention replaces the softmax read with a "
            "state read and reuses the same four projections. Both are still declared and "
            "both are still emitted, because §7.2 asks for both and because a later "
            "architecture that does change the count must not find the parameter-matched "
            "path unexercised. "
            "Operating point: the rung preset's own — d=16 at R2 (the condition's "
            "d_recommended) and d=64 at R3 and R4 (config.OPERATING_POINT_EVIDENCE, chosen "
            "on A0 before A1 existed). Prompt 11's R2 boundary screen recorded A1 at 0.096 "
            "exact recall at d=64 against A0's 0.481, both still rising at 2000 steps and "
            "both with active mechanisms: behind, not collapsed. Prompt 13 owns the choice of "
            "operating point and if the intersection of the two competence envelopes says "
            "another width is better it declares a *new* comparison at that width — a new "
            "plan, committed before its runs — rather than editing this one. "
            "The claim packet is a forward reference on purpose: claims/"
            "a1-vs-a0-t1-capability-gap.yml states the prediction, the kill condition and the "
            "rung being aimed at, and bin/check_claims.sh requires a packet claiming any rung "
            "above 0 to arrive with the gates file its evidence wrote. That packet therefore "
            "belongs to the mission that runs the comparison, and this plan refuses to run "
            "until it exists and its primary_metric_key matches the echo committed here."
        ),
    },
}
"""Every comparison this laboratory declares, as data.

Written here rather than typed into JSON so the committed plans are regenerable
and a test can assert that the files on disk are what this source tree produces.
A comparison whose declaration cannot be regenerated is a file nobody can check.
"""


def declare(name: str, *, lab: Path | None = None, write: bool = True) -> list[ComparisonPlan]:
    """Build (and by default write) every plan for one declared comparison."""
    if name not in DECLARED_COMPARISONS:
        raise ComparisonError(
            f"unknown comparison {name!r}; declared: {sorted(DECLARED_COMPARISONS)}"
        )
    spec = DECLARED_COMPARISONS[name]
    out: list[ComparisonPlan] = []
    for ladder, n_seeds in spec["rungs"].items():
        plans = matched_plans(
            name=name,
            claim_id=spec["claim_id"],
            ladder=ladder,
            control_arch=spec["control_arch"],
            candidate_archs=spec["candidate_archs"],
            seeds=seed_family(n_seeds),
            cell_names=spec["cells"],
            task=spec["task"],
            primary_metric=spec["primary_metric"],
            notes=spec["notes"],
        )
        for plan in plans:
            if write:
                plan.write(lab)
            out.append(plan)
    return out


def check(lab: Path | None = None, *, verbose: bool = True) -> int:
    """Re-verify every committed plan, and refuse a plan in the gate's directory.

    No GPU, no data, no model beyond the parameter counts. What it proves is that
    every declared comparison is still constructible and still matched under this
    source tree — which is the property that decays silently as later missions
    edit rung presets.
    """
    lab = Path(lab or lab_root())
    problems: list[str] = []
    lines: list[str] = []

    for path in sorted((lab / COMPARISONS_DIR).glob("*.json")):
        try:
            schema = json.loads(path.read_text()).get("schema")
        except Exception as error:  # noqa: BLE001 — a malformed file is the finding
            problems.append(f"{path.name}: does not parse ({error})")
            continue
        if schema == PLAN_SCHEMA:
            problems.append(
                f"{path.name}: a plan in the gate's own directory. Plans name planned arms, "
                f"not run IDs, so bin/check_no_rescue.sh must fail on it; move it to "
                f"{PLANNED_DIR}."
            )
        elif schema != DECLARATION_SCHEMA:
            problems.append(f"{path.name}: unexpected schema {schema!r}")

    planned = sorted((lab / PLANNED_DIR).glob("*.json"))
    for path in planned:
        try:
            plan = load_plan(path)
            pairs = plan.pairs()
        except (ComparisonError, RunConfigError) as error:
            problems.append(f"{path.name}: {error}")
            continue
        packet = lab / "claims" / f"{plan.claim_id}.yml"
        claim_state = "committed" if packet.is_file() else "PENDING — see notes"
        lines.append(
            f"ok   {path.name}: {len(pairs)} pair(s), seeds {list(plan.seeds)}, "
            f"claim {plan.claim_id} {claim_state}"
        )
        if packet.is_file():
            try:
                _primary_metric(plan, lab)
            except ComparisonError as error:
                problems.append(f"{path.name}: {error}")

    if verbose:
        for line in lines:
            print(line)
        for line in problems:
            print(f"FAIL {line}")
        print(f"\n{len(planned)} plans checked, {len(problems)} problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Declare and check §7.2 matched comparisons. Running one is "
        "architecture_mechanics.experiments.runner --comparison."
    )
    parser.add_argument("--declare", metavar="NAME", default=None,
                        help=f"write every plan for a declared comparison "
                             f"({sorted(DECLARED_COMPARISONS)})")
    parser.add_argument("--check", action="store_true",
                        help="re-verify every committed plan under this source tree")
    parser.add_argument("--lab", default=None, help="laboratory root; defaults to this tree's")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lab = Path(args.lab) if args.lab else None
    if not args.declare and not args.check:
        build_parser().print_help()
        return 2
    if args.declare:
        for plan in declare(args.declare, lab=lab):
            path = Path(lab or lab_root()) / PLANNED_DIR / plan.filename
            if not args.quiet:
                print(f"declared {path.relative_to(Path(lab or lab_root()))}")
    if args.check:
        return check(lab, verbose=not args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
