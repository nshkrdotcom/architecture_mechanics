"""The matched-comparison harness: what it refuses, and that it cannot be dodged.

Every test here is about a comparison that *should not be runnable*. A harness
whose refusals are untested is a harness that will be found not to refuse on the
day it matters, which is the day someone is debugging a candidate at 2am and the
learning rate moves.

Three things are checked that ``bin/check_no_rescue.sh`` structurally cannot see:

* the refusal happens at **construction**, before any GPU time — the gate can
  only speak about runs that already exist;
* the two arms share the **same seed set**, not merely the same count, so the
  §7.4 test over the arm is the paired one prompt 08 calibrated;
* the **measured** work of the two runs agreed, not only their declared
  ``max_steps`` — a screen that stopped early did less work than its partner
  while recording an identical config.

The gate itself is exercised on real fixtures in
``tests/controls/test_gates_can_fail.py``; this file is about the half of the
discipline that lives in Python.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from architecture_mechanics.experiments import comparison as C
from architecture_mechanics.experiments.claim_packet import REQUIRED_FIELDS, ClaimPacket
from architecture_mechanics.experiments.config import (
    SEED_FAMILY,
    RunConfig,
    config_fingerprint,
    seed_family,
)
from architecture_mechanics.experiments.runner import build_parser, check_comparison_flags

METRIC = "associative_recall_accuracy"
CLAIM = "a1-vs-a0-t1-capability-gap"


def a_plan(**overrides) -> C.ComparisonPlan:
    """The R3 A0-vs-A1 plan, or a deliberately broken variant of it."""
    fields = {
        "name": "fixture",
        "claim_id": CLAIM,
        "ladder": "R3",
        "matching_strategy": "width_matched",
        "primary_metric": METRIC,
    }
    fields.update(overrides)
    return C.ComparisonPlan(**fields)


# --------------------------------------------------------------------------- #
# The frozen variables are all of them
# --------------------------------------------------------------------------- #


def test_every_config_field_is_a_declared_frozen_variable():
    """The drift guard. A field added to :class:`RunConfig` by a later mission is
    a §7.2 variable nobody froze until it appears in ``FROZEN_VARIABLES``, and
    this test is where that is noticed rather than in a comparison that quietly
    stopped being matched."""
    keys = set(C.flatten_config(RunConfig().as_dict()))
    declared = {key for entry in C.FROZEN_VARIABLES for key in entry["keys"]}
    assert keys - declared == {"arch.arch"}, "unfrozen configuration fields"
    # ``data.generator_overrides`` is a container and is empty in the default
    # config, so it flattens to no key at all; a cell's override appears as
    # ``data.generator_overrides.<field>``. It is one frozen variable either way,
    # and an override present in one arm and absent from the other is a
    # difference the gate reports as a key it can see on one side only.
    assert declared - keys == {"data.generator_overrides"}
    from architecture_mechanics.experiments.t1_ladder import cell_config, cells

    sparse = next(cell for cell in cells(include_base=False) if cell.overrides)
    with_override = set(C.flatten_config(cell_config(sparse).as_dict()))
    assert {key for key in with_override if key.startswith("data.generator_overrides.")}


def test_a_candidate_given_a_different_difficulty_cell_is_a_visible_difference():
    """The empty-container case, checked rather than reasoned about: a candidate
    run at another cell of the matrix differs from its control in a key the diff
    reports, even though the control has no such key at all."""
    from architecture_mechanics.experiments.t1_ladder import cell_config, cells

    base, other = cell_config(cells()[0]), cell_config(
        next(cell for cell in cells(include_base=False) if cell.overrides)
    )
    differences = C.config_differences(base.as_dict(), other.as_dict())
    assert any(key.startswith("data.generator_overrides.") for key in differences)
    assert C._undeclared(differences, {})


def test_the_architecture_is_the_only_always_permitted_difference():
    keys = set(C.flatten_config(RunConfig().as_dict()))
    assert {key for key in keys if C._identity_key(key)} == {"arch.arch"}


def test_compute_budget_keys_are_frozen_variables_too():
    keys = set(C.flatten_config(RunConfig().as_dict()))
    assert set(C.COMPUTE_BUDGET_KEYS) <= keys


# --------------------------------------------------------------------------- #
# Construction refuses the unfair comparison
# --------------------------------------------------------------------------- #


def test_a_matched_plan_builds_and_declares_nothing():
    """The base case: both arms from one rung preset, nothing to declare."""
    pairs = a_plan().pairs()
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.differences == {}
    assert pair.permitted_differences == {}
    assert pair.control.arch.arch == "softmax"
    assert pair.candidate.arch.arch == "linear"
    # Everything except the mechanism is the same object.
    assert pair.control.optim == pair.candidate.optim
    assert pair.control.data == pair.candidate.data
    assert pair.control.seed == pair.candidate.seed


def test_an_undeclared_candidate_learning_rate_is_refused_at_construction():
    """§7.4's named failure, caught before a model is built."""
    with pytest.raises(C.ComparisonError) as error:
        a_plan(candidate_overrides={"linear": {"optim.learning_rate": 1e-3}}).pairs()
    assert "optim.learning_rate" in str(error.value)
    assert "undeclared" in str(error.value)


def test_the_same_difference_declared_with_a_justification_is_allowed():
    """Non-vacuity: the harness refuses undeclared differences, not differences."""
    pairs = a_plan(
        candidate_overrides={"linear": {"optim.learning_rate": 1e-3}},
        permitted_differences={"optim.learning_rate": "fixture: declared deliberately"},
    ).pairs()
    assert pairs[0].differences["optim.learning_rate"] == {
        "control": 3e-3,
        "candidate": 1e-3,
    }


def test_a_declaration_with_an_empty_justification_is_refused():
    with pytest.raises(C.ComparisonError, match="empty justification"):
        a_plan(permitted_differences={"optim.learning_rate": "   "})


def test_more_steps_for_the_slower_candidate_needs_more_than_a_justification():
    """§7.2's compute-parity rule. A justification is exactly what a rescue has —
    the candidate needed it — so a budget difference additionally requires the
    plan to state that wall clock is what the *claim* is about."""
    with pytest.raises(C.ComparisonError, match="wall clock"):
        a_plan(
            candidate_overrides={"linear": {"optim.max_steps": 6000}},
            permitted_differences={"optim.max_steps": "linear attention is slower per step"},
        )


def test_a_wall_clock_claim_may_declare_a_budget_difference():
    plan = a_plan(
        candidate_overrides={"linear": {"optim.max_steps": 6000}},
        permitted_differences={"optim.max_steps": "the claim is about fixed wall clock"},
        wall_clock_claim=True,
    )
    assert plan.pairs()[0].differences["optim.max_steps"]["candidate"] == 6000


def test_an_override_cannot_rename_an_arm():
    with pytest.raises(C.ComparisonError, match="architecture identity"):
        a_plan(candidate_overrides={"linear": {"arch.arch": "softmax"}}).pairs()


def test_an_override_of_a_field_that_does_not_exist_is_refused():
    with pytest.raises(C.ComparisonError, match="names no configuration field"):
        a_plan(candidate_overrides={"linear": {"optim.learning_rate_v2": 1.0}}).pairs()


def test_an_architecture_cannot_be_its_own_control():
    with pytest.raises(C.ComparisonError, match="both the control and a candidate"):
        a_plan(candidate_archs=("softmax",))


def test_an_unimplemented_architecture_is_refused():
    with pytest.raises(C.ComparisonError, match="unknown architecture"):
        a_plan(candidate_archs=("delta_memory",))


# --------------------------------------------------------------------------- #
# The seed set, not the seed count
# --------------------------------------------------------------------------- #


def test_both_arms_take_the_same_seed_set():
    pairs = a_plan(ladder="R4", seeds=seed_family(5)).pairs()
    assert [pair.seed for pair in pairs] == list(seed_family(5))
    for pair in pairs:
        assert pair.control.seed == pair.candidate.seed


def test_seeds_that_are_not_the_frozen_prefix_are_refused():
    """Five seeds nobody else ran is a different experiment with the same name,
    and it silently un-pairs the comparison."""
    with pytest.raises(C.ComparisonError, match="same seed set"):
        a_plan(ladder="R4", seeds=(20260809, 20260810, 20260811, 20260812, 99999999))


def test_the_seed_set_may_not_be_a_shuffle_of_the_family():
    with pytest.raises(C.ComparisonError, match="same seed set"):
        a_plan(ladder="R4", seeds=tuple(reversed(SEED_FAMILY[:5])))


# --------------------------------------------------------------------------- #
# Both matching strategies
# --------------------------------------------------------------------------- #


def test_both_strategies_are_produced_by_default():
    plans = C.matched_plans(name="fixture", claim_id=CLAIM, ladder="R3", primary_metric=METRIC)
    assert [plan.matching_strategy for plan in plans] == list(C.MATCHING_STRATEGIES)


def test_producing_one_strategy_costs_a_written_justification():
    with pytest.raises(C.ComparisonError, match="requires both"):
        C.matched_plans(
            name="fixture", claim_id=CLAIM, ladder="R3", strategies=("width_matched",)
        )
    plans = C.matched_plans(
        name="fixture",
        claim_id=CLAIM,
        ladder="R3",
        strategies=("width_matched",),
        single_strategy_justification="fixture: a reason, recorded in the plan",
    )
    assert len(plans) == 1
    assert plans[0].as_dict()["single_strategy_justification"]


def test_a0_and_a1_have_equal_parameter_counts_so_the_strategies_coincide():
    """The parameter accounting for the comparison this mission declared. Linear
    attention reuses A0's four projections and replaces the softmax read with a
    state read, so at equal width the two are the same size — which is worth
    recording, because it means the parameter-matched comparison adds no
    information here and would be silently vacuous if nobody said so."""
    width = a_plan(matching_strategy="width_matched").pairs()[0]
    matched = a_plan(matching_strategy="parameter_matched").pairs()[0]
    assert width.parameters["control"]["parameters"] == width.parameters["candidate"]["parameters"]
    assert matched.parameters["adjustment"]["adjusted"] is False
    assert matched.parameters["adjustment"]["coincides_with_width_matched"] is True
    assert matched.control.arch.d_model == width.control.arch.d_model
    assert matched.permitted_differences == {}


def test_parameter_matching_moves_the_control_and_declares_it(monkeypatch):
    """The path no real pair in this laboratory exercises yet.

    A candidate is made artificially larger, and the parameter-matched
    construction must widen *the control* — never the candidate, whose mechanism
    is the thing under study — and declare ``arch.d_model`` with the accounting
    in the justification.
    """
    real = C._parameters

    def bigger(model_config):
        count = real(model_config)
        return int(count * 1.6) if model_config.arch == "linear" else count

    monkeypatch.setattr(C, "_parameters", bigger)
    pair = a_plan(matching_strategy="parameter_matched").pairs()[0]

    assert pair.candidate.arch.d_model == 64, "the candidate must not be re-sized"
    assert pair.control.arch.d_model > 64
    assert pair.parameters["adjustment"]["adjusted"] is True
    assert pair.parameters["adjustment"]["control_d_model_before"] == 64
    assert "arch.d_model" in pair.permitted_differences
    assert "parameter matching" in pair.permitted_differences["arch.d_model"]
    # Both bracketing widths are recorded, so a reader can see which way the
    # residual error leans rather than being told it is small.
    assert pair.parameters["adjustment"]["narrower"]
    assert pair.parameters["adjustment"]["wider"]


def test_an_inexact_parameter_match_is_still_reported_as_a_pair(monkeypatch):
    """§7.2: when the counts cannot be equalised exactly, report both
    comparisons. The width-matched one is the other half and is always emitted;
    what this checks is that the parameter-matched one does not quietly become
    exact."""
    real = C._parameters
    monkeypatch.setattr(
        C,
        "_parameters",
        lambda mc: real(mc) + 7 if mc.arch == "linear" else real(mc),
    )
    adjustment = a_plan(matching_strategy="parameter_matched").pairs()[0].parameters["adjustment"]
    assert adjustment["adjusted"] is True
    assert adjustment["exact"] is False
    assert adjustment["relative_error"] != 0


# --------------------------------------------------------------------------- #
# A task family with no ladder cannot be compared
# --------------------------------------------------------------------------- #


def test_a_t0_comparison_is_refused_because_no_rung_declares_t0():
    """T0 is implemented in the generator and has no §7.3 operating point: every
    §4.4 condition is built from T1. The honest refusal names what is missing
    instead of inventing a rung, because "declare the difference" is the rule and
    a new operating point is not this module's to declare."""
    with pytest.raises(C.ComparisonError, match="T0"):
        a_plan(task="T0")


# --------------------------------------------------------------------------- #
# The plan file round-trips and cannot silently drift
# --------------------------------------------------------------------------- #


def test_a_plan_round_trips_through_its_file(tmp_path):
    plan = a_plan()
    path = plan.write(tmp_path)
    assert C.load_plan(path).as_dict() == plan.as_dict()


def test_a_plan_whose_committed_arms_no_longer_match_the_source_is_refused(tmp_path):
    path = a_plan().write(tmp_path)
    payload = json.loads(path.read_text())
    payload["arms"][0]["control"]["config"]["optim"]["learning_rate"] = 1e-4
    path.write_text(json.dumps(payload))
    with pytest.raises(C.ComparisonError, match="no longer produces"):
        C.load_plan(path)


def test_the_committed_declarations_are_what_this_source_tree_produces():
    """The real files under ``reports/comparisons/planned/`` are regenerable.

    A committed pre-registration that cannot be regenerated is a file nobody can
    check; one that can is a file a reviewer can diff against the code that
    claims to have produced it."""
    for name in C.DECLARED_COMPARISONS:
        for plan in C.declare(name, write=False):
            path = C.lab_root() / C.PLANNED_DIR / plan.filename
            assert path.is_file(), f"{path} is declared in code but not committed"
            assert json.loads(path.read_text()) == plan.as_dict(), (
                f"{path.name} differs from what experiments/comparison.py produces; "
                "re-declare it"
            )


def test_the_declared_comparison_names_both_strategies_at_every_rung():
    for name, spec in C.DECLARED_COMPARISONS.items():
        for ladder in spec["rungs"]:
            plans = C.plans_for(name, ladder)
            assert {plan.matching_strategy for plan in plans} == set(C.MATCHING_STRATEGIES)


def test_a_missing_declaration_is_refused_by_name():
    with pytest.raises(C.ComparisonError, match="no comparison"):
        C.plans_for("a_comparison_nobody_declared", "R3")


def test_plans_live_outside_the_gates_directory():
    """``bin/check_no_rescue.sh`` globs ``reports/comparisons/*.json`` and must
    find only resolved declarations: a plan names planned arms, not run IDs, so
    the gate would have to fail on it."""
    assert C.PLANNED_DIR.parent == C.COMPARISONS_DIR
    for path in (C.lab_root() / C.COMPARISONS_DIR).glob("*.json"):
        assert json.loads(path.read_text()).get("schema") != C.PLAN_SCHEMA, path


# --------------------------------------------------------------------------- #
# Resolution: plan plus runs to the object the gate reads
# --------------------------------------------------------------------------- #


def _fake_run(lab, config: RunConfig, *, run_id: str, steps: int | None = None,
              split: str = "h", learning_rate: float | None = None) -> str:
    """A recorded run directory carrying only what the resolver reads.

    Hand-built rather than trained: the resolver's job is to compare recorded
    configs, split hashes and measured work, and none of those need a GPU to be
    wrong in the ways that matter.
    """
    record = config.as_dict()
    if learning_rate is not None:
        record["optim"]["learning_rate"] = learning_rate
    steps = config.optim.max_steps if steps is None else steps
    directory = lab / "runs" / run_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": record,
                "seed": config.seed,
                "split_hashes": {"train": split, "eval": split},
                "parameter_count": 1,
            }
        )
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": record,
                "history": [{"step": steps, "eval_associative_recall_accuracy": 0.5}],
            }
        )
    )
    (directory / "cost.json").write_text(
        json.dumps({"run_id": run_id, "wall_clock_seconds": 10.0, "train_seconds": 8.0})
    )
    return run_id


@pytest.fixture
def resolvable(tmp_path):
    """A laboratory with a committed packet, a plan, and both arms recorded."""
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    ClaimPacket(
        claim_id=CLAIM,
        claimed_rung=1,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
        primary_metric_key=METRIC,
    ).write(lab / "claims" / f"{CLAIM}.yml")
    plan = a_plan()
    plan.write(lab)
    pair = plan.pairs()[0]
    ids = {
        config_fingerprint(pair.control): _fake_run(lab, pair.control, run_id="control-run"),
        config_fingerprint(pair.candidate): _fake_run(lab, pair.candidate, run_id="candidate-run"),
    }
    return lab, plan, pair, ids


def test_resolution_writes_the_five_fields_the_gate_reads(resolvable):
    lab, plan, pair, ids = resolvable
    (record,) = C.resolve(plan, ids, lab=lab)
    path = lab / C.COMPARISONS_DIR / f"{pair.name}.json"
    assert json.loads(path.read_text()) == record
    assert record["control_run"] == "control-run"
    assert record["candidate_runs"] == ["candidate-run"]
    assert record["primary_metric"] == METRIC
    assert record["matching_strategy"] == "width_matched"
    assert record["permitted_differences"] == {}
    assert record["primary_metric_source"].endswith("#primary_metric_key")


def test_the_metric_comes_from_the_packet_and_a_renamed_echo_is_refused(resolvable):
    lab, _, _, ids = resolvable
    renamed = replace(a_plan(), primary_metric="reconstruction_loss")
    with pytest.raises(C.ComparisonError, match="choosing an outcome"):
        C.resolve(renamed, ids, lab=lab)


def test_resolution_refuses_a_run_that_is_not_the_arm_the_plan_declared(tmp_path):
    lab, plan, pair, ids = None, a_plan(), None, None
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    ClaimPacket(
        claim_id=CLAIM,
        claimed_rung=1,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
        primary_metric_key=METRIC,
    ).write(lab / "claims" / f"{CLAIM}.yml")
    pair = plan.pairs()[0]
    ids = {
        config_fingerprint(pair.control): _fake_run(lab, pair.control, run_id="control-run"),
        # The candidate really ran at a different learning rate than the plan
        # declared. Its config no longer matches the pre-registration, and the
        # comparison must not be recorded as if it did.
        config_fingerprint(pair.candidate): _fake_run(
            lab, pair.candidate, run_id="candidate-run", learning_rate=1e-4
        ),
    }
    with pytest.raises(C.ComparisonError, match="does not match the configuration"):
        C.resolve(plan, ids, lab=lab)


def test_resolution_refuses_arms_that_trained_on_different_data(tmp_path):
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    ClaimPacket(
        claim_id=CLAIM,
        claimed_rung=1,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
        primary_metric_key=METRIC,
    ).write(lab / "claims" / f"{CLAIM}.yml")
    plan = a_plan()
    pair = plan.pairs()[0]
    ids = {
        config_fingerprint(pair.control): _fake_run(
            lab, pair.control, run_id="control-run", split="one"
        ),
        config_fingerprint(pair.candidate): _fake_run(
            lab, pair.candidate, run_id="candidate-run", split="another"
        ),
    }
    with pytest.raises(C.ComparisonError, match="different data"):
        C.resolve(plan, ids, lab=lab)


def _lab_with_work(tmp_path, *, control_steps=None, candidate_steps=None):
    """A resolvable laboratory whose two arms did different amounts of work."""
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    ClaimPacket(
        claim_id=CLAIM,
        claimed_rung=1,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
        primary_metric_key=METRIC,
    ).write(lab / "claims" / f"{CLAIM}.yml")
    plan = a_plan()
    pair = plan.pairs()[0]
    ids = {
        config_fingerprint(pair.control): _fake_run(
            lab, pair.control, run_id="control-run", steps=control_steps
        ),
        config_fingerprint(pair.candidate): _fake_run(
            lab, pair.candidate, run_id="candidate-run", steps=candidate_steps
        ),
    }
    return lab, plan, ids


def test_a_candidate_that_did_more_work_than_its_control_is_refused(tmp_path):
    """The case config equality cannot see, in the direction §7.4 names: identical
    ``max_steps``, and the *control* stopped early, so the candidate got more
    training compute than the architecture it is being compared against."""
    lab, plan, ids = _lab_with_work(tmp_path, control_steps=1500)
    with pytest.raises(C.ComparisonError, match="candidate did more of the work"):
        C.resolve(plan, ids, lab=lab)


def test_a_candidate_that_did_less_work_is_recorded_and_named(tmp_path):
    """The other direction is a result, not a rescue: a screen stopped the
    candidate. Refusing it would discard the evidence of the collapse, so it is
    recorded with the disparity and its direction stated."""
    lab, plan, ids = _lab_with_work(tmp_path, candidate_steps=1500)
    (record,) = C.resolve(plan, ids, lab=lab)
    assert record["checks"]["measured_work_matched"] is False
    assert record["checks"]["unequal_work_favours_the_control"] is True
    assert "favouring the control" in record["compute_ledger"]["parity"]["verdict"]


def test_the_ledger_records_the_work_and_the_wall_clock(resolvable):
    lab, plan, _, ids = resolvable
    (record,) = C.resolve(plan, ids, lab=lab)
    ledger = record["compute_ledger"]
    assert ledger["control"]["measured_steps"] == 3000
    assert ledger["control"]["examples_processed"] == 3000 * 128
    assert ledger["control"]["wall_clock_seconds"] == 10.0
    assert ledger["parity"]["measured_steps_equal"] is True
    assert ledger["parity"]["examples_processed_equal"] is True
    assert ledger["parity"]["verdict"] == "matched"


def test_an_unrun_arm_cannot_be_declared(resolvable):
    lab, plan, _, ids = resolvable
    partial = dict(list(ids.items())[:1])
    with pytest.raises(C.ComparisonError, match="no recorded run"):
        C.resolve(plan, partial, lab=lab)


def test_prompt_08s_reader_accepts_what_resolution_writes(resolvable):
    """The declaration is consumed by
    ``metrics.statistics.load_comparison``/``primary_metric_for``, which is what
    turns it into a §7.4 comparison record. A file this module wrote that they
    refuse would be discovered after the runs."""
    from architecture_mechanics.metrics.statistics import load_comparison, primary_metric_for

    lab, plan, pair, ids = resolvable
    C.resolve(plan, ids, lab=lab)
    declaration = load_comparison(lab / C.COMPARISONS_DIR / f"{pair.name}.json")
    metric, source = primary_metric_for(declaration, claims_dir=lab / "claims")
    assert metric == METRIC
    assert declaration.matching_strategy == "width_matched"
    assert declaration.control_run == "control-run"
    assert source.endswith("#primary_metric_key")


def test_a_comparison_cannot_run_before_its_pre_registration_exists(tmp_path):
    """The refusal that costs no GPU time. ``bin/check_prereg.sh`` catches a
    post-hoc claim after the fact; this catches it before the fact."""
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    plan = a_plan()
    plan.write(lab)
    with pytest.raises(C.ComparisonError, match="must exist, and be committed"):
        C.resolve(plan, {}, lab=lab)


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def _args(*argv):
    return build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "flag",
    ["--arch=linear", "--seed=20260809", "--seeds=5", "--d-model=32", "--max-steps=10",
     "--task=t1", "--claim=claims/x.yml", "--config-json=/tmp/x.json"],
)
def test_a_comparison_refuses_a_command_line_that_overrides_it(flag):
    from architecture_mechanics.experiments.config import RunConfigError

    with pytest.raises(RunConfigError, match="declared plan"):
        check_comparison_flags(_args("--comparison", "a0_vs_a1", "--ladder", "R3", flag))


def test_the_ordinary_flags_still_work_with_a_comparison():
    check_comparison_flags(
        _args("--comparison", "a0_vs_a1", "--ladder", "R3", "--emit-bundle", "--quiet")
    )


def test_a_comparison_that_records_nothing_is_refused():
    """``--out none`` writes no run directory, so there would be no manifest to
    declare a comparison over and nothing for the gate to check."""
    with pytest.raises(C.ComparisonError, match="must be recorded"):
        C.run_comparison("a0_vs_a1", ladder="R3", out_dir=None, verbose=False)


def test_dry_run_without_a_comparison_is_refused():
    from architecture_mechanics.experiments.config import RunConfigError

    with pytest.raises(RunConfigError, match="only meaningful"):
        check_comparison_flags(_args("--dry-run"))


def test_a_dry_run_of_the_declared_comparison_checks_every_arm(capsys):
    """What a comparison command does before it spends anything: build both
    strategies, check every pair, and report what is still missing.

    Prompt 12 wrote this asserting a non-zero exit and ``NOT READY TO RUN``,
    because the packet ``a0_vs_a1`` names was a forward reference until the
    mission that ran the comparison committed it. Prompt 13 committed
    ``claims/a1-vs-a0-t1-capability-gap.yml``, so the readiness half of this test
    now asserts the other outcome — the pre-registration exists and the plan is
    ready. The refusal it used to observe has not been dropped: it is asserted
    against a missing packet in
    ``test_a_comparison_cannot_run_before_its_pre_registration_exists``, which is
    where it belongs, since it is a statement about a missing packet and not
    about this laboratory's state on one day."""
    exit_code = C.run_comparison("a0_vs_a1", ladder="R3", dry_run=True)
    out = capsys.readouterr().out
    assert "width_matched" in out and "parameter_matched" in out
    assert "2 distinct arm configurations" in out, "the coinciding controls run once"
    assert "every pair above is matched" in out
    assert exit_code == 0 and "NOT READY TO RUN" not in out
    assert "a1-vs-a0-t1-capability-gap" in out
    assert "associative_recall_accuracy" in out
