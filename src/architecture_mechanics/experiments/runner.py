"""The single training/eval entry point: config in, metrics out.

One function, :func:`run`, takes a :class:`~.config.RunConfig` and returns a
:class:`RunResult`. There is no experiment framework here and there must not
become one — §13.3 is explicit that building orchestration before the baseline
is solved is the failure mode this whole program exists to avoid. Everything
that varies between runs is a field of the config; everything that does not is
written once, here.

The rungs of §7.3:

``R0``  build the model and check the §8.5 invariants. No optimisation, no data
        beyond a handful of examples. The authoritative versions of these checks
        are in ``tests/``; this rung is the one-command gate before GPU time.
``R1``  the known-easy positive control. A0 must solve it rapidly and its
        mechanism must become active. ``--assert-pass`` turns the verdict into
        an exit code.
``R2``  the capacity-stressed kill screen: short, one seed, stop on collapse,
        inactivity, or numerical failure.

R1's verdict is deliberately not computed here. It comes from
:func:`~architecture_mechanics.metrics.capability.positive_control`, which was
written and calibrated in prompt 03 against the oracle, chance, and the
frequency ceiling — before any architecture existed that its threshold could
have been chosen to flatter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from architecture_mechanics.data.feature_program import (
    FeatureProgramDataset,
    condition_config,
    generate_dataset,
)
from architecture_mechanics.device import resolve_device
from architecture_mechanics.experiments.config import (
    LADDERS,
    RunConfig,
    config_fingerprint,
    ladder_config,
)
from architecture_mechanics.instrumentation.hooks import NO_HOOKS, CaptureContext, capture_all
from architecture_mechanics.metrics.capability import (
    EvaluationReference,
    Predictions,
    ProgramOracle,
    associative_recall_accuracy,
    evaluate_all,
    feature_detection,
    fit_marginal,
    normalized_skill,
    positive_control,
    positive_control_datasets,
    reconstruction_loss,
)
from architecture_mechanics.metrics.mechanism import (
    MECHANISM_VERSION,
    attention_retrieval,
    mechanism_is_active,
)
from architecture_mechanics.models.common import FeatureModel, ModelOutput, parameter_report
from architecture_mechanics.seeding import SeedRecord, seed_everything

__all__ = ["RunResult", "evaluate", "main", "run", "run_r0_checks"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Everything one run produced.

    Split across two files by :func:`_write`. Everything the experiment
    determines goes to ``summary.json``; the measurements that belong to the
    machine and the moment rather than to the experiment — wall clock, peak
    VRAM, free VRAM at start — go to ``cost.json``. Re-running an identical
    config therefore rewrites ``summary.json`` byte for byte, which is what
    makes "the run directory is unchanged" a meaningful statement.
    """

    run_id: str
    config: dict
    device: dict
    seeding: dict
    model: dict
    parameters: dict
    checks: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    references: dict = field(default_factory=dict)
    mechanism: dict = field(default_factory=dict)
    positive_control: dict | None = None
    kill: dict | None = None
    cost: dict = field(default_factory=dict)
    passed: bool = False
    verdict: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Batchable:
    """One split as device tensors, plus the columns the loss looks at."""

    inputs: torch.Tensor
    targets: torch.Tensor
    supervised: torch.Tensor
    target_active: torch.Tensor
    content: torch.Tensor
    weights: torch.Tensor

    @property
    def n_examples(self) -> int:
        return int(self.inputs.shape[0])


def _to_device(dataset: FeatureProgramDataset, device: torch.device) -> _Batchable:
    content = torch.as_tensor(dataset.content_indices, dtype=torch.long, device=device)
    weights = dataset.importance.to(device)[content]
    return _Batchable(
        inputs=dataset.inputs.to(device),
        targets=dataset.targets.to(device),
        supervised=dataset.target_mask.to(device),
        target_active=dataset.target_active_mask.to(device),
        content=content,
        weights=weights,
    )


def _datasets(config: RunConfig) -> tuple[FeatureProgramDataset, FeatureProgramDataset]:
    """Train and evaluation splits for this rung.

    R1 goes through :func:`positive_control_datasets` rather than building its
    own, so that no run can accidentally train on a positive control it quietly
    made easier than the one it is judged on.
    """
    spec = config.data
    if spec.condition == "positive_control":
        return positive_control_datasets(n_examples=spec.n_train, seed=spec.data_seed)
    overrides: dict = {}
    if spec.data_seed is not None:
        overrides["seed"] = spec.data_seed
    train = generate_dataset(
        condition_config(spec.condition, split="train", n_examples=spec.n_train, **overrides)
    )
    evaluation = generate_dataset(
        condition_config(spec.condition, split="test", n_examples=spec.n_eval, **overrides)
    )
    return train, evaluation


# --------------------------------------------------------------------------- #
# Loss and optimisation
# --------------------------------------------------------------------------- #


def compute_loss(output: ModelOutput, batch: _Batchable, rows: torch.Tensor, config: RunConfig):
    """Importance-weighted value MSE plus activity BCE, at supervised positions.

    The value term is exactly
    :func:`~architecture_mechanics.metrics.capability.reconstruction_loss`, down
    to the importance weighting and the restriction to the content bank, so the
    objective and the primary reconstruction metric cannot drift apart. The
    activity term is BCE against the ground-truth active mask, which is what the
    detection and answer-set metrics score.

    Unsupervised positions contribute nothing. They carry an all-zero target by
    construction, and training a model to emit zeros at ten of twelve positions
    would make "predicts nothing" the dominant gradient.
    """
    content = batch.content
    supervised = batch.supervised[rows].float()
    denominator = supervised.sum().clamp_min(1.0)

    values = output.values.index_select(-1, content)
    logits = output.active_logits.index_select(-1, content)
    targets = batch.targets[rows].index_select(-1, content)
    active = batch.target_active[rows].index_select(-1, content).to(values.dtype)

    weights = batch.weights
    residual = values - targets
    value_per_position = (residual * residual * weights).sum(-1) / weights.sum()
    value_loss = (value_per_position * supervised).sum() / denominator

    bce = F.binary_cross_entropy_with_logits(logits, active, reduction="none").mean(-1)
    active_loss = (bce * supervised).sum() / denominator

    total = (
        config.optim.value_loss_weight * value_loss
        + config.optim.activity_loss_weight * active_loss
    )
    return total, {
        "value_loss": float(value_loss.detach()),
        "activity_loss": float(active_loss.detach()),
    }


def build_optimizer(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
    optim = config.optim
    if optim.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer {optim.optimizer!r}")
    if optim.decay_matrices_only:
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        groups = [{"params": list(model.parameters()), "weight_decay": optim.weight_decay}]
    return torch.optim.AdamW(
        groups, lr=optim.learning_rate, betas=(optim.beta1, optim.beta2), eps=optim.eps
    )


def learning_rate_at(step: int, config: RunConfig) -> float:
    """Linear warmup into a cosine decay. One schedule, shared by every run."""
    optim = config.optim
    if optim.schedule != "cosine":
        raise ValueError(f"unsupported schedule {optim.schedule!r}")
    total = max(1, optim.max_steps)
    warmup = max(1, round(optim.warmup_fraction * total))
    if step < warmup:
        return optim.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = optim.min_lr_fraction
    return optim.learning_rate * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict(model: FeatureModel, inputs: torch.Tensor, *, chunk: int = 512) -> Predictions:
    """Value and activity channels for a whole split."""
    model.eval()
    values: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], chunk):
        output = model(inputs[start : start + chunk], hooks=NO_HOOKS)
        values.append(output.values.detach().float().cpu().numpy())
        probabilities.append(output.active_prob.detach().float().cpu().numpy())
    return Predictions(
        values=np.concatenate(values, axis=0),
        active_prob=np.concatenate(probabilities, axis=0),
    )


def evaluate(
    model: FeatureModel, dataset: FeatureProgramDataset, batch: _Batchable
) -> tuple[dict, Predictions, EvaluationReference]:
    """Every §6.1 metric on one split, plus the loss the objective optimises."""
    reference = EvaluationReference.from_dataset(dataset)
    predictions = predict(model, batch.inputs)
    metrics = {
        name: value.value for name, value in evaluate_all(predictions, reference).items()
    }
    return metrics, predictions, reference


def training_curve_metrics(
    model: FeatureModel, dataset: FeatureProgramDataset, batch: _Batchable
) -> dict:
    """The four numbers the training curve plots, without the full metric suite.

    ``evaluate_all`` walks every program record twice and sweeps twenty-one
    thresholds; at the R1 example budget that is slower than the training it is
    meant to monitor. The curve gets the primary metric and its two companions,
    and the *reported result* still comes from ``evaluate_all`` at the end, so
    nothing is decided from the cheap path.
    """
    reference = EvaluationReference.from_dataset(dataset)
    predictions = predict(model, batch.inputs)
    detection = feature_detection(predictions, reference)
    return {
        "eval_reconstruction_loss": reconstruction_loss(predictions, reference).value,
        "eval_feature_f1": detection.f1,
        "eval_associative_recall_accuracy": associative_recall_accuracy(
            predictions, reference
        ).value,
    }


@torch.no_grad()
def capture_mechanism(
    model: FeatureModel, batch: _Batchable, dataset: FeatureProgramDataset, limit: int
) -> dict:
    """One instrumented forward pass, then the §6.3 activity report.

    Bounded to ``limit`` examples because the attention weight tensor is the one
    object here whose size is quadratic in sequence length.
    """
    model.eval()
    rows = min(limit, batch.n_examples)
    hooks = CaptureContext(capture=("weights",))
    model(batch.inputs[:rows], hooks=hooks)
    captures = dict(hooks.captures)

    distribution = model.mechanism_activity(captures)
    retrieval: dict = {}
    for key, weights in captures.items():
        if not key.endswith(".weights"):
            continue
        layer = key[: -len(".weights")]
        report = attention_retrieval(weights, dataset.programs[:rows], layer=layer)
        retrieval[layer] = report

    verdict = mechanism_is_active(distribution, retrieval)
    return {
        "n_examples": rows,
        "distribution": distribution,
        "retrieval": {
            layer: (None if report is None else report.as_dict())
            for layer, report in retrieval.items()
        },
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# R0 — the §8.5 invariants, as a runnable gate
# --------------------------------------------------------------------------- #


def run_r0_checks(model: FeatureModel, inputs: torch.Tensor) -> dict:
    """Shapes, causality, determinism, gradients, hooks, and accounting.

    Returns a name-to-record mapping rather than raising, so the CLI can print
    every failure at once. ``tests/identity`` and ``tests/equations`` hold the
    authoritative, adversarial versions of each of these; this is the gate an
    operator runs before spending GPU time.
    """
    config = model.config
    results: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: object = "") -> None:
        results[name] = {"ok": bool(ok), "detail": detail}

    model.eval()
    with torch.no_grad():
        output = model(inputs)
    expected = (inputs.shape[0], inputs.shape[1], config.n_features)
    record(
        "shapes",
        tuple(output.values.shape) == expected and tuple(output.active_logits.shape) == expected,
        {"values": tuple(output.values.shape), "expected": expected},
    )

    # Causal masking, by perturbation: change the last position's input and
    # assert that no earlier output moved by even one bit.
    perturbed = inputs.clone()
    perturbed[:, -1] = perturbed[:, -1] + 7.5
    with torch.no_grad():
        after = model(perturbed)
    leak = (after.values[:, :-1] - output.values[:, :-1]).abs().max()
    record("causal_masking", bool(leak.item() == 0.0), {"max_earlier_change": float(leak)})

    with torch.no_grad():
        again = model(inputs)
    record(
        "deterministic_forward",
        bool(torch.equal(again.values, output.values)),
        "two forward passes on identical input are bitwise equal",
    )

    # Gradients finite under large-magnitude inputs. 50x the generator's scale:
    # inputs are Uniform(0, 1) where active, so this is far outside anything a
    # run will see, which is the point.
    model.train()
    model.zero_grad(set_to_none=True)
    large = model(inputs * 50.0)
    (large.values.square().mean() + large.active_logits.square().mean()).backward()
    finite = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    largest = max(
        (float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None), default=0.0
    )
    record("gradients_finite_under_large_inputs", finite, {"max_abs_grad": largest})
    model.zero_grad(set_to_none=True)
    model.eval()

    # Hook no-op equivalence: register *every* site, not a sample.
    hooks = capture_all()
    with torch.no_grad():
        hooked = model(inputs, hooks=hooks)
    record(
        "hooks_are_no_ops",
        bool(
            torch.equal(hooked.values, output.values)
            and torch.equal(hooked.active_logits, output.active_logits)
        ),
        {"n_sites_captured": len(hooks.captures)},
    )
    declared = set(model.hook_sites())
    visited = set(hooks.visited)
    record(
        "declared_sites_are_reached",
        declared == visited,
        {"declared_only": sorted(declared - visited), "visited_only": sorted(visited - declared)},
    )

    report = parameter_report(model)
    record("parameter_accounting", True, report)
    return results


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run(config: RunConfig, *, out_dir: Path | None = None, verbose: bool = True) -> RunResult:
    """Train, evaluate, and report. The whole entry point."""
    started = time.perf_counter()
    seed_record: SeedRecord = seed_everything(config.seed)
    torch.set_float32_matmul_precision(config.optim.float32_matmul_precision)
    device, device_record = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_dataset, eval_dataset = _datasets(config)
    train_batch = _to_device(train_dataset, device)
    eval_batch = _to_device(eval_dataset, device)

    model_config, d_source = config.arch.bind(
        n_features=train_dataset.n_features,
        seq_len=int(train_dataset.inputs.shape[1]),
        d_recommended=train_dataset.config.d_recommended,
    )
    model = FeatureModel(model_config).to(device)
    parameters = parameter_report(model)

    run_id = f"{config.ladder}-{config.arch.arch}-{config.data.condition}-s{config.seed}-{config_fingerprint(config)}"
    result = RunResult(
        run_id=run_id,
        config=config.as_dict(),
        device=device_record.as_dict(),
        seeding=seed_record.as_dict(),
        model=model_config.as_dict() | {"d_model_source": d_source},
        parameters=parameters,
    )
    result.model["d_recommended"] = train_dataset.config.d_recommended
    result.model["honours_d_recommended"] = (
        model_config.d_model == train_dataset.config.d_recommended
    )
    result.references = {
        "train": train_dataset.summary(),
        "eval": eval_dataset.summary(),
    }

    if verbose:
        print(f"[{run_id}] {LADDERS[config.ladder]['description']}")
        print(f"  device   {device_record.device_name} ({device_record.resolved})")
        print(
            f"  model    d={model_config.d_model} ({d_source}) layers={model_config.n_layers} "
            f"heads={model_config.n_heads} params={parameters['total']}"
        )
        print(
            f"  data     {config.data.condition} F={train_dataset.n_features} "
            f"T={train_dataset.inputs.shape[1]} train={train_dataset.n_examples} "
            f"eval={eval_dataset.n_examples}"
        )

    result.checks = run_r0_checks(model, train_batch.inputs[: min(8, train_batch.n_examples)])
    failed_checks = [name for name, record in result.checks.items() if not record["ok"]]

    if config.ladder == "R0":
        result.passed = not failed_checks
        result.verdict = (
            "R0 invariants hold" if result.passed else f"R0 failures: {failed_checks}"
        )
        result.cost = _cost(started, device)
        _write(result, out_dir)
        if verbose:
            _print_checks(result.checks)
        return result

    if failed_checks:
        result.passed = False
        result.verdict = f"refusing to train: R0 failures {failed_checks}"
        result.cost = _cost(started, device)
        _write(result, out_dir)
        return result

    train_started = time.perf_counter()
    history, stopped_early = _train(model, config, train_batch, eval_batch, eval_dataset, verbose)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_started
    result.history = history

    metrics, _predictions, reference = evaluate(model, eval_dataset, eval_batch)
    result.final = metrics
    result.mechanism = capture_mechanism(model, eval_batch, eval_dataset, config.capture_examples)
    result.mechanism["mechanism_version"] = MECHANISM_VERSION

    # Reference predictors on the same evaluation data, so every number in the
    # summary has a floor and a ceiling attached rather than standing alone.
    marginal = fit_marginal(train_dataset)
    oracle = ProgramOracle(fallback=marginal)
    result.references["oracle"] = _reference_scores(oracle.predict(eval_dataset), reference)
    result.references["marginal"] = _reference_scores(marginal.predict(eval_dataset), reference)
    result.references["skill"] = {
        name: normalized_skill(
            metrics.get(name),
            result.references["marginal"].get(name),
            result.references["oracle"].get(name),
        )
        for name in ("associative_recall_accuracy", "answer_set_accuracy", "feature_f1",
                     "reconstruction_loss")
    }

    if config.ladder == "R1":
        verdict = positive_control(
            lambda dataset: _predict_checked(model, dataset, eval_dataset, device),
            n_examples=config.data.n_train,
            seed=config.data.data_seed,
        )
        result.positive_control = verdict.as_dict()
        active = result.mechanism["verdict"]["active"]
        result.passed = bool(verdict.passed and active)
        result.verdict = (
            f"{verdict.summary()}; mechanism {'active' if active else 'INERT'}"
            + ("" if active else f" ({'; '.join(result.mechanism['verdict']['reasons'])})")
        )
    elif config.ladder == "R2":
        result.kill = _kill_screen(result, history, stopped_early)
        result.passed = not result.kill["fired"]
        result.verdict = (
            "kill screen survived"
            if result.passed
            else f"kill conditions fired: {result.kill['fired']}"
        )

    result.cost = _cost(started, device) | {"train_seconds": round(train_seconds, 3)}
    result.cost["r4_five_seed_estimate_seconds"] = round(5 * (time.perf_counter() - started), 1)
    _write(result, out_dir)
    if verbose:
        _print_result(result)
    return result


def _predict_checked(
    model: FeatureModel,
    dataset: FeatureProgramDataset,
    expected: FeatureProgramDataset,
    device: torch.device,
) -> Predictions:
    """Predict, after proving the judge handed us the data we trained against.

    ``positive_control`` regenerates its own evaluation split. Generation is
    deterministic, so it must be bitwise the split this run already holds — and
    if it ever is not, a silent config drift has just turned the positive
    control into a different task. Hashes are cheap; finding that out later is
    not.
    """
    if dataset.content_hash != expected.content_hash:
        raise RuntimeError(
            "the positive control evaluated a different dataset than this run trained "
            f"against ({dataset.content_hash[:12]} vs {expected.content_hash[:12]})"
        )
    return predict(model, dataset.inputs.to(device))


def _reference_scores(predictions: Predictions, reference: EvaluationReference) -> dict:
    return {name: value.value for name, value in evaluate_all(predictions, reference).items()}


def _train(
    model: FeatureModel,
    config: RunConfig,
    train: _Batchable,
    evaluation: _Batchable,
    eval_dataset: FeatureProgramDataset,
    verbose: bool,
) -> tuple[list[dict], bool]:
    optim = config.optim
    optimizer = build_optimizer(model, config)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    history: list[dict] = []
    order = torch.randperm(train.n_examples, generator=generator)
    cursor = 0

    for step in range(optim.max_steps):
        learning_rate = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        if cursor + optim.batch_size > train.n_examples:
            order = torch.randperm(train.n_examples, generator=generator)
            cursor = 0
        rows = order[cursor : cursor + optim.batch_size].to(train.inputs.device)
        cursor += optim.batch_size

        model.train()
        output = model(train.inputs[rows])
        loss, parts = compute_loss(output, train, rows, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            history.append(
                {"step": step, "train_loss": loss_value, "learning_rate": learning_rate,
                 "numerical_failure": True}
            )
            return history, True

        last = step == optim.max_steps - 1
        if last or (step + 1) % optim.eval_every == 0:
            metrics = training_curve_metrics(model, eval_dataset, evaluation)
            entry = {
                "step": step + 1,
                "train_loss": loss_value,
                "learning_rate": learning_rate,
                **parts,
                **metrics,
            }
            history.append(entry)
            if verbose:
                recall = entry["eval_associative_recall_accuracy"]
                shown = "n/a" if recall is None else f"{recall:.4f}"
                print(
                    f"  step {step + 1:>5}  loss {loss_value:.5f}  "
                    f"recon {metrics['eval_reconstruction_loss']:.5f}  recall {shown}"
                )
    return history, False


def _kill_screen(result: RunResult, history: Sequence[dict], stopped_early: bool) -> dict:
    """§7.3 R2: stop on collapse, inactivity, numerical failure, or gross failure.

    Declared before the run, evaluated after it, and not renegotiated. Each
    condition names what it would mean if it fired.
    """
    fired: list[str] = []
    numerical = stopped_early or any(
        not math.isfinite(entry.get("train_loss", 0.0)) for entry in history
    )
    if numerical:
        fired.append("numerical_failure")

    skill = result.references.get("skill", {}).get("associative_recall_accuracy")
    if skill is not None and skill <= 0.0:
        fired.append("gross_baseline_failure")

    recall = result.final.get("associative_recall_accuracy")
    if recall is not None and history:
        first = history[0].get("eval_associative_recall_accuracy")
        if first is not None and recall < 0.5 * first:
            fired.append("collapse")

    if not result.mechanism.get("verdict", {}).get("active", False):
        fired.append("mechanism_inactive")

    return {
        "fired": fired,
        "conditions": {
            "numerical_failure": "any non-finite training loss",
            "gross_baseline_failure": "final recall skill at or below the training marginal",
            "collapse": "final recall below half its first measured value",
            "mechanism_inactive": "the §6.3 activity gates did not all pass",
        },
        "final_recall": recall,
        "recall_skill": skill,
    }


def _cost(started: float, device: torch.device) -> dict:
    record = {"wall_clock_seconds": round(time.perf_counter() - started, 3)}
    if device.type == "cuda":
        record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        record["peak_allocated_mib"] = round(record["peak_allocated_bytes"] / 2**20, 2)
        record["peak_reserved_mib"] = round(record["peak_reserved_bytes"] / 2**20, 2)
    return record


def _write(result: RunResult, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    directory = Path(out_dir) / result.run_id
    directory.mkdir(parents=True, exist_ok=True)

    payload = result.as_dict()
    cost = payload.pop("cost", {})
    # Free VRAM at start is a fact about the machine at that instant, not about
    # the run; it travels with the timings so that what remains is reproducible.
    free_memory = payload.get("device", {}).pop("free_memory_bytes", None)
    cost = {"run_id": result.run_id, "device_free_memory_bytes": free_memory} | cost

    (directory / "summary.json").write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    (directory / "cost.json").write_text(json.dumps(cost, indent=2, default=_json_default) + "\n")
    with (directory / "metrics.jsonl").open("w") as handle:
        for entry in result.history:
            handle.write(json.dumps(entry, default=_json_default) + "\n")


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _print_checks(checks: dict) -> None:
    for name, record in checks.items():
        print(f"  {'ok  ' if record['ok'] else 'FAIL'} {name}")


def _print_result(result: RunResult) -> None:
    print(f"  verdict  {result.verdict}")
    mechanism = result.mechanism.get("verdict", {})
    print(
        f"  mechanism off_diag={mechanism.get('best_off_diagonal_mass')} "
        f"entropy_ratio={mechanism.get('best_entropy_ratio')} "
        f"retrieval_lift={mechanism.get('best_retrieval_lift')}"
    )
    print(
        f"  cost     {result.cost.get('wall_clock_seconds')} s wall, "
        f"{result.cost.get('peak_allocated_mib')} MiB peak allocated"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one rung of the §7.3 ladder.")
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="R1")
    parser.add_argument("--arch", default="softmax")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=None,
                        help="override the condition's d_recommended; recorded in the run identity")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default="runs", help="run directory root, or 'none'")
    parser.add_argument("--assert-pass", action="store_true",
                        help="exit non-zero unless the rung's verdict passes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = ladder_config(
        args.ladder, arch=args.arch, seed=args.seed, device=args.device, d_model=args.d_model
    )
    if args.max_steps is not None:
        config = replace(config, optim=replace(config.optim, max_steps=args.max_steps))

    out_dir = None if args.out.lower() == "none" else Path(args.out)
    result = run(config, out_dir=out_dir, verbose=not args.quiet)

    if args.assert_pass and not result.passed:
        print(f"FAILED: {result.verdict}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
