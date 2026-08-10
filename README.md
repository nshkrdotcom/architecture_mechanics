# Architecture Mechanics

> Given matched small models and synthetic tasks with **known ground-truth
> features**, how do alternative sequence-mixing and state-writing mechanisms
> change feature packing, cross-token transport, overwrite behavior, and causal
> legibility?

The laboratory generates its own data and trains its own tiny models, so every
result is reproducible from this repository and a GPU.

## Run the ladder

```bash
make r0            # build A0 and check the section 8.5 invariants; no training
make r1            # the known-easy positive control; exits non-zero if A0 fails it
make r2            # the capacity-stressed kill screen
```

A0 — standard causal softmax attention — is the reference architecture. Its fast
path is held to a loop-written slow reference in `tests/equations`, because every
later claim in this program is stated relative to A0 and a bug there is a bug in
everything.

## How much A0 differs from itself

```bash
make t1-ladder     # R1 -> R2 -> R3 -> R4 on T1, then both reports
```

Every architecture comparison in this program is a claim that two architectures
differ by more than an architecture differs from itself, so that second number
has to exist first. Measured on the T1 capacity-stressed operating point over
eight seeds that differ only in initialisation and batch order:

| | across-seed sd | smallest difference visible at 5 seeds |
|---|---|---|
| `associative_recall_accuracy` | **0.054** on a mean of 0.491 | **0.150**, which is 31% of the mean |
| `geometry.mean_purity` | 0.021 on 0.184 | 0.056, 31% |
| `geometry.participation_ratio` | 2.74 on 14.5 | 5.82, 41% |

**Five seeds cannot see an architecture difference on T1 recall smaller than
about 0.15 exact recall.** Ten seeds reach 0.076 and twenty reach 0.051. The
spread is training variance and not scoring noise — the evaluation split is
bitwise identical across seeds and its binomial noise bound is 0.0078, seven
times smaller — and it shrinks monotonically through training, so it is mostly
*when* the retrieval circuit forms rather than what it converges to.

The same number decides what the R3 difficulty sweep can say. Of §4.3's five T1
axes, **only feature sparsity moves A0 further than its own seed noise**; source
distance, distractor count, key collisions and simultaneous associations all have
ranges inside it at one seed per cell. `reports/a0_t1_difficulty_curves.json`
records that verdict per axis rather than leaving it to whoever plots the curve.

## Reproduce a figure

```bash
make figure1       # regenerates paper/figures/fig1_benchmark_schematic.png
```

Figures are generated only from recorded artifacts — figure 1 from one real
generated example, later figures from recorded run outputs — and never from
hand-drawn numbers. Regeneration is byte-identical: delete the PNG, run the
command, and the file comes back with the same sha256, so a changed figure
means changed evidence. The caption beside each PNG carries every parameter
needed to reproduce it.

## What this laboratory can and cannot see

```bash
make statistics-selftest      # re-runs both §7.4 calibrations and checks every recorded threshold
```

The estimators in `metrics/statistics.py` are calibrated against data with **no
effect at all**, so their false-positive rates are measured rather than assumed,
and against injected effects of known size, so their power is too. At the five
seeds §10.1 asks for, the smallest paired difference detected with 80% power is
**1.68 standard deviations of the run-to-run difference**; a half-standard-deviation
difference needs 34 seeds. Below that line a null result means *underpowered*,
not *no effect*, and every comparison in this repository is designed knowing it.

The run is the experimental unit and the API enforces it: every test takes one
`RunSummary` per run and refuses a bare array. Pooling 64 tokens per run narrows
the standard error by a factor of five, which is the difference between a null
result and a finding.

## Check the GPU

```bash
make gpu-check     # resolves CUDA, prints the manifest record, runs a real matmul
```

It fails loudly if CUDA is missing. Nothing here silently falls back to CPU: a
run whose manifest says `cuda` but which executed on CPU is a lie in the
provenance record.

## Test and lint

```bash
make test          # uv run pytest tests -q
make lint          # uv run ruff check src tests
```

`tests/determinism` proves that one seed gives bitwise-identical tensors across
two **fresh processes**, not two calls in one. `tests/identity` proves that
requesting CUDA without CUDA raises.

## Layout

```text
src/architecture_mechanics/
  seeding.py device.py       seed once per entry point; resolve the device honestly
  data/ models/              feature program and task families; one file per mechanism
  instrumentation/           hooks, state capture, interventions
  metrics/ experiments/      capability, geometry, mechanism activity; config and runner
  reporting/                 tables, figures, evidence bundles
configs/  tests/             baselines, screens, replications; equations, identity,
                             interventions, determinism
claims/   runs/              pre-registration packets; one directory of evidence per run
reports/  paper/             comparisons and figures; the publishable artifact
```

Most modules are empty stubs whose docstring names the prompt that fills them.
A stub that returned a plausible tensor would be worse than an empty one.

## Requirements

Python 3.11–3.12 and `uv`. `uv sync` installs `torch 2.11.0+cu128` from the
PyTorch CUDA 12.8 index — the default PyPI wheels carry no `sm_120` kernels and
cannot drive a Blackwell GPU. Runtime dependencies are torch, numpy, matplotlib,
and pyyaml; pytest and ruff are dev-only. Every added dependency is a
reproduction liability and needs a written justification.
