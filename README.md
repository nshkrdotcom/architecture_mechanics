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
