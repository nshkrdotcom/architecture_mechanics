.PHONY: test lint gpu-check selftest metrics-selftest geometry-selftest \
        statistics-selftest statistics-calibration t0 gates \
        r0 r1 r2 index figure1 figures geometry-table geometry-across-seeds \
        t1-r1 t1-r2 t1-r3 t1-r4 t1-r4-extended t1-report t1-ladder \
        comparisons comparisons-check comparison-dry-run \
        a0-a1-r1 a0-a1-screen a0-a1-pilot a0-a1-report a0-a1 \
        phase-r1 phase-null phase-d16 phase-d32 phase-d64 phase-length \
        phase-sweep phase-report figure2 phase

# The pre-registration every ladder run below is a child of. A recorded run
# must have one: bin/check_prereg.sh refuses a manifest without it, and the
# runner refuses to write a run directory without it. Named here explicitly;
# omitting --claim resolves the same packet from its committed covers: block.
CLAIM ?= claims/a0-baseline-solves-t0.yml

test:
	uv run pytest tests -q

lint:
	uv run ruff check src tests

gpu-check:
	uv run python -m architecture_mechanics.device

# The data gate: generates all six §4.4 control conditions and asserts the
# invariants that make every downstream measurement meaningful. Run it before
# interpreting anything.
selftest:
	uv run python -m architecture_mechanics.data.feature_program --selftest

# The metric gate: calibrates every §6.1 ruler against the oracle, chance, the
# training marginal, and the frequency ceiling, then re-checks every retirement
# decision recorded in METRIC_SPECS. A retired metric that starts passing fails
# this gate just as loudly as a retained one that stops passing.
metrics-selftest:
	uv run python -m architecture_mechanics.metrics.capability --selftest

# The geometry gate: measures every §6.2 ruler against five constructed
# representations whose answer is known in advance — orthogonal, superposed,
# rotated, collapsed, and pure noise — and re-derives which of them can carry a
# claim alone. A measure whose noise null moves fails this gate.
geometry-selftest:
	uv run python -m architecture_mechanics.metrics.geometry --selftest

# The statistics gate: re-runs both §7.4 calibrations at reduced replicate counts
# and fails if any estimator's false-positive rate has left its recorded
# tolerance — including if one recorded as unusable starts behaving. Twenty
# seconds of CPU; no GPU, no model, no data.
statistics-selftest:
	uv run python -m architecture_mechanics.metrics.statistics --selftest

# The recorded calibration at full replicate counts: 2000 null replicates per
# estimator per noise shape per seed count, the power sweep, and the minimum
# detectable effect. About four minutes of CPU. Its numbers are what
# state/08_statistics.md quotes and what tests/metrics/test_statistics_selftest_gate.py
# checks the register against.
statistics-calibration:
	uv run python -m architecture_mechanics.metrics.statistics --calibrate \
	  --json reports/statistics_calibration.json

# The recorded expected-versus-measured table, regenerated into reports/.
geometry-table:
	uv run python -m architecture_mechanics.metrics.geometry --table \
	  --json reports/geometry_calibration.json

# How much A0 differs from itself: the reference every later "architecture X
# differs from architecture Y" claim has to be read against. Reads the recorded
# summaries and npz files; runs no model.
geometry-across-seeds:
	uv run python -m architecture_mechanics.metrics.geometry \
	  --across-runs $(wildcard runs/R1-softmax-positive_control-*) \
	  --json reports/geometry_across_seeds.json

# T0 end to end with no model anywhere: generate, apply each reference
# predictor, compute every metric, print the table.
t0:
	uv run python -m architecture_mechanics.metrics.capability --t0

gates: selftest metrics-selftest geometry-selftest statistics-selftest \
       comparisons-check

# The §7.2 matched-comparison declarations. `comparisons` rewrites every
# committed plan from experiments/comparison.py#DECLARED_COMPARISONS — a
# pre-registration that cannot be regenerated is a file nobody can check — and
# `comparisons-check` re-verifies that each is still constructible and still
# matched under this source tree, which is the property that decays silently as
# later missions edit rung presets. No GPU, no data, no training.
comparisons:
	uv run python -m architecture_mechanics.experiments.comparison --declare all

comparisons-check:
	uv run python -m architecture_mechanics.experiments.comparison --check

# What a declared comparison would run, and what is still missing, without
# spending anything. Prompt 13's command without the --dry-run.
comparison-dry-run:
	uv run python -m architecture_mechanics.experiments.runner \
	  --comparison a0_vs_a1 --ladder $(LADDER) --dry-run

LADDER ?= R3

# ------------------------------------------------------------------ P13 ---
# The first architecture comparison, in the order §7.3 requires it be run.
# Every stage is a child of claims/a1-vs-a0-t1-capability-gap.yml and each is
# refused until that packet is committed.
#
# a0-a1-r1 is the comparison's own positive control and comes first: both
# architectures are known to solve the known-easy condition, so the pair must
# come out null, and a harness that reports a gap there is measuring itself.
# a0-a1-screen locates the intersection of the two competence envelopes; the
# pilot cells are chosen from its output and declared before the pilot runs.
CMP = uv run python -m architecture_mechanics.experiments.runner --comparison
TABLE = uv run python -m architecture_mechanics.reporting.tables

a0-a1-r1:
	$(CMP) a0_vs_a1_r1 --ladder R1 --emit-bundle

a0-a1-screen:
	$(CMP) a0_vs_a1_screen --ladder R2 --emit-bundle

a0-a1-pilot:
	$(CMP) a0_vs_a1_pilot --ladder R3 --emit-bundle

# Read back from recorded artifacts only; runs no model.
a0-a1-report:
	$(TABLE) --comparison a0_vs_a1_r1 --ladder R1 --json reports/a0_a1_r1_control.json
	$(TABLE) --comparison a0_vs_a1_screen --ladder R2 --json reports/a0_a1_screens.json
	$(TABLE) --comparison a0_vs_a1_pilot --ladder R3 --json reports/a0_a1_pilots.json

a0-a1: a0-a1-r1 a0-a1-screen a0-a1-pilot a0-a1-report

# ------------------------------------------------------------------ P14 ---
# The §10.2 figure 2 sweep: architecture x sparsity x bottleneck ratio, over
# the §4.5 grid, at §7.3's R2 screening depth, one seed. The grid, its price
# and everything cut from it are in experiments/phase_grid.py.
#
# The order below is §7.3's and is not a convenience. phase-r1 is this
# mission's own positive control and comes first, with --assert-pass, because
# seventy screening runs from a broken instrument are seventy measurements of
# the bug. phase-null is the map's own information-destroyed control and comes
# second, because if the task leaks the map is void and that costs two runs to
# find out. The three width panels are the map. The length ribbon is last
# because it is the part that would be dropped if the budget ran out.
#
# No --assert-pass on the R2 stages: a cell where an arm collapses is a
# measured corner of the map and stopping there would discard it.
phase-r1:
	$(CMP) phase_r1 --ladder R1 --emit-bundle --assert-pass

phase-null:
	$(CMP) phase_negative_control_d32 --ladder R2 --emit-bundle

phase-d16:
	$(CMP) phase_T32_d16 --ladder R2 --emit-bundle

phase-d32:
	$(CMP) phase_T32_d32 --ladder R2 --emit-bundle

phase-d64:
	$(CMP) phase_T32_d64 --ladder R2 --emit-bundle

phase-length:
	$(CMP) phase_length_d32 --ladder R2 --emit-bundle

phase-sweep: phase-r1 phase-null phase-d16 phase-d32 phase-d64 phase-length

# Read back from recorded artifacts only; runs no model.
phase-report:
	$(TABLE) --phase --json reports/phase_diagram.json

figure2:
	uv run python -m architecture_mechanics.reporting.figures --figure 2 \
	  --verify-deterministic

phase: phase-sweep phase-report figure2

# The section 7.3 run ladder. R0 builds the model and checks the section 8.5
# invariants without training; R1 is the known-easy positive control and exits
# non-zero if A0 does not solve it; R2 is the capacity-stressed kill screen.
#
# --emit-bundle refuses to finish having written an incomplete §8.4 bundle.
r0:
	uv run python -m architecture_mechanics.experiments.runner --ladder R0 \
	  --claim $(CLAIM) --emit-bundle --assert-pass

r1:
	uv run python -m architecture_mechanics.experiments.runner --ladder R1 \
	  --claim $(CLAIM) --emit-bundle --assert-pass

r2:
	uv run python -m architecture_mechanics.experiments.runner --ladder R2 \
	  --claim $(CLAIM) --emit-bundle

# A0 through the ladder on T1 — the task that cannot be solved without
# transport. Every stage is a child of claims/a0-t1-associative-recall.yml, and
# the stages must be run in this order: t1-r3 refuses to start until a recorded
# R1 for that packet has passed, because a sixteen-cell task matrix run on a
# broken instrument is sixteen measurements of the bug.
#
# R3 and R4 write full §8.4 bundles, so the working tree must be clean or
# bin/check_evidence.sh will (correctly) refuse the result.
T1 = uv run python -m architecture_mechanics.experiments.t1_ladder

t1-r1:
	$(T1) --stage r1

t1-r2:
	$(T1) --stage r2

t1-r3:
	$(T1) --stage r3

t1-r4:
	$(T1) --stage r4

# Three seeds beyond §10.1's five. Whether the five-seed interval was honest is
# only checkable against seeds it did not see, and here that costs four minutes.
t1-r4-extended:
	$(T1) --stage r4 --seeds 20260814 20260815 20260816

# The two reports this mission exists to produce: A0's competence envelope along
# §4.3's five axes, and A0's seed-to-seed spread — the reference every later
# "architecture X differs from architecture Y" has to be read against.
t1-report:
	$(T1) --stage report

t1-ladder: t1-r1 t1-r2 t1-r3 t1-r4 t1-report

# Every run in the laboratory against its claim packet, in one screen. The
# partial defence against selection over experiments; prompt 29 is told to look.
index:
	uv run python -m architecture_mechanics.experiments.index

# The section 10.2 figures, from recorded artifacts only. --verify-deterministic
# deletes the PNG, regenerates it, and exits non-zero unless the bytes match:
# a figure that moves when nothing moved makes review impossible.
figure1:
	uv run python -m architecture_mechanics.reporting.figures --figure 1 \
	  --verify-deterministic

figures: figure1 figure2
