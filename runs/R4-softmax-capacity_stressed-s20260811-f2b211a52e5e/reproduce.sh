#!/usr/bin/env bash
# Regenerate R4-softmax-capacity_stressed-s20260811-f2b211a52e5e from its own recorded identity.
#
# Pins three things and re-runs:
#
#   source   git commit 3589c23feb92c8490bda5d32fc2754ebb8f409be, exported with `git archive` — read-only, no
#            worktree is added and nothing in the laboratory is mutated;
#   config   the `config` block of manifest.json, handed back to the runner
#            verbatim, so options that never had a command-line flag still
#            reproduce;
#   seed     20260811, which is inside that config.
#
# Then it checks itself: the regenerated summary.json must agree with the
# recorded one on the primary metric, or this script exits non-zero. A
# reproduction script that cannot fail is not evidence of anything.
#
#   ./reproduce.sh [output_dir]

set -euo pipefail

RUN_ID="R4-softmax-capacity_stressed-s20260811-f2b211a52e5e"
COMMIT="3589c23feb92c8490bda5d32fc2754ebb8f409be"
LAB="${AM_LAB_DIR:-/home/home/p/g/n/architecture_mechanics}"
CLAIM="claims/a0-t1-associative-recall.yml"
PRIMARY_METRIC="associative_recall_accuracy"
LOCK_HASH="6e018e7cf306742270ee2ff7a986657860cf18e656e14f1bcf7f28797abfb81e"
SOURCE_TREE_HASH="e1ec3cee8599b9e0a28007a8b053defb02cc6835637743df5734d33c124ff7d4"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(mktemp -d -t am-reproduce-XXXXXX)}"
mkdir -p "$OUT"
OUT="$(cd -- "$OUT" && pwd)"

echo "reproducing $RUN_ID"
echo "  commit    $COMMIT"
echo "  lab       $LAB"
echo "  output    $OUT"

if [ ! -d "$LAB/.git" ]; then
  echo "FAIL: $LAB is not a git repository; set AM_LAB_DIR to the laboratory" >&2
  exit 2
fi
if ! git -C "$LAB" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  echo "FAIL: commit $COMMIT is not in $LAB" >&2
  exit 2
fi

SRC="$OUT/source"
mkdir -p "$SRC"
git -C "$LAB" archive "$COMMIT" | tar -x -C "$SRC"

lock_now="$(sha256sum "$SRC/uv.lock" | cut -d' ' -f1)"
if [ "$lock_now" != "$LOCK_HASH" ]; then
  echo "FAIL: uv.lock at $COMMIT hashes $lock_now, manifest recorded $LOCK_HASH" >&2
  exit 3
fi

python3 - "$HERE/manifest.json" > "$OUT/config.json" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))["config"], indent=2))
PY

# The pinned source is put ahead of the installed package on PYTHONPATH, and the
# laboratory's own environment supplies the dependencies. Building a fresh
# environment here would reach the network, which no mission of this program is
# authorised to do, and would prove less: uv.lock is verified above, so the
# dependency set is already known to be the recorded one.
export PYTHONPATH="$SRC/src"
export AM_SOURCE_COMMIT="$COMMIT"
( cd "$LAB" && uv run --no-sync python -m architecture_mechanics.experiments.runner \
    --config-json "$OUT/config.json" \
    --claim "$CLAIM" \
    --out "$OUT/runs" \
    --emit-bundle \
    --quiet )

NEW="$OUT/runs/$RUN_ID"
if [ ! -d "$NEW" ]; then
  echo "FAIL: expected $NEW; the run identity did not reproduce" >&2
  ls -1 "$OUT/runs" >&2
  exit 4
fi

python3 - "$HERE/summary.json" "$NEW/summary.json" "$PRIMARY_METRIC" "$SOURCE_TREE_HASH" \
        "$HERE/manifest.json" "$NEW/manifest.json" <<'PY'
import json, sys

original, regenerated, metric, source_hash, m_old, m_new = sys.argv[1:7]
a, b = json.load(open(original)), json.load(open(regenerated))
ma, mb = json.load(open(m_old)), json.load(open(m_new))

rows, bad = [], 0
def row(name, x, y):
    global bad
    same = x == y
    bad += 0 if same else 1
    rows.append(f"  {'ok  ' if same else 'DIFF'} {name}: {x} vs {y}")

row(f"final.{metric}", a.get("final", {}).get(metric), b.get("final", {}).get(metric))
row("verdict.passed", a.get("passed"), b.get("passed"))
row("run_id", a.get("run_id"), b.get("run_id"))
row("manifest.source_tree_hash", source_hash, mb.get("source_tree_hash"))
row("manifest.split_hashes", ma.get("split_hashes"), mb.get("split_hashes"))
row("manifest.parameter_count", ma.get("parameter_count"), mb.get("parameter_count"))

print("\n".join(rows))
if bad:
    print(f"\nFAIL: {bad} field(s) did not reproduce")
    sys.exit(5)
print("\nok   the primary metric and the run identity reproduced exactly")
PY

echo "regenerated run: $NEW"
