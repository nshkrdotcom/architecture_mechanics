# Corrections to `a0-t1-associative-recall.yml`

Recorded by **prompt 10** (instrument review), 2026-08-10. The packet itself is
**not edited**, and that is deliberate: `bin/check_prereg.sh` establishes that a
pre-registration predates its runs by reading the *last* commit that touched the
file, so any edit here — however honest — would move that timestamp past every
run this packet governs and turn 28 legitimately pre-registered runs into
post-hoc ones. A prediction that can be revised is not a prediction. So the
prediction stands as written and this file records, beside it, which of its
components the evidence does not support.

Nothing below changes `claimed_rung` or the gates file. Rung 0
(`implementation_survives`) and rung 1 (`mechanism_is_active`) are unaffected:
neither depends on any of the clauses corrected here, and both were re-derived
from the recorded checkpoints during this review. No kill condition fires under
any of these corrections; each was checked against its own declared bar.

---

## 1. `NEGATIVE_CONTROL` — "I(input; target) = 0 by construction" is false

**Withdrawn as stated. The control's verdict survives; the mechanism by which
it survives is not the one the packet claims.**

Measured on the `negative_control` test split, 4096 examples, reading only the
input tensor:

| channel | measurement |
|---|---|
| H(answer content group) | 1.4594 bits |
| I(answer group ; the query position's own key bits) | **0.2074 bits** (14.2% of H) |
| I(answer group ; which content group has most mass in the input) | **0.0152 bits** |
| P(the input's dominant content group *is* the answer's) | **0.3149** against a chance of 0.2500 |

Two independent causes, both structural:

- a program template is `(operation, content_group, key_group, distance_bucket)`
  and the coverage-preserving holdout removes 6 of 24 cells, so `content_group`
  and `key_group` are **not independent within a split**. The query's key bits
  name its key group (`data/feature_program.py:546` puts a key's bits inside one
  contiguous block), and the source's content group is the answer's
  (`data/task_families.py:279,310`);
- `_destroy_sources` redraws the destroyed binding's content from *the answer's
  own content group* (`data/feature_program.py:1057-1064` against `:1066`), so
  that group is over-represented among the sequence's content cells.

`answer_appears_in_input` is still exactly 0 and the 33-strategy perfect-memory
battery is still at R² 0.0000, so the answer itself is genuinely absent. What
leaks is *which 24 of the 96 content features the answer lives in*.

**Consequence, measured, on the two metrics the kill condition names:** none.
A predictor exploiting the strongest channel scores `associative_recall_accuracy`
**0.0000** and normalized skill **0.0000**, identically to the frequency ceiling
— exact answer-set equality over 24 candidate features is unreachable from a
group label. What moves is `feature_f1`: the ceiling reads 0.0700 and the
key-group-conditional predictor 0.0832, +18.7% relative. A0 scored 0.0492 on
that metric, *below* both, so prompt 09's "at chance" verdict is strengthened
rather than weakened by the correction.

**Not bounded, and not fixed here.** Removing either channel is a change to the
generator's semantics, which requires a `GENERATOR_VERSION` bump, which
invalidates the recorded dataset hash of all 40 runs. Owner: **prompt 02's
generator, at whichever mission next bumps `GENERATOR_VERSION`** (prompt 18 is
the first scheduled one). Two candidate repairs, both cheap at that point: draw
the destroyed source's replacement content from a group *other* than the answer
group; and hold out template cells so that `content_group ⟂ key_group` survives
the split.

## 2. `STRUCTURALLY_ENFORCED_PROPERTIES` — "the seed moves initialisation and batch order only"

**Corrected. The clause is false; the property it was protecting holds.**

`experiments/runner.py:468-469` also derives the geometry probe partition
(`split_seed = config.seed + 2`) and the matched-site random projection
(`projection_seed = config.seed + 3`) from the run seed, so each of the eight R4
seeds scores its geometry on a different random half of the evaluation examples.
The offsets are arithmetic and the frozen seed family is consecutive, so within
one arm seed *k*'s projection stream and seed *k+1*'s probe-split stream are the
same PCG64 stream, and seed *k*'s batch-order stream (`runner.py:796`,
`config.seed + 1`) is the stream seed *k+1* initialises its weights from.

The property the clause exists to protect — that the reported spread is the
spread of the training procedure — was re-measured directly by re-running the
geometry pass over the eight recorded checkpoints three ways:

| measure | recorded sd | eight models, each at its own split | eight models, one fixed split | one model, eight splits | split's share of variance |
|---|---|---|---|---|---|
| `probe_macro_r2` | 0.009713 | 0.009713 | 0.009537 | 0.000486 | **0.3%** |
| `probe_macro_auc` | 0.019654 | 0.019654 | 0.020046 | 0.000940 | 0.2% |
| `mean_purity` | 0.020792 | 0.020792 | 0.019937 | 0.000821 | 0.2% |
| `interference_fraction` | 0.006869 | 0.006869 | 0.006437 | 0.000386 | 0.3% |
| `effective_rank` | 2.814916 | 2.814916 | 2.814916 | 0.000000 | 0.0% |
| `participation_ratio` | 2.738412 | 2.738412 | 2.738412 | 0.000000 | 0.0% |
| `capacity_total` | 1.508591 | 1.508591 | 1.533443 | 0.091891 | 0.4% |

The probe partition accounts for **0.0% to 0.4%** of the recorded variance. The
seed-spread numbers in `reports/a0_t1_seed_variance.json` are model variance, as
claimed. The clause overstates what is enforced; the deliverable is unaffected.

**Not fixed here.** Replacing the arithmetic offsets with `derive_seed` is a
two-line change and is the right one — but it would move every future probe
partition and batch order, so every recorded metric would become
irreproducible from the current source for a measured benefit of under half a
percent of one variance. Owner: **prompt 12 or 15**, at whichever mission next
has a reason to re-record the arm.

## 3. `POSITIVE_CONTROL` — R1 tests transport, not content addressing

**Corrected. R1's verdict stands; its scope is narrower than the packet's
wording implies.**

The known-easy condition sets `n_associations = 1`, so each example holds
exactly one keyed binding and "return the value bound to this key" and "return
the value at the one keyed position" are the same instruction. Measured on the
positive control: the ordinal strategy `copy_first_keyed` scores R² **1.0000**,
tying `key_match_exact`, and the best fixed-offset copier scores
`associative_recall_accuracy` **0.4968** — half the task, with no addressing at
all. On the mechanism half, a fixed "attend to *t*−1" pattern that performs no
retrieval clears the `min_retrieval_lift = 2.0` gate at **6.09×**.

Prompt 02's own oracle table already printed `copy_first_keyed 1.0000` for this
condition; nothing gated on it and no artifact read it as a limitation on R1's
scope. It now has an invariant of its own —
`positive_control_addressing_is_ordinal` in
`data/feature_program.run_selftest` — so the property is checked rather than
implied, and a later mission that makes the positive control harder will be told.

R1 remains a valid instrument gate: it is the difference between a mixer that
moves information and one that does not, which is exactly what §7.3 asks R1 to
establish, and a crippled A0 fails it (0.0028 and 0.0000 against 0.9055; see
`tests/controls/test_positive_control_can_fail.py`). What a green R1 does **not**
license is the sentence "A0 solves associative recall".

## 4. `heldout_composition_accuracy` is a duplicate column in every recorded run

The runner scores on the evaluation split alone, and every template in that
split is a held-out template, so `heldout_template_ids` covers every row:
`heldout_composition_accuracy` equals `associative_recall_accuracy` to the last
digit in all 40 recorded summaries and `heldout_composition_gap` is `null`
throughout. It is not wrong, it is uninformative, and it must not be cited as a
generalisation measure. Prompt 03 built it to be read on a *concatenated*
seen-plus-held-out reference (`make t0` does this); no recorded run does.
Owner: **prompt 12**, which should either score a concatenated reference or stop
reporting the column.
