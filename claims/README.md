# claims

One pre-registration packet per candidate: `<claim_id>.yml` with the twelve
7.1 fields and `claimed_rung`, alongside `<claim_id>.gates.json` written only
by code that evaluated evidence. Schemas in `ml/04_SCIENCE_GATES.md`.

A packet must be committed before the run it predicts; `bin/check_prereg.sh`
compares commit time against the run manifest's `started_utc`.

**A packet is never edited after its runs.** The gate reads the *last* commit
that touched the file, so any edit — including an honest correction — moves that
timestamp past every run the packet governs and turns pre-registered runs into
post-hoc ones. A component of a committed prediction that the evidence does not
support is therefore recorded in a sibling `<claim_id>.corrections.md`, naming
the mission that found it, what was measured, whether any kill condition fires,
and who owns the repair. See `a0-t1-associative-recall.corrections.md`.

A packet may also declare which runs it governs:

```yaml
covers:
  ladder: [R0, R1, R2]
  arch: [softmax]
  condition: [positive_control, capacity_stressed]
```

A recorded run given no `--claim` takes the one packet whose `covers:` block
names its rung, architecture and condition; none, or more than one, is refused
before the first gradient step. All three axes are required — a scope that
leaves an axis open would adopt every future run on it — so widening a
pre-registration's reach costs a commit, which is the same discipline the
packet is already under.
