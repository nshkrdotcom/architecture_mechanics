# runs

One directory per run, named from stable config and source identity. R0-R2
screens carry `manifest.json`, `metrics.jsonl`, `summary.json`; R3+ evidence
bundles carry the full 8.4 set. Checkpoints and raw tensors are gitignored;
everything else here is committed, including runs that failed.

`cost.json` is the exception, and the reason is worth stating. Wall clock,
peak VRAM, and free VRAM are measurements of this machine at that instant, not
of the experiment; they change on every re-run while nothing else does. Keeping
them out of `summary.json` buys a property worth more than the record: an
identical config re-run reproduces `summary.json` byte for byte, so a dirty run
directory means the science changed. The cost floor itself is committed, in
`reports/cost_floor.md`.
