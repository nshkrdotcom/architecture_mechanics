# runs

One directory per run, named from stable config and source identity. R0-R2
screens carry `manifest.json`, `metrics.jsonl`, `summary.json`; R3+ evidence
bundles carry the full 8.4 set. Checkpoints and raw tensors are gitignored;
everything else here is committed, including runs that failed.
