# Cost floor — A0 softmax, prompt 04

What one run of each screened rung costs on this machine. Later rungs and later
replications are priced against this. The per-run `runs/*/cost.json` these come
from is deliberately not committed (see `runs/README.md`); this file is the
committed record.

Machine: RTX 5060 Ti, 16 GB (17,102,864,384 bytes total, ~12 GB usable),
compute capability 12.0, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0,
`float32_matmul_precision=highest`.

| rung | condition | shape | params | wall clock | peak allocated | peak reserved |
|---|---|---|---:|---:|---:|---:|
| R0 | positive_control | `F=36 T=12 d=48` | 62,520 | 1.4 s | 66.6 MiB | 68.0 MiB |
| R1 | positive_control | `F=36 T=12 d=48`, 32768 train | 62,520 | 83–86 s | 378.3 MiB | 394.0 MiB |
| R2 | capacity_stressed | `F=124 T=48 d=16`, 16384 train | 13,576 | 43.6 s | 1177.8 MiB | 1210.0 MiB |
| R2 | capacity_stressed | same, `d=32` | 39,192 | 43.6 s | 1188.8 MiB | 1230.0 MiB |
| R2 | capacity_stressed | same, `d=64` | 127,288 | 43.0 s | 1213.9 MiB | 1296.0 MiB |

R1's wall clock is quoted as a range because it is the only quantity here that
moves between identical runs; four runs at the final budget spanned 82.2–86.0 s
while every metric in `summary.json` stayed byte-identical. Roughly 45% of it is
the training loop and the rest is generation, the reference predictors, and the
mechanism capture.

R2 costs half R1's wall clock and three times its memory. The data tensors, not
the model, dominate at this scale — four times the sequence length and three
times the feature bank against a model an order of magnitude smaller. That is
the number to watch when a later prompt raises an example budget; parameter
count is not.

**R4 estimate — 5 seeds of R1, run sequentially: ~7 minutes, ~380 MiB peak.**
Memory is per-run and does not accumulate, so the 12 GB usable would hold on the
order of thirty concurrent R1s; replication at this scale is bounded by wall
clock alone. A five-seed R2 at each of three widths is ~11 minutes.

Nothing in this program's near future is compute-bound. §13.3 applies directly:
runs this cheap are not a reason to build orchestration.
