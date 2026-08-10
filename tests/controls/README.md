# `tests/controls/` — the controls, watched failing

Every other directory here tests that something *works*. This one tests that the
laboratory's controls can still **fail**, by constructing the situation each was
built to catch and requiring it to be caught.

Prompt 10 ran these once by hand and wrote the output into
`state/10_instrument_review.md`. A one-off demonstration protects nothing: a
later mission can loosen a threshold, widen an exemption, or route around a gate
and every green tick in the suite will stay green. As tests they are re-run on
every commit.

Each test also carries its own **non-vacuity control** — the thing that must
still pass — because a check that fails for everything is not a check.

| file | the control, and what makes it fail |
|---|---|
| `test_positive_control_can_fail.py` | R1's known-easy task, fed models and predictors that cannot transport |
| `test_negative_control_can_fail.py` | the information-destroyed condition, fed an honest oracle and a cheating one |
| `test_metrics_report_their_null.py` | capability, geometry, mechanism and statistics, each fed pure noise |
| `test_gates_can_fail.py` | the four science gates, each fed the artifact it exists to refuse |

The thresholds asserted here are the laboratory's own frozen constants, imported
rather than restated. A test that hard-coded `0.80` would keep passing after
somebody moved `POSITIVE_CONTROL_THRESHOLD`.
