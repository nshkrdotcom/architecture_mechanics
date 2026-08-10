# claims

One pre-registration packet per candidate: `<claim_id>.yml` with the twelve
7.1 fields and `claimed_rung`, alongside `<claim_id>.gates.json` written only
by code that evaluated evidence. Schemas in `ml/04_SCIENCE_GATES.md`.

A packet must be committed before the run it predicts; `bin/check_prereg.sh`
compares commit time against the run manifest's `started_utc`.
