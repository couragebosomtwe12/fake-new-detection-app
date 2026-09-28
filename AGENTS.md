# Project Guidance

- Run the complete test suite from the repository root with `python -m pytest -q`.
- Real-world evidence fixtures live in `tests/evidence_cases.json`; keep claim text separate from verdict evidence and use published review or correction URLs for traceability.
- Evidence tests must remain offline and deterministic. Do not require live websites or API credentials during pytest runs.
- Do not treat missing fact-check results as evidence that a claim is true, and do not count syndicated copies as independent corroboration.
