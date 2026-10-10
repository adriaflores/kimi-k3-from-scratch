# Pretraining-study scalar fixtures

`diagnostic_run.json` and `diagnostic_metrics.jsonl` come from the 82M/A=8 diagnostic in the user's `branch3_results(2).zip` (10 October 2026). The optimizer layout and parameter-name lists were removed from the manifest to keep the fixture small; both algorithm identities remain. The JSONL scalar records are unchanged.

These are **Branch 3 diagnostic logs**, with a constant scheduler and only 204,800 training tokens. A regression test requires the Branch 4 analyzer to reject them as research-grid results.

The test module builds separate **synthetic control fixtures** using this schema and the existing trainer's schedule. Their known losses test ranking, ties, rollback, incomplete output and ZIP handling. They do not represent any training result. Generated control figures/reports are temporary and excluded from the deliverable.
