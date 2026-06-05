# Data Layout

`data/raw/` is reserved for local full datasets and is ignored by Git.

`data/samples/sample_repair_text.csv` is the public sample used by the smoke pipeline. It follows this schema:

| column | description |
|--------|-------------|
| `text` | repair text |
| `fault_category` | fault category label |
| `risk_level` | downtime risk level |
| `department` | responsible department label |

Generated files such as standardized datasets, train/validation/test splits, and reports should be written to `data/processed/` or `artifacts/reports/`; both are ignored by Git.

