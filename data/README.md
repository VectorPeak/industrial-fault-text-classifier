# Data Layout

`data/full/chemical_repair_text_dataset_cn.csv` is the committed full constructed dataset used by the default pipeline examples.

`data/samples/sample_repair_text.csv` is the smaller public sample used by the smoke pipeline.

`data/raw/` is reserved for local raw TXT/TSV source files and is ignored by Git.

Both committed CSV files follow this schema:

| column | description |
|--------|-------------|
| `text` | repair text |
| `fault_category` | fault category label |
| `risk_level` | downtime risk level |
| `department` | responsible department label |

Generated files such as standardized datasets, train/validation/test splits, and reports should be written to `data/processed/` or `artifacts/reports/`; both are ignored by Git.
