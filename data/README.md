# Datasets

Download every dataset with

```bash
python scripts/download_data.py            # or: python scripts/download_data.py mocap pump
```

Seven datasets are hosted on the Hugging Face Hub in
[fchavelli/tsb-seg](https://huggingface.co/datasets/fchavelli/tsb-seg), one
configuration per dataset, each with its license and Croissant metadata: they
land as `data/<name>.parquet`,
exactly the series of the paper. USC-HAD and UTSA have no explicit license and
are not redistributed: the script fetches them from their source into
`data/usc-had/` and `data/utsa/`. `data/registry.json` pins the revision and
sha256 of every file. The loader (`src/tsbseg/data_loading/loaders.py`) reads
`data/<name>.parquet` when it exists, the source layout otherwise.

| Dataset | Series | Channels | Source | License |
|---|---|---|---|---|
| MoCap | 9 | 4 | CMU Graphics Lab, subject 86; labels of Time2State (embedded in `tsseg`) | CMU terms (research use, no resale) |
| HAS | 250 | 6 or 9 | Human Activity Segmentation Challenge (ECML/PKDD 2023), via the ClaP repository | CC BY-NC-SA 4.0 |
| ActRecTut | 2 | 10 | Bulling et al. 2014, `subject{1,2}_walk`, via Time2State | GPL-3.0-or-later |
| PUMP | 120 | 9 | Hydraulic End-of-Line test cycles (Bosch Research; Gaugel and Reichert, 2023) | CC BY 4.0 |
| SKAB | 34 | 8 | Skoltech Anomaly Benchmark, via the ClaP repository | GPL-3.0 |
| USC-HAD | 70 | 6 | USC Human Activity Dataset | no explicit license (citation requested) |
| UTSA | 32 | 1 | UTSA archive (Gharghabi et al.), via the ClaP repository | no explicit license at the source |
| TSSB | 75 | 1 | Time Series Segmentation Benchmark (Ermshaus et al.) | BSD-3-Clause |
| PAMAP2 | 9 | 9 | UCI, DOI 10.24432/C5NW2H (scalability case study) | CC BY 4.0 |

Series are indexed from 1 (`series` in the results), in the order of
`src/tsbseg/data_loading/series.csv`, which names the file of each series.
