# CMC Evolving Necrotic Front

[![Tests](https://github.com/azeemali7009-star/CMC-Evolving-Necrotic-Front/actions/workflows/tests.yml/badge.svg)](https://github.com/azeemali7009-star/CMC-Evolving-Necrotic-Front/actions/workflows/tests.yml)

Reproducible Python implementation of an evolving compartment-complex
formulation for forecasting growth-inhibited and necrotic fronts in
multicellular tumour spheroids.

This repository contains the analysis code, algebraic tests, permitted public
data, a downloader for non-redistributed source data, and reproducibility
metadata.

It intentionally excludes manuscript source files, submitted PDFs, cover
letters, generated figures, and generated result tables.

## Repository structure

- `src/cmc_remodelling/`: incidence matrices, event maps, and shell-capacity
  transformations.
- `experiments/run_all.py`: forecasting, comparator, treatment-remodelling,
  and numerical-audit pipeline.
- `tests/`: algebraic and reconstruction tests.
- `data/`: data provenance, permitted redistributed data, and the downloader
  for the WM793b longitudinal data.
- `requirements.txt`: fixed Python dependencies.

## Installation

```bash
git clone https://github.com/azeemali7009-star/CMC-Evolving-Necrotic-Front.git
cd CMC-Evolving-Necrotic-Front

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Obtain the longitudinal data

```bash
make data
```

This downloads the required WM793b longitudinal CSV files from the original
authors' repository. The downloaded files are excluded from Git tracking and
remain subject to the source repository's terms.

## Run the tests

```bash
make check
```

## Reproduce the analysis

```bash
make results
```

Generated result files and figures are written locally to `results/` and
`figures/`. These directories are ignored by Git and are not included in the
public repository.

## Data provenance and licensing

Original software developed for this repository is released under the MIT
License.

Third-party experimental data retain their original copyright and licence
terms. See:

- `NOTICE.md`
- `data/README.md`
- `data/public_data_manifest.csv`

## Scientific scope

This repository supports a computational reanalysis of public preclinical
cell-culture data. It is not a patient-specific model, clinical
decision-support system, or validated treatment-selection system.

## Citation

Citation metadata are provided in `CITATION.cff`.

## Additional leakage-controlled early-origin audit (revision)

After obtaining the longitudinal data, run:

```bash
make early-audit
```

This writes `results/early_origin_one_step_metrics.csv` and
`results/early_origin_one_step_cases.csv`. For each cutoff (6, 9, 12, 15,
18 days by default), models are trained **only on transitions whose target
was observed by that cutoff**. They then predict the *next observed step*
from that cutoff. Empty or undersized splits are marked `NOT_EVALUATED`.
This is **not** an early-to-day-24 rollout, nor an independent-spheroid
validation: the source observations are aggregated by condition and day.
No performance claim should be made without executing this analysis on
retrieved source data. The treatment dataset remains descriptive only.
