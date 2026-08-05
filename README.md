# CMC Evolving Necrotic Front

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
