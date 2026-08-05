# Third-party data notice

The MIT License in this repository applies only to original software written
for this project. It does not relicense third-party experimental data.

## Browning et al. confocal data

`data/AllConfocalData.csv` is obtained from:

Alexander P. Browning et al.,
"Quantitative analysis of tumour spheroid structure",
eLife 10:e73020 (2021).
DOI: https://doi.org/10.7554/eLife.73020

Original repository:
https://github.com/ap-browning/Spheroids

Original file:
https://github.com/ap-browning/Spheroids/blob/main/Data/AllConfocalData.csv

The source repository is licensed under the Creative Commons Attribution
4.0 International License (CC BY 4.0). The CSV values are included without
scientific modification.

## Diosdi et al. group-mean data

`data/diosdi_2025_group_means.csv` is a structured transcription of the
group-level measurements reported in Table 2 of:

Akos Diosdi et al.,
"Single-cell light-sheet fluorescence 3D images of tumour-stroma spheroid
multicultures",
Scientific Data 12, 492 (2025).
DOI: https://doi.org/10.1038/s41597-025-04832-0

The article is licensed under CC BY 4.0. The published values were transcribed
into CSV format, and machine-readable column names were introduced. This
repository does not claim authorship of the underlying experimental data.

## Murphy et al. longitudinal data

The following files are required by the analysis but are not redistributed
through this repository:

- `data/WM793b_outer.csv`
- `data/WM793b_inhibited.csv`
- `data/WM793b_necrotic.csv`

They originate from:

Ryan J. Murphy et al.,
"Designing and interpreting 4D tumour spheroid experiments",
Communications Biology 5, 91 (2022).
DOI: https://doi.org/10.1038/s42003-022-03018-3

Original repository:
https://github.com/ryanmurphy42/4DSpheroids_Murphy2021

The command `make data` downloads the files directly from the authors'
repository and stores them under the local `data/` directory. These downloaded
files are excluded from Git tracking.
