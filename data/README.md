# Data

This directory distinguishes redistributed CC BY 4.0 material from
download-only third-party data.

## Included files

- `AllConfocalData.csv`: Browning et al. confocal structural measurements,
  redistributed under CC BY 4.0 with attribution.
- `diosdi_2025_group_means.csv`: machine-readable transcription of published
  group means from Diosdi et al., distributed under CC BY 4.0 with attribution.
- `public_data_manifest.csv`: provenance and licence summary.
- `fetch_murphy_data.py`: downloader for the WM793b longitudinal CSV files.

## Download required longitudinal files

Run:

```bash
python data/fetch_murphy_data.py
```

or:

```bash
make data
```

This creates:

- `WM793b_outer.csv`
- `WM793b_inhibited.csv`
- `WM793b_necrotic.csv`

These downloaded files are excluded from Git tracking and remain subject to
the terms of their original source repository.

See `NOTICE.md` for complete attribution and licensing information.
