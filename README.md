# Evaluation Protocol Matters More Than Model Complexity

Code and data for the paper *"Evaluation Protocol Matters More Than Model Complexity: Lessons from Monthly Streamflow Forecasting in a Snowmelt-Dominated Basin"* (IEEE Al-Khwarizmi 2026, paper UID 6502).

Authors: Rustamjon Nasridinov, Tulkin Delov, Orif Erdonov, Voxidjon Maxmudov
(Tashkent University of Information Technologies named after Muhammad al-Khwarizmi)

## Contents

| File | Description |
|---|---|
| `AmuDarya_v4_anomaly.ipynb` | Full pipeline (Google Colab): data loading, protocols P1–P3, all models, forecasts at h = 1, 3, 6 months |
| `predictions.csv` | Test-period forecasts (93 months, 2010-04 to 2017-12) of all models under protocol P3 |
| `revision_statistics.py` | Skill scores, NSE, KGE, RMSE, block-bootstrap confidence intervals, Diebold–Mariano (HLN-corrected) and Clark–West tests |

## Data

- Discharge: CA-discharge dataset, gauge 16198 (39.69° N, 71.11° E), doi: 10.1038/s41597-023-02474-8
- Meteorological forcing: ERA5-Land monthly means, doi: 10.5194/essd-13-4349-2021

Raw data are not redistributed here; the notebook downloads them from the original sources.

## `predictions.csv` columns

- `observed_Q` – observed monthly discharge (m³/s)
- `climatology` – training-period monthly climatology (m³/s)
- `observed_anomaly` – `observed_Q − climatology`
- `<Model>_anom_h<k>` – forecast anomaly of each model at horizon k; the discharge forecast is `climatology + anomaly`

## Reproducing the revision statistics

```bash
pip install numpy pandas scipy
python revision_statistics.py
```

Random seed is fixed (42); bootstrap uses 12-month blocks and 2000 resamples.

## License

MIT
