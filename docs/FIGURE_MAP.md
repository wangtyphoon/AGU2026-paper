# Figure-to-workflow map

| Figure | Retained plotting module / CLI | Required upstream inputs |
|---|---|---|
| 1 | `plot graph` | Raw mesh7 graph from graph construction, required graph helpers |
| 2 | Manual draw.io schematic | Editable `model.drawio` and original PNG in the separate plotting-data bundle; no generated-code equivalent |
| 3 | `plot metrics --step 28 --style lines` | Four-model RMSE/Bias/ACC/Activity summaries, ERA5 target activity; 700q and 850v |
| 4 | `plot metrics --step 28 --style tradeoff` | Same aligned summaries; relative RMSE–Activity/ERA5 and ACC–Bias |
| 5 | `plot energy --period 7d` | Aggregate forecast/ERA5 power for 700q, 850u, 850v; exclude DC and use λ<1000 km |
| 6 | `plot coherence --channels 700q 850v` | Real training-set MSE coherence/energy, three test-model coherence/energy, matching features/k/counts |
| 7 | `plot tc` | Four tracking/scalar CSVs; jointly available cases and paired 24-hour windows |
| 8 | `plot physics` | Three GNN axisymmetric w/divergence profile NPZs, tracking/structure CSVs; 72 h, ERA5 pressure groups |
| 9 | `plot physics` | Wind–pressure and 100-km SIM/GWB columns from scalar tracking and GWB diagnostics |
| 10 | `plot physics` | Rmax, B, signed -V_T^L / -V_T^U columns; native 925–600 / 600–300 hPa levels; see METRIC_DEFINITIONS.md#signed-thermal-wind |
| 11 | `plot krathon-track` | Four-model Krathon tracking, 2024 best track, separately sourced IBTrACS USA/JTWC fields |
| 12 | `plot krathon-fields` | 2024092912 map exports at 24/48/72 h; best-track centers and latitude/longitude |
| 13 | `plot monthly` (700q output) | 2024102100 MAE/MSE channel_076 maps at steps 36/72/108, 120-step aggregate energy |
| 14 | `plot monthly` (850v_850wind output) | Same initialization, 850-hPa speed maps; **850v** component aggregate spectra, not spectra inferred from wind-speed maps |


Download `plotting_data.zip` from the associated Zenodo deposit (DOI **PENDING**)
and extract it at the repository root. Its `file_index.csv` maps inputs to
figures. Use software version **0.1.1** and run all commands from the code root:

```bash
python -m agu_james.workflow --config plotting_data/figures.example.json plot graph
python -m agu_james.workflow --config plotting_data/figures.example.json plot metrics --step 28 --style lines
python -m agu_james.workflow --config plotting_data/figures.example.json plot metrics --step 28 --style tradeoff
python -m agu_james.workflow --config plotting_data/figures.example.json plot energy --period 7d
python -m agu_james.workflow --config plotting_data/figures.example.json plot coherence --channels 700q 850v
python -m agu_james.workflow --config plotting_data/figures.example.json plot tc
python -m agu_james.workflow --config plotting_data/figures.example.json plot physics
python -m agu_james.workflow --config plotting_data/figures.example.json plot krathon-track
python -m agu_james.workflow --config plotting_data/figures.example.json plot krathon-fields
python -m agu_james.workflow --config plotting_data/figures.example.json plot monthly --run-tag 2024102100 --days 9 18 27
```

Figure 2 is the supplied `plotting_data/inputs/model/model.drawio` schematic
and PNG; it has no code-generation step. Other outputs go to
`artifacts/figure_reproduction/figures/`. Add `--output-dir PATH` to choose
another destination. The graph command produces Figure 1, physics produces
Figures 8–10, and monthly produces Figures 13–14.

Spectral totals use mode-count-weighted power with Parseval normalization on
the 174×174 FFT domain, nominal wavelengths 4872/k km, and **k=5–87** for the
nominal <1000-km band. See [metric definitions](METRIC_DEFINITIONS.md#spectral-power).
