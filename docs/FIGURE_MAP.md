# Manuscript V1 figure-to-workflow map

| Figure | Retained plotting module / CLI | Required upstream inputs |
|---|---|---|
| 1 | `plot graph` | Raw mesh7 graph from graph construction, required graph helpers |
| 2 | Manual draw.io schematic | Original editable `model.drawio` in the author's paper source; no generated-code equivalent or bundled image |
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

The extra IKE–R8 / 500-km distribution plotting script, heatmap-only experiments, old single-variable plotting variants, interactive debugging and profiling are not needed for Figure 1–14 and were not selected as publication entry points. IKE calculation remains because it is described in the methodology.

The corrected 174×174 FFT domain uses nominal wavelength 4872/k km (28 km isotropic approximation). Nominal λ<1000 km selects **k=5–87**. Totals now sum mode-count-weighted power with Parseval normalization; they are scalar band mean-square diagnostics, not domain-integrated kinetic energy. Latitude-dependent zonal spacing is not resolved by this nominal mapping. See METRIC_DEFINITIONS.md#spectral-power.
