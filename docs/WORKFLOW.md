# Ordered execution guide

Every command uses the installed `agu_james` package. Put shared paths in a JSON configuration and pass `--config PATH` **before** the subcommand. Relative paths resolve beside the configuration file. Leave a path unset to use generated artifacts where a default exists; required external assets produce an explicit error. Use `--dry-run` before any stage to inspect its controls without executing downloads or training.

After data/graphs/statistics are available, `python -m tools.run_curriculum --config config.json` prints all Stage1→2→3 segments, training-set spectrum/calibration, both Stage4 branches and three final test forecasts, with checkpoint handoff paths. Add `--execute` to run sequentially on a prepared single-process environment. This example is not an automatic download/HPC launcher; distributed runs use the ordinary stage CLI with the researcher's cluster launcher.

## 1. Data preparation

Inputs: JTWC western North Pacific ATCF `.dat`/`.txt` files or a normalized CSV with `DATE_TIME,CYCLONE_NUM,LAT,LON,VMAX`; ERA5 CDS credentials belong in the user's CDS configuration, never in this repository.

```bash
python -m agu_james.workflow --config config.json prepare best-track --input /external/jtwc
python -m agu_james.workflow --config config.json prepare download
python -m agu_james.workflow --config config.json prepare convert
python -m agu_james.workflow --config config.json prepare statistics
python -m agu_james.workflow --config config.json prepare climatology
```

Order: best-track format/duplicate removal → inside-domain dates ±3 days → three ERA5 request groups → valid-time conversion/field alignment → SST/SWH background and nearest-neighbor fill → calendar-year split → six-hour local time features → training-only feature statistics and normalized increment scales → WeatherBench 1990–2017 climatology crop.

Outputs: `data/{train,val,test}/{prs,sfc/regular,sfc/average,time}/YYYYMMDDHH.npz`, `data/{lat,lon}.npy`, training-only pre-fill SST/SWH masks, best-track/selection manifests, `statistics/` and `climatology/`. Statistics are generated rather than bundled. Input archives remain untouched. SST/SWH physical statistics exclude masked background cells; normalized increment scales retain the source all-cell convention. Existing filled NPZs require explicit external `sst_mask`/`swh_mask` paths. Use historical masks and normalization statistics when loading historical checkpoints (see DATA_SOURCES and MODIFICATIONS).

For training, a sequence must include the previous analysis at t−6 h and **all** future frames through t+rollout×6 h in the same split. Calendar boundary sequences are excluded rather than assigned to a different year or filled with another split.

## 2. Graph preparation

```bash
python -m agu_james.workflow --config config.json graph --mesh 6
python -m agu_james.workflow --config config.json graph --mesh 7
```

Order: heterogeneous grid/mesh connectivity → edge geometry → node features → positional/structural encoding → self-loops → feature normalization. Outputs are under `graphs/mesh6/mesh_grid_data_unet/` and `graphs/mesh7/mesh_grid_data_unet7/`. Stage 1 uses mesh 3–6; later stages use mesh 3–7. A mesh7 normalization-directory typo is corrected, and explicit `weights_only=False` supports generated PyG graph serialization on recent PyTorch.

## 3. Stage 1, transfer and Stage 2

```bash
python -m agu_james.workflow --config config.json train --stage 1 --rollout 1 --micro-batch 1
python -m agu_james.workflow --config config.json train --stage 2 --rollout 1 --micro-batch 1 --checkpoint artifacts/training/stage1_mse_step01/last_model.pth
```

Stage 1: 150 epochs, effective batch 16, peak LR 1e−3. Stage 2: 30 epochs, effective batch 16, peak LR 1e−4. Stage1 uses the corrected ver_unet_scale model at levels3–6; Stage2–4 use the AR model at levels3–7 with shared child layers. Both single-step stages use MSE and one-epoch warm-up. The transfer copies same-name/same-shape tensors, then initializes level-7 counterparts from level-6 prefixes using the original transfer rules. Keys are recorded in the checkpoint manifest. Stages2–4 select the same AR model class. Synthetic source/curated predictions match exactly for the two level variants.

## 4. Stage 3 curriculum

Run rollout **2, 4, 8, 12, 20, 28** in that order, passing each preceding checkpoint via `--checkpoint`. Every new curriculum segment starts its own optimizer/schedule; use `--resume` only for a compatible checkpoint from the same segment.

| Rollout | Effective batch | Epochs / optimizer updates | Peak LR |
|---|---:|---|---:|
| 2 | 16 | 4 epochs | 1e−5 |
| 4 | 16 | 2 epochs | 1e−5 |
| 8 | 16 | 1 epoch | 1e−5 |
| 12 | 16 | 1 epoch | 1e−5 |
| 20 | 8 | 1000 optimizer updates | 1e−6 |
| 28 | 8 | 1000 optimizer updates | 1e−6 |

```bash
python -m agu_james.workflow --config config.json train --stage 3 --rollout 2 --checkpoint artifacts/training/stage2_mse_step01/last_model.pth
```

The source autoregressive feature assembly is retained: two preceding states, forcing/static/time features, increment prediction, and prescribed boundary replacement on only the most recent forecast before the next model call. Nine edge cells are excluded from supervision. All rollout steps have equal weight.

## 5. Training-set spectrum and cutoff

```bash
python -m agu_james.workflow --config config.json validate --split train --rollout 28 --checkpoint artifacts/training/stage3_mse_step28/last_model.pth --spectrum-output artifacts/calibration/mse_train_coherence_energy_rollout28.npz
python -m agu_james.workflow --config config.json calibrate --input artifacts/calibration/mse_train_coherence_energy_rollout28.npz
```

Calibration accepts only training-set metadata with matching forecast/target counts and frozen feature order. Outputs: `calibration/cutoffs/cutoff_step01.csv` through step28. Select the smallest positive radial k where forecast power ≤0.9×target power. NaN indicates no crossing; retained loss loaders use their explicit cap/fallback (MAE 46, MSE 28). No test-set spectra calibrate training.

## 6. Stage 4, two branches

Both branches begin from the Stage-3 checkpoint and follow **1, 2, 4, 8, 12, 20, 28**, continuing within their own branch.

| Rollout | Effective batch | Epochs / optimizer updates | Peak LR |
|---|---:|---|---:|
| 1 | 16 | 8 epochs | 1e−5 |
| 2 | 16 | 4 epochs | 1e−5 |
| 4 | 16 | 2 epochs | 1e−5 |
| 8 | 16 | 1 epoch | 1e−5 |
| 12 | 16 | 1 epoch | 1e−5 |
| 20 | 8 | 1000 optimizer updates | 1e−6 |
| 28 | 8 | 1000 optimizer updates | 1e−6 |

```bash
python -m agu_james.workflow --config config.json train --stage 4 --rollout 1 --loss mae --checkpoint artifacts/training/stage3_mse_step28/last_model.pth
python -m agu_james.workflow --config config.json train --stage 4 --rollout 1 --loss spectral_mse --checkpoint artifacts/training/stage3_mse_step28/last_model.pth
```

Optimizer: AdamW, betas (0.9,0.95), decay 0.1. All stages retain the corrected coarse/AR zero-decay bias/normalization partition. Warm-up is 10% of update budget, followed by cosine decay. Source coefficients (filtered MSE, spectrum, bias) are MAE (1,1,2), MSE (1,8,1). Stage 4 uses uniform channel weights, as the source entry points pass `channel_weights=None`; Stage 1–3 use frozen channel-weight configurations. Alpha=1, order=12 and spectral bounds=87 follow the local spectral entry points, configurable in `training`; they are not independently established as national HPC settings. Helper-level validation defaults of 56 are not the effective entry-point bounds.

Effective batch = micro-batch × world size × accumulation. Incomplete effective batches at epoch end are dropped. Optimizer updates, LR and checkpoint history share one counter. The MSE distributed collective loss is retained; real distributed execution is not validated. MAE uses one GPU with accumulation because exact global batch-bias behavior under DDP is not yet established; requesting MAE with `WORLD_SIZE>1` fails clearly.

## 7. Test forecasts and Pangu

```bash
python -m agu_james.workflow --config config.json validate --split test --rollout 28 --checkpoint /external/mse.pth
python -m agu_james.workflow --config config.json validate --split test --rollout 120 --initial-time 2024102100 --checkpoint /external/mae.pth
python -m agu_james.workflow --config config.json prepare pangu-inputs --initial-times 2024092912 --rollout 28
python -m agu_james.workflow --config config.json pangu --initial-times 2024092912 --rollout 28 --weights-dir /external/pangu-models
```

Repeat GNN validation for three final checkpoints. Default initialization hours are 00/06/12/18 UTC. `--initial-time` filters a complete sequence for Krathon (`2024092912`) or monthly (`2024102100`) experiments. Monthly integration needs separately available continuous 120-step ERA5 sequences; TC-window sampling alone may not contain them. Do not invent missing frames.

Pangu preparation downloads global ERA5 analyses for initial and future target times, preserving field/level ordering. Inference evolves full global fields, anchors every 24 hours with the 24-h model, uses the 6-h model between anchors, and saves every 6-h lead before regional cropping. Supply external 6/24-h ONNX weights; their binary files are not in Git.

## 8. Analysis and map export

```bash
python -m agu_james.workflow --config config.json analyze rmse --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze acc_activity --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze target_activity --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze energy --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze coherence --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze tc --model mse --forecast-dir /external/forecasts --target-dir /external/targets --rollout 28
python -m agu_james.workflow --config config.json analyze rmse --model pangu --rollout 28
python -m agu_james.workflow --config config.json analyze tc --model pangu --rollout 28
python -m agu_james.workflow --config config.json analyze export-gnn --model mae --forecast-dir /external/case-forecasts --target-dir /external/targets --initial-time 2024092912 --steps 4 8 12
python -m agu_james.workflow --config config.json analyze export-pangu --model pangu --initial-time 2024092912 --steps 4 8 12
```

Repeat `rmse`, `acc_activity`, `energy` and `tc` for all models; `coherence` applies to GNNs, and ERA5 target activity needs one compatible target population. Monthly aggregate energy is a separate 120-step GNN analysis. Scalar tracking → axisymmetric wind → GWB → B/VT → IKE → vertical-motion/divergence profiles precede composite figures. A generated tracking CSV is augmented in the output directory; source CSVs are never modified. Provide `lsm`, `surface_nc`, `upper_nc` and a normalized best track.

Outputs use the paper input layout under `paper_inputs/{rmse_multi,acc_activity_multi,spectrum_multi,tracker_multi,struc_multi,vis_multi_steps}`. The original case plot expects the 2024 best track beside `tracker_multi`; copy the generated `data/cyclone_data_2024.csv` there or provide the existing normalized paper input tree via the plotting CLI. Map exports preserve `channel_076` (700q), `wind_speed_086_087` (850-hPa speed) and source Pangu variable names.

## 9. Figures

```bash
python -m agu_james.workflow --config config.json plot metrics --step 28 --style lines
python -m agu_james.workflow --config config.json plot metrics --step 28 --style tradeoff
python -m agu_james.workflow --config config.json plot energy --period 7d
python -m agu_james.workflow --config config.json plot coherence --channels 700q 850v --mse-train-npz /external/mse_train_coherence.npz
python -m agu_james.workflow --config config.json plot tc
python -m agu_james.workflow --config config.json plot physics
python -m agu_james.workflow --config config.json plot krathon-track
python -m agu_james.workflow --config config.json plot krathon-fields
python -m agu_james.workflow --config config.json plot monthly --run-tag 2024102100 --days 9 18 27
python -m agu_james.workflow --config config.json plot graph
```

For existing aggregate inputs, add `--input-root /external/paper-input-tree --output-dir /new/figures`. Outputs are PNG/PDF and relevant CSV/manifests. `Figure 2` has no code-generation stage: it is the author's manual draw.io model schematic. Figure 13/14 maps are one initialization; spectra are aggregate populations. They must not be relabeled as spectra of that single case.
