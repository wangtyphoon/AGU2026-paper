# Curation and scientific change audit

This is a **paper-setting implementation variant**. It preserves the source checkouts and excludes source data, weights, figures, caches, profiling, editor settings and unrelated experiments. Source provenance is recorded in `provenance/sources.json`; category-level old/new/source/reason records are in `provenance/changes.json`. The original manuscript is a scientific reference, not executable instructions.

## Path and package organization

| Original | Curated value / behavior | Source and reason |
|---|---|---|
| Loose `train_script`, `layer`, plotting sibling imports | Qualified `agu_james.*` modules, one CLI | Selected source closures in provenance; imports remain usable after relocation |
| C:/D: paths, cwd-relative statistics/graphs | JSON `paths`, CLI inputs and ignored artifact root | Source assignments and common adapters; measured assets remain external |
| Top-level downloads/graph jobs/NumPy loads | Explicit `main()` / `initialize()` | Importing modules must not download, train or mutate data |
| Repeated layer files | Shared only where original parsed AST is identical | Original byte and AST hashes retained; distinct model roots/losses retained |
| Full original trainers with history plotting/early stop/AMP setup | One paper curriculum controller around retained models, feature assembly and losses | Original entry/control scripts remain reference sources; controller is newly authored and requires HPC comparison |
| Bulk in-place NPZ repair/demo helpers | Only pure nearest-fill / wave resize helper retained | `npz/fill_nan_nearest.py`, `resize_swh.py`; conversion writes new artifact inputs |

## Paper setting synchronization

| Original local setting | Curated setting | Source / reason and expected effect |
|---|---|---|
| Coarse config 1 epoch, microbatch 4×accumulation 4, LR1e−3 | Stage 1 150 epochs, effective16, LR1e−3 | `ver_unet_scale/train_script/auto_config.py`; manuscript Table 3; materially different training history |
| Fine config 31 epochs, microbatch1×accumulation4, LR1e−3 | Stage 2 30 epochs, effective16, LR1e−4 | `ver_unet7/train_script/auto_config.py`; Table 3 |
| AR config 31 epochs, LR1e−3, one local rollout setup | Stage 3 curriculum 2/4/8/12/20/28 and paper LR/budgets | `ver_autoregressive` entry/config; approved plan; last two limits count optimizer updates |
| Spectrum-v2 config 8 epochs, microbatch1×accumulation1, LR1e−5 | Stage 4 separate MAE/MSE curricula, effective16 through12 then8 | Local spectral entries; Table 3 plus user's Stage-4 rollout12 effective16 clarification |
| Corrected coarse source weighted MSE objective | Stage1 weighted MSE retained | `ver_unet_scale/train_script/training.py`; source and paper use MSE |
| Source warmup LR1e−8 with entry-specific schedulers | One-epoch warmup in Stages1/2; 10% updates in3/4; cosine to1% peak | Approved schedule; update counter drives LR and checkpoint. This is a new controller, not byte-equivalent scheduler behavior |
| Corrected coarse and AR source bias/norm exemptions | Preserved for all stages, betas(.9,.95), decay.1 | Actual corrected source optimizer grouping retained |
| Local dataset initialization filter 00/12 UTC | 00/06/12/18 UTC with complete same-split sequences | `datasetM.py`; paper population; counts can increase; no crossing of train/val/test |
| Historical folder assignment / continuity only | Calendar1979–2023 excluding five val years, test2024 | Manuscript data section and approved plan; cross-set sequences removed |
| Loss mask `distance <=8`; validation crop `distance <boundary` | Explicit width9; metric adapter retains8 for inclusive source mask and9 for crop | Actual training mask unchanged where <=8 already meant9 cells; inconsistent forcing/validation margins corrected |
| Source forcing routines with different mask conventions | First call unchanged; replace only newest forecast boundary before call2+ | Approved plan; earlier state preserved; source first-call treatment retained |
| Climatology URL1990–2019 | Official1990–2017 dataset | `output/download_clim.py`; manuscript reference period; ACC/Activity changes expected when recomputed |
| Annular means summed without mode counts; 192-cell wavelength length | Mode-count-weighted Parseval band mean square; nominal 174×28/k km and k5–87 | Approved spectral correction; not exact geographic wavelength or kinetic energy |

## Required fixes and source preservation

| Original value / issue | Curated value | Source / reason |
|---|---|---|
| mesh7 normalization writes mesh6 `norm_stats/` | mesh7 directory | `pre_norm6_unet7.py`; prevents wrong graph asset locations |
| Implicit PyTorch graph deserialization | Explicit `weights_only=False` for generated PyG graphs | Graph preparation; recent serialization behavior requires graph objects. Use only researcher-controlled assets |
| Non-Windows Pangu `_normalize_path` returns None | `os.fspath` + `normpath` | Upstream `func_inference.py`; portable string/Path support |
| Original static ocean masks selected using snapshot-specific NaN counts | Preserve pre-fill training reference; choose maximal missing count; explicit external historical-mask override | `create_weight_mask.py`, `stat_all_variables.py`; avoid statistics over artificial land backgrounds. Generated masks may differ from historical reference |
| Original source statistics contain optional distribution moments | Only required mean/std/min/max/count plus ordered increment scales | Dataset/loss consumers; exclude unnecessary full-array moment calculations |
| Original entry-point Stage4 `channel_weights=None` | Uniform Stage4 channel weights | Spectral entries; fixed nonuniform weights remain in Stage1–3 |
| Spectral entry bounds87; helper validation defaults56 | Effective entry bounds87 for train/validation | Actual `TrainingConfig` overrides helper defaults. Alpha1/order12, MAE cap46/MSE cap28 and branch coefficients preserved as local reference settings |
| True radial cross-spectrum coherence | Formula preserved, tested against source | `val_autoregressive_with_spectrum.py`; no energy-ratio/SNR substitution |
| Figure2 schematic | Manual draw.io dependency documented | No fabricated code equivalent; original editable asset is external |

## User-supplied Pangu integration

The additional source is `C:\Users\王子祐\Documents\Pangu-Weather-ReadyToGo`. Its inference functions, regional subset utility, pressure/surface metrics and spectra, TC tracker and wind/GWB/B–VT/IKE diagnostics are retained with notices. Global state evolution, regional outputs and 24-hour anchors with intervening 6-hour steps are preserved. The wrapper changes `save_after_hours=72` to **0**, and extends 24-hour anchor times across the requested rollout, so leads6/12/18 and subsequent six-hour outputs exist. This is a deliberate output-coverage change. No ONNX model binaries were copied or downloaded.

## Unresolved experiment-to-manuscript discrepancies

**MAE bias:** the source `spectrum_loss.py` uses the spectral energy of the batch-mean forecast error without target-power normalization; the manuscript Eq.7 and MSE-v2 branch use normalized bias. The MAE source formula is retained and regression-tested. Normalizing it now would silently create a different trained method. Establish the actual national HPC implementation and either correct the equation/text or introduce a separately labeled model variant before claiming reproduction.

**Corrected model source:** the initial curation mistakenly selected ver_unetG/model_unetT.py. The author identified ver_unet_scale/model_unet.py as the correct coarse source. It now uses the same GAT/SG child modules as ver_autoregressive/model_unet.py; the model-root difference is the level lists and their associated default per-level depths. Stages2–4 select the AR root directly. Original child ASTs are identical and source/curated synthetic predictions match exactly. The old block-attention closure and its dependency requirement are removed. See MODEL_ARCHITECTURE.md for explicit trainer hyperparameter differences.

**Execution controls:** the common trainer uses float32, successful optimizer-update limits, drops incomplete effective batches, and runs the prescribed full budget without source early stopping. Checkpoint compatibility is checked at the model key/shape level and same-segment resume validates loss/settings. This does not establish bitwise historical equivalence, matching stochastic order/AMP, or reproducibility of the historical HPC optimizer state.

**Distributed batch bias:** original MSE-v2 distributed collectives are retained but untested on multiple GPUs. MAE distributed execution is rejected until its batch-mean spectral-bias semantics are established. Single-GPU accumulation computes the source spectral loss per microbatch, as original accumulation did; it does not turn the batch-bias term into a single full-effective-batch loss. Effective sample count and bias-statistic batch size are distinct and must be recorded for HPC comparison.

**Results status:** spectral totals and cumulative products deliberately differ from the legacy implementation after the approved modal-power correction; individual annular ratios are preserved. Other aggregate-input regressions retain their stated scope. New split/hour/climatology/forcing/training settings can change newly generated results. No corrected paper numbers, full retraining, national HPC run, GitHub publication or archive DOI have been produced.

**Fixed single-head API (2026-10-13):** model roots and shared layers no longer accept `heads` or `num_heads`. Widths use the former `heads=1` values; singleton attention axes and operation order are preserved. The factory specification omits the head count. Existing single-head state dictionaries load strictly; callers must omit the removed arguments.


## GWB endpoint correction

See [exact-radius recomputation](METRIC_DEFINITIONS.md#exact-radius-gwb) for the strict integral embedded in the original GNN/Pangu modules, analytic tests, completed four-model rerun and output verification policy. The two temporary standalone modules were removed; the external original scripts were also replaced after preserving backups. Full-run completion and counts are recorded in the generated manifest and completion report.


## ACC/Activity formula correction

ACC removes only the climatological mean. Forecast and ERA5 activity both use
sqrt(mean case centered spatial variance). Explicit batch-case handling,
finite-case counts, input fingerprints, case audit arrays, and direct summary
reconstruction checks accompany the corrected calculations. The AGU analysis
adapter also aligns Spectrum MSE CSV naming with the existing plotting loader
(`spectrum_ver2`). Historical aggregate equality checks in VALIDATION.md refer
to the previous metric definitions and must remain labeled historical.

## Signed native-level thermal wind

Thermal-wind regressions now use only native 925/850/700/600 and
600/500/400/300 hPa height ranges, with signed -V_T^L / -V_T^U labels.
Delta Z_core is explicitly distinguished from radial Delta Z_sim. CSV column
names remain compatible; counts and method metadata are written beside phase
entrypoint outputs. AGU and source-local analysis/plotting programs were
replaced after complete four-model replay and QA. Recomputed CSVs and Figure 10
are in `artifacts/thermal_wind_native_levels/`; source CSVs remain preserved.
See METRIC_DEFINITIONS.md#signed-thermal-wind.


## Workspace cleanup (2026-10-14)

Four correction/definition documents are consolidated into METRIC_DEFINITIONS.md. Five completed one-time curation/staging/installation utilities are retired. Active workflows, numerical tests, replay/report tools, and research outputs are retained. Original documents, retired tools, and pre-cleanup metadata are recoverable from `artifacts/cleanup/before_cleanup.zip`. Build products and Python caches are regenerable.
