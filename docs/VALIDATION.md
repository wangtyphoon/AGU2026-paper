# Executed local verification and limits

Date: 2026-10-13. This report validates local curation, not a new reproduction of the manuscript results. Original reference files were read; no source checkout was edited.

## Executed checks

| Check | Result and scope |
|---|---|
| Retained-source integrity | 91 source/config records; all original SHA-256 match; all curated-output hashes match. Identical model layers are shared; records remain one per original file |
| Syntax | All package/tools/tests parse with the Python3.10 grammar. Curation was regenerated with Python3.11 to avoid Python3.12-only AST f-string output |
| Imports | 89 package modules import in `atm312`, no unexpected exceptions. Three coarse-model closure modules cannot import because torch-scatter is missing; torch-sparse is also absent. Pangu ONNX module imports in the separate `pangu` environment |
| Command interface | Top-level and all nine subcommand help pages checked without input-data execution |
| Relocation | Copied package/config to a new temporary directory, removed PYTHONPATH, ran CLI help and resolved relative config paths/Stage4 rollout28 update budget correctly |
| Packaging | Built a pure-Python wheel locally with no dependency downloads/installation; data/weights/figures excluded |
| Main test suite | 20 discovered: **19 passed, 1 ONNX test skipped** in `atm312`. Four tests use read-only original-code/paper-input paths; no reference tests skipped when those paths were configured |
| Pangu control test | The skipped ONNX test **passed separately** in Python3.11/Pangu environment using fake model operators. Confirms output leads6..54, global-state retention, regional saves, and 24h anchors from states0/24h; not real pretrained inference |

Tests cover JTWC conversion/deduplication/basin/region/±3-day selection, calendar-year splits, complete six-hour sequences, all four UTC initialization hours, frozen116-channel order, nine-cell masks, latest-state forcing with earlier-state preservation, effective batch arithmetic, Stage1–4 epoch/update budgets, LR tied to optimizer updates, positive-k90% power cutoff, λ<1000km k6 selection, training-only pre-fill ocean-mask selection, source-equivalent spectral losses and true radial cross-spectrum coherence, finite gradients, and a synthetic fine/AR GNN forward/backward.

The common trainer's update-count test actually runs a tiny parameterized model for three optimizer updates with four-way accumulation and checks the saved optimizer state/count. Its fake dataset/objective validates the controller only; it does not simulate the full192×192 atmospheric model.

## Existing external-input regressions

Original and curated numerical inputs were compared using `D:\paper\paper_with_code` and original functions extracted from `D:\RDL_TC_MODEL`:

- Complete normalized RMSE/Bias/ACC/Activity table: exact DataFrame equality.
- Seven-day energy arrays/ratios/totals for four models × three channels ×28 leads: agreement to1e−13, including NaN handling; zero-target bins absent in this available input population.
- TC common-case tables and summary outputs: exact equality; **265/211/162** common cases at24/48/72h, **1,622** paired24h windows.
- True coherence on synthetic paired fields: source/curated agreement to1e−13. Both source MAE and MSE-v2 losses agree on controlled inputs to1e−6 and have finite gradients.

These comparisons use historical aggregate inputs with their original climatology/selection settings. They do **not** validate newly recomputed1990–2017 ACC, expanded initialization populations or regenerated models.

## Representative figure runs

Generated fresh PNG/PDF outputs under the ignored `qa/` directory, using external original inputs rather than copying inputs into this repository:

| Group | Generated output / evidence |
|---|---|
| Metrics | `qa/metrics/james_700q_850v_lines_step28.*`; Figure3 composite |
| Energy | `qa/energy/spectral_energy_summary_7d_700q_850u_850v_lt1000km.*` and numerical CSVs; Figure5 |
| Coherence | `qa/coherence/{076_700q,087_850v}.png`; Figure6 using external original training/test spectra; no invalid points, 24 DC points excluded across both composites |
| TC verification | `qa/tc/tc_verification.*` and summaryCSV; Figure7 |
| TC physics | `qa/physics/tc_axisymmetric_vertical_motion.*`, `tc_wind_pressure.*`, `tc_structure_parameters.*`; Figures8–10; scalar common cases249/199/151 and pressure-group profiles25/60/66 |
| Monthly | `qa/monthly/monthly_spatial_spectrum_{700q,850v_850wind}_2024102100.*`; Figures13–14; maps single initialization, energy six series×60 leads from aggregate population |

Representative metric, energy, coherence700q, verification, vertical-profile and monthly700q images were inspected visually: axes/legends/text are readable and panels/colors render correctly. Other entry points are mapped and import-checked; Figure1/2/4/11/12 have not all been rerendered in this release. Figure2 requires its external manual draw.io asset. Coherence exports use the retained source PNG format.

## Re-run local checks

```powershell
$env:AGU_RDL_REFERENCE='D:\RDL_TC_MODEL'
$env:AGU_PAPER_REFERENCE='D:\paper\paper_with_code'
python -m unittest discover -s tests -v
python tools/verify_release.py
python -m agu_james.workflow check
```

Run `test_pangu.py` in an ONNX Runtime environment. Without the two reference environment variables, the four source/input regression tests skip explicitly. Maintainer regeneration uses Python3.10/3.11: `python tools/curate_sources.py`; then refresh `tools/prepare_manifest.py`. Regular execution does not require author source paths.

Detailed local reports are under ignored `artifacts/verification/{check.json,release_checks.json,tests.log,pangu_tests.log}`. Dependency inventory is an observed local snapshot; it is not an HPC environment lock. The Pangu environment emitted a NumPy/netCDF4 binary-size warning on import, while the fake-operator control test passed; actual NetCDF/weighted inference still requires environment validation.

## Not executed / not established

Full ERA5 archive downloads/conversion/statistics, full graph preparation, all-stage retraining and Stage3-train spectrum calibration, actual GNN forecasts from historical weights, real Pangu6/24h ONNX inference, full field-level TC recomputation, multi-GPU collective-loss equivalence, national HPC execution and manuscript-number reproduction remain pending. Coarse required dependencies are unavailable locally. MAE DDP is explicitly unsupported. Scientific mismatches and anticipated setting-induced changes are in `MODIFICATIONS.md`. GitHub publication, archive deposits, author-owned licenses and DOI assignment were not performed.
