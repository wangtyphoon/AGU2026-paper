# Executed local verification and limits

Date: 2026-10-13. This report validates local curation, not a new reproduction of the manuscript results. Original reference files were read; no source checkout was edited.

## Executed checks

| Check | Result and scope |
|---|---|
| Retained-source integrity | 81 source/config records after correcting and reducing model closures; all original SHA-256 match; all curated-output hashes match. Identical model layers are shared; records remain one per original file |
| Syntax | All package/tools/tests parse with the Python3.10 grammar. Curation was regenerated with Python3.11 to avoid Python3.12-only AST f-string output |
| Imports | All retained GNN model closures import in `atm312`, with no unexpected exceptions or missing coarse-model extensions. Pangu ONNX module imports in the separate `pangu` environment |
| Command interface | Top-level and all nine subcommand help pages checked without input-data execution |
| Relocation | Copied package/config to a new temporary directory, removed PYTHONPATH, ran CLI help and resolved relative config paths/Stage4 rollout28 update budget correctly |
| Packaging | Built a pure-Python wheel locally with no dependency downloads/installation; data/weights/figures excluded |
| Main test suite | 23 discovered: **22 passed, 1 ONNX test skipped** in `atm312`. Five tests use read-only original-code/paper-input paths; no reference tests skipped when those paths were configured |
| Pangu control test | The skipped ONNX test **passed separately** in Python3.11/Pangu environment using fake model operators. Confirms output leads6..54, global-state retention, regional saves, and 24h anchors from states0/24h; not real pretrained inference |

Tests cover JTWC conversion/deduplication/basin/region/±3-day selection, calendar-year splits, complete six-hour sequences, all four UTC initialization hours, frozen116-channel order, nine-cell masks, latest-state forcing with earlier-state preservation, effective batch arithmetic, Stage1–4 epoch/update budgets, LR tied to optimizer updates, positive-k90% power cutoff, nominal λ<1000km k5 selection, training-only pre-fill ocean-mask selection, source-equivalent spectral losses and true radial cross-spectrum coherence, finite gradients, and synthetic coarse/AR GNN forward/backward, shared Stage2/3 factory class, corrected child-source AST identity and exact original/curated model predictions.

The common trainer's update-count test actually runs a tiny parameterized model for three optimizer updates with four-way accumulation and checks the saved optimizer state/count. Its fake dataset/objective validates the controller only; it does not simulate the full192×192 atmospheric model.

## Existing external-input regressions

Original and curated numerical inputs were compared using `D:\paper\paper_with_code` and original functions extracted from `D:\RDL_TC_MODEL`:

- Complete normalized RMSE/Bias/ACC/Activity table: exact DataFrame equality.
- Spectral correction: 696 model/channel/lead band-power comparisons; annular arrays and individual ratios preserved. Modal-weighted totals deliberately differ from legacy unweighted totals. Parseval/mode counting, synthetic raw analyzer, zero-denominator and anisotropic Cartesian cutoff checks passed. Four updated composite figures inspected. See METRIC_DEFINITIONS.md#spectral-power.
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

Run `test_pangu.py` in an ONNX Runtime environment. Without the two reference environment variables, the four source/input regression tests skip explicitly. The completed one-time source curation utilities have been retired; regular execution does not require author source paths. Use `python tools/verify_release.py` for syntax, CLI and relocation checks.

Detailed local reports are under ignored `artifacts/verification/{check.json,release_checks.json,tests.log,pangu_tests.log}`. Dependency inventory is an observed local snapshot; it is not an HPC environment lock. The Pangu environment emitted a NumPy/netCDF4 binary-size warning on import, while the fake-operator control test passed; actual NetCDF/weighted inference still requires environment validation.

## Not executed / not established

Full ERA5 archive downloads/conversion/statistics, full graph preparation, all-stage retraining and Stage3-train spectrum calibration, actual GNN forecasts from historical weights, real Pangu6/24h ONNX inference, full field-level TC recomputation, multi-GPU collective-loss equivalence, national HPC execution and manuscript-number reproduction remain pending. The former coarse dependency limitation was resolved by selecting the correct GAT/SG model source. MAE DDP is explicitly unsupported. Scientific mismatches and anticipated setting-induced changes are in `MODIFICATIONS.md`. GitHub publication, archive deposits, author-owned licenses and DOI assignment were not performed.

## Fixed single-head follow-up (2026-10-13)

Before/after checks covered eight synthetic coarse/AR combinations: FFN ratios 1/2, checkpoint off/on, two processor layers per level. State-dict keys and shapes matched and strict loading succeeded. Evaluation outputs, seeded training outputs, loss and every gradient were bitwise equal on CPU. Independent original child classes and model roots were also checked for exact prediction/gradient equivalence. The updated suite ran 24 tests: 21 passed, two external paper-input regressions and one ONNX test skipped because their environments were not configured. No full-size atmospheric checkpoint or HPC training run was performed for this change.


## GWB endpoint correction

See [exact-radius recomputation](METRIC_DEFINITIONS.md#exact-radius-gwb) for the strict integral embedded in the original GNN/Pangu modules, analytic tests, completed four-model rerun and output verification policy. The two temporary standalone modules were removed; the external original scripts were also replaced after preserving backups. Full-run completion and counts are recorded in the generated manifest and completion report.


## ACC/Activity corrected-definition verification

The corrected 28-step runs evaluate 487 common initializations (13,636 cases
per model). Independent scalar formulas, staged AGU/local parity, direct
reconstruction from case arrays, finite-case counts, and source fingerprint
checks are recorded in artifacts/acc_activity_corrected. Both pressure-level
and surface Pangu outputs are reconstructed from per-case CSVs. Figure 3-4
use corrected ACC/Activity and target activity, retaining original RMSE/Bias.
Historical aggregate equality checks above refer to the prior definitions.
See METRIC_DEFINITIONS.md#acc-and-activity for the precise corrected equations.

## Native-level thermal-wind verification

All 37,305 tracking rows were replayed for four models, including forecast and
ERA5: 106,586 finite signed thermal-wind diagnostics. The remaining 42,634 values
are missing best-track targets; no finite values were added or lost. Every
non-thermal CSV cell is exactly preserved, including corrected GWB values.
The previous interpolated algorithm is reproduced from the saved native height
ranges with maximum discrepancy below 8e-12 m. New common-case counts remain
249/199/151 at 24/48/72 h. Analytic sign, missing-layer, auxiliary-function
preservation, and GNN/Pangu entrypoint tests pass. AGU and external originals
were replaced only after full rerun and Figure 10 review, with backups, hashes
and installed real-profile parity checks under
`artifacts/thermal_wind_native_levels/code_replacement/`.
See METRIC_DEFINITIONS.md#signed-thermal-wind and the generated completion report for the
paired numerical changes and exact native-layer specification.
