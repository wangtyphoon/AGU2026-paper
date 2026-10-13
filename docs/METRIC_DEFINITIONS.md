# Metric definitions and numerical corrections

Definitions, replay evidence, and limitations for the retained paper workflows.

- [ACC and Activity](#acc-and-activity)
- [Exact-radius GWB](#exact-radius-gwb)
- [Spectral power](#spectral-power)
- [Signed thermal wind](#signed-thermal-wind)

## ACC and Activity

For case j and grid point i, a_ji = forecast_ji - climatology_ji and
b_ji = ERA5_ji - climatology_ji. Spatial weights w_i retain the existing cosine
latitude weighting and nine-grid-point boundary exclusion.

ACC_j = sum_i(w_i a_ji b_ji) / sqrt(sum_i(w_i a_ji^2) sum_i(w_i b_ji^2)).
ACC is the arithmetic mean of finite ACC_j values at each feature and lead time.
There is no extra case spatial demeaning in ACC.

v_j = sum_i[w_i (a_ji - weighted_mean_i(a_ji))^2] / sum_i(w_i).
Activity = sqrt(mean_j(v_j)). ERA5 target activity uses b_ji in the same formula.
Activity/ERA5 is the ratio of these two aggregated activities, rather than a
mean of per-case ratios. Overall channel activity uses the same variance-first
aggregation across all processed cases and lead times.

Each batch member is a separate case. Zero-weight cells are excluded before
arithmetic; nonfinite values on positive-weight support invalidate the affected
diagnostic. A zero ACC denominator gives NaN; zero spatial activity is valid.
Finite-case counts are recorded separately for each feature, lead, and metric.
Per-file channel means are diagnostic fields, not case aggregation formulas.

The 28-step recalculation preserves input rosters, climatology, reference-time
mapping, RMSE/Bias inputs, and masks. Twelve-step files are not replaced.

The corrected 28-step programs, results, Figure 3-4 inputs and figures were
installed after explicit user confirmation on 2026-10-14. Backups and
post-installation verification are recorded in artifacts/acc_activity_corrected.

## Exact-radius GWB

The original 25-km ring-center trapezoid integrates 12.5–112.5 km for the
100-km diagnostic, and 12.5–512.5 km for the 500-km diagnostic. This differs
from the equation's interval [0, R]. For solid rotation the former 100-km
result is 1.25 times the analytic integral; this is not a universal correction
factor for real storms.

The existing GNN and Pangu calculation modules each embed `integrate_gwb`,
which uses SI units, assumes a
regular vortex with Vtheta=O(r), sets the origin integrand to zero, linearly
interpolates the integrand at exactly R, and integrates over [0, R]. Ring
means and the original center, tangential projection and Coriolis definitions
are retained. The upper ring supplies interpolation support only. Missing
first/interior/endpoint-support rings invalidate that radius; gaps are not
bridged. This changes both quadrature endpoints and missing-data eligibility.
The regular-origin assumption is a numerical boundary condition, not a new
observation of the unresolved storm core.

The correction is embedded directly in these existing AGU modules:

- `agu_james.analysis.gnn.tc_gwb`
- `agu_james.analysis.pangu.axisymmetric_gwb_metrics_all`

The two temporary modules `gwb_integral.py` and `recompute_gwb.py` were removed
at the user's request after completing the four-model rerun. The same corrected
quadrature was installed directly into the original scripts in both GNN
checkouts and Pangu validation. Their original contents, replacements, SHA-256
hashes and verification are retained under
`artifacts/gwb_endpoint_corrected/code_replacement`. Existing script input/output
settings are retained; running those legacy entry points still writes to their
configured output CSVs. No rerun was performed during this code replacement.

Run the numerical checks:

```powershell
& 'C:\miniconda\envs\pangu\python.exe' -B -m unittest discover -s tests -p test_gwb.py -v
```

The completed rerun covered AUTO MSE, Spectrum MAE, Spectrum V2 MSE and Pangu,
steps 1–28, forecast and target, 100 and 500 km. `configs/gwb.local.json` is
retained as an input-path record for that completed run, rather than an active
standalone runner configuration. Outputs live in `artifacts/gwb_endpoint_corrected`:

- `tracker_multi`: corrected CSVs with original columns and order.
- Per-model case changes, missing reasons, source audits, and manifests.
- `paired_change_summary.csv`: old/new height and residual distributions on
  identical finite rows; separate old/new/lost counts.
- `paired_error_summary.csv`: forecast-minus-own-target GWB and residual
  bias, RMSE, and error quantiles on identical valid pairs.
- `lead_paired_distribution.csv`: paired 24/48/72-h height/residual quantiles.
- `output_verification.csv`: exact non-GWB text, all rows/reasons, read-back
  and original-input fingerprint checks.
- `lead_paired_metrics.csv`: 24/48/72-h old/new scores on the same four-model
  all-scalar-parameter intersection, with shared MSE ERA5 reference.
- `target_agreement.csv`, `figure_sample_audit.csv`, and Figure 9 PNG/PDF,
  relationship statistics and case keys under `figure9`.

The existing geographic bounds, scalar eligibility, 72-h plotting lead,
KDE and Figure 9 layout are retained. Original per-model target discrepancies
are checked before replacing targets with shared MSE ERA5 values in figures.
SIM and all other source columns are retained rather than recalculated.

CSV, grid, channel-table and calculation-code fingerprints use SHA-256.
Large meteorological input files use absolute path, size and modification
time, verified unchanged after processing; these are metadata fingerprints,
not full content hashes. Read-back checks verify all rows and all non-GWB
columns. A `complete` manifest confirms computation and report generation;
visual artifact QA is recorded separately in the completion report.

Analytic checks cover solid rotation, zero wind, exact and arbitrary endpoints,
required and irrelevant missing rings, unsupported endpoints, a nonlinear
Vtheta=a*r^2 convergence example, and the two original calculation modules.

## Spectral power

本版選擇「模態加權的尺度帶功率診斷＋明確標示近似尺度」：修正總量計算，保留可由現有徑向 NPZ 重建的資訊，不宣稱已得到精確投影後的中尺度物理能量。使用者於 2026-10-14 批准替換正式程式、已知既有光譜 NPZ 及目前圖／CSV。安裝紀錄見 artifacts/spectral_power_review/install_log.json。

### 定義

每個 annulus 維持原始平均功率，個別 k 的 forecast/ERA5 比值不變。新 band mean square 為

`V_B = sum_{k in B}(N_k * mean(|FFT(field)|^2)_k) / (H*W)^2`。

`N_k` 以實際 174×174 FFT 網格與原始 `floor(hypot(integer FFT indices))` 分箱直接計數；並非 2πk 近似，也不是樣本數。NumPy 預設 forward FFT 未正規化，因此除以 `(H*W)^2`；與全模態加總的 Parseval mean square 相容。參考：https://numpy.org/doc/stable/reference/routines.fft.html 。

採用非 DC 的選定模態時，此量是對應帶通場的格點平均平方值／變異量。它不是區域積分能量；700q 單位為 `(kg/kg)^2`，單獨 850u／850v 為 `(m/s)^2`，不稱為動能。若要水平動能需另外對同一模態集合加總 u、v 並乘 1/2。既有來源的 sample population 與各模型自有 ERA5 分母保持原樣，未新增共同樣本交集。

保留環帶 `k=1..87`，也就是 `1 <= floor(r) <= 87`；FFT 方形角落的模態未被收錄，不能稱為整個 FFT 的總量。分析函式 cumulative products 現改以模態數加權，包含 DC 的 cumulative quantity 仍為 mean-square diagnostic，而不是零均值變異量。新 NPZ 添加形狀、模態數、功率／幅度、正規化與去均值／面積權重 metadata。振幅不能用於本功率彙總。零分母標為缺值；避免 Parseval 正規化後仍加固定 epsilon 造成相對偏差。

### 尺度與分界

尺度軸採 `nominal_lambda = 174*28/k = 4872/k km`，明確標示 **Nominal wavelength (km; 28 km grid)**。這是原 index FFT 的各向同性名義尺度，沒有採用代表緯度，也没有重新投影，不能當成精確 geographic physical wavelength。`k` 是 `[k,k+1)` 的環帶下緣；同一環帶內模態波長不同。用整個環帶的下緣檢查名義 `<1000 km`，選 `k=5..87`；`k=4` 跨越邊界的部分模態無法從已平均 NPZ 分離。本版不把這些模式假裝成精確 1000 km 切分。

`aggregation_sensitivity.csv` 分列 k=6 舊分界、k=5 新分界及 900／1100 km 名義分界，覆蓋 7 天四模型三通道×28 leads、30 天兩模型三通道×60 leads，共 696 列。900 km 對應 k=6；1100 km 仍對應 k=5，兩者受整數環帶解析度限制。

MSE／700q／168 h：

| 定義 | Forecast / ERA5 |
|---|---:|
| 舊版 k=6..87，環帶平均直接加總 | 0.5963105599 |
| 固定 k=6..87，模態加權 | 0.5665559556 |
| 修正名義尺度 k=5..87，模態加權 | 0.6087482664 |

不能把上述第三列解讀為精確地理波長 <1000 km 的物理能量保留率。若要求精確物理尺度，需要從每個原始 forecast/ERA5 場重新投影／重採樣並重算全部模態。候選 `spectral.py` 提供 `physical_band_power`，要求呼叫端提供均勻 Cartesian grid，以每個模態的 `fftfreq(n,d)` 徑向頻率切分；已用等向／非等向正弦場驗證，但沒有執行整套地理重新投影及全樣本重算。

### 驗證與產物

執行命令（在 D:\paper\AGU）：

```powershell
& 'C:\miniconda\envs\atm312\python.exe' artifacts\spectral_power_review\build_candidate.py
& 'C:\miniconda\envs\atm312\python.exe' artifacts\spectral_power_review\run_review.py
```

`atm312` 環境完成全套檢查；`pangu` 環境在 NumPy／PyTorch OpenMP runtime 衝突時停止，未以允許重複 runtime 的方式繼續。

已通過：全 FFT Parseval、環帶模態計數對直接 FFT 模態求和、1000 km 正弦場切分、dx≠dy 方向敏感性、零分母、合成檔案的多製程 aggregate/per-step NPZ 與 metadata/cumulative ratio、AGU／standalone 計算一致、全部 candidate syntax compilation、原始來源雜湊保護。合成 raw analyzer 是端到端驗證，不代表所有歷史 raw forecast 重算。歷史實際資料的 696 列彙總與四張圖則已完整重算。

`results/` 包含四張 PNG／PDF、spectral/totals CSV、月模擬 map RMSE CSV／manifest、`aggregation_sensitivity.csv`、`validation.json`。視覺檢查覆蓋四張 PNG；圖中 spectra 仍是 aggregate population，月模擬地圖仍是單一起報案例。

### 已批准替換範圍

只依 `replacement_manifest.json` 的逐檔來源／候選映射替換；**不要把整個 candidate/agu_james 覆蓋回去**，其中含複製用來執行的其他現有檔案。

AGU 候選修改五個現有模組：`analysis/gnn/energy.py`、`analysis/pangu/spectrum_multistep_prs.py`、`analysis/pangu/spectrum_multistep_sfc.py`、`plotting/energy.py`、`plotting/monthly.py`；新增 `analysis/spectral.py`。

本地候選修改：`paper_with_code/spectrum analysis/plot_spectral_energy_summary.py`、`paper_with_code/case_study and monthly simulation/plot_monthly_spatial_spectrum.py`，以及 `RDL_TC_MODEL/gnn/{ver_autoregressive,ver_spectrum_v2,ver_unet7}/output/spectrum_all_stat.py`。各 standalone 來源所在目錄需同步新增 `spectral_contract.py`。本地 Pangu 原程式尚無候選替換，AGU Pangu 已有 metadata 修正版。

已備份並替換上述來源、已知既有 NPZ 與指定正式圖，同步 AGU docs／provenance／k=5 contract test。NPZ 的 mean/std/SNR 陣列逐元素保留，cumulative 欄位已從既有 annular means 重建為模態加權定義。完整回復用原檔與逐檔 hash 列於 install_log.json；這次 migration 不代表重算 raw forecast 或重新投影。

### 建議 caption 文字

Figure 5: “Annular-mean Fourier power ratios and mode-count-weighted band-power ratios relative to ERA5 for 700-hPa specific humidity and the 850-hPa wind components. Nominal wavelengths are defined as 4872/k km using the 174×174 FFT domain and an isotropic 28-km grid approximation. Band-power ratios are the ratio of sums of N_k P̄_k over k=5–87, excluding DC. These scalar mean-square diagnostics do not represent domain-integrated kinetic energy; the nominal scale mapping does not account for latitude-dependent zonal spacing.”

Figures 13–14: “Maps show the single initialization at 0000 UTC 21 October 2024; spectral panels show aggregate forecast populations. Annular and band-power ratios use the same definitions as Figure 5. The wind-speed maps in Figure 14 are accompanied by spectra of the 850-hPa meridional wind component.”


### 安裝資料政策

已知 paper spectrum_multi NPZ 的原始 mean/std/SNR 陣列逐元素保留；cumulative products 從環帶平均重建為模態加權版本，另新增 band mean-square 與 metadata。研究 output/spectrum 中與這些 published input 位元組相同的 NPZ 同步替換。其他來源／振幅／訓練校正資料不猜測其幾何或正規化，不在本次 migration 之列。所有被替換原檔集中備份，包含來源雜湊。歷史 lt1200km 或無尺度後綴的探索圖保留原名，正式更新的現行圖為 lt1000km 後綴及指定月模擬 composites。

## Signed thermal wind

The stored `forecast_vtl`, `target_vtl`, `forecast_vtu`, and `target_vtu` columns
represent signed **-V_T^L** and **-V_T^U**, in metres. Positive values indicate a
warm core, and negative values a cold core. No absolute value or extra sign flip
is applied to the regression coefficient.

At each native pressure level, define

\[
\Delta Z_{\mathrm{core}}(p)=\max_{d\le500\,\mathrm{km}}Z(p)
                              -\min_{d\le500\,\mathrm{km}}Z(p).
\]

Use equal-weight ordinary least squares of this range against `ln(p / 1 hPa)`.
The lower layer is exactly `(925, 850, 700, 600)` hPa and the upper layer exactly
`(600, 500, 400, 300)` hPa. No interpolated pressure samples are included. At least
two finite native-layer values are required; missing layers are excluded without
filling or interpolation. Forecast centres and best-track target centres follow
the existing respective workflows. Geopotential is converted to height using
`g = 9.80665 m s^-2` before spatial extrema are taken.

This differs from the radial diagnostic in Eq. (15),
`Delta Z_sim(R) = Z(R) - Z(0)`, which is unchanged. The 925–600 hPa lower layer is
this study's choice; Hart's author help page describes 900–600 hPa:
<https://moe.met.fsu.edu/cyclonephase/help.html>.

Legacy tracking CSV names and columns remain compatible. Running a phase-analysis
entrypoint additionally writes `.thermal_wind_counts.csv` and
`.thermal_wind_method.json` beside its output CSV. Counts are indexed by output
row; zero counts also identify cases where loading or centre availability
prevented diagnosis. Full replay audits additionally identify individual reasons
and store all seven native height ranges.

The complete replay is under `artifacts/thermal_wind_native_levels/`. It starts
from the preceding GWB-corrected CSVs, replaces only the four thermal-wind cells,
and verifies all other CSV cells exactly. The reproduction report checks the old
900–600 and 600–300 hPa interpolated regressions against the original columns.
`paired_change_summary.csv` uses each model's finite old/new pairs; lead 0 means
all 28 steps. `lead_paired_metrics.csv` uses the fixed intersection of old/new
four-model all-parameter common cases, with shared MSE ERA5 targets per version.
The new Figure 10 uses the new four-model intersection and the same reference
policy. Original target discrepancies are exported separately.

Reproduction commands, from the AGU root with the plotting Python environment:

```powershell
python tools/recompute_thermal_wind.py
python tools/report_thermal_wind.py
python -m unittest tests.test_thermal_wind
```

The one-time source staging and installation utilities have been retired.
Original utilities and documents are backed up in `artifacts/cleanup/before_cleanup.zip`.
