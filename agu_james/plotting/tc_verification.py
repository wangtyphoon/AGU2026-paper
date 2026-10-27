# Derived from tc_verification/plot_tc_verification.py
"""Paper figure for tropical-cyclone track and S10 intensity verification.

Run from any working directory with the existing plotting environment::

    python <path-to-this-script>

Inputs are the four existing step28 CSVs under ../tracker_multi. Outputs are
tc_verification.png (600 dpi), tc_verification.pdf (vector), and a long-format
tc_verification_summary.csv beside this script. No source data are copied.

Panels (a)-(c) use common cases at 24, 48, and 72 h. Boxes show individual
track distances, absolute S10 errors, and signed S10 errors, respectively;
symbols show mean track distance, S10 RMSE, and mean signed S10 error.
Whiskers extend to 1.5 IQR; outliers are hidden but included in all statistics.
Mean/bias intervals are mean +/- 1.96 SE. RMSE intervals are percentile
intervals from 2,000 case-wise bootstrap resamples (seed 42). These intervals
retain the original script's method; they are not storm-cluster intervals.

Panel (d) pools common 24 h windows ending by 72 h. Each pair belongs to the
same initialization and cyclone and is separated by exactly four 6 h steps.
Step 0 is not synthesized, so window endpoints span 30-72 h. The plotted
quantity is a 24 h wind change in m/s, not a change per hour. ERA5 is taken
once from MSE's target values; model error metrics retain each CSV's targets.

The original geographic bounds and non-missing target MSL eligibility filter
are retained for sample compatibility; no MSL or S850 metric is calculated.
CSV distribution summaries use all cases, including hidden boxplot outliers.
For the RMSE row, distribution statistics describe absolute errors, while
the value and confidence interval describe RMSE. Change rows pool windows
rather than a single lead and have no confidence interval.
"""
from __future__ import annotations
from agu_james import runtime
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
STEP_HOURS = 6
MAX_LEAD_HOURS = 72
TARGET_STEPS = (4, 8, 12)
CHANGE_HOURS = 24
CHANGE_STEPS = CHANGE_HOURS // STEP_HOURS
CI_Z = 1.96
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
FIGSIZE = (7.2, 6.2)
OUTPUT_DPI = 600
CASE_KEY_COLS = ['time', 'forecast_init_time', 'CYCLONE_NUM', 'step']
WINDOW_KEY_COLS = ['forecast_init_time', 'CYCLONE_NUM', 'step_start', 'step_end']

@dataclass(frozen=True)
class ModelSpec:
    label: str
    relative_path: str
    latitude_col: str
    longitude_col: str
    color: str
    marker: str
    linestyle: str
MODEL_SPECS = (ModelSpec('MSE', 'ours/tc_aurora_intensity35_step28_auto.csv', 'forecast_center_lat', 'forecast_center_lon', '#1f77b4', 'o', '-'), ModelSpec('Spectrum MAE', 'ours/tc_aurora_intensity35_step28_spectrum.csv', 'forecast_center_lat', 'forecast_center_lon', '#ff7f0e', 's', '--'), ModelSpec('Spectrum MSE', 'ours/tc_aurora_intensity35_step28_spectrum_ver2.csv', 'forecast_center_lat', 'forecast_center_lon', '#2ca02c', '^', '-.'), ModelSpec('Pangu', 'pangu/tc_aurora_metrics35_step28.csv', 'forecast_lat', 'forecast_lon', '#d62728', 'D', ':'))

def normalize_case_keys(df: pd.DataFrame, label: str) -> pd.DataFrame:
    aliases = {'init_time': 'forecast_init_time', 'cyclone_num': 'CYCLONE_NUM'}
    df = df.rename(columns={k: v for k, v in aliases.items() if v not in df}).copy()
    missing = [col for col in CASE_KEY_COLS if col not in df]
    if missing:
        raise ValueError(f'{label}: missing case keys {missing}')
    if df[CASE_KEY_COLS].isna().any().any():
        raise ValueError(f'{label}: missing values in case keys')
    steps = pd.to_numeric(df['step'], errors='raise')
    if not np.isfinite(steps).all() or (steps != np.floor(steps)).any():
        raise ValueError(f'{label}: forecast steps must be finite integers')
    df['step'] = steps.astype(int)
    for col in ('time', 'forecast_init_time'):
        df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d%H', errors='raise')
    df['CYCLONE_NUM'] = df['CYCLONE_NUM'].astype(str)
    expected_time = df['forecast_init_time'] + pd.to_timedelta(df['step'] * STEP_HOURS, unit='h')
    if df['time'].ne(expected_time).any():
        raise ValueError(f'{label}: valid time does not match initialization plus lead')
    for keys in (CASE_KEY_COLS, ['forecast_init_time', 'CYCLONE_NUM', 'step']):
        if df.duplicated(keys).any():
            raise ValueError(f'{label}: duplicate cases for keys {keys}')
    return df

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    a = np.clip(a, 0, 1)
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def load_model_data(tracker_dir: Path, spec: ModelSpec) -> pd.DataFrame:
    path = tracker_dir / spec.relative_path
    df = normalize_case_keys(pd.read_csv(path), spec.label)
    value_cols = [spec.latitude_col, spec.longitude_col, 'bst_lat', 'bst_lon', 'forecast_s10_max', 'target_s10_max']
    required = value_cols + ['target_msl_min_hPa']
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}')
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='raise')
    eligible = df['step'].between(1, MAX_LEAD_HOURS // STEP_HOURS) & df[spec.latitude_col].between(5.0, 42.25, inclusive='neither') & df[spec.longitude_col].between(107.5, 142.5, inclusive='neither') & df['target_msl_min_hPa'].notna() & np.isfinite(df[value_cols]).all(axis=1)
    valid = df.loc[eligible].copy()
    valid['track_error_km'] = haversine_km(valid[spec.latitude_col], valid[spec.longitude_col], valid['bst_lat'], valid['bst_lon'])
    valid['s10_error'] = valid['forecast_s10_max'] - valid['target_s10_max']
    valid['s10_absolute_error'] = valid['s10_error'].abs()
    print(f'{spec.label}: {len(df):,} input rows -> {len(valid):,} eligible rows (6-72 h)')
    return valid

def filter_common(datasets: dict[str, pd.DataFrame], keys: list[str]) -> dict[str, pd.DataFrame]:
    common = None
    for label, df in datasets.items():
        if df.duplicated(keys).any():
            raise ValueError(f'{label}: duplicate keys {keys}')
        current = df[keys]
        common = current if common is None else common.merge(current, on=keys, how='inner', validate='one_to_one')
    if common is None or common.empty:
        raise ValueError(f'No common samples for keys {keys}')
    return {label: df.merge(common, on=keys, how='inner', validate='one_to_one') for label, df in datasets.items()}

def compute_change_windows(df: pd.DataFrame) -> pd.DataFrame:
    cols = CASE_KEY_COLS + ['forecast_s10_max', 'target_s10_max']
    start = df[cols].rename(columns={'time': 'time_start', 'step': 'step_start', 'forecast_s10_max': 'forecast_s10_start', 'target_s10_max': 'target_s10_start'})
    start = start.assign(step_end=start['step_start'] + CHANGE_STEPS)
    end = df[cols].rename(columns={'time': 'time_end', 'step': 'step_end', 'forecast_s10_max': 'forecast_s10_end', 'target_s10_max': 'target_s10_end'})
    paired = start.merge(end, on=['forecast_init_time', 'CYCLONE_NUM', 'step_end'], how='inner', validate='one_to_one')
    paired = paired.loc[paired['step_end'] <= MAX_LEAD_HOURS // STEP_HOURS].copy()
    if (paired['time_end'] - paired['time_start']).ne(pd.Timedelta(hours=CHANGE_HOURS)).any():
        raise ValueError('A change window does not span exactly 24 hours')
    paired['forecast_change_24h'] = paired['forecast_s10_end'] - paired['forecast_s10_start']
    paired['target_change_24h'] = paired['target_s10_end'] - paired['target_s10_start']
    return paired

def load_common_data(tracker_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    raw = {spec.label: load_model_data(tracker_dir, spec) for spec in MODEL_SPECS}
    common = filter_common(raw, CASE_KEY_COLS)
    reference = common['MSE'].set_index(CASE_KEY_COLS)['target_s10_max']
    for label, df in common.items():
        targets = df.set_index(CASE_KEY_COLS)['target_s10_max'].reindex(reference.index)
        if not np.allclose(targets, reference, rtol=0, atol=0.001):
            raise ValueError(f'{label}: ERA5 S10 targets disagree across common cases')
    windows = filter_common({label: compute_change_windows(df) for label, df in common.items()}, WINDOW_KEY_COLS)
    counts = common['MSE'].groupby('step').size()
    missing = [step for step in TARGET_STEPS if step not in counts]
    if missing:
        raise ValueError(f'No common cases at required forecast steps {missing}')
    print('Common cases: ' + ', '.join((f'{step * STEP_HOURS} h = {counts[step]:,}' for step in TARGET_STEPS)))
    print(f"Common 24 h windows: {len(windows['MSE']):,}")
    return (common, windows)

def bootstrap_rmse_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))]
    estimates = np.sqrt(np.mean(samples ** 2, axis=1))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return (float(low), float(high))

def distribution_statistics(values: np.ndarray) -> dict[str, float]:
    return {'distribution_mean': float(np.mean(values)), 'distribution_std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 'distribution_min': float(np.min(values)), 'q25': float(np.quantile(values, 0.25)), 'median': float(np.median(values)), 'q75': float(np.quantile(values, 0.75)), 'distribution_max': float(np.max(values))}

def summarize(datasets: dict[str, pd.DataFrame], windows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for step in TARGET_STEPS:
            group = datasets[spec.label].loc[datasets[spec.label]['step'] == step]
            for metric, column, unit in (('track_mean_error', 'track_error_km', 'km'), ('s10_rmse', 's10_absolute_error', 'm/s'), ('s10_bias', 's10_error', 'm/s')):
                values = group[column].to_numpy(dtype=float)
                stats = distribution_statistics(values)
                if metric == 's10_rmse':
                    value = float(np.sqrt(np.mean(values ** 2)))
                    low, high = bootstrap_rmse_ci(values, rng)
                    ci_method = 'case_bootstrap_percentile_2000_seed42'
                else:
                    value = stats['distribution_mean']
                    margin = CI_Z * stats['distribution_std'] / np.sqrt(len(values))
                    low, high = (value - margin, value + margin)
                    ci_method = 'mean_plus_minus_1.96_se'
                rows.append({'metric': metric, 'model': spec.label, 'unit': unit, 'lead_step': step, 'lead_hours': step * STEP_HOURS, 'value': value, 'ci95_low': low, 'ci95_high': high, 'ci_method': ci_method, 'count': len(group), 'cyclone_count': group['CYCLONE_NUM'].nunique(), 'window_hours': np.nan, 'window_end_min_hours': np.nan, 'window_end_max_hours': np.nan, **stats})
    change_series = [(spec.label, windows[spec.label], 'forecast_change_24h') for spec in MODEL_SPECS]
    change_series.append(('ERA5', windows['MSE'], 'target_change_24h'))
    for label, group, column in change_series:
        stats = distribution_statistics(group[column].to_numpy(dtype=float))
        rows.append({'metric': 's10_change_24h', 'model': label, 'unit': 'm/s per 24 h', 'lead_step': np.nan, 'lead_hours': np.nan, 'value': stats['distribution_mean'], 'ci95_low': np.nan, 'ci95_high': np.nan, 'ci_method': 'not_estimated', 'count': len(group), 'cyclone_count': group['CYCLONE_NUM'].nunique(), 'window_hours': CHANGE_HOURS, 'window_end_min_hours': group['step_end'].min() * STEP_HOURS, 'window_end_max_hours': group['step_end'].max() * STEP_HOURS, **stats})
    return pd.DataFrame(rows)

def style_axis(ax: plt.Axes) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.7)
    ax.grid(axis='y', color='#dddddd', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', width=0.6, length=3, labelsize=8)

def plot_box_panel(ax: plt.Axes, datasets: dict[str, pd.DataFrame], summary: pd.DataFrame, metric: str, value_col: str, title: str, ylabel: str, zero_line: bool=False) -> None:
    positions = np.arange(1, len(TARGET_STEPS) + 1, dtype=float)
    offsets = (np.arange(len(MODEL_SPECS)) - (len(MODEL_SPECS) - 1) / 2) * 0.2
    for spec, offset in zip(MODEL_SPECS, offsets):
        data = [datasets[spec.label].loc[datasets[spec.label]['step'] == step, value_col].to_numpy() for step in TARGET_STEPS]
        boxes = ax.boxplot(data, positions=positions + offset, widths=0.16, patch_artist=True, showfliers=False, whis=1.5, manage_ticks=False, medianprops={'color': '#333333', 'linewidth': 0.8}, whiskerprops={'color': spec.color, 'linewidth': 0.7}, capprops={'color': spec.color, 'linewidth': 0.7})
        for box in boxes['boxes']:
            box.set(facecolor=spec.color, edgecolor=spec.color, alpha=0.22, linewidth=0.7)
        rows = summary.loc[(summary['model'] == spec.label) & (summary['metric'] == metric)].set_index('lead_step').loc[list(TARGET_STEPS)]
        x = positions + offset
        low, high = (rows['ci95_low'].to_numpy(), rows['ci95_high'].to_numpy())
        ax.vlines(x, low, high, colors=spec.color, linewidth=1.2, zorder=4)
        ax.hlines(low, x - 0.035, x + 0.035, colors=spec.color, linewidth=1.2, zorder=4)
        ax.hlines(high, x - 0.035, x + 0.035, colors=spec.color, linewidth=1.2, zorder=4)
        ax.scatter(x, rows['value'], color=spec.color, marker=spec.marker, s=20, edgecolors='white', linewidths=0.4, zorder=5)
    if zero_line:
        ax.axhline(0, color='#666666', linewidth=0.8, linestyle='--', zorder=2)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(positions, [str(step * STEP_HOURS // 24) for step in TARGET_STEPS])
    ax.set_xlim(0.45, len(TARGET_STEPS) + 0.55)
    ax.set_xlabel('Forecast lead (day)', fontsize=9, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=5)
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold', pad=8)
    style_axis(ax)

def plot_change_panel(ax: plt.Axes, windows: dict[str, pd.DataFrame]) -> None:
    target = windows['MSE']['target_change_24h'].to_numpy(dtype=float)
    forecasts = [windows[spec.label]['forecast_change_24h'].to_numpy(dtype=float) for spec in MODEL_SPECS]
    joined = np.concatenate([target, *forecasts])
    low, high = np.quantile(joined, [0.01, 0.99])
    pad = max((high - low) * 0.08, 1.0)
    grid = np.linspace(low - pad, high + pad, 500)
    for label, values in [('ERA5', target), *[(spec.label, values) for spec, values in zip(MODEL_SPECS, forecasts)]]:
        if len(values) < 2 or np.unique(values).size < 2:
            raise ValueError(f'{label}: KDE requires at least two distinct 24 h changes')
    target_density = gaussian_kde(target)(grid)
    ax.fill_between(grid, 0, target_density, color='#999999', alpha=0.22, zorder=1)
    ax.plot(grid, target_density, color='#555555', linewidth=1.2, zorder=2)
    for spec, values in zip(MODEL_SPECS, forecasts):
        ax.plot(grid, gaussian_kde(values)(grid), color=spec.color, linestyle=spec.linestyle, linewidth=1.4, zorder=3)
    ax.axvline(0, color='#666666', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)
    ax.set_xlabel('24 h S10 change (m s$^{-1}$)', fontsize=9, labelpad=5)
    ax.set_ylabel('Probability density', fontsize=9, labelpad=5)
    ax.set_title('(d) 24 h S10 change', loc='left', fontsize=10, fontweight='bold', pad=8)
    style_axis(ax)

def build_figure(datasets: dict[str, pd.DataFrame], windows: dict[str, pd.DataFrame], summary: pd.DataFrame, output_dir: Path) -> None:
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
        fig.subplots_adjust(left=0.1, right=0.985, bottom=0.105, top=0.885, wspace=0.32, hspace=0.49)
        plot_box_panel(axes[0, 0], datasets, summary, 'track_mean_error', 'track_error_km', '(a) Track position error', 'Distance / mean (km)')
        plot_box_panel(axes[0, 1], datasets, summary, 's10_rmse', 's10_absolute_error', '(b) S10 intensity error', 'Absolute error / RMSE (m s$^{-1}$)')
        plot_box_panel(axes[1, 0], datasets, summary, 's10_bias', 's10_error', '(c) S10 intensity bias', 'Signed error / bias (m s$^{-1}$)', zero_line=True)
        plot_change_panel(axes[1, 1], windows)
        handles = [Line2D([0], [0], color=spec.color, marker=spec.marker, linestyle=spec.linestyle, linewidth=1.4, markersize=4, label=spec.label) for spec in MODEL_SPECS]
        handles.append(Patch(facecolor='#999999', edgecolor='#555555', alpha=0.35, label='ERA5'))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.535, 0.97), ncol=5, frameon=False, fontsize=8.2, handlelength=2.2, columnspacing=1.1)
        fig.savefig(output_dir / 'tc_verification.png', dpi=OUTPUT_DPI, facecolor='white')
        fig.savefig(output_dir / 'tc_verification.pdf', metadata={'Title': 'Tropical cyclone track and S10 intensity verification'})
        plt.close(fig)

def main() -> None:
    output_dir = runtime.setting('agu_james.plotting.tc_verification.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.tc_verification')
    tracker_dir = runtime.setting('agu_james.plotting.tc_verification.tracker_dir', lambda: output_dir.parent / 'tracker_multi')
    datasets, windows = load_common_data(tracker_dir)
    summary = summarize(datasets, windows)
    build_figure(datasets, windows, summary, output_dir)
    summary.to_csv(output_dir / 'tc_verification_summary.csv', index=False, float_format='%.10g')
    for filename in ('tc_verification.png', 'tc_verification.pdf', 'tc_verification_summary.csv'):
        print(f'Saved {output_dir / filename}')
