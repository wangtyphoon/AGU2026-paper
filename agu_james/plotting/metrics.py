# Derived from evaluation_metric/plot_james_metrics.py
# See provenance/sources.json for original hash and modifications.
"""Publication figures for 700q and 850v: relative RMSE, bias, ACC and activity.

Run with the local environment's Python from any working directory. Input CSVs
remain untouched. Values retain their source scale; source files provide no unit
metadata. Activity/ERA5 is a ratio of the supplied aggregate activity values,
not an average of individual forecast/target ratios. No uncertainty is inferred.
"""
from __future__ import annotations
from agu_james import runtime
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
ROOT = runtime.setting('agu_james.plotting.metrics.ROOT', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.metrics')
FEATURES = ('700q', '850v')
FEATURE_LABELS = {'700q': '$q_{700}$', '850v': '$v_{850}$'}
MODELS = ('MSE', 'Spectrum MAE', 'Spectrum MSE', 'Pangu')
COLORS = dict(zip((*MODELS, 'ERA5'), ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#222222')))
MARKERS = dict(zip((*MODELS, 'ERA5'), ('o', 's', '^', 'D', '')))
SUFFIXES = {'MSE': ('auto', 'auto'), 'Spectrum MAE': ('spectrum_new', 'spectrum'), 'Spectrum MSE': ('spectrum_ver2', 'spectrum_ver2')}
METRICS = ('rmse', 'bias', 'acc', 'activity')
DAY_MARKERS = {1: 'o', 3: 's', 5: '^', 7: 'D'}

class DataError(ValueError):
    """A source CSV does not satisfy the plotting data contract."""

def read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, encoding='utf-8-sig')
    except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise DataError(f'{path}: cannot read CSV: {exc}') from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataError(f"{path}: missing columns: {', '.join(missing)}")
    return frame

def finite_values(values: pd.Series, path: Path, column: str, *, positive: bool=False) -> np.ndarray:
    try:
        array = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataError(f'{path}: {column} contains non-numeric values') from exc
    if not np.isfinite(array).all():
        raise DataError(f'{path}: {column} contains missing or non-finite values')
    if positive and (array <= 0).any():
        raise DataError(f'{path}: {column} must be strictly positive')
    return array

def load_ours(path: Path, metrics: tuple[str, ...], step: int) -> pd.DataFrame:
    required = ['feature', *(f'{metric}_step{i:02d}' for metric in metrics for i in range(1, step + 1))]
    frame = read_csv(path, required)
    frame = frame[frame.feature.isin(FEATURES)].copy()
    if frame.feature.duplicated().any() or set(frame.feature) != set(FEATURES):
        raise DataError(f'{path}: expected exactly one row for each of {FEATURES}')
    for metric in metrics:
        actual = [c for c in frame.columns if c.startswith(f'{metric}_step')]
        expected = [f'{metric}_step{i:02d}' for i in range(1, step + 1)]
        if set(actual) != set(expected):
            raise DataError(f'{path}: {metric} step columns must cover exactly 1..{step}')
        for column in expected:
            frame[column] = finite_values(frame[column], path, column, positive=metric in ('rmse', 'target_activity'))
    records = []
    for _, row in frame.iterrows():
        for i in range(1, step + 1):
            records.append({'variable': row.feature, 'lead_step': i, **{metric: float(row[f'{metric}_step{i:02d}']) for metric in metrics}})
    return pd.DataFrame(records)

def load_pangu(path: Path, metrics: tuple[str, ...], step: int) -> pd.DataFrame:
    prefixes = {'700q': 'q700', '850v': 'v850'}
    required = ['lead_hours', *(f'{prefix}_{metric}' for prefix in prefixes.values() for metric in metrics)]
    frame = read_csv(path, required)
    hours = finite_values(frame.lead_hours, path, 'lead_hours')
    if len(hours) != step or not np.array_equal(np.sort(hours), np.arange(1, step + 1) * 6):
        raise DataError(f'{path}: lead_hours must contain each of 6,12,...,{step * 6} exactly once')
    for column in required[1:]:
        frame[column] = finite_values(frame[column], path, column, positive=column.endswith('_rmse'))
    records = []
    for _, row in frame.iterrows():
        for variable, prefix in prefixes.items():
            records.append({'variable': variable, 'lead_step': int(row.lead_hours / 6), **{metric: float(row[f'{prefix}_{metric}']) for metric in metrics}})
    return pd.DataFrame(records)

def load_data(root: Path, step: int) -> pd.DataFrame:
    keys = ['variable', 'lead_step']
    frames = []
    for model, (rmse_suffix, acc_suffix) in SUFFIXES.items():
        errors = load_ours(root / 'rmse_multi' / 'ours' / f'per_channel_rmse_summary_macro_2024_bias_fix_attention_step{step}_{rmse_suffix}.csv', ('rmse', 'bias'), step)
        skill = load_ours(root / 'acc_activity_multi' / 'ours' / f'per_channel_acc_activity_summary_step{step}_{acc_suffix}.csv', ('acc', 'activity'), step)
        frames.append(errors.merge(skill, on=keys, validate='one_to_one').assign(model=model))
    errors = load_pangu(root / 'rmse_multi' / 'pangu' / f'rmse_multistep_prs{step * 6}_by_lead.csv', ('rmse', 'bias'), step)
    skill = load_pangu(root / 'acc_activity_multi' / 'pangu' / f'acc_activity_prs{step * 6}_by_lead.csv', ('acc', 'activity'), step)
    frames.append(errors.merge(skill, on=keys, validate='one_to_one').assign(model='Pangu'))
    target = load_ours(root / 'acc_activity_multi' / 'target' / f'target_forecast_activity_summary_step{step}.csv', ('target_activity',), step)
    data = pd.concat(frames, ignore_index=True).merge(target, on=keys, validate='many_to_one')
    baseline = data.loc[data.model == 'Pangu', [*keys, 'rmse']].rename(columns={'rmse': 'pangu_rmse'})
    data = data.merge(baseline, on=keys, validate='many_to_one')
    data['relative_rmse'] = (data.rmse / data.pangu_rmse - 1) * 100
    data['activity_ratio'] = data.activity / data.target_activity
    data['lead_day'] = data.lead_step / 4
    if not np.isfinite(data[[*METRICS, 'relative_rmse', 'activity_ratio']].to_numpy()).all():
        raise DataError('Derived metrics contain non-finite values after baseline alignment')
    return data.sort_values(['variable', 'model', 'lead_step']).reset_index(drop=True)

def configure_style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def scientific_axis(ax: plt.Axes, axis: str='y') -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    getattr(ax, f'{axis}axis').set_major_formatter(formatter)

def panel_title(ax: plt.Axes, index: int, feature: str) -> None:
    ax.set_title(f'({chr(97 + index)}) {FEATURE_LABELS[feature]}', loc='left', pad=7)

def legend(fig: plt.Figure, *, era5: bool=False) -> None:
    models = (*MODELS, 'ERA5') if era5 else MODELS
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=4, linewidth=1.5, linestyle='--' if m == 'Pangu' else ':' if m == 'ERA5' else '-', label=m) for m in models]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=len(models), frameon=False, handlelength=2.2, columnspacing=1.5)

def day_axis(ax: plt.Axes, step: int) -> None:
    ax.set_xlim(0.25, step / 4)
    ax.set_xticks([0.25, *range(1, step // 4 + 1)])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.set_xlabel('Forecast lead time (days)')

def build_lines(data: pd.DataFrame, step: int) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0))
    specs = (('relative_rmse', 'Relative RMSE vs Pangu (%)'), ('bias', 'Bias (source scale)'), ('acc', 'ACC'), ('activity', 'Forecast activity (source scale)'))
    for row, (metric, label) in enumerate(specs):
        for col, feature in enumerate(FEATURES):
            ax = axes[row, col]
            subset = data[data.variable == feature]
            for model in MODELS[:-1] if metric == 'relative_rmse' else MODELS:
                values = subset[subset.model == model].sort_values('lead_step')
                ax.plot(values.lead_day, values[metric], color=COLORS[model], marker=MARKERS[model], markevery=[i for i, s in enumerate(values.lead_step) if s == 1 or s % 4 == 0], markersize=3.6, linewidth=1.5, linestyle='--' if model == 'Pangu' else '-')
            if metric == 'activity':
                target = subset[subset.model == 'MSE'].sort_values('lead_step')
                ax.plot(target.lead_day, target.target_activity, color=COLORS['ERA5'], linestyle=':', linewidth=1.8)
            if metric in ('relative_rmse', 'bias'):
                ax.axhline(0, color='0.4', linestyle='--', linewidth=0.8, zorder=0)
            if feature == '700q' and metric in ('bias', 'activity'):
                scientific_axis(ax)
            ax.grid(axis='y', color='0.9', linewidth=0.6)
            panel_title(ax, row * 2 + col, feature)
            ax.set_ylabel(label)
            day_axis(ax, step)
            if row < 3:
                ax.set_xlabel('')
            if metric in ('relative_rmse', 'acc'):
                limits = data[metric] if metric == 'acc' else data.loc[data.model != 'Pangu', metric]
                lo, hi = (min(float(limits.min()), 0) if metric == 'relative_rmse' else float(limits.min()), float(limits.max()))
                pad = max((hi - lo) * 0.08, 0.005)
                ax.set_ylim(lo - pad, hi + pad)
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.085, top=0.96, wspace=0.44, hspace=0.48)
    legend(fig, era5=True)
    return fig

def diverging_norm(values: np.ndarray, center: float) -> TwoSlopeNorm:
    span = max(float(np.max(np.abs(values - center))), 1e-12)
    return TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)

def build_heatmap(data: pd.DataFrame, step: int) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0))
    specs = (('relative_rmse', 'Relative RMSE (%)'), ('bias', 'Bias (source scale)'), ('acc', 'ACC'), ('activity_ratio', 'Activity / ERA5'))
    for row, (metric, label) in enumerate(specs):
        models = MODELS[:-1] if metric == 'relative_rmse' else MODELS
        pooled = data.loc[data.model.isin(models), metric].to_numpy()
        for col, feature in enumerate(FEATURES):
            ax = axes[row, col]
            subset = data[data.variable == feature]
            matrix = subset.pivot(index='model', columns='lead_step', values=metric).reindex(index=models, columns=range(1, step + 1))
            if metric == 'bias':
                norm = diverging_norm(matrix.to_numpy(), 0)
            elif metric in ('relative_rmse', 'activity_ratio'):
                norm = diverging_norm(pooled, 1 if metric == 'activity_ratio' else 0)
            else:
                lo, hi = (float(pooled.min()), float(pooled.max()))
                norm = Normalize(lo, hi if hi > lo else lo + 1e-12)
            image = ax.pcolormesh(np.arange(step + 1) / 4 + 0.125, np.arange(len(models) + 1), matrix.to_numpy(), cmap='viridis' if metric == 'acc' else 'RdBu_r', norm=norm, shading='flat', rasterized=False)
            ax.invert_yaxis()
            ax.set_yticks(np.arange(len(models)) + 0.5, labels=models)
            ax.tick_params(axis='y', length=0)
            ax.set_xlim(0.125, step / 4 + 0.125)
            ax.set_xticks(range(1, step // 4 + 1))
            ax.set_xlabel('Forecast lead time (days)' if row == 3 else '')
            panel_title(ax, row * 2 + col, feature)
            bar = fig.colorbar(image, ax=ax, pad=0.035, fraction=0.055)
            bar.solids.set_rasterized(False)
            bar.solids.set_edgecolor('face')
            bar.set_label(label, fontsize=8)
            if metric == 'bias' and feature == '700q':
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                formatter.set_useOffset(False)
                bar.formatter = formatter
                bar.update_ticks()
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.075, top=0.96, wspace=0.8, hspace=0.48)
    return fig

def build_tradeoff(data: pd.DataFrame, step: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    days = [day for day in DAY_MARKERS if day * 4 <= step]
    for col, feature in enumerate(FEATURES):
        subset = data[data.variable == feature]
        for row, (x, y, xlabel, ylabel) in enumerate((('relative_rmse', 'activity_ratio', 'Relative RMSE vs Pangu (%)', 'Activity / ERA5'), ('acc', 'bias', 'ACC', 'Bias (source scale)'))):
            ax = axes[row, col]
            for model in MODELS:
                values = subset[subset.model == model].sort_values('lead_step')
                ax.plot(values[x], values[y], color=COLORS[model], linewidth=1.1, alpha=0.6, linestyle='--' if model == 'Pangu' else '-')
                for day in days:
                    point = values[values.lead_step == day * 4].iloc[0]
                    ax.scatter(point[x], point[y], s=34, marker=DAY_MARKERS[day], color=COLORS[model], edgecolors='white', linewidths=0.5, zorder=3)
            if row == 0:
                ax.axvline(0, color='0.45', linewidth=0.8, linestyle='--', zorder=0)
                ax.axhline(1, color='0.45', linewidth=0.8, linestyle='--', zorder=0)
            else:
                ax.axhline(0, color='0.45', linewidth=0.8, linestyle='--', zorder=0)
                if feature == '700q':
                    scientific_axis(ax, 'y')
            ax.grid(color='0.92', linewidth=0.6)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            panel_title(ax, row * 2 + col, feature)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.94, wspace=0.38, hspace=0.43)
    legend(fig)
    day_handles = [Line2D([], [], linestyle='', marker=DAY_MARKERS[d], markersize=5, color='0.35', label=f'Day {d}') for d in days]
    fig.legend(handles=day_handles, loc='lower center', bbox_to_anchor=(0.5, 0.064), ncol=len(days), frameon=False, title='Highlighted forecast lead times', title_fontsize=8)
    return fig

def save_figure(fig: plt.Figure, output: Path, style: str, step: int) -> None:
    stem = output / f'james_700q_850v_{style}_step{step}'
    try:
        fig.savefig(stem.with_suffix('.pdf'), metadata={'Title': f'700q / 850v {style}, {step * 6} h', 'Subject': 'Relative RMSE vs Pangu; bias; ACC; forecast activity'})
        fig.savefig(stem.with_suffix('.png'), dpi=600)
    finally:
        plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=(12, 28), default=28)
    parser.add_argument('--style', choices=('all', 'lines', 'heatmap', 'tradeoff'), default='all')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args(argv)
    try:
        data = load_data(ROOT, args.step)
    except DataError as exc:
        print(f'Data error: {exc}', file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    builders = {'lines': build_lines, 'heatmap': build_heatmap, 'tradeoff': build_tradeoff}
    for style in builders if args.style == 'all' else (args.style,):
        save_figure(builders[style](data, args.step), args.output_dir, style, args.step)
    return 0
