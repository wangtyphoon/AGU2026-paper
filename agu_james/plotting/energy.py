"""Annular ratios and mode-count-weighted band mean-square ratios.
Nominal scale = 174*28/k km; not an exact geographic wavelength.
Retained disk k=1..87; nominal <1000 km selects complete bins k=5..87.
No kinetic-energy interpretation for humidity or individual wind components.
Legacy raw annular means are preserved; only band totals are normalized.
"""
from agu_james.analysis.spectral import modal_weights, NOMINAL_LENGTH_KM, FFT_SHAPE
from agu_james import runtime
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
BASE_DIR = runtime.setting('agu_james.plotting.energy.BASE_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.energy')
DATA_DIR = runtime.setting('agu_james.plotting.energy.DATA_DIR', lambda: BASE_DIR.parent / 'spectrum_multi')
CHANNELS = (('700q', 'q700'), ('850u', 'u850'), ('850v', 'v850'))
MODEL_STYLES = {'MSE': ('auto', '#1f77b4'), 'Spectrum MAE': ('spectrum', '#ff7f0e'), 'Spectrum MSE': ('spectrum_ver2', '#2ca02c'), 'Pangu': (None, '#d62728')}
PERIODS = {'7d': {'steps': 28, 'stride': 1, 'days': (1, 3, 7), 'models': ('MSE', 'Spectrum MAE', 'Spectrum MSE', 'Pangu')}, '30d': {'steps': 120, 'stride': 2, 'days': (7, 14, 21, 28), 'models': ('Spectrum MAE', 'Spectrum MSE')}}
LEAD_STYLES = ('-', '--', '-.', ':')
EFFECTIVE_LENGTH_KM = NOMINAL_LENGTH_KM
TOTAL_WAVELENGTH_MAX_KM = 1000.0

@dataclass
class EnergySeries:
    model: str
    channel: str
    lead_hours: np.ndarray
    k: np.ndarray
    forecast: np.ndarray
    target: np.ndarray
    ratio: np.ndarray
    forecast_total: np.ndarray
    target_total: np.ndarray
    total_ratio: np.ndarray

def total_energy_mask(k):
    """Select non-DC bins with wavelength strictly below the cutoff."""
    k = np.asarray(k)
    return (k > 0) & (k > EFFECTIVE_LENGTH_KM / TOTAL_WAVELENGTH_MAX_KM)

def compute_energy_ratios(k, forecast, target):
    """Return non-DC spectra, bin ratios, totals and total ratios in float64.

    Invalid powers fail validation. A zero target bin produces NaN, including
    0/0; zero forecast power with positive target remains a valid zero ratio.
    """
    k = np.asarray(k)
    forecast = np.asarray(forecast, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if k.ndim != 1 or not np.all(np.isfinite(k)) or np.any(k < 0) or np.any(np.diff(k) <= 0):
        raise ValueError('k must be finite, nonnegative, and strictly increasing.')
    if forecast.shape != target.shape or forecast.ndim != 2 or forecast.shape[1] != len(k):
        raise ValueError('Forecast and target must have matching (lead, k) shapes.')
    mask = k > 0
    if not mask.any():
        raise ValueError('No non-DC wavenumbers available.')
    forecast, target = (forecast[:, mask], target[:, mask])
    if not np.all(np.isfinite(forecast)) or not np.all(np.isfinite(target)) or np.any(forecast < 0) or np.any(target < 0):
        raise ValueError('Selected spectral powers must be finite and nonnegative.')
    total_mask = total_energy_mask(k[mask])
    if not total_mask.any():
        raise ValueError('No wavenumbers with wavelength <1000 km available.')
    forecast_total = (forecast[:, total_mask] * modal_weights(k[mask][total_mask])).sum(axis=1, dtype=np.float64)
    target_total = (target[:, total_mask] * modal_weights(k[mask][total_mask])).sum(axis=1, dtype=np.float64)
    if np.any(target_total <= 0) or not np.all(np.isfinite(target_total)) or (not np.all(np.isfinite(forecast_total))):
        raise ValueError('Total energies must be finite and target totals positive.')
    ratio = np.full(target.shape, np.nan, dtype=np.float64)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        np.divide(forecast, target, out=ratio, where=target > 0)
        total_ratio = forecast_total / target_total
    if not np.all(np.isfinite(total_ratio)) or np.any(~np.isfinite(ratio[target > 0])):
        raise ValueError('Energy ratio overflow.')
    return (k[mask], forecast, target, ratio, forecast_total, target_total, total_ratio)

def load_period(period):
    """Validate all required leads/models/channels before exporting a period."""
    config = PERIODS[period]
    with (DATA_DIR / 'next_feature_table_weight.csv').open(encoding='utf-8-sig', newline='') as stream:
        features = [row['feature'] for row in csv.DictReader(stream)]
    if not features or len(set(features)) != len(features):
        raise ValueError('Feature table must contain nonempty, unique feature names.')
    indices = {name: features.index(name) for name, _ in CHANNELS}
    steps = np.arange(config['stride'], config['steps'] + 1, config['stride'])
    lead_hours = steps * 6
    datasets = []
    reference_k = None
    for model in config['models']:
        suffix, _ = MODEL_STYLES[model]
        path = DATA_DIR / 'Pangu' / 'all_energy_multistep_prs168.npz' if model == 'Pangu' else DATA_DIR / 'ours' / f"per_channel_energy_2024_fix_attention_step{config['steps']}_{suffix}.npz"
        with np.load(path, allow_pickle=False) as source:
            k = np.asarray(source['k']).copy()
            if 'fft_shape' in source and not np.array_equal(source['fft_shape'], FFT_SHAPE):
                raise ValueError(f'{path}: incompatible FFT geometry')
            if 'spectral_quantity' in source and str(source['spectral_quantity']) != 'power':
                raise ValueError(f'{path}: power spectra required')
            if 'density_norm' in source and bool(source['density_norm']):
                raise ValueError(f'{path}: expected unnormalized annular means')
            if reference_k is not None and (not np.array_equal(k, reference_k)):
                raise ValueError(f'{path}: wavenumbers differ across models.')
            reference_k = k
            if not np.array_equal(k, np.arange(88)):
                raise ValueError(f'{path}: expected wavenumbers 0..87, got {k}.')
            for channel, alias in CHANNELS:
                forecast, target = ([], [])
                for step, hours in zip(steps, lead_hours):
                    if model == 'Pangu':
                        keys = (f'pangu_mean_{alias}_lead{hours:02d}', f'era5_mean_{alias}_lead{hours:02d}')
                        expected = (len(k),)
                    else:
                        keys = (f'forecast_mean_step{step:02d}', f'target_mean_step{step:02d}')
                        expected = (len(features), len(k))
                    values = []
                    for key in keys:
                        if key not in source:
                            raise KeyError(f'{path}: missing required key {key}.')
                        value = np.asarray(source[key], dtype=np.float64)
                        if value.shape != expected:
                            raise ValueError(f'{path}: {key} shape {value.shape}, expected {expected}.')
                        values.append(value if model == 'Pangu' else value[indices[channel]])
                    forecast.append(values[0])
                    target.append(values[1])
                result = compute_energy_ratios(k, forecast, target)
                series = EnergySeries(model, channel, lead_hours.copy(), *result)
                missing = int(np.count_nonzero(np.isnan(series.ratio)))
                print(f'[{period}] {model} {channel}: {len(steps)} leads, zero-target bins={missing} (ratios marked missing)', flush=True)
                datasets.append(series)
    return datasets

def ratio_limits(arrays, include_zero=False):
    """Common limits include every finite value and the unit reference."""
    values = np.concatenate([np.asarray(a).ravel() for a in arrays] + [np.array([1.0])])
    values = values[np.isfinite(values)]
    lower, upper = (float(values.min()), float(values.max()))
    if include_zero:
        lower = min(0.0, lower)
    margin = max(0.04 * (upper - lower), 0.02)
    return (lower if include_zero else lower - margin, upper + margin)

def plot_period(period, datasets, output_dir):
    config = PERIODS[period]
    spectral_limits = ratio_limits([s.ratio[np.isin(s.lead_hours, np.array(config['days']) * 24)] for s in datasets], include_zero=True)
    total_limits = ratio_limits([s.total_ratio for s in datasets])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(3, 2, figsize=(12, 12.5))
        try:
            for row, (channel, _) in enumerate(CHANNELS):
                spectral, cumulative = axes[row, :]
                for series in datasets:
                    if series.channel != channel:
                        continue
                    color = MODEL_STYLES[series.model][1]
                    for day, linestyle in zip(config['days'], LEAD_STYLES):
                        matches = np.flatnonzero(series.lead_hours == day * 24)
                        if len(matches) != 1:
                            raise ValueError(f'{series.model}: missing spectral lead day {day}.')
                        spectral.plot(EFFECTIVE_LENGTH_KM / series.k, series.ratio[matches[0]], color=color, linestyle=linestyle, linewidth=1.6)
                    cumulative.plot(series.lead_hours / 24, series.total_ratio, color=color, linewidth=1.8)
                for column, ax in enumerate((spectral, cumulative)):
                    ax.axhline(1, color='#555555', linestyle=(0, (2, 3)), linewidth=0.9)
                    ax.grid(True, alpha=0.23, linewidth=0.6)
                    ax.set_axisbelow(True)
                    ax.set_title(f'({chr(97 + row * 2 + column)}) {channel}', loc='left', fontsize=13)
                    ax.set_ylabel('Annular power ratio' if column == 0 else 'Band power ratio (nominal <1000 km)')
                    ax.set_ylim(*(spectral_limits if column == 0 else total_limits))
                spectral.set_xscale('log')
                spectral.set_xlim(EFFECTIVE_LENGTH_KM, EFFECTIVE_LENGTH_KM / 87)
                spectral.xaxis.set_major_locator(FixedLocator([4000, 2000, 1000, 500, 200, 100]))
                spectral.xaxis.set_major_formatter(ScalarFormatter())
                spectral.set_xlabel('Nominal wavelength (km; 28 km grid)')
                cumulative.set_xlabel('Forecast lead time (days)')
                cumulative.set_xlim(0, config['steps'] / 4)
                cumulative.set_xticks([0, 1, 2, 3, 4, 5, 6, 7] if period == '7d' else [0, 5, 10, 15, 20, 25, 30])
            model_handles = [Line2D([0], [0], color=MODEL_STYLES[m][1], linewidth=2, label=m) for m in config['models']]
            lead_handles = [Line2D([0], [0], color='#333333', linestyle=ls, linewidth=1.7, label=f'Day {day}') for day, ls in zip(config['days'], LEAD_STYLES)]
            fig.suptitle(f"Annular and band power ratios | {config['steps'] // 4}-day forecasts", fontsize=17, y=0.98)
            fig.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=len(model_handles), frameon=False)
            fig.legend(handles=lead_handles, loc='upper center', bbox_to_anchor=(0.5, 0.902), ncol=len(lead_handles), frameon=False, title='Left panels: forecast lead')
            fig.subplots_adjust(left=0.085, right=0.98, bottom=0.085, top=0.83, wspace=0.28, hspace=0.4)
            stem = output_dir / f'spectral_energy_summary_{period}_700q_850u_850v_lt1000km'
            for extension in ('png', 'pdf'):
                path = stem.with_suffix(f'.{extension}')
                fig.savefig(path, dpi=600, facecolor='white')
                print(f'[saved] {path}', flush=True)
        finally:
            plt.close(fig)

def write_csvs(period, datasets, output_dir):
    stem = f'spectral_energy_summary_{period}_700q_850u_850v_lt1000km'
    spectral_path = output_dir / f'{stem}_spectral.csv'
    totals_path = output_dir / f'{stem}_totals.csv'
    with spectral_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('period', 'model', 'channel', 'lead_hours', 'lead_days', 'k', 'forecast_annular_mean_power', 'target_annular_mean_power', 'power_ratio'))
        for s in datasets:
            for row, hours in enumerate(s.lead_hours):
                for column, k in enumerate(s.k):
                    ratio = s.ratio[row, column]
                    writer.writerow((period, s.model, s.channel, int(hours), hours / 24, int(k), s.forecast[row, column], s.target[row, column], ratio if np.isfinite(ratio) else ''))
    with totals_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('period', 'model', 'channel', 'lead_hours', 'lead_days', 'k_min', 'k_max', 'forecast_band_mean_square', 'target_band_mean_square', 'band_power_ratio', 'wavelength_cutoff_km', 'wavelength_comparison', 'effective_length_km'))
        for s in datasets:
            selected_k = s.k[total_energy_mask(s.k)]
            for row, hours in enumerate(s.lead_hours):
                writer.writerow((period, s.model, s.channel, int(hours), hours / 24, int(selected_k[0]), int(selected_k[-1]), s.forecast_total[row], s.target_total[row], s.total_ratio[row], TOTAL_WAVELENGTH_MAX_KM, '<', EFFECTIVE_LENGTH_KM))
    print(f'[saved] {spectral_path}\n[saved] {totals_path}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', choices=('7d', '30d', 'both'), default='both')
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'outputs')
    args = parser.parse_args()
    periods = ('7d', '30d') if args.period == 'both' else (args.period,)
    datasets = {period: load_period(period) for period in periods}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for period in periods:
        plot_period(period, datasets[period], args.output_dir)
        write_csvs(period, datasets[period], args.output_dir)
