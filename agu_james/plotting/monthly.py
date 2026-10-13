"""Combine monthly case maps with the existing 30-day aggregate spectra.

Run from any directory. Map lead days must exist in both monthly exports.
"""
from __future__ import annotations
from agu_james import runtime
import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
BASE = runtime.setting('agu_james.plotting.monthly.BASE', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.monthly')
MAP_BASE = runtime.setting('agu_james.plotting.monthly.MAP_BASE', lambda: BASE.parent / 'vis_multi_steps')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
import agu_james.plotting.monthly_maps as maps
import agu_james.plotting.energy as energy
MODEL_ROOTS = {'Spectrum MAE': MAP_BASE / 'ours_spectrum_monthly', 'Spectrum MSE': MAP_BASE / 'ours_spectrum_ver2_monthly'}
FIGURES = (('700q', 'channel_076', '700q', '700 hPa specific humidity', 'Specific humidity (kg/kg)', 0.014), ('850v_850wind', 'wind_speed_086_087', '850v', '850 hPa wind speed', 'Wind speed (m/s)', 50.0))

def read_maps(run_tag, days, channel):
    rows, sources = ([], [])
    for day in days:
        step = round(day * 4)
        fields, targets = ({}, [])
        for model, root in MODEL_ROOTS.items():
            path = root / run_tag / f'step{step:02d}' / f'{channel}.npz'
            with np.load(path, allow_pickle=False) as data:
                forecast = np.asarray(data['forecast'], dtype=np.float64)
                target = np.asarray(data['target'], dtype=np.float64)
            if forecast.shape != (192, 192) or target.shape != forecast.shape:
                raise ValueError(f'{path}: expected matching 192x192 fields')
            if not np.isfinite(forecast).all() or not np.isfinite(target).all():
                raise ValueError(f'{path}: nonfinite map values')
            fields[model] = forecast
            targets.append(target)
            sources.append(str(path))
        if not np.allclose(targets[0], targets[1], rtol=1e-07, atol=1e-10):
            raise ValueError(f'{channel} day {day}: model ERA5 targets differ')
        fields['ERA5'] = targets[0]
        rows.append((day, fields))
    return (rows, sources)

def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

def render(config, rows, series, context, extent, days, run_tag, output):
    name, channel, spectral_channel, map_title, units, maximum = config
    stem = output / f'monthly_spatial_spectrum_{name}_{run_tag}'
    rmse_rows = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig = plt.figure(figsize=(14.5, max(8.0, 2.05 * len(days) + 1.5)))
        outer = fig.add_gridspec(1, 2, left=0.055, right=0.975, bottom=0.12, top=0.84, width_ratios=(1.7, 1), wspace=0.23)
        left = outer[0].subgridspec(len(days), 4, width_ratios=(1, 1, 1, 0.07), wspace=0.08, hspace=0.1)
        right = outer[1].subgridspec(2, 1, hspace=0.52)
        weights = maps.create_spatial_weights_np()
        for row, (day, fields) in enumerate(rows):
            for col, (model, field) in enumerate(fields.items()):
                ax = fig.add_subplot(left[row, col], projection=context['projection'])
                im = ax.imshow(field, origin='upper', extent=extent, transform=context['data_crs'], cmap='jet', vmin=0, vmax=maximum, zorder=1)
                ax.set_extent(extent, crs=context['data_crs'])
                maps.draw_map_background(ax, context)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(model, fontsize=11, pad=8)
                if col == 0:
                    ax.text(-0.08, 0.5, f'Day {day:g}', transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=11)
                if model != 'ERA5':
                    rmse = maps.weighted_rmse(field, fields['ERA5'], weights)
                    ax.text(0.5, 0.97, f'RMSE = {rmse:.4g}', transform=ax.transAxes, ha='center', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 2})
                    rmse_rows.append((run_tag, channel, model, day, day * 24, rmse))
        color_ax = fig.add_subplot(left[:, 3])
        colorbar = fig.colorbar(im, cax=color_ax)
        colorbar.set_label(units, fontsize=10, labelpad=8)
        colorbar.ax.tick_params(labelsize=9)
        spectral = fig.add_subplot(right[0])
        total = fig.add_subplot(right[1])
        for s in series:
            color = energy.MODEL_STYLES[s.model][1]
            for day, style in zip(days, energy.LEAD_STYLES):
                idx = np.flatnonzero(s.lead_hours == day * 24)[0]
                spectral.plot(energy.EFFECTIVE_LENGTH_KM / s.k, s.ratio[idx], color=color, linestyle=style, linewidth=1.6)
            total.plot(s.lead_hours / 24, s.total_ratio, color=color, linewidth=1.8)
        for ax in (spectral, total):
            ax.axhline(1, color='0.4', linestyle=(0, (2, 3)), linewidth=0.9)
            ax.grid(alpha=0.22)
            ax.set_axisbelow(True)
        spectral.set_title(f'(b) {spectral_channel}: annular power ratio', loc='left', fontsize=11)
        spectral.set_xlabel('Nominal wavelength (km; 28 km grid)')
        spectral.set_ylabel('Forecast / ERA5')
        spectral.set_xscale('log')
        spectral.set_xlim(energy.EFFECTIVE_LENGTH_KM, energy.EFFECTIVE_LENGTH_KM / 87)
        spectral.xaxis.set_major_locator(FixedLocator([4000, 2000, 1000, 500, 200, 100]))
        spectral.xaxis.set_major_formatter(ScalarFormatter())
        spectral.set_ylim(*energy.ratio_limits([s.ratio[np.isin(s.lead_hours, np.array(days) * 24)] for s in series], include_zero=True))
        total.set_title(f'(c) {spectral_channel}: band power ratio (nominal <1000 km)', loc='left', fontsize=11)
        total.set_xlabel('Forecast lead time (days)')
        total.set_ylabel('Band power ratio (nominal <1000 km)')
        total.set_xlim(0, 30)
        total.set_xticks(np.arange(0, 31, 5))
        total.set_ylim(*energy.ratio_limits([s.total_ratio for s in series]))
        fig.suptitle(f'{map_title} and {spectral_channel} power ratios', fontsize=16, y=0.965)
        fig.text(0.055, 0.887, f'(a) {map_title} | Single case: {run_tag} UTC', fontsize=11)
        handles = [Line2D([], [], color=energy.MODEL_STYLES[s.model][1], label=s.model, linewidth=2) for s in series]
        handles += [Line2D([], [], color='0.3', linestyle=style, label=f'Day {day:g}') for day, style in zip(days, energy.LEAD_STYLES)]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles), frameon=False, fontsize=10)
        for extension in ('png', 'pdf'):
            fig.savefig(stem.with_suffix(f'.{extension}'), dpi=400, facecolor='white')
        plt.close(fig)
    write_csv(Path(f'{stem}_map_rmse.csv'), ('run_tag', 'map_channel', 'model', 'lead_days', 'lead_hours', 'weighted_rmse'), rmse_rows)
    spectral_rows, total_rows = ([], [])
    for s in series:
        for i, hours in enumerate(s.lead_hours):
            for j, k in enumerate(s.k):
                ratio = s.ratio[i, j]
                spectral_rows.append((s.model, s.channel, int(hours), hours / 24, int(k), energy.EFFECTIVE_LENGTH_KM / k, s.forecast[i, j], s.target[i, j], ratio if np.isfinite(ratio) else ''))
            total_rows.append((s.model, s.channel, int(hours), hours / 24, s.forecast_total[i], s.target_total[i], s.total_ratio[i], 1000, '<'))
    write_csv(Path(f'{stem}_spectral.csv'), ('model', 'channel', 'lead_hours', 'lead_days', 'k', 'wavelength_km', 'forecast_annular_mean_power', 'target_annular_mean_power', 'power_ratio'), spectral_rows)
    write_csv(Path(f'{stem}_totals.csv'), ('model', 'channel', 'lead_hours', 'lead_days', 'forecast_band_mean_square', 'target_band_mean_square', 'band_power_ratio', 'wavelength_cutoff_km', 'wavelength_comparison'), total_rows)
    return stem

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', default='2024102100')
    parser.add_argument('--days', type=float, nargs='+', default=[9.0, 18.0, 27.0])
    parser.add_argument('--output-dir', type=Path, default=BASE / 'monthly_spatial_spectrum_outputs')
    args = parser.parse_args()
    datetime.strptime(args.run_tag, '%Y%m%d%H')
    if not 1 <= len(args.days) <= len(energy.LEAD_STYLES) or args.days != sorted(set(args.days)) or any((not np.isfinite(d) or d <= 0 or d > 30 or (not (d * 2).is_integer()) for d in args.days)):
        parser.error('Use 1-4 increasing unique days in (0,30], in half-day increments.')
    datasets = energy.load_period('30d')
    maps.LAT_PATH, maps.LON_PATH = (MAP_BASE / 'lat.npy', MAP_BASE / 'lon.npy')
    lat, lon = maps.load_lat_lon()
    if lat is None or lon is None or lat.shape != (192, 192) or (lon.shape != (192, 192)):
        raise ValueError('Matching 192x192 latitude/longitude grids are required')
    context = maps.setup_cartopy(lat, lon)
    if context is None:
        raise RuntimeError('Cartopy is required for the coastline maps')
    extent = maps.full_extent(lat, lon)
    prepared = []
    for config in FIGURES:
        rows, paths = read_maps(args.run_tag, args.days, config[1])
        series = [s for s in datasets if s.channel == config[2]]
        for s in series:
            for day in args.days:
                if np.count_nonzero(s.lead_hours == day * 24) != 1:
                    raise ValueError(f'{s.model}/{s.channel}: missing day {day}')
        prepared.append((config, rows, paths, series))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'run_tag': args.run_tag, 'days': args.days, 'steps': [int(d * 4) for d in args.days], 'extent': extent, 'spectrum_population': 'existing aggregate statistics (not the single map case)', 'spectral_sources': [str(energy.DATA_DIR / 'ours' / f'per_channel_energy_2024_fix_attention_step120_{suffix}.npz') for suffix in ('spectrum', 'spectrum_ver2')], 'feature_table': str(energy.DATA_DIR / 'next_feature_table_weight.csv'), 'coordinate_sources': [str(maps.LAT_PATH), str(maps.LON_PATH)], 'effective_length_km': energy.EFFECTIVE_LENGTH_KM, 'fft_shape': [174, 174], 'band_aggregation': 'sum(N_k * annular_mean)/(174*174)^2', 'scale_interpretation': 'nominal 28 km isotropic index grid; not projected wavelength', 'exclude_dc': True, 'energy_wavelength_cutoff_km': 1000, 'energy_wavelength_comparison': '<', 'rmse_weights': 'cos(latitude), zero boundary at min distance <=8 pixels, normalized mean', 'model_styles': {s: energy.MODEL_STYLES[s][1] for s in MODEL_ROOTS}, 'figures': []}
    for config, rows, paths, series in prepared:
        stem = render(config, rows, series, context, extent, args.days, args.run_tag, args.output_dir)
        manifest['figures'].append({'name': config[0], 'map_channel': config[1], 'spectral_channel': config[2], 'map_limits': [0, config[5]], 'map_sources': paths, 'outputs': [str(stem.with_suffix(f'.{ext}')) for ext in ('png', 'pdf')] + [f'{stem}_{kind}.csv' for kind in ('map_rmse', 'spectral', 'totals')]})
        print(f'Saved: {stem}', flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
