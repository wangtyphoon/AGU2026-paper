# Derived from case_study and monthly simulation/plot_krathon_wind850_q700.py
# See provenance/sources.json for original hash and modifications.
"""Plot Krathon wind850/q700 around the valid-time best-track centers.

Run from any working directory with a Python providing numpy, matplotlib and
cartopy. Outputs are a 300 dpi PNG and PDF beside this script. Data conventions
follow vis_multi_steps/vis_multi_model_forecast_compare_all.py.
"""
from agu_james import runtime
from pathlib import Path
from datetime import datetime, timedelta
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import numpy as np
HERE = runtime.setting('agu_james.plotting.krathon_fields.HERE', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.krathon_fields')
ROOT = runtime.setting('agu_james.plotting.krathon_fields.ROOT', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.krathon_fields')
DATA = runtime.setting('agu_james.plotting.krathon_fields.DATA', lambda: ROOT / 'vis_multi_steps')
BEST_TRACK = runtime.setting('agu_james.plotting.krathon_fields.BEST_TRACK', lambda: ROOT / 'tracker_multi' / 'cyclone_data_2024.csv')
INIT = datetime(2024, 9, 29, 12)
LEADS = (24, 48, 72)
HALF_WIDTH_DEG = 5.0
MODEL_ROOTS = (('Spectrum MAE', 'ours_spectrum'), ('Spectrum MSE', 'ours_spectrum_ver2'), ('MSE', 'ours'))
LABELS = ('Pangu', 'Spectrum MAE', 'Spectrum MSE', 'MSE', 'ERA5')
VARIABLES = (('wind850', 'wind_speed_086_087', '850 hPa wind', 50.0, 'Wind speed (m s$^{-1}$)'), ('q700', 'channel_076', '700 hPa humidity', 0.015, 'Specific humidity (kg kg$^{-1}$)'))

def load_grid():
    lat = np.load(DATA / 'lat.npy', allow_pickle=False)
    lon = np.load(DATA / 'lon.npy', allow_pickle=False)
    if lat.ndim == lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != (192, 192) or lon.shape != lat.shape:
        raise ValueError(f'Expected 192 x 192 coordinates, got {lat.shape}, {lon.shape}')
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError('Nonfinite grid coordinates')
    if not (np.allclose(lat, lat[:, :1]) and np.allclose(lon, lon[:1, :])):
        raise ValueError('Expected a rectilinear latitude/longitude grid')
    if not (np.all(np.diff(lat[:, 0]) < 0) and np.all(np.diff(lon[0]) > 0)):
        raise ValueError('Expected latitude descending and longitude ascending')
    if not (np.allclose(np.diff(lat[:, 0]), -0.25) and np.allclose(np.diff(lon[0]), 0.25)):
        raise ValueError('Expected the source 0.25 degree grid')
    return (lat, lon)

def load_centers():
    with BEST_TRACK.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'BASIN', 'CYCLONE_NUM', 'TECH', 'DATE_TIME', 'LON', 'LAT'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing best-track columns: {required - set(reader.fieldnames or [])}')
        rows = [row for row in reader if row['BASIN'] == 'WP' and int(row['CYCLONE_NUM']) == 20 and (row['TECH'] == 'BEST')]
    centers = {}
    for lead in LEADS:
        stamp = (INIT + timedelta(hours=lead)).strftime('%Y%m%d%H')
        matches = [row for row in rows if row['DATE_TIME'] == stamp]
        if len(matches) != 1:
            raise ValueError(f'Expected one Krathon BEST center at {stamp}, found {len(matches)}')
        lon, lat = (float(matches[0]['LON']), float(matches[0]['LAT']))
        if not (np.isfinite([lon, lat]).all() and -180 <= lon <= 180 and (-90 <= lat <= 90)):
            raise ValueError(f'Invalid best-track center at {stamp}: {lon}, {lat}')
        centers[lead] = (lon, lat)
    return centers

def load_field(path, key):
    if not path.is_file():
        raise FileNotFoundError(f'Missing field file: {path}')
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f'Missing {key!r} in {path}')
        field = np.asarray(archive[key], dtype=float)
    if field.shape != (192, 192):
        raise ValueError(f'Expected 192 x 192 field, got {field.shape}: {path}')
    if not np.isfinite(field).any():
        raise ValueError(f'No finite field values: {path}')
    return field

def load_panels(var, channel, lead):
    valid = INIT + timedelta(hours=lead)
    pangu = DATA / 'pangu' / INIT.strftime('%Y-%m-%d-%H-%M') / valid.strftime('%Y-%m-%d-%H-%M') / f'{var}.npz'
    panels = {'Pangu': load_field(pangu, 'forecast')}
    for label, directory in MODEL_ROOTS:
        path = DATA / directory / INIT.strftime('%Y%m%d%H') / f'step{lead // 6:02d}' / f'{channel}.npz'
        panels[label] = load_field(path, 'forecast')
    panels['ERA5'] = load_field(pangu, 'truth')
    return panels

def crop_grid(lat, lon, center):
    cx, cy = center
    bounds = (cx - HALF_WIDTH_DEG, cx + HALF_WIDTH_DEG, cy - HALF_WIDTH_DEG, cy + HALF_WIDTH_DEG)
    west, east, south, north = bounds
    if west < lon.min() or east > lon.max() or south < lat.min() or (north > lat.max()):
        raise ValueError(f'Crop exceeds source grid: {bounds}')
    ys = np.flatnonzero((lat[:, 0] >= south - 0.25) & (lat[:, 0] <= north + 0.25))
    xs = np.flatnonzero((lon[0] >= west - 0.25) & (lon[0] <= east + 0.25))
    crop = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    return (crop, bounds)

def main():
    lat, lon = load_grid()
    centers = load_centers()
    fields = {(var, lead): load_panels(var, channel, lead) for var, channel, *_ in VARIABLES for lead in LEADS}
    crops = {lead: crop_grid(lat, lon, centers[lead]) for lead in LEADS}
    projection = ccrs.PlateCarree()
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(10.8, 12.8))
    outer = fig.add_gridspec(2, 1, left=0.125, right=0.875, bottom=0.035, top=0.94, hspace=0.16)
    for group, (var, channel, title, maximum, unit) in enumerate(VARIABLES):
        grid = outer[group].subgridspec(3, 5, wspace=0.045, hspace=0.065)
        axes = []
        norm = Normalize(0.0, maximum)
        for row, lead in enumerate(LEADS):
            crop, bounds = crops[lead]
            valid = INIT + timedelta(hours=lead)
            for col, label in enumerate(LABELS):
                ax = fig.add_subplot(grid[row, col], projection=projection)
                axes.append(ax)
                mesh = ax.pcolormesh(lon[crop], lat[crop], fields[var, lead][label][crop], shading='nearest', cmap='jet', norm=norm, transform=projection, rasterized=True)
                ax.set_extent(bounds, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.55, color='0.2')
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(label, fontsize=10, pad=6)
                if col == 0:
                    ax.text(-0.07, 0.5, f'+{lead} h\n{valid:%b %d, %H UTC}', transform=ax.transAxes, ha='right', va='center', fontsize=9)
        top = axes[0].get_position().y1
        bottom = axes[-1].get_position().y0
        fig.text(0.125, top + 0.031, f'({chr(97 + group)}) {title}', fontsize=12, weight='bold')
        cax = fig.add_axes([0.898, bottom + 0.025, 0.016, top - bottom - 0.05])
        cbar = fig.colorbar(mesh, cax=cax, extend='max')
        cbar.set_label(unit, labelpad=9)
        if var == 'q700':
            cbar.set_ticks([0, 0.003, 0.006, 0.009, 0.012, 0.015])
        else:
            cbar.set_ticks([0, 10, 20, 30, 40, 50])
    for lead in LEADS:
        print(f'+{lead} h: BEST center={centers[lead]}, bounds={crops[lead][1]}')
    for suffix in ('png', 'pdf'):
        output = HERE / f'krathon_wind850_q700.{suffix}'
        fig.savefig(output, dpi=300, facecolor='white')
        print(f'Saved: {output}')
    plt.close(fig)
