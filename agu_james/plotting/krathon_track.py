# Derived from case_study and monthly simulation/plot_krathon_track_intensity.py
"""Reproduce the Krathon case study from tracker_multi CSVs.

Run with the project interpreter (pandas, matplotlib, numpy, scipy and cartopy):
    python "case_study and monthly simulation/plot_krathon_track_intensity.py"
Paths are resolved relative to this file, independently of the working directory.
Outputs: vector PDF, 300 dpi PNG, raw paired/best-track CSVs and smoothed CSV.
"""
from agu_james import runtime
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as dates
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
HERE = runtime.setting('agu_james.plotting.krathon_track.HERE', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.krathon_track')
DATASETS = {'Pangu': 'pangu/tc_aurora_metrics35_step28.csv', 'Spectrum MAE': 'ours/tc_aurora_intensity35_step28_spectrum.csv', 'Spectrum MSE': 'ours/tc_aurora_intensity35_step28_spectrum_ver2.csv', 'MSE': 'ours/tc_aurora_intensity35_step28_auto.csv'}
KEYS = ['CYCLONE_NUM', 'forecast_init_time', 'time', 'step']
VALUES = ['bst_lon', 'bst_lat', 'forecast_center_lon', 'forecast_center_lat', 'forecast_msl_min_hPa', 'forecast_s10_max', 'target_msl_min_hPa', 'target_s10_max']
CADENCE = pd.Timedelta(hours=12)
SMOOTHING_SIGMA_HOURS = 12.0
KNOT_TO_MPS = 0.514444
MODEL_COLORS = {'Pangu': '#2878b5', 'Spectrum MAE': '#d97716', 'Spectrum MSE': '#289458', 'MSE': '#8553ad'}

def smooth_segment(part, columns):
    """Gaussian time smoothing (sigma=12 h), then hourly PCHIP sampling.

    Separate contiguous segments avoid bridging gaps. Endpoints are fixed to
    raw values; one/two-point segments use raw/linear interpolation.
    """
    hours = (part.valid_time_utc - part.valid_time_utc.iloc[0]).dt.total_seconds().to_numpy() / 3600
    dense = np.arange(0, hours[-1] + 0.5, 1.0)
    result = pd.DataFrame({'valid_time_utc': part.valid_time_utc.iloc[0] + pd.to_timedelta(dense, unit='h')})
    for column in columns:
        raw = part[column].to_numpy(dtype=float)
        if len(part) >= 3:
            filtered = gaussian_filter1d(raw, sigma=SMOOTHING_SIGMA_HOURS / np.median(np.diff(hours)), mode='nearest', truncate=2.0)
            filtered[[0, -1]] = raw[[0, -1]]
            result[column] = PchipInterpolator(hours, filtered)(dense)
        else:
            result[column] = np.interp(dense, hours, raw)
    return result

def td_mask(wind, pressure):
    return (wind < 17.2) & (pressure > 1002.0)

def segments(frame):
    """Keep time gaps larger than CADENCE disconnected."""
    frame = frame.sort_values('valid_time_utc')
    groups = frame.valid_time_utc.diff().gt(CADENCE).cumsum()
    return [part for _, part in frame.groupby(groups)]

def load_cases(root):
    frames = {}
    for model, relative in DATASETS.items():
        source = root / relative
        frame = pd.read_csv(source).rename(columns={'cyclone_num': 'CYCLONE_NUM', 'init_time': 'forecast_init_time', 'forecast_lon': 'forecast_center_lon', 'forecast_lat': 'forecast_center_lat'})
        frame = frame.loc[frame.CYCLONE_NUM.eq(20), KEYS + VALUES].copy()
        if frame.duplicated(KEYS).any():
            raise ValueError(f'{model}: duplicate case keys')
        for column in KEYS + VALUES:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        frame = frame.dropna(subset=KEYS + VALUES)
        frame[KEYS] = frame[KEYS].astype('int64')
        frame['source_csv'] = relative
        frame['model'] = model
        frames[model] = frame
    common = None
    for frame in frames.values():
        keys = frame[KEYS]
        common = keys if common is None else common.merge(keys, on=KEYS, validate='one_to_one')
    if common is None or common.empty:
        raise ValueError('No complete common Krathon cases')
    paired = pd.concat([frame.merge(common, on=KEYS, validate='one_to_one') for frame in frames.values()], ignore_index=True)
    paired['valid_time_utc'] = pd.to_datetime(paired.time.astype(str), format='%Y%m%d%H', utc=True)
    paired['initialization_time_utc'] = pd.to_datetime(paired.forecast_init_time.astype(str), format='%Y%m%d%H', utc=True)
    elapsed = (paired.valid_time_utc - paired.initialization_time_utc).dt.total_seconds() / 3600
    if not np.array_equal(elapsed.to_numpy(), paired.step.to_numpy() * 6):
        raise ValueError('Time/step mismatch: expected six hours per step')
    paired['forecast_td'] = td_mask(paired.forecast_s10_max, paired.forecast_msl_min_hPa)
    paired['era5_td'] = td_mask(paired.target_s10_max, paired.target_msl_min_hPa)
    for model, frame in paired.groupby('model'):
        spread = frame.groupby('time')[['target_msl_min_hPa', 'target_s10_max']].agg(lambda x: x.max() - x.min())
        if spread.to_numpy().max() > 0.001:
            raise ValueError(f'{model}: inconsistent ERA5 values at the same valid time')
    best = pd.read_csv(root / 'cyclone_data_2024.csv')
    best = best.loc[best.CYCLONE_NUM.eq(20) & best.TECH.eq('BEST')].copy()
    best['valid_time_utc'] = pd.to_datetime(best.DATE_TIME.astype(str), format='%Y%m%d%H', utc=True)
    best = best[best.valid_time_utc.isin(paired.valid_time_utc.unique())]
    best = best.dropna(subset=['LON', 'LAT']).sort_values('valid_time_utc')
    if best.valid_time_utc.duplicated().any() or best.empty:
        raise ValueError('Missing or duplicate best-track positions')
    return (paired, best)

def point_colors(frame, cmap, norm):
    """Use valid time only, independently of forecast initialization."""
    values = np.asarray(dates.date2num(frame.valid_time_utc.to_numpy()), dtype=float)
    return cmap(norm(values))

def mark_reference_ends(ax, frame, x, y, transform=None):
    """Mark chronological endpoints at their original reference coordinates."""
    ordered = frame.sort_values('valid_time_utc')
    kwargs = {} if transform is None else {'transform': transform}
    for row, marker, size in [(ordered.iloc[0], 'o', 30), (ordered.iloc[-1], 'X', 34)]:
        ax.scatter([row[x]], [row[y]], marker=marker, s=size, facecolors='white' if marker == 'o' else 'black', edgecolors='black', linewidths=1.0, zorder=9, **kwargs)

def load_ibtracs(paired, path):
    """Read the USA-only source subset without requiring unused WMO fields."""
    frame = pd.read_csv(path, dtype=str).apply(lambda s: s.str.strip())
    required = {'NAME', 'SEASON', 'TRACK_TYPE', 'SID', 'ISO_TIME', 'USA_LAT', 'USA_LON', 'USA_WIND', 'USA_PRES'}
    if missing := required - set(frame.columns):
        raise ValueError(f'Missing IBTrACS USA columns: {sorted(missing)}')
    frame = frame[frame.NAME.str.upper().eq('KRATHON') & frame.SEASON.eq('2024') & frame.TRACK_TYPE.eq('main')].copy()
    if frame.SID.nunique() != 1:
        raise ValueError('Expected exactly one Krathon 2024 SID in IBTrACS')
    frame['valid_time_utc'] = pd.to_datetime(frame.ISO_TIME, utc=True, errors='raise')
    frame = frame[frame.valid_time_utc.between(paired.valid_time_utc.min(), paired.valid_time_utc.max())]
    for column in ['USA_LAT', 'USA_LON', 'USA_WIND', 'USA_PRES']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
        if column.endswith(('WIND', 'PRES')):
            frame.loc[frame[column] <= 0, column] = np.nan
    frame['USA_wind_mps'] = frame['USA_WIND'] * KNOT_TO_MPS
    frame = frame.dropna(subset=['USA_LAT', 'USA_LON']).sort_values('valid_time_utc')
    if frame.empty or frame.valid_time_utc.duplicated().any():
        raise ValueError('Missing or duplicate IBTrACS positions')
    frame['forecast_td'] = False
    frame['source_csv'] = str(Path(path).resolve())
    return frame

def comparison_cases(paired, ibtracs):
    era5 = paired[paired.model.eq('MSE')].drop_duplicates('valid_time_utc')
    columns = ['valid_time_utc', 'target_msl_min_hPa', 'target_s10_max']
    usa = ibtracs.dropna(subset=['USA_PRES', 'USA_wind_mps'])
    return usa.merge(era5[columns], on='valid_time_utc', how='inner', validate='one_to_one').sort_values('valid_time_utc')

def draw_curve(ax, frame, x, y, color=None, transform=None, reference=False, cmap=None, norm=None, linestyle='-', marker_frame=None):
    kwargs = {} if transform is None else {'transform': transform}
    for part in segments(frame):
        smooth = smooth_segment(part, [x, y])
        if reference:
            ax.plot(smooth[x], smooth[y], color=color, lw=1.25, ls='--', zorder=4, **kwargs)
        elif len(smooth) > 1:
            points = smooth[[x, y]].to_numpy()
            lines = np.stack([points[:-1], points[1:]], axis=1)
            times = dates.date2num(smooth.valid_time_utc.to_numpy())
            collection = LineCollection(lines, cmap=cmap, norm=norm, linewidths=0.85, alpha=0.78, zorder=2, linestyles=linestyle, **kwargs)
            collection.set_array((times[:-1] + times[1:]) / 2)
            ax.add_collection(collection)
    if reference:
        mark_reference_ends(ax, frame, x, y, transform)
        return
    shown = frame if marker_frame is None else marker_frame
    ax.scatter(shown[x], shown[y], s=12 if marker_frame is not None else 5, c=point_colors(shown, cmap, norm), zorder=3, **kwargs)
    for row, marker, size in [(frame.iloc[0], 'o', 22), (frame.iloc[-1], 'X', 26)]:
        color = cmap(norm(dates.date2num(row.valid_time_utc)))
        ax.scatter([row[x]], [row[y]], marker=marker, s=size, facecolors='white' if marker == 'o' else color, edgecolors=color, linewidths=0.8, zorder=5, **kwargs)
    td = frame[frame.forecast_td]
    ax.scatter(td[x], td[y], marker='v', s=28, facecolors='white', edgecolors=point_colors(td, cmap, norm), linewidths=0.9, zorder=6, **kwargs)

def annotate_comparison_times(ax, sampled):
    """Place daily 00 UTC labels near raw points, with leaders in crowded regions."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    region = ax.get_window_extent(renderer)
    occupied = []
    points = []
    for x, y in [('USA_PRES', 'USA_wind_mps'), ('target_msl_min_hPa', 'target_s10_max')]:
        for _, row in sampled.iterrows():
            px, py = ax.transData.transform((row[x], row[y]))
            points.append(Bbox.from_extents(px - 3, py - 3, px + 3, py + 3))
    for x, y, direction in [('USA_PRES', 'USA_wind_mps', 1), ('target_msl_min_hPa', 'target_s10_max', -1)]:
        for _, row in sampled.iterrows():
            label = ax.annotate(row.valid_time_utc.strftime('%m-%d\n%HZ'), xy=(row[x], row[y]), xytext=(0, direction * 12), textcoords='offset points', ha='center', va='center', fontsize=6.5, color='0.15', zorder=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.3), arrowprops=dict(arrowstyle='-', color='0.55', lw=0.4))
            candidates = [(dx, sign * dy) for sign in [direction, -direction] for dy in [12, 24, 36, 48, 60] for dx in [0, -20, 20, -40, 40, -60, 60]]
            best, best_score = (None, float('inf'))
            for offset in candidates:
                label.set_position(offset)
                label.update_positions(renderer)
                box = Text.get_window_extent(label, renderer).expanded(1.1, 1.15)
                inside = region.contains(box.x0, box.y0) and region.contains(box.x1, box.y1)
                score = (0 if inside else 1000) + 100 * sum((box.overlaps(b) for b in occupied))
                score += 10 * sum((box.overlaps(b) for b in points))
                if score < best_score:
                    best, best_score = (offset, score)
                if score == 0:
                    break
            label.set_position(best)
            label.update_positions(renderer)
            occupied.append(Text.get_window_extent(label, renderer).expanded(1.1, 1.15))

def export_smoothed(paired, best, output, ibtracs):
    tables = []
    forecast_columns = ['forecast_center_lon', 'forecast_center_lat', 'forecast_msl_min_hPa', 'forecast_s10_max']
    reference_columns = ['target_msl_min_hPa', 'target_s10_max']
    for model in DATASETS:
        frame = paired[paired.model.eq(model)]
        series = [('forecast', init, track, forecast_columns) for init, track in frame.groupby('forecast_init_time')]
        series.append(('ERA5', None, frame.drop_duplicates('time'), reference_columns))
        for kind, init, track, columns in series:
            for segment_id, part in enumerate(segments(track)):
                smooth = smooth_segment(part, columns)
                smooth['model'] = model
                smooth['series_type'] = kind
                if init is not None:
                    smooth['forecast_init_time'] = init
                smooth['segment_id'] = segment_id
                tables.append(smooth)
    for segment_id, part in enumerate(segments(best)):
        smooth = smooth_segment(part, ['LON', 'LAT'])
        smooth['model'] = 'Best track'
        smooth['series_type'] = 'best_track'
        smooth['segment_id'] = segment_id
        tables.append(smooth)
    for kind, columns in [('IBTrACS USA', ['USA_PRES', 'USA_wind_mps'])]:
        for segment_id, part in enumerate(segments(ibtracs.dropna(subset=columns))):
            smooth = smooth_segment(part, columns)
            smooth['model'] = 'IBTrACS'
            smooth['series_type'] = kind
            smooth['segment_id'] = segment_id
            tables.append(smooth)
    table = pd.concat(tables, ignore_index=True)
    table['forecast_init_time'] = table.forecast_init_time.astype('Int64')
    table['smoothing_sigma_hours'] = SMOOTHING_SIGMA_HOURS
    table.to_csv(output / 'krathon_track_intensity_smoothed.csv', index=False)

def plot(paired, best, output, ibtracs):
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    cmaps = {}
    for model, base in MODEL_COLORS.items():
        rgb = np.array(colors.to_rgb(base))
        cmaps[model] = colors.LinearSegmentedColormap.from_list(model, [rgb * 0.45 + 0.55, rgb * 0.78])
    numbers = dates.date2num(paired.valid_time_utc.to_numpy())
    norm = colors.Normalize(numbers.min(), numbers.max() if numbers.max() > numbers.min() else numbers.min() + 0.25)
    fig = plt.figure(figsize=(9, 14.5))
    grid = fig.add_gridspec(5, 2, left=0.075, right=0.975, bottom=0.1, top=0.88, hspace=0.36, wspace=0.17, width_ratios=[0.45, 1.55], height_ratios=[1, 1, 1, 1, 1.35])
    geo = ccrs.PlateCarree()
    lon = np.r_[paired.forecast_center_lon, best.LON]
    lat = np.r_[paired.forecast_center_lat, best.LAT]
    extent = [np.floor(lon.min() - 0.7), np.ceil(lon.max() + 0.7), np.floor(lat.min() - 0.7), np.ceil(lat.max() + 0.7)]
    pressure = np.r_[paired.forecast_msl_min_hPa, paired.target_msl_min_hPa]
    wind = np.r_[paired.forecast_s10_max, paired.target_s10_max]
    xlim = (5 * np.floor((pressure.min() - 2) / 5), 5 * np.ceil((pressure.max() + 2) / 5))
    ylim = (2 * np.floor((wind.min() - 1) / 2), 2 * np.ceil((wind.max() + 1) / 2))
    for index, model in enumerate(DATASETS):
        cmap = cmaps[model]
        frame = paired[paired.model.eq(model)].sort_values(['forecast_init_time', 'time'])
        map_ax = fig.add_subplot(grid[index, 0], projection=geo)
        map_ax.set_extent(extent, crs=geo)
        map_ax.set_aspect('equal')
        map_ax.set_anchor('E')
        if index == 0:
            first_map_ax = map_ax
        map_ax.coastlines(resolution='50m', color='0.5', linewidth=0.5)
        gl = map_ax.gridlines(draw_labels=True, linewidth=0.35, color='0.8', alpha=0.7, xlocs=np.arange(118, 133, 4), ylocs=np.arange(16, 33, 2))
        gl.top_labels = gl.right_labels = False
        gl.bottom_labels = index == 3
        gl.xlabel_style = gl.ylabel_style = {'size': 8}
        intensity_ax = fig.add_subplot(grid[index, 1])
        intensity_ax.set(xlim=xlim, ylim=ylim)
        intensity_ax.grid(alpha=0.25, lw=0.4)
        intensity_ax.set_ylabel('S10 max (m/s)')
        if index == 3:
            intensity_ax.set_xlabel('MSL min (hPa)')
            map_ax.set_xlabel('Longitude')
        else:
            intensity_ax.tick_params(labelbottom=False)
        for _, track in frame.groupby('forecast_init_time', sort=True):
            draw_curve(map_ax, track, 'forecast_center_lon', 'forecast_center_lat', transform=geo, cmap=cmap, norm=norm)
            draw_curve(intensity_ax, track, 'forecast_msl_min_hPa', 'forecast_s10_max', cmap=cmap, norm=norm)
        for part in segments(best):
            smooth = smooth_segment(part, ['LON', 'LAT'])
            map_ax.plot(smooth.LON, smooth.LAT, color='black', lw=1.35, transform=geo, zorder=7)
        mark_reference_ends(map_ax, best, 'LON', 'LAT', geo)
        reference = frame.drop_duplicates('time').sort_values('time')
        draw_curve(intensity_ax, reference, 'target_msl_min_hPa', 'target_s10_max', 'black', reference=True)
        td = reference[reference.era5_td]
        intensity_ax.scatter(td.target_msl_min_hPa, td.target_s10_max, marker='s', s=24, facecolors='white', edgecolors='black', zorder=7)
        for column, ax in enumerate([map_ax, intensity_ax]):
            ax.set_title(f'({chr(97 + 2 * index + column)}) {model}', loc='left', pad=5)
    gray_cmap = colors.LinearSegmentedColormap.from_list('valid_time_gray', ['0.78', '0.18'])
    intensity_ax = fig.add_subplot(grid[4, :])
    fig.canvas.draw()
    bounds = intensity_ax.get_position()
    left_edge = first_map_ax.get_position().x0
    intensity_ax.set_position([left_edge, bounds.y0, bounds.x1 - left_edge, bounds.height])
    comparison = comparison_cases(paired, ibtracs)
    sampled = comparison[comparison.valid_time_utc.dt.hour.eq(0)]
    pressures, winds = ([], [])
    for agency, linestyle in [('USA', '-')]:
        columns = [f'{agency}_PRES', f'{agency}_wind_mps']
        frame = comparison
        if frame.empty:
            raise ValueError(f'IBTrACS {agency}: no paired wind/pressure values')
        draw_curve(intensity_ax, frame, *columns, cmap=gray_cmap, norm=norm, linestyle=linestyle, marker_frame=sampled)
        mark_reference_ends(intensity_ax, frame, *columns)
        pressures.extend(frame[columns[0]])
        winds.extend(frame[columns[1]])
    draw_curve(intensity_ax, comparison, 'target_msl_min_hPa', 'target_s10_max', color='black', reference=True)
    intensity_ax.scatter(sampled.target_msl_min_hPa, sampled.target_s10_max, color='black', s=12, zorder=4)
    pressures.extend(comparison.target_msl_min_hPa)
    winds.extend(comparison.target_s10_max)
    intensity_ax.set(xlim=(5 * np.floor((min(pressures) - 3) / 5), 5 * np.ceil((max(pressures) + 3) / 5)), ylim=(5 * np.floor((min(winds) - 2) / 5), 5 * np.ceil((max(winds) + 2) / 5)), xlabel='Minimum pressure (hPa)', ylabel='Maximum wind (m/s)')
    intensity_ax.grid(alpha=0.25, lw=0.4)
    intensity_ax.set_title('(i) USA (JTWC) and ERA5', loc='left', pad=5)
    intensity_ax.legend(handles=[Line2D([], [], color='0.25', ls='-', label='USA (JTWC)'), Line2D([], [], color='black', ls='--', label='ERA5')], loc='upper right', fontsize=8, frameon=False)
    annotate_comparison_times(intensity_ax, sampled)
    fig.suptitle('Typhoon Krathon: track and intensity evolution', y=0.982, fontsize=15)
    fig.text(0.18, 0.905, 'Track', ha='center', fontsize=11, weight='bold')
    fig.text(0.655, 0.905, 'Pressure-wind evolution', ha='center', fontsize=11, weight='bold')
    handles = [Line2D([], [], color='black', lw=1.3, label='Best track (map)'), Line2D([], [], color='black', ls='--', lw=1.3, label='ERA5 (intensity)'), Line2D([], [], color='0.4', marker='o', mfc='white', ls='none', label='Start (all curves)'), Line2D([], [], color='0.4', marker='X', ls='none', label='End (all curves)'), Line2D([], [], color='0.4', marker='v', mfc='white', ls='none', label='Forecast TD'), Line2D([], [], color='black', marker='s', mfc='white', ls='none', label='ERA5 TD')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.956), ncol=3, fontsize=11, markerscale=1.3, frameon=False, columnspacing=1.3, handlelength=2, labelspacing=0.6)
    ticks = pd.date_range(paired.valid_time_utc.min().ceil('D'), paired.valid_time_utc.max().floor('D'), freq='D')
    ticks = ticks.union(pd.DatetimeIndex([paired.valid_time_utc.min()]))
    cax = fig.add_axes([0.2, 0.041, 0.72, 0.012])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=gray_cmap), cax=cax, orientation='horizontal')
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_ticks(dates.date2num(ticks.to_numpy()))
    bar.ax.tick_params(labelsize=8, length=2)
    bar.set_ticklabels([t.strftime('%m-%d %HZ') for t in ticks])
    bar.set_label('Valid time (UTC)')
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'krathon_track_intensity.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker-root', type=Path, default=HERE.parent / 'tracker_multi')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--ibtracs-path', type=Path, default=HERE / 'krathon_ibtracs_source.csv', help='Krathon subset or original IBTrACS CSV (includes units row).')
    args = parser.parse_args()
    paired, best = load_cases(args.tracker_root)
    ibtracs = load_ibtracs(paired, args.ibtracs_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / 'krathon_track_intensity.csv', index=False)
    best.to_csv(args.output_dir / 'krathon_track_intensity_best_track.csv', index=False)
    ibtracs.drop(columns='forecast_td').to_csv(args.output_dir / 'krathon_track_intensity_ibtracs.csv', index=False)
    comparison = comparison_cases(paired, ibtracs)
    comparison[['valid_time_utc', 'USA_WIND', 'USA_wind_mps', 'USA_PRES', 'target_s10_max', 'target_msl_min_hPa']].to_csv(args.output_dir / 'krathon_usa_era5_comparison.csv', index=False)
    export_smoothed(paired, best, args.output_dir, ibtracs)
    plot(paired, best, args.output_dir, ibtracs)
    print(paired.groupby('model', sort=False).agg(cases=('time', 'size'), initializations=('forecast_init_time', 'nunique')).to_string())
    print(f'Valid time: {paired.valid_time_utc.min()} to {paired.valid_time_utc.max()}')
    print(f'Saved PNG/PDF/CSV to {args.output_dir}')
