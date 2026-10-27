# Derived from vis_multi_steps/vis_multi_model_forecast_compare_monthly_all.py
from __future__ import annotations
from agu_james import runtime
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Literal
import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('matplotlib is required. Install with pip install matplotlib.') from exc
OURS_SPECTRUM_MONTHLY_ROOT = runtime.setting('agu_james.plotting.monthly_maps.OURS_SPECTRUM_MONTHLY_ROOT', lambda: Path('ours_spectrum_monthly'))
OURS_SPECTRUM_VER2_MONTHLY_ROOT = runtime.setting('agu_james.plotting.monthly_maps.OURS_SPECTRUM_VER2_MONTHLY_ROOT', lambda: Path('ours_spectrum_ver2_monthly'))
RUN_TAG = '2024102100'
STEPS_TO_SAVE = [36, 72, 108]
OUTPUT_DIR = runtime.setting('agu_james.plotting.monthly_maps.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.monthly_maps')
LAT_PATH = runtime.setting('agu_james.plotting.monthly_maps.LAT_PATH', lambda: Path('lat.npy'))
LON_PATH = runtime.setting('agu_james.plotting.monthly_maps.LON_PATH', lambda: Path('lon.npy'))
USE_CARTOPY = True
COASTLINE_RESOLUTION = '50m'
LAND_COLOR = '0.9'
OCEAN_COLOR = '0.85'
COMPACT_DOMAIN_LAYOUT = True
PANEL_WIDTH = 1.7
PANEL_HEIGHT = 1.7
PANEL_WSPACE = 0.1
PANEL_HSPACE = 0.1
TITLE_FONT_SIZE = 14
TITLE_Y = 0.97
DEFAULT_XLIM = (102.25, 150.0)
DEFAULT_YLIM = (0.0, 47.75)
ScaleMode = Literal['global', 'fixed', 'symmetric', 'symmetric_percentile']
PRECIP_CMAP = ListedColormap(['#ffffff', '#cfefff', '#7ec8ff', '#1f78ff', '#22b14c', '#fff200', '#ff9f1c', '#e31a1c', '#b100b8'], name='weather_precip')

@dataclass(frozen=True)
class VarSpec:
    var_name: str
    channel_name: str
    cmap: str | Any
    scale_mode: ScaleMode
    colorbar_label: str
    value_range: tuple[float, float] | None = None
    percentile: float = 99.0
VAR_SPECS = [VarSpec('z500', 'channel_059', 'turbo', 'global', 'Geopotential height'), VarSpec('q700', 'channel_076', 'jet', 'fixed', 'Specific humidity', (0.0, 0.014)), VarSpec('t850', 'channel_085', 'turbo', 'global', 'Temperature'), VarSpec('u850', 'channel_086', 'RdBu_r', 'symmetric', 'U wind'), VarSpec('v850', 'channel_087', 'RdBu_r', 'symmetric', 'V wind'), VarSpec('wind850', 'wind_speed_086_087', 'jet', 'fixed', 'Wind speed', (0.0, 50.0)), VarSpec('500pv', 'channel_053', 'magma', 'symmetric_percentile', 'Potential vorticity'), VarSpec('700cc', 'channel_074', 'Blues', 'fixed', 'Cloud cover', (0.0, 1.0)), VarSpec('850w', 'channel_088', 'RdBu_r', 'fixed', 'Vertical velocity', (-1.0, 1.0)), VarSpec('tp', 'channel_115', PRECIP_CMAP, 'fixed', 'Total precipitation', (0.0, 0.005))]
MODEL_LABELS = ['Spectrum_MAE', 'Spectrum_MSE', 'ERA5']

def format_step_days(step_num: int) -> str:
    days = step_num * 6 / 24
    if days.is_integer():
        return f'Day {int(days)}'
    return f'Day {days:g}'

def wrap_figure_title(title: str, fig_width: float) -> str:
    max_chars = max(28, int(fig_width * 8.5))
    return '\n'.join(textwrap.wrap(title, width=max_chars, break_long_words=False))

def load_lat_lon() -> tuple[np.ndarray | None, np.ndarray | None]:
    if not (LAT_PATH.exists() and LON_PATH.exists()):
        return (None, None)
    lat = np.load(LAT_PATH)
    lon = np.load(LON_PATH)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return (lat2d, lon2d)
    if lat.ndim == 2 and lon.ndim == 2:
        return (lat, lon)
    print(f'Cannot use lat/lon: unexpected shapes lat={lat.shape}, lon={lon.shape}')
    return (None, None)

def create_spatial_weights_np(h: int=192, w: int=192, boundary: int=8) -> np.ndarray:
    lats_deg = np.linspace(0, 47.75, h)
    lat_w = np.cos(np.deg2rad(lats_deg))
    w2d = np.tile(lat_w[:, None], (1, w))
    w2d = np.flipud(w2d).copy()
    y_idx, x_idx = np.indices((h, w))
    min_dist = np.minimum.reduce([y_idx, h - 1 - y_idx, x_idx, w - 1 - x_idx])
    w2d[min_dist <= boundary] = 0.0
    w2d = w2d / np.mean(w2d)
    return w2d.astype(np.float32)

def discover_common_steps() -> list[tuple[str, int]]:
    a = OURS_SPECTRUM_MONTHLY_ROOT / RUN_TAG
    b = OURS_SPECTRUM_VER2_MONTHLY_ROOT / RUN_TAG
    a_steps = {p.name for p in a.glob('step*') if p.is_dir()}
    b_steps = {p.name for p in b.glob('step*') if p.is_dir()}
    wanted = {f'step{hours:02d}' for hours in STEPS_TO_SAVE}
    out = []
    for name in a_steps & b_steps & wanted:
        try:
            step_num = int(name.replace('step', ''))
        except ValueError:
            continue
        out.append((name, step_num))
    out.sort(key=lambda x: x[1])
    return out

def load_forecast_target(root: Path, run_tag: str, step_name: str, channel_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    file_path = root / run_tag / step_name / f'{channel_name}.npz'
    if not file_path.exists():
        return (None, None)
    with np.load(file_path) as data:
        return (np.asarray(data['forecast']), np.asarray(data['target']))

def collect_rows(spec: VarSpec, common_steps: list[tuple[str, int]]) -> list[dict]:
    rows = []
    for step_name, step_num in common_steps:
        s_forecast, s_target = load_forecast_target(OURS_SPECTRUM_MONTHLY_ROOT, RUN_TAG, step_name, spec.channel_name)
        v2_forecast, _v2_target = load_forecast_target(OURS_SPECTRUM_VER2_MONTHLY_ROOT, RUN_TAG, step_name, spec.channel_name)
        if s_forecast is None or v2_forecast is None or s_target is None:
            print(f'Skip {spec.var_name} {step_name}: missing monthly data')
            continue
        panels = {'Spectrum_MAE': s_forecast, 'Spectrum_MSE': v2_forecast, 'ERA5': s_target}
        rows.append({'label': format_step_days(step_num), 'step_num': step_num, 'panels': panels, 'target': s_target})
    return rows

def compute_norm(spec: VarSpec, rows: list[dict]):
    values = [np.asarray(data) for row in rows for data in row['panels'].values() if np.size(data) > 0 and np.isfinite(data).any()]
    if not values:
        return (None, None, None)
    if spec.scale_mode == 'fixed':
        if spec.value_range is None:
            raise ValueError(f'{spec.var_name} uses fixed scale without value_range')
        return (None, spec.value_range[0], spec.value_range[1])
    if spec.scale_mode == 'global':
        return (None, float(min((np.nanmin(arr) for arr in values))), float(max((np.nanmax(arr) for arr in values))))
    if spec.scale_mode == 'symmetric_percentile':
        abs_limit = float(max((np.nanpercentile(np.abs(arr), spec.percentile) for arr in values)))
    else:
        abs_limit = float(max((np.nanmax(np.abs(arr)) for arr in values)))
    abs_limit = max(abs_limit, 1e-12)
    return (TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit), None, None)

def setup_cartopy(lat2d: np.ndarray | None, lon2d: np.ndarray | None):
    if not (USE_CARTOPY and lat2d is not None and (lon2d is not None)):
        return None
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError:
        print('Cartopy is not installed; falling back to plain imshow.')
        return None
    return {'projection': ccrs.PlateCarree(), 'data_crs': ccrs.PlateCarree(), 'feature': cfeature, 'extent': [float(np.nanmin(lon2d)), float(np.nanmax(lon2d)), float(np.nanmin(lat2d)), float(np.nanmax(lat2d))]}

def full_extent(lat2d: np.ndarray | None, lon2d: np.ndarray | None) -> tuple[float, float, float, float]:
    if lat2d is None or lon2d is None:
        return (*DEFAULT_XLIM, *DEFAULT_YLIM)
    return (float(np.nanmin(lon2d)), float(np.nanmax(lon2d)), float(np.nanmin(lat2d)), float(np.nanmax(lat2d)))

def draw_map_background(ax, cartopy_ctx) -> None:
    if not cartopy_ctx:
        return
    ax.add_feature(cartopy_ctx['feature'].LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cartopy_ctx['feature'].OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.coastlines(resolution=COASTLINE_RESOLUTION, linewidth=0.6)

def compact_axis(ax) -> None:
    if not COMPACT_DOMAIN_LAYOUT:
        return
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0)

def weighted_rmse(data: np.ndarray, target: np.ndarray, spatial_weights: np.ndarray) -> float:
    weights = spatial_weights
    if weights.shape != data.shape:
        weights = create_spatial_weights_np(*data.shape)
    diff = data - target
    denom = float(np.sum(weights))
    return float(np.sqrt(np.sum(diff ** 2 * weights) / max(denom, 1e-12)))

def plot_var(spec: VarSpec, rows: list[dict], cartopy_ctx, extent: tuple[float, float, float, float], spatial_weights: np.ndarray) -> Path:
    n_rows = len(rows)
    n_cols = len(MODEL_LABELS)
    norm, vmin, vmax = compute_norm(spec, rows)
    subplot_kwargs = {'figsize': (PANEL_WIDTH * n_cols, PANEL_HEIGHT * n_rows)}
    if cartopy_ctx:
        subplot_kwargs['subplot_kw'] = {'projection': cartopy_ctx['projection']}
    fig, axes = plt.subplots(n_rows, n_cols, gridspec_kw={'wspace': PANEL_WSPACE, 'hspace': PANEL_HSPACE}, **subplot_kwargs)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    last_im = None
    for step_idx, row in enumerate(rows):
        xlim = (extent[0], extent[1])
        ylim = (extent[2], extent[3])
        for col, model_label in enumerate(MODEL_LABELS):
            ax = axes[step_idx, col]
            data = row['panels'][model_label]
            im_kwargs = {'cmap': spec.cmap, 'origin': 'upper', 'zorder': 1, 'extent': cartopy_ctx['extent'] if cartopy_ctx else extent}
            if cartopy_ctx:
                im_kwargs['transform'] = cartopy_ctx['data_crs']
            if norm is not None:
                im_kwargs['norm'] = norm
            else:
                im_kwargs['vmin'] = vmin
                im_kwargs['vmax'] = vmax
            last_im = ax.imshow(data, **im_kwargs)
            if cartopy_ctx:
                ax.set_extent([*xlim, *ylim], crs=cartopy_ctx['data_crs'])
            else:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            draw_map_background(ax, cartopy_ctx)
            compact_axis(ax)
            if step_idx == 0:
                ax.set_title(model_label, fontsize=12)
            if col == 0:
                ax.set_ylabel(row['label'], fontsize=12)
            if model_label != 'ERA5':
                rmse = weighted_rmse(data, row['target'], spatial_weights)
                ax.text(0.5, 0.98, f'RMSE={rmse:.4g}', transform=ax.transAxes, ha='center', va='top', fontsize=8.5, bbox={'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'none'})
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(left=0.045, right=0.89, bottom=0.025, top=0.84, wspace=PANEL_WSPACE, hspace=PANEL_HSPACE)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.91, 0.07, 0.018, 0.76])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.ax.set_ylabel(spec.colorbar_label, rotation=270, labelpad=8)
        cbar.ax.tick_params(labelsize=8, pad=2)
    fig.suptitle(wrap_figure_title(f'Monthly forecast comparison: {spec.var_name} ({RUN_TAG})', fig.get_figwidth()), fontsize=TITLE_FONT_SIZE, y=TITLE_Y, va='top')
    output_path = OUTPUT_DIR / f'monthly_3x3_{spec.var_name}_{RUN_TAG}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return output_path

def main() -> None:
    matplotlib.use('Agg')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_weights = create_spatial_weights_np()
    common_steps = discover_common_steps()
    if not common_steps:
        raise RuntimeError('No common step folders found between monthly roots.')
    lat2d, lon2d = load_lat_lon()
    cartopy_ctx = setup_cartopy(lat2d, lon2d)
    extent = full_extent(lat2d, lon2d)
    saved = []
    for spec in VAR_SPECS:
        rows = collect_rows(spec, common_steps)
        if not rows:
            print(f'No usable steps for {spec.var_name}; no figure written.')
            continue
        output_path = plot_var(spec, rows, cartopy_ctx, extent, spatial_weights)
        saved.append(output_path)
        print(f'Saved to {output_path}')
    print(f'Finished. Wrote {len(saved)} figure(s) to {OUTPUT_DIR}.')
