# Derived from gnn/ver_autoregressive/output/tracker_ver_aurora.py
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
from agu_james import runtime
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, minimum_filter
__all__ = ['Tracker']
logger = logging.getLogger(__file__)

class NoEyeException(Exception):
    """Raised when no eye can be found."""

def get_box(variable: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """Get a square box for a variable."""
    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[..., lat_mask, :]
    lats = lats[lat_mask]
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[..., lon_mask]
        lons = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[..., lon_mask1], box[..., lon_mask2]), axis=-1)
        lons = np.concatenate((lons[lon_mask1], lons[lon_mask2]))
    return (lats, lons, box)

def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude-longitude coordinates."""
    lat1, lat2 = (np.deg2rad(lat1), np.deg2rad(lat2))
    lon1, lon2 = (np.deg2rad(lon1), np.deg2rad(lon2))
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))

def get_closest_min(variable: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float, delta_lat: float=5, delta_lon: float=5, minimum_cap_size: int=8) -> tuple[float, float]:
    """Get the minimum in `variable` that is closest to `lat` and `lon`."""
    lats, lons, box = get_box(variable, lats, lons, lat - delta_lat, lat + delta_lat, lon - delta_lon, lon + delta_lon)
    box = gaussian_filter(box, sigma=1)
    if box.size == 0 or box.shape[-2] == 0 or box.shape[-1] == 0:
        raise NoEyeException()
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box
    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0
    if local_minima.sum() == 0:
        raise NoEyeException()
    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats[list(lat_inds)], lons[list(lon_inds)], lat, lon)
    i = np.argmin(dists)
    return (lats[lat_inds[i]], lons[lon_inds[i]])

def extrapolate(lats: list[float], lons: list[float]) -> tuple[float, float]:
    """Guess an initial latitude and longitude by extrapolating `lats` and `lons`."""
    assert len(lats) == len(lons)
    if len(lats) == 0:
        raise ValueError('Cannot extrapolate from empty lists.')
    elif len(lats) == 1:
        return (lats[0], lons[0])
    else:
        lats = lats[-8:]
        lons = lons[-8:]
        n = len(lats)
        fit = np.polyfit(np.arange(n), np.stack((lats, lons), axis=-1), 1)
        return np.polyval(fit, n)

class Tracker:
    """Simple tropical cyclone tracker.

    This algorithm was originally designed and implemented by Anna Allen. This particular
    implementation is by Wessel Bruinsma and features various improvements over the original design.
    """

    def __init__(self, init_lat: float | None=None, init_lon: float | None=None, init_time: datetime | None=None) -> None:
        self.tracked_times: list[datetime] = []
        self.tracked_lats: list[float] = []
        self.tracked_lons: list[float] = []
        self.tracked_msls: list[float] = []
        self.tracked_winds: list[float] = []
        if init_lat is not None or init_lon is not None or init_time is not None:
            if init_lat is None or init_lon is None or init_time is None:
                raise ValueError('init_lat/init_lon/init_time must all be set or all be None.')
            self.tracked_times.append(init_time)
            self.tracked_lats.append(init_lat)
            self.tracked_lons.append(init_lon)
            self.tracked_msls.append(np.nan)
            self.tracked_winds.append(np.nan)
        self.fails: int = 0

    def results(self) -> pd.DataFrame:
        """Assemble the track into a convenient DataFrame."""
        return pd.DataFrame({'time': self.tracked_times, 'lat': self.tracked_lats, 'lon': self.tracked_lons, 'msl': self.tracked_msls, 'wind': self.tracked_winds})

    def _track_step(self, z700: np.ndarray, msl: np.ndarray, u10: np.ndarray, v10: np.ndarray, lsm: np.ndarray, lats: np.ndarray, lons: np.ndarray, time: datetime, guess: tuple[float, float] | None=None) -> None:
        wind = np.sqrt(u10 * u10 + v10 * v10)
        if guess is None:
            if not self.tracked_lats:
                raise ValueError('No previous track available; provide guess for the first step.')
            lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        else:
            lat, lon = guess
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        def is_clear(lat: float, lon: float, delta: float) -> bool:
            """Is a box centred at `lat` and `lon` with "radius" `delta` clear of land?"""
            _, _, lsm_box = get_box(lsm, lats, lons, lat - delta, lat + delta, lon - delta, lon + delta)
            if lsm_box.size == 0:
                return False
            return lsm_box.max() < 0.5
        snap = False
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                if is_clear(lat, lon, delta):
                    lat, lon = get_closest_min(msl, lats, lons, lat, lon, delta_lat=delta, delta_lon=delta)
                    snap = True
                    break
            except NoEyeException:
                pass
        if not snap:
            try:
                lat, lon = get_closest_min(z700, lats, lons, lat, lon, delta_lat=5, delta_lon=5)
                snap = True
                for delta in [5, 4, 3, 2, 1.5]:
                    try:
                        if is_clear(lat, lon, delta):
                            lat, lon = get_closest_min(msl, lats, lons, lat, lon, delta_lat=delta, delta_lon=delta)
                            break
                    except NoEyeException:
                        pass
            except NoEyeException:
                pass
        if not snap:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info(f'Failed at time {time}. Extrapolating in a silly way.')
            else:
                raise NoEyeException('Completely failed at the first step.')
        lat_axis = lats[:, 0] if lats.ndim == 2 else lats
        lon_axis = lons[0, :] if lons.ndim == 2 else lons
        center_i, center_j = safe_window_center(lat_axis, lon_axis, lat, lon)
        lat, lon = weighted_centroid_latlon(msl, lat_axis, lon_axis, center_i, center_j, CENTROID_RADIUS, mode='min')
        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)
        _, _, msl_crop = get_box(msl, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        _, _, wind_crop = get_box(wind, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        self.tracked_msls.append(msl_crop.min())
        self.tracked_winds.append(wind_crop.max())

    def step(self, batch) -> None:
        """Track the next step.

        Args:
            batch (:class:`aurora.Batch`): Prediction.
        """
        if len(batch.metadata.time) != 1:
            raise RuntimeError("Predictions don't have batch size one.")
        batch = batch.to('cpu')
        z700_index = list(batch.metadata.atmos_levels).index(700)
        z700 = batch.atmos_vars['z'][0, 0, z700_index].numpy()
        msl = batch.surf_vars['msl'][0, 0].numpy()
        u10 = batch.surf_vars['10u'][0, 0].numpy()
        v10 = batch.surf_vars['10v'][0, 0].numpy()
        lsm = batch.static_vars['lsm'].numpy()
        lats = batch.metadata.lat.numpy()
        lons = batch.metadata.lon.numpy()
        time = batch.metadata.time[0]
        self._track_step(z700, msl, u10, v10, lsm, lats, lons, time)

    def step_fields(self, z700: np.ndarray, msl: np.ndarray, u10: np.ndarray, v10: np.ndarray, lsm: np.ndarray, lats: np.ndarray, lons: np.ndarray, time: datetime, guess: tuple[float, float] | None=None) -> None:
        """Track the next step from numpy arrays."""
        self._track_step(z700, msl, u10, v10, lsm, lats, lons, time, guess=guess)
BASE_DIR = runtime.setting('agu_james.analysis.gnn.tc_tracker.BASE_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_tracker')
CSV_PATH = runtime.setting('agu_james.analysis.gnn.tc_tracker.CSV_PATH', lambda: BASE_DIR / 'cyclone_data_2024.csv')
NPZ_DIR = runtime.setting('agu_james.analysis.gnn.tc_tracker.NPZ_DIR', lambda: BASE_DIR / 'UnetGNNgate7_fix_attention_autoregressive_step28_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
TARGET_DIR = runtime.setting('agu_james.analysis.gnn.tc_tracker.TARGET_DIR', lambda: BASE_DIR / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
FEATURE_TABLE = runtime.setting('agu_james.analysis.gnn.tc_tracker.FEATURE_TABLE', lambda: BASE_DIR / 'next_feature_table_weight.csv')
LAT_NPY = runtime.setting('agu_james.analysis.gnn.tc_tracker.LAT_NPY', lambda: BASE_DIR / 'lat.npy')
LON_NPY = runtime.setting('agu_james.analysis.gnn.tc_tracker.LON_NPY', lambda: BASE_DIR / 'lon.npy')
LSM_NPY = runtime.setting('agu_james.analysis.gnn.tc_tracker.LSM_NPY', lambda: BASE_DIR / 'lsm.npy')
SURFACE_NC_PATH = runtime.setting('agu_james.analysis.gnn.tc_tracker.SURFACE_NC_PATH', lambda: BASE_DIR.parent / 'surface.nc')
STEP_HOURS = 6
CENTROID_RADIUS = 6
INTENSITY_RADIUS = 6
BOUNDARY_MARGIN_DEG = 5.0
VMAX_MIN = 35.0
SUBSET_LAT_RANGE = (0.0, 47.75)
SUBSET_LON_RANGE = (102.25, 150.0)
OUTPUT_CSV = runtime.setting('agu_james.analysis.gnn.tc_tracker.OUTPUT_CSV', lambda: BASE_DIR / 'tc_aurora_intensity35_step28_auto.csv')
PANGU_CHANNEL_LEVEL_MAP = {'z700': (0, 3), 'u850': (3, 2), 'v850': (4, 2)}
SURFACE_CHANNEL_MAP = {'msl': 0, 'u10': 1, 'v10': 2, 't2m': 3}

def parse_timestamp_from_filename(path: str | Path) -> datetime | None:
    name = os.path.splitext(os.path.basename(str(path)))[0]
    try:
        return datetime.strptime(name, '%Y%m%d%H')
    except ValueError:
        pass
    digits = re.findall('(\\d{10})', name)
    for cand in digits:
        try:
            return datetime.strptime(cand, '%Y%m%d%H')
        except ValueError:
            continue
    return None

def parse_step_from_filename(path: str | Path) -> int | None:
    name = os.path.basename(str(path))
    m = re.search('_step(\\d+)', name)
    if m:
        return int(m.group(1))
    m = re.search('step(\\d+)', name)
    return int(m.group(1)) if m else None

def normalize_to_CLHW(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-2:] != (H, W):
            raise ValueError(f'Grid mismatch: {arr.shape[-2:]} vs {(H, W)}')
        return arr
    if arr.ndim == 3 and arr.shape[-1] == H * W:
        return arr.reshape(arr.shape[0], arr.shape[1], H, W)
    raise ValueError(f'Unsupported forecast shape: {arr.shape}')

def normalize_to_CHW(arr: np.ndarray, H: int, W: int, channels: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == channels and arr.shape[1:] == (H, W):
            return arr
        if arr.shape[-1] == channels and arr.shape[:2] == (H, W):
            return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 2:
        if arr.shape == (H * W, channels):
            return arr.reshape(H, W, channels).transpose(2, 0, 1)
        if arr.shape == (channels, H * W):
            return arr.reshape(channels, H, W)
    raise ValueError(f'Unsupported surface shape: {arr.shape}')

def normalize_forecast_to_CHW(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[1] == H and arr.shape[2] == W:
            return arr
        if arr.shape[0] == H and arr.shape[1] == W:
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(f'Unsupported 3D shape: {arr.shape}, expected H={H}, W={W}')
    if arr.ndim == 2:
        n0, n1 = arr.shape
        if n0 == H * W:
            C = n1
            return np.transpose(arr.reshape(H, W, C), (2, 0, 1))
        if n1 == H * W:
            C = n0
            return arr.reshape(C, H, W)
        raise ValueError(f'Unsupported 2D shape: {arr.shape}, expected H*W={H * W}')
    raise ValueError(f'Only 2D/3D arrays supported, got {arr.ndim}D')

def to_1d_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lat.ndim == 2:
        lat1d = lat[:, 0]
    elif lat.ndim == 1:
        lat1d = lat
    else:
        raise ValueError(f'Unsupported lat shape: {lat.shape}')
    if lon.ndim == 2:
        lon1d = lon[0, :]
    elif lon.ndim == 1:
        lon1d = lon
    else:
        raise ValueError(f'Unsupported lon shape: {lon.shape}')
    return (lat1d.astype(float), lon1d.astype(float))

def align_lon_system(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    lo_min, lo_max = (np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d))
    v = float(lon_value)
    if lo_min >= 0 and v < 0:
        v = v + 360.0
    if lo_max <= 180 and v > 180:
        v = v - 360.0
    return v

def build_global_lat_lon(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(90.0, -90.0, H)
    lon = np.linspace(0.0, 360.0 - 360.0 / W, W)
    return (lat, lon)

def safe_window_center(lat1d: np.ndarray, lon1d: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    i = int(np.argmin(np.abs(lat1d - lat0)))
    lon0_adj = align_lon_system(lon1d, lon0)
    j = int(np.argmin(np.abs(lon1d - lon0_adj)))
    i = max(0, min(lat1d.size - 1, i))
    j = max(0, min(lon1d.size - 1, j))
    return (i, j)

def make_window(i: int, j: int, radius: int, H: int, W: int) -> tuple[int, int, int, int]:
    i0, i1 = (max(0, i - radius), min(H, i + radius + 1))
    j0, j1 = (max(0, j - radius), min(W, j + radius + 1))
    if i0 >= i1:
        i0, i1 = (i, i + 1)
    if j0 >= j1:
        j0, j1 = (j, j + 1)
    return (i0, i1, j0, j1)

def weighted_centroid_latlon(field: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_i: int, center_j: int, radius: int, mode: str) -> tuple[float, float]:
    H, W = field.shape
    i0, i1, j0, j1 = make_window(center_i, center_j, radius, H, W)
    sub = field[i0:i1, j0:j1]
    if np.all(np.isnan(sub)):
        return (float(lat1d[center_i]), float(lon1d[center_j]))
    if mode == 'min':
        weights = np.nanmax(sub) - sub
    else:
        weights = sub - np.nanmin(sub)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if not np.any(weights > 0):
        return (float(lat1d[center_i]), float(lon1d[center_j]))
    lat_grid = lat1d[i0:i1][:, None]
    lon_grid = lon1d[j0:j1][None, :]
    wsum = float(np.sum(weights))
    lat_c = float(np.sum(weights * lat_grid) / wsum)
    lon_c = float(np.sum(weights * lon_grid) / wsum)
    return (lat_c, lon_c)

def select_bbox_slices(lat1d: np.ndarray, lon1d: np.ndarray, lat_range, lon_range) -> tuple[slice, slice]:
    lat_min, lat_max = (min(lat_range), max(lat_range))
    lon_min, lon_max = (min(lon_range), max(lon_range))
    lat_mask = (lat1d >= lat_min) & (lat1d <= lat_max)
    lon_mask = (lon1d >= lon_min) & (lon1d <= lon_max)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError('Subset range does not overlap grid.')
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    return (slice(lat_idx.min(), lat_idx.max() + 1), slice(lon_idx.min(), lon_idx.max() + 1))

def subset_global_chw(data_chw: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> np.ndarray:
    _, H, W = data_chw.shape
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    return data_chw[:, lat_sl, lon_sl]

def subset_global_upper(data: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> dict[str, np.ndarray]:
    if data.ndim != 4:
        raise ValueError(f'Unsupported upper shape: {data.shape}')
    max_ch = max((ch for ch, _ in PANGU_CHANNEL_LEVEL_MAP.values())) + 1
    max_lv = max((lv for _, lv in PANGU_CHANNEL_LEVEL_MAP.values())) + 1
    if data.shape[0] == max_ch and data.shape[1] >= max_lv:
        arr = data
    elif data.shape[1] == max_ch and data.shape[0] >= max_lv:
        arr = np.transpose(data, (1, 0, 2, 3))
    else:
        raise ValueError(f'Unsupported upper shape: {data.shape}')
    H, W = arr.shape[-2:]
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = arr[ch, lv, lat_sl, lon_sl]
    return fields

def subset_global_field(field: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> np.ndarray:
    H, W = field.shape
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    return field[lat_sl, lon_sl]

def extract_pangu_fields(forecast: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    data = normalize_to_CLHW(forecast, H, W)
    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = data[ch, lv, :, :]
    return fields

def surface_to_dict(data: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    channels = len(SURFACE_CHANNEL_MAP)
    try:
        chw = normalize_to_CHW(data, H, W, channels)
        return {name: chw[ch, :, :] for name, ch in SURFACE_CHANNEL_MAP.items()}
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 3:
            if arr.shape[0] == channels:
                chw = arr
            elif arr.shape[-1] == channels:
                chw = np.transpose(arr, (2, 0, 1))
            else:
                chw = None
            if chw is not None:
                sub = subset_global_chw(chw, SUBSET_LAT_RANGE, SUBSET_LON_RANGE)
                return {name: sub[ch, :, :] for name, ch in SURFACE_CHANNEL_MAP.items()}
        raise ValueError(f'Unsupported surface shape: {arr.shape}')

def upper_to_dict(data: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    try:
        return extract_pangu_fields(data, H, W)
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 4 and (arr.shape[-2], arr.shape[-1]) != (H, W):
            try:
                return subset_global_upper(arr, SUBSET_LAT_RANGE, SUBSET_LON_RANGE)
            except ValueError:
                pass
        raise ValueError(f'Unsupported upper shape: {arr.shape}')

def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))

def safe_nanmin(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else float('nan')

def safe_nanmax(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else float('nan')

def compute_intensity_metrics(msl: np.ndarray, u10: np.ndarray, v10: np.ndarray, u850: np.ndarray, v850: np.ndarray, center_i: int, center_j: int, radius: int) -> tuple[float, float, float]:
    H, W = msl.shape
    i0, i1, j0, j1 = make_window(center_i, center_j, radius, H, W)
    s10 = compute_wind_speed(u10, v10)
    s850 = compute_wind_speed(u850, v850)
    msl_min_hpa = safe_nanmin(msl[i0:i1, j0:j1]) / 100.0
    s10_max = safe_nanmax(s10[i0:i1, j0:j1])
    s850_max = safe_nanmax(s850[i0:i1, j0:j1])
    return (msl_min_hpa, s10_max, s850_max)

def parse_best_track_datetime(df: pd.DataFrame) -> pd.DataFrame:
    dt1 = pd.to_datetime(df['DATE_TIME'].astype(str), format='%Y%m%d%H', errors='coerce')
    if dt1.isna().any():
        dt2 = pd.to_datetime(df['DATE_TIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        dt1 = dt1.fillna(dt2)
    if dt1.isna().any():
        dt3 = pd.to_datetime(df['DATE_TIME'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
        dt1 = dt1.fillna(dt3)
    if dt1.isna().any():
        bad = df.loc[dt1.isna(), 'DATE_TIME'].head(5).tolist()
        raise ValueError(f'Unparsed DATE_TIME values (first 5): {bad}')
    df = df.copy()
    df['DATE_TIME'] = dt1
    return df

def get_channel_map(feature_csv: Path, varnames: dict[str, list[str]]) -> dict[str, int]:
    df = pd.read_csv(feature_csv)
    features = df['feature'].astype(str).tolist()
    result: dict[str, int] = {}
    for name, keys in varnames.items():
        found = None
        for i, f in enumerate(features):
            if any((k.lower() in f.lower() for k in keys)):
                found = i
                break
        if found is None:
            raise ValueError(f'{name} not found in feature table (keywords={keys})')
        result[name] = found
    return result

def filter_bst_records(df: pd.DataFrame, lat1d: np.ndarray, lon1d: np.ndarray, margin_deg: float, vmax_min: float) -> pd.DataFrame:
    lat_min, lat_max = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
    lon_min, lon_max = (float(np.nanmin(lon1d)), float(np.nanmax(lon1d)))
    lat_lo, lat_hi = (min(lat_min, lat_max) + margin_deg, max(lat_min, lat_max) - margin_deg)
    lon_lo, lon_hi = (min(lon_min, lon_max) + margin_deg, max(lon_min, lon_max) - margin_deg)
    lat_vals = pd.to_numeric(df['LAT'], errors='coerce')
    lon_vals = pd.to_numeric(df['LON'], errors='coerce')
    lon_adj = lon_vals.apply(lambda v: align_lon_system(lon1d, v) if np.isfinite(v) else v)
    if 'VMAX' not in df.columns:
        raise KeyError('Missing VMAX column in cyclone data.')
    vmax_vals = pd.to_numeric(df['VMAX'], errors='coerce')
    mask = lat_vals.between(lat_lo, lat_hi) & lon_adj.between(lon_lo, lon_hi) & (vmax_vals >= vmax_min)
    return df.loc[mask].copy()

def load_forecast_and_target(npz_path: Path, target_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        keys = set(getattr(data, 'files', []))
        forecast = None
        target = None
        for k_pred in ('forecast', 'pred', 'y_pred'):
            if k_pred in keys:
                forecast = data[k_pred]
                break
        for k_label in ('target', 'label', 'y_true'):
            if k_label in keys:
                target = data[k_label]
                break
    if target is None and target_dir is not None:
        step_idx = parse_step_from_filename(npz_path) or 1
        ts = parse_timestamp_from_filename(npz_path)
        candidate_name = npz_path.name
        if step_idx > 1 and ts is not None:
            shifted_ts = ts + timedelta(hours=STEP_HOURS * (step_idx - 1))
            ts_str = shifted_ts.strftime('%Y%m%d%H')
            candidate_name = re.sub('\\d{10}', ts_str, candidate_name, count=1)
            candidate_name = re.sub('_step\\d+', '_step01', candidate_name, count=1)
        target_path = target_dir / candidate_name
        if not target_path.exists():
            raise FileNotFoundError(f'Target file not found: {target_path}')
        with np.load(target_path) as tgt:
            tgt_keys = set(getattr(tgt, 'files', []))
            for k_label in ('target', 'label', 'y_true'):
                if k_label in tgt_keys:
                    target = tgt[k_label]
                    break
        if target is None:
            raise KeyError(f'{target_path} has no target field (target/label/y_true)')
    if forecast is None:
        raise KeyError(f'{npz_path} has no forecast field (forecast/pred/y_pred)')
    if target is None:
        raise KeyError(f'{npz_path} has no target field and TARGET_DIR is not set')
    return (forecast, target)

def collect_npz_files(npz_dir: Path) -> list[Path]:
    return sorted(npz_dir.glob('*.npz'))

def group_step_files_by_init(files: list[Path]) -> dict[datetime, list[Path]]:
    groups: dict[datetime, list[Path]] = {}
    for path in files:
        init_time = parse_timestamp_from_filename(path)
        if init_time is None:
            continue
        groups.setdefault(init_time, []).append(path)
    for paths in groups.values():
        paths.sort(key=lambda p: (parse_step_from_filename(p) or 0, p.name))
    return groups

def load_lsm(lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    lsm = np.load(LSM_NPY)
    target_shape = (lat1d.size, lon1d.size)
    if lsm.shape == target_shape:
        return lsm
    lat_range = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
    lon_range = (float(np.nanmin(lon1d)), float(np.nanmax(lon1d)))
    if lsm.ndim != 2:
        raise ValueError(f'Unexpected LSM shape: {lsm.shape}')
    lsm_sub = subset_global_field(lsm, lat_range, lon_range)
    if lsm_sub.shape != target_shape:
        raise ValueError(f'LSM shape mismatch: {lsm.shape} -> {lsm_sub.shape}, expected {target_shape}')
    return lsm_sub

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = parse_best_track_datetime(df)
    if 'TECH' in df.columns:
        df = df[df['TECH'].astype(str).str.upper() == 'BEST'].copy()
    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    if np.nanmin(lon1d) < 0 and np.nanmax(lon1d) <= 180:
        lon1d = np.mod(lon1d, 360.0)
    df = filter_bst_records(df, lat1d, lon1d, BOUNDARY_MARGIN_DEG, VMAX_MIN)
    lsm = load_lsm(lat1d, lon1d)
    H, W = (lat1d.size, lon1d.size)
    wanted = {'msl': ['msl'], 'u10': ['u10'], 'v10': ['v10'], 'u850': ['850u', 'u850'], 'v850': ['850v', 'v850'], 'z700': ['700z', 'z700']}
    ch_map = get_channel_map(FEATURE_TABLE, wanted)
    npz_files = collect_npz_files(NPZ_DIR)
    if not npz_files:
        raise FileNotFoundError(f'No npz files found in {NPZ_DIR}')
    init_groups = group_step_files_by_init(npz_files)
    if not init_groups:
        raise FileNotFoundError(f'No valid init time found in {NPZ_DIR}')
    cyclone_groups: dict[int, dict[datetime, tuple[float, float]]] = {}
    for row in df.itertuples(index=False):
        bst_time = pd.Timestamp(row.DATE_TIME).to_pydatetime()
        cyclone_num = int(row.CYCLONE_NUM)
        cyclone_groups.setdefault(cyclone_num, {})[bst_time] = (float(row.LAT), float(row.LON))
    records: list[dict] = []
    for cyclone_num, bst_map in cyclone_groups.items():
        for init_time, step_files in sorted(init_groups.items()):
            start_center = bst_map.get(init_time)
            if start_center is None:
                print(f'[WARN] No BST start center at {init_time:%Y%m%d%H} (CYCLONE_NUM={cyclone_num})')
                continue
            seed_lat, seed_lon = start_center
            seed_lon = align_lon_system(lon1d, seed_lon)
            print(f'Processing init {init_time:%Y-%m-%d %H:%M}, cyclone #{cyclone_num}')
            tracker = Tracker()
            for idx, npz_path in enumerate(step_files):
                step_idx = parse_step_from_filename(npz_path) or 1
                valid_time = init_time + timedelta(hours=STEP_HOURS * step_idx)
                try:
                    forecast_raw, target_raw = load_forecast_and_target(npz_path, TARGET_DIR)
                except (FileNotFoundError, KeyError) as exc:
                    print(f'[WARN] {exc}')
                    continue
                forecast = normalize_forecast_to_CHW(forecast_raw, H, W)
                target = normalize_forecast_to_CHW(target_raw, H, W)
                z700 = forecast[ch_map['z700'], :, :]
                msl = forecast[ch_map['msl'], :, :]
                u10 = forecast[ch_map['u10'], :, :]
                v10 = forecast[ch_map['v10'], :, :]
                u850 = forecast[ch_map['u850'], :, :]
                v850 = forecast[ch_map['v850'], :, :]
                guess = (seed_lat, seed_lon) if idx == 0 else None
                try:
                    tracker.step_fields(z700, msl, u10, v10, lsm, lat1d, lon1d, valid_time, guess=guess)
                except NoEyeException as exc:
                    print(f'[WARN] Tracking failed at {valid_time:%Y-%m-%d %H:%M}: {exc}')
                    break
                bst_lat = float('nan')
                bst_lon = float('nan')
                bst_center = bst_map.get(valid_time)
                if bst_center is not None:
                    bst_lat, bst_lon = bst_center
                    bst_lon = align_lon_system(lon1d, bst_lon)
                rec = {'time': valid_time.strftime('%Y%m%d%H'), 'forecast_init_time': init_time.strftime('%Y%m%d%H'), 'step': step_idx, 'CYCLONE_NUM': cyclone_num, 'bst_lat': bst_lat, 'bst_lon': bst_lon, 'forecast_center_lat': float(tracker.tracked_lats[-1]), 'forecast_center_lon': float(tracker.tracked_lons[-1]), 'seed_distance_flag': 0, 'forecast_msl_min_hPa': float('nan'), 'forecast_s10_max': float('nan'), 'forecast_s850_max': float('nan'), 'target_msl_min_hPa': float('nan'), 'target_s10_max': float('nan'), 'target_s850_max': float('nan'), 'source_file': npz_path.name}
                f_i, f_j = safe_window_center(lat1d, lon1d, rec['forecast_center_lat'], rec['forecast_center_lon'])
                rec['forecast_msl_min_hPa'], rec['forecast_s10_max'], rec['forecast_s850_max'] = compute_intensity_metrics(msl, u10, v10, u850, v850, f_i, f_j, INTENSITY_RADIUS)
                if np.isfinite(bst_lat) and np.isfinite(bst_lon):
                    t_i, t_j = safe_window_center(lat1d, lon1d, bst_lat, bst_lon)
                    t_msl = target[ch_map['msl'], :, :]
                    t_u10 = target[ch_map['u10'], :, :]
                    t_v10 = target[ch_map['v10'], :, :]
                    t_u850 = target[ch_map['u850'], :, :]
                    t_v850 = target[ch_map['v850'], :, :]
                    rec['target_msl_min_hPa'], rec['target_s10_max'], rec['target_s850_max'] = compute_intensity_metrics(t_msl, t_u10, t_v10, t_u850, t_v850, t_i, t_j, INTENSITY_RADIUS)
                records.append(rec)
    if not records:
        print('[WARN] No records collected; check BST times and results directories.')
        return
    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values(['time', 'CYCLONE_NUM', 'step'])
    out_df.to_csv(OUTPUT_CSV, index=False, float_format='%.3f')
    print(f'Saved: {OUTPUT_CSV} ({len(out_df)} rows)')
