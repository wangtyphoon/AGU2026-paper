"""Modal-power contract for legacy integer-radius annular means.

Units are field units squared after Parseval normalization. This is a scalar
mean-square diagnostic, not kinetic energy or a domain-integrated energy.
The retained k<=87 disk excludes corner modes and is not the complete FFT.
"""
import numpy as np

FFT_SHAPE = (174, 174)
NOMINAL_SPACING_KM = 28.0
NOMINAL_LENGTH_KM = FFT_SHAPE[0] * NOMINAL_SPACING_KM

def safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan, dtype=float), where=denominator > 0)

def mode_counts(shape=FFT_SHAPE):
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError('FFT dimensions must be positive')
    yy, xx = np.indices((h, w))
    radius = np.floor(np.hypot(yy-h//2, xx-w//2)).astype(int)
    return np.bincount(radius.ravel())

def modal_weights(k, shape=FFT_SHAPE, density_norm=False):
    k = np.asarray(k)
    counts = mode_counts(shape)
    if np.any(k != np.floor(k)) or np.any(k < 0) or np.any(k >= len(counts)):
        raise ValueError('Invalid integer annular indices for FFT shape')
    return counts[k.astype(int)] / (1.0 if density_norm else float(np.prod(shape))**2)

def spectrum_metadata(shape=FFT_SHAPE, amplitude=False, density_norm=False,
                      detrend_mean=False, apply_lat_weight=False):
    return dict(spectral_contract=np.array('annular_mean_v2'),
                fft_shape=np.asarray(shape, dtype=np.int64),
                mode_count=mode_counts(shape)[:min(shape)//2+1],
                spectral_quantity=np.array('amplitude' if amplitude else 'power'),
                density_norm=np.array(density_norm), detrend_mean=np.array(detrend_mean),
                apply_lat_weight=np.array(apply_lat_weight),
                radial_bin_definition=np.array('floor(hypot(integer_fft_indices)); retained disk'),
                scale_interpretation=np.array('index radius; 28 km isotropic spacing is nominal only'))

def physical_band_power(field, dx_km, dy_km, wavelength_max_km=1000.,
                        wavelength_min_km=None, detrend_mean=True):
    """Mean-square power on a caller-supplied uniform Cartesian grid.

    Use only after reprojection/resampling to a regular physical grid. Geographic
    degree-grid fields must not be passed using one constant zonal spacing.
    Cutoffs operate on each mode's fftfreq radius, without annular rounding.
    """
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Require a finite 2D Cartesian field')
    if not np.isfinite([dx_km, dy_km, wavelength_max_km]).all() or min(dx_km, dy_km, wavelength_max_km) <= 0:
        raise ValueError('Spacing and cutoff must be finite and positive')
    if wavelength_min_km is not None and (not np.isfinite(wavelength_min_km) or not 0 < wavelength_min_km < wavelength_max_km):
        raise ValueError('Invalid lower wavelength cutoff')
    if detrend_mean:
        x = x - x.mean()
    fy = np.fft.fftfreq(x.shape[0], d=dy_km)
    fx = np.fft.fftfreq(x.shape[1], d=dx_km)
    nu = np.hypot(fy[:, None], fx[None, :])
    selected = nu > 1. / wavelength_max_km
    if wavelength_min_km is not None:
        selected &= nu <= 1. / wavelength_min_km
    power = np.abs(np.fft.fft2(x, norm='forward'))**2
    return float(power[selected].sum())
