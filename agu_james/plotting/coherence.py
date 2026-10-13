# Derived from spectrum analysis/coherence_energy_ratio_scatter.py
# See provenance/sources.json for original hash and modifications.
"""Plot energy ratio versus mean magnitude-squared coherence at days 1, 3, and 7.

Run with no arguments to export all channels. Use --channels 700q 850v
for a subset. Inputs are resolved relative to this script, independently
of the working directory. Each point represents one wavenumber, 1 <= k <= 45.
Rows show days; columns show MSE train, MSE test, Spectrum MAE, and Spectrum MSE.
"""
from agu_james import runtime
import argparse
import csv
from pathlib import Path
import tempfile
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
BASE_DIR = runtime.setting('agu_james.plotting.coherence.BASE_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.coherence')
DATA_DIR = runtime.setting('agu_james.plotting.coherence.DATA_DIR', lambda: BASE_DIR.parent / 'spectrum_multi')
DEFAULT_MSE_TRAIN_NPZ = runtime.setting('agu_james.plotting.coherence.DEFAULT_MSE_TRAIN_NPZ', lambda: BASE_DIR / 'UnetGNNgate7_fix_attention_step28_auto_GS192x192_BS4_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2' / 'mse_train_coherence_energy_rollout28.npz')
K_CUTOFF = 45
OUTPUT_DIR = runtime.setting('agu_james.plotting.coherence.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.coherence')
STEPS = (4, 12, 28)
DPI = 300
MODELS = (('autoregressive', 'auto', 'MSE'), ('spectrumv1', 'spectrum', 'Spectrum MAE'), ('spectrumv2', 'spectrum_ver2', 'Spectrum MSE'))

def load_inputs(mse_train_npz=DEFAULT_MSE_TRAIN_NPZ):
    """Validate all input arrays before writing any figures."""
    with (DATA_DIR / 'next_feature_table_weight.csv').open(encoding='utf-8-sig', newline='') as stream:
        features = [row['feature'] for row in csv.DictReader(stream)]
    if not features or len(set(features)) != len(features):
        raise ValueError('Feature names must be nonempty and unique.')
    datasets = []
    reference_k = None
    for coherence_name, energy_name, label in MODELS:
        coherence_path = DATA_DIR / 'coherence' / f'coherence_per_channel_step28_{coherence_name}.npz'
        energy_path = DATA_DIR / 'ours' / f'per_channel_energy_2024_fix_attention_step28_{energy_name}.npz'
        with np.load(coherence_path) as coherence, np.load(energy_path) as energy:
            k = np.asarray(coherence['k'])
            if k.ndim != 1 or not np.all(np.isfinite(k)):
                raise ValueError(f'{label}: k must be a finite one-dimensional array.')
            if len(np.unique(k)) != len(k) or np.any(k < 0):
                raise ValueError(f'{label}: k must contain unique nonnegative values.')
            if not np.array_equal(k, energy['k']):
                raise ValueError(f'{label}: coherence and energy wavenumbers differ.')
            if reference_k is not None and (not np.array_equal(k, reference_k)):
                raise ValueError(f'{label}: wavenumbers differ across models.')
            reference_k = k.copy()
            arrays = {}
            expected_shape = (len(features), len(k))
            for step in STEPS:
                keys = (f'coherence_mean_step{step:02d}', f'forecast_mean_step{step:02d}', f'target_mean_step{step:02d}')
                values = []
                for source, key in zip((coherence, energy, energy), keys):
                    if key not in source:
                        raise KeyError(f'{label}: missing {key}')
                    value = np.asarray(source[key], dtype=np.float64)
                    if value.shape != expected_shape:
                        raise ValueError(f'{label}: {key} shape {value.shape}, expected {expected_shape}')
                    values.append(value)
                arrays[step] = tuple(values)
            datasets.append((label, arrays))
    if mse_train_npz is not None:
        with np.load(mse_train_npz, allow_pickle=False) as train:
            if not np.array_equal(train['k'], reference_k):
                raise ValueError('MSE train: wavenumbers differ from comparison data')
            if 'feature_names' not in train or train['feature_names'].tolist() != features:
                raise ValueError('MSE train: feature names/order differ from comparison data')
            arrays = {}
            for step in STEPS:
                keys = (f'coherence_mean_step{step:02d}', f'forecast_mean_step{step:02d}', f'target_mean_step{step:02d}')
                values = []
                for key in keys:
                    if key not in train:
                        raise KeyError(f'MSE train: missing {key}')
                    value = np.asarray(train[key], dtype=np.float64)
                    if value.shape != (len(features), len(reference_k)):
                        raise ValueError(f'MSE train: invalid shape for {key}: {value.shape}')
                    values.append(value)
                arrays[step] = tuple(values)
            datasets[0] = ('MSE test', datasets[0][1])
            datasets.insert(0, ('MSE train', arrays))
    if not np.any((reference_k > 0) & (reference_k <= K_CUTOFF)):
        raise ValueError('No wavenumbers available within the selected cutoff.')
    return (features, reference_k, datasets)

def paired_points(k, coherence, forecast, target):
    """Return aligned finite points and counts of DC/invalid exclusions."""
    non_dc = k > 0
    selected_k = non_dc & (k <= K_CUTOFF)
    valid = selected_k & np.isfinite(coherence) & np.isfinite(forecast) & np.isfinite(target) & (target > 0)
    ratio = np.full(k.shape, np.nan, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        np.divide(forecast, target, out=ratio, where=valid)
    valid &= np.isfinite(ratio)
    return (ratio[valid], coherence[valid], k[valid], int(np.count_nonzero(selected_k & ~valid)), int(np.count_nonzero(~non_dc)))

def plot_channel(channel_idx, feature, k, datasets, output_dir):
    points = {}
    invalid_count = dc_count = 0
    x_min, x_max = (0.0, 1.0)
    for column, (label, arrays) in enumerate(datasets):
        for row, step in enumerate(STEPS):
            values = paired_points(k, *(arr[channel_idx] for arr in arrays[step]))
            points[row, column] = values[:3]
            invalid_count += values[3]
            dc_count += values[4]
            if values[0].size:
                x_min = min(x_min, float(values[0].min()))
                x_max = max(x_max, float(values[0].max()))
    margin = 0.05 * (x_max - x_min)
    x_limits = (x_min - margin if x_min < 0 else 0, x_max + margin)
    positive_k = k[(k > 0) & (k <= K_CUTOFF)]
    norm = Normalize(float(positive_k.min()), float(positive_k.max()))
    reference_ratio = np.linspace(0.0, 1.0, 256)
    reference_coherence = reference_ratio
    fig, axes = plt.subplots(len(STEPS), len(datasets), figsize=(4.6 * len(datasets) + 1.0, 3.15 * len(STEPS) + 0.9), sharex=True, sharey=True, squeeze=False, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.035)
    try:
        for column, (label, _) in enumerate(datasets):
            for row, step in enumerate(STEPS):
                ax = axes[row, column]
                ratio, coherence, point_k = points[row, column]
                ax.scatter(ratio, coherence, c=point_k, cmap='viridis', norm=norm, s=23, alpha=0.85, linewidths=0)
                ax.plot(reference_ratio, reference_coherence, '--', color='#444444', linewidth=1.3, label='MSE projection reference\n' + '$Q=R$ ($y=x$)')
                if row == 0 and column == 0:
                    ax.legend(loc='upper left', fontsize=12, borderpad=0.45, handlelength=1.8)
                ax.set_title(f'{label}\nDay {step // 4} ({step * 6} h) | n={len(ratio)}', fontsize=16, pad=8)
                ax.set_xlim(*x_limits)
                ax.set_ylim(0, 1.05)
                ax.tick_params(labelsize=13, labelbottom=row == len(STEPS) - 1, labelleft=column == 0)
                ax.grid(alpha=0.2)
                ax.set_axisbelow(True)
                if not len(ratio):
                    ax.text(0.5, 0.5, 'No valid points', transform=ax.transAxes, ha='center', va='center', fontsize=15)
        fig.suptitle(f'{feature} | Coherence vs energy ratio | k ≤ {K_CUTOFF}', fontsize=21)
        fig.supxlabel('Energy ratio  P / T', fontsize=18)
        fig.supylabel('Mean magnitude-squared coherence Q', fontsize=17)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        colorbar = fig.colorbar(scalar, ax=axes, pad=0.025, fraction=0.025, shrink=0.88, aspect=35)
        colorbar.set_label('Wavenumber k', fontsize=17, labelpad=10)
        colorbar.ax.tick_params(labelsize=13)
        colorbar.set_ticks([1, 10, 20, 30, 40, K_CUTOFF])
        safe_name = ''.join((c if c.isalnum() or c in '-_' else '_' for c in feature))
        output_path = output_dir / f'{channel_idx:03d}_{safe_name}.png'
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.png', delete=False) as stream:
            temporary_path = Path(stream.name)
        try:
            fig.savefig(temporary_path, dpi=DPI, facecolor='white')
            try:
                temporary_path.replace(output_path)
            except PermissionError:
                output_path = output_path.with_stem(output_path.stem + '_reference')
                temporary_path.replace(output_path)
                print(f'[Info] Previous PNG is locked; saved {output_path.name}', flush=True)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        plt.close(fig)
    return (output_path, invalid_count, dc_count)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs='+', help='Exact feature names to export')
    parser.add_argument('--mse-train-npz', type=Path, default=DEFAULT_MSE_TRAIN_NPZ, help='Combined MSE training coherence/energy NPZ (external training-set NPZ)')
    args = parser.parse_args()
    features, k, datasets = load_inputs(args.mse_train_npz)
    selected = set(args.channels) if args.channels else set(features)
    unknown = selected - set(features)
    if unknown:
        parser.error(f"Unknown channels: {', '.join(sorted(unknown))}")
    output_dir = runtime.setting('agu_james.plotting.coherence.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.coherence')
    output_dir.mkdir(parents=True, exist_ok=True)
    total_invalid = total_dc = exported = 0
    for channel_idx, feature in enumerate(features):
        if feature not in selected:
            continue
        path, invalid, dc = plot_channel(channel_idx, feature, k, datasets, output_dir)
        exported += 1
        total_invalid += invalid
        total_dc += dc
        print(f'[{exported}/{len(selected)}] {path.name} | invalid skipped={invalid}, DC excluded={dc}', flush=True)
    print(f'Saved {exported} PNGs to {output_dir}; invalid skipped={total_invalid}; DC excluded={total_dc}', flush=True)
