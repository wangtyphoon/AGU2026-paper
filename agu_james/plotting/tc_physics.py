# Derived from tc_verification/plot_tc_physics_structure.py
"""Compact manuscript figures combining tracker_multi and struc_multi.

Run with python from any working directory.
No source files are modified. Defaults: scalar verification at 24/48/72 h,
relationship and omega composites at 72 h, PNG at 600 dpi and vector PDF.
See docs/METRIC_DEFINITIONS.md for diagnostic meanings.
"""
from __future__ import annotations
from agu_james import runtime
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agu_james.plotting.tc_verification import CASE_KEY_COLS, MODEL_SPECS, STEP_HOURS, filter_common, normalize_case_keys, style_axis
from agu_james.plotting.tc_distributions import RELATIONSHIP_VARIABLES, STRUCTURE_VARIABLES, plot_wind_physics, plot_structure
OUTPUT_DIR = runtime.setting('agu_james.plotting.tc_physics.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.tc_physics')
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
TARGET_STEPS = (4, 8, 12)
STRUCTURE_MODELS = MODEL_SPECS[:3]
NPZ_DIRS = {'MSE': 'axisym_wcc_step28_auto_npz', 'Spectrum MAE': 'axisym_wcc_step28_spectrum_npz', 'Spectrum MSE': 'axisym_wcc_step28_spectrumv2_npz'}
STRUCTURE_CSVS = {'MSE': 'tc_aurora_intensity35_step28_auto.csv', 'Spectrum MAE': 'tc_aurora_intensity35_step28_spectrum.csv', 'Spectrum MSE': 'tc_aurora_intensity35_step28_spectrumv2.csv'}
PRESSURE_GROUPS = (('le975', 'MSL $\\leq$ 975 hPa', lambda p: p <= 975), ('975_990', '975 < MSL < 990 hPa', lambda p: (p > 975) & (p < 990)), ('ge990', 'MSL $\\geq$ 990 hPa', lambda p: p >= 990))
ALIASES = {'forecast_rmax': 'forecast_rmax_km', 'target_rmax': 'target_rmax_km', 'forecast_delta_z_sim_100': 'delta_z_sim_100km', 'forecast_delta_z_gwb_100': 'delta_z_gwb_100km', 'target_delta_z_sim_100': 'target_delta_z_sim_100km', 'target_delta_z_gwb_100': 'target_delta_z_gwb_100km'}

@dataclass(frozen=True)
class Variable:
    key: str
    title: str
    unit: str
    forecast: str
    target: str
VARIABLES = (Variable('s10', 'S10', 'm/s', 'forecast_s10_max', 'target_s10_max'), Variable('msl', 'MSL', 'hPa', 'forecast_msl_min_hPa', 'target_msl_min_hPa'), Variable('vmax', 'Vmax', 'm/s', 'forecast_vmax', 'target_vmax'), Variable('sim100', 'SIM height difference', 'm', 'delta_z_sim_100km', 'target_delta_z_sim_100km'), Variable('gwb100', 'GWB height difference', 'm', 'delta_z_gwb_100km', 'target_delta_z_gwb_100km'), Variable('balance_residual', 'SIM - GWB', 'm', 'forecast_balance_residual', 'target_balance_residual'), Variable('rmax', 'Rmax', 'km', 'forecast_rmax_km', 'target_rmax_km'), Variable('b', 'B', 'm', 'forecast_b_param', 'target_b_param'), Variable('vtl', '$\\!-V_T^L$', 'm', 'forecast_vtl', 'target_vtl'), Variable('vtu', '$\\!-V_T^U$', 'm', 'forecast_vtu', 'target_vtu'))
VARS = {v.key: v for v in VARIABLES}
WIND_KEYS = ('s10', 'msl', 'vmax', 'sim100', 'gwb100', 'balance_residual')
STRUCTURE_KEYS = ('rmax', 'b', 'vtl', 'vtu')
ALL_KEYS = WIND_KEYS + STRUCTURE_KEYS

def normalized_name(value: str) -> str:
    return str(value).replace("['", '').replace("']", '')

def load_scalar_data(tracker_dir: Path, max_step: int) -> tuple[dict, list]:
    datasets, audit = ({}, [])
    for spec in MODEL_SPECS:
        df = normalize_case_keys(pd.read_csv(tracker_dir / spec.relative_path), spec.label)
        df = df.rename(columns={old: new for old, new in ALIASES.items() if old in df and new not in df})
        required = [spec.latitude_col, spec.longitude_col, 'bst_lat', 'bst_lon']
        required += [col for v in VARIABLES if v.key != 'balance_residual' for col in (v.forecast, v.target)]
        missing = sorted(set(required) - set(df))
        if missing:
            raise ValueError(f'{spec.label}: missing columns {missing}')
        df[required] = df[required].apply(pd.to_numeric, errors='raise')
        selected = df['step'].between(1, max_step)
        geographic = df[spec.latitude_col].between(5, 42.25, inclusive='neither') & df[spec.longitude_col].between(107.5, 142.5, inclusive='neither') & np.isfinite(df[['bst_lat', 'bst_lon']]).all(axis=1)
        keep = selected & geographic
        audit.append({'model': spec.label, 'input_rows': len(df), 'lead_rows': int(selected.sum()), 'geographic_rows': int(keep.sum())})
        df = df.loc[keep].copy()
        for role in ('forecast', 'target'):
            prefix = '' if role == 'forecast' else 'target_'
            df[f'{role}_balance_residual'] = df[f'{prefix}delta_z_sim_100km'] - df[f'{prefix}delta_z_gwb_100km']
        datasets[spec.label] = df
    return (datasets, audit)

def common_for_variables(datasets: dict, keys: tuple, audit: list, theme: str) -> dict:
    columns = [col for key in keys for col in (VARS[key].forecast, VARS[key].target)]
    finite = {label: df.loc[np.isfinite(df[columns]).all(axis=1)].copy() for label, df in datasets.items()}
    common = filter_common(finite, CASE_KEY_COLS)
    common = {label: df.sort_values(CASE_KEY_COLS).reset_index(drop=True) for label, df in common.items()}
    for label, df in common.items():
        for step in sorted(datasets[label]['step'].unique()):
            audit.append({'theme': theme, 'model': label, 'lead_hours': int(step * STEP_HOURS), 'geographic_rows': int((datasets[label]['step'] == step).sum()), 'finite_rows': int((finite[label]['step'] == step).sum()), 'common_rows': int((df['step'] == step).sum())})
    return common

def target_agreement(datasets: dict, keys: tuple, strict: bool) -> list:
    reference = datasets['MSE']
    rows = []
    for label, df in datasets.items():
        if not df[CASE_KEY_COLS].equals(reference[CASE_KEY_COLS]):
            raise ValueError(f'{label}: common case alignment failed')
        for key in keys:
            delta = df[VARS[key].target].to_numpy() - reference[VARS[key].target].to_numpy()
            maximum = float(np.max(np.abs(delta)))
            rows.append({'model': label, 'variable': key, 'common_rows': len(df), 'target_max_abs_difference_from_mse': maximum, 'target_differences_above_0.002': int((np.abs(delta) > 0.002).sum())})
            if strict and maximum > 0.002:
                raise ValueError(f'{label}: incompatible ERA5 {key} targets (max difference {maximum})')
    return rows

def clustered_statistics(errors: np.ndarray, storms: np.ndarray) -> dict:
    """Case-weighted estimate; resample whole cyclone clusters for its interval."""
    if not len(errors):
        raise ValueError('No errors to summarize')
    _, cluster = np.unique(storms, return_inverse=True)
    size = int(cluster.max()) + 1
    result = {'rmse': float(np.sqrt(np.mean(errors ** 2))), 'bias': float(np.mean(errors)), 'mae': float(np.mean(np.abs(errors))), 'count': len(errors), 'cyclone_count': size}
    if size < 2:
        return {**result, 'rmse_ci95_low': np.nan, 'rmse_ci95_high': np.nan, 'bias_ci95_low': np.nan, 'bias_ci95_high': np.nan, 'ci_method': 'unavailable_single_cyclone'}
    sums = np.bincount(cluster, weights=errors)
    squares = np.bincount(cluster, weights=errors ** 2)
    counts = np.bincount(cluster)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, size, size=(BOOTSTRAP_SAMPLES, size))
    den = counts[draws].sum(axis=1)
    rmse = np.sqrt(squares[draws].sum(axis=1) / den)
    bias = sums[draws].sum(axis=1) / den
    result.update(zip(('rmse_ci95_low', 'rmse_ci95_high'), np.quantile(rmse, [0.025, 0.975])))
    result.update(zip(('bias_ci95_low', 'bias_ci95_high'), np.quantile(bias, [0.025, 0.975])))
    result['ci_method'] = f'cyclone_cluster_percentile_{BOOTSTRAP_SAMPLES}_seed{BOOTSTRAP_SEED}'
    return result

def summarize_variables(datasets: dict, keys: tuple) -> list:
    rows = []
    for key in keys:
        var = VARS[key]
        for label, df in datasets.items():
            for step in TARGET_STEPS:
                sub = df.loc[df['step'] == step]
                stats = clustered_statistics((sub[var.forecast] - sub[var.target]).to_numpy(), sub['CYCLONE_NUM'].to_numpy())
                rows.append({'model': label, 'variable': key, 'unit': var.unit, 'lead_hours': step * STEP_HOURS, 'forecast_mean': float(sub[var.forecast].mean()), 'target_mean': float(sub[var.target].mean()), **stats})
    return rows

def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int, title: str) -> None:
    fig.savefig(output_dir / f'{stem}.png', dpi=dpi, facecolor='white')
    fig.savefig(output_dir / f'{stem}.pdf', metadata={'Title': title})
    plt.close(fig)
    print(f'Saved {output_dir / stem}.png / .pdf')

def index_profiles(structure_dir: Path, label: str, step: int) -> pd.DataFrame:
    folder = structure_dir / NPZ_DIRS[label]
    rows = []
    for path in sorted(folder.glob(f'*_step{step:02d}.npz')):
        with np.load(path, allow_pickle=False) as data:
            count = len(data['time'])
            for field in ('cyclone_num', 'step'):
                if len(data[field]) != count:
                    raise ValueError(f'{path}: inconsistent metadata lengths')
            for i in range(count):
                rows.append({'time': int(data['time'][i]), 'CYCLONE_NUM': int(data['cyclone_num'][i]), 'step': int(data['step'][i]), 'forecast_init_time': int(path.name[:10]), 'name': path.name, 'case_index': i})
    if not rows:
        raise ValueError(f'{label}: no step {step} profiles in {folder}')
    meta = normalize_case_keys(pd.DataFrame(rows), label)
    source = normalize_case_keys(pd.read_csv(structure_dir / STRUCTURE_CSVS[label]), label)
    source = source.loc[source.step == step].copy()
    source['name'] = source['source_file'].map(normalized_name)
    return meta.merge(source[CASE_KEY_COLS + ['name', 'target_msl_min_hPa']], on=CASE_KEY_COLS + ['name'], how='inner', validate='one_to_one')

def load_vertical_composites(structure_dir: Path, scalar_raw: dict, step: int, audit: list, profile_rows: list, selected_rows: list) -> tuple:
    tables = {}
    for spec in STRUCTURE_MODELS:
        indexed = index_profiles(structure_dir, spec.label, step)
        indexed = indexed.merge(scalar_raw[spec.label][CASE_KEY_COLS + ['target_msl_min_hPa']], on=CASE_KEY_COLS, how='inner', validate='one_to_one', suffixes=('_profile', ''))
        finite = np.isfinite(indexed.target_msl_min_hPa) & np.isfinite(indexed.target_msl_min_hPa_profile)
        indexed = indexed.loc[finite].copy()
        if not np.allclose(indexed.target_msl_min_hPa, indexed.target_msl_min_hPa_profile, atol=0.002, rtol=0):
            raise ValueError(f'{spec.label}: tracker and struc CSV pressure targets disagree')
        tables[spec.label] = indexed
        audit.append({'theme': 'axisymmetric', 'model': spec.label, 'lead_hours': step * STEP_HOURS, 'matched_npz_eligible_rows': len(indexed)})
    tables = filter_common(tables, CASE_KEY_COLS)
    tables = {label: df.sort_values(CASE_KEY_COLS).reset_index(drop=True) for label, df in tables.items()}
    reference = tables['MSE']
    for spec in STRUCTURE_MODELS:
        if not np.allclose(tables[spec.label].target_msl_min_hPa, reference.target_msl_min_hPa, atol=0.002, rtol=0):
            raise ValueError(f'{spec.label}: profile pressure groups do not share identical ERA5 targets')
    levels = radius = None
    arrays = {}
    for spec in STRUCTURE_MODELS:
        storage = {key: [] for key in ('forecast_w_axisym', 'forecast_div_axisym', 'target_w_axisym', 'target_div_axisym')}
        for record in tables[spec.label].to_dict('records'):
            with np.load(structure_dir / NPZ_DIRS[spec.label] / record['name'], allow_pickle=False) as data:
                p, r = (data['pressure_levels_hpa'].astype(float), data['radius_centers_deg'].astype(float))
                if levels is None:
                    levels, radius = (p, r)
                if not np.array_equal(levels, p) or not np.allclose(radius, r, atol=1e-07, rtol=0):
                    raise ValueError(f'{spec.label}: incompatible profile grids')
                for key in storage:
                    field = data[key][record['case_index']].astype(float)
                    if field.shape != (len(levels), len(radius)):
                        raise ValueError(f'{spec.label}: incompatible {key} shape')
                    storage[key].append(field)
        arrays[spec.label] = {key: np.stack(value) for key, value in storage.items()}
    for spec in STRUCTURE_MODELS:
        for field in ('target_w_axisym', 'target_div_axisym'):
            if not np.allclose(arrays[spec.label][field], arrays['MSE'][field], atol=1e-05, rtol=1e-05, equal_nan=True):
                raise ValueError(f'{spec.label}: profile ERA5 targets disagree')
    results = []
    for group_id, group_label, choose in PRESSURE_GROUPS:
        mask = choose(reference.target_msl_min_hPa.to_numpy())
        if not mask.any():
            raise ValueError(f'No common profile cases in pressure group {group_id}')
        comps = {}
        for label in ('ERA5', *NPZ_DIRS):
            src = 'MSE' if label == 'ERA5' else label
            role = 'target' if label == 'ERA5' else 'forecast'
            fields = []
            for suffix in ('w_axisym', 'div_axisym'):
                stack = arrays[src][f'{role}_{suffix}'][mask]
                valid = np.isfinite(arrays['MSE'][f'target_{suffix}'][mask])
                for model in NPZ_DIRS:
                    valid &= np.isfinite(arrays[model][f'forecast_{suffix}'][mask])
                counts = valid.sum(axis=0)
                value = np.divide(np.where(valid, stack, 0).sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
                fields.append(value)
                for pi, pressure in enumerate(levels):
                    for ri, distance in enumerate(radius):
                        profile_rows.append({'group': group_id, 'model': label, 'lead_hours': step * STEP_HOURS, 'pressure_hpa': pressure, 'radius_deg': distance, 'field': 'omega' if suffix == 'w_axisym' else 'divergence', 'mean': value[pi, ri], 'finite_case_count': int(counts[pi, ri])})
            comps[label] = tuple(fields)
        for record in reference.loc[mask].to_dict('records'):
            selected_rows.append({**{key: record[key] for key in CASE_KEY_COLS}, 'group': group_id, 'name': record['name'], 'target_msl_min_hPa': record['target_msl_min_hPa']})
        results.append((group_id, group_label, int(mask.sum()), reference.loc[mask, 'CYCLONE_NUM'].nunique(), comps))
    print('Axisymmetric common cases: ' + ', '.join((f'{label}: {n}' for _, label, n, _, _ in results)))
    return (levels, radius, results)

def plot_vertical(levels: np.ndarray, radius: np.ndarray, groups: list, output_dir: Path, step: int, dpi: int, radius_max: float) -> dict:
    columns = ('ERA5', *NPZ_DIRS)
    shown = radius <= radius_max
    omega = np.stack([comps[label][0][:, shown] for _, _, _, _, comps in groups for label in columns])
    if not np.isfinite(omega).any():
        raise ValueError('No finite composite omega in displayed radius')
    limit = max(float(np.ceil(np.nanmax(np.abs(omega)) / 0.25) * 0.25), 0.25)
    color_levels = np.linspace(-limit, limit, 25)
    rr, pp = np.meshgrid(radius, levels)
    fig, axes = plt.subplots(len(groups), 4, figsize=(7.2, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.08, top=0.9, wspace=0.13, hspace=0.43)
    divergence_levels = np.array([-4, -2, 2, 4]) * 1e-05
    clipped_contours = 0
    for row, (_, group_label, count, storms, comps) in enumerate(groups):
        for col, label in enumerate(columns):
            ax = axes[row, col]
            w, div = comps[label]
            cf = ax.contourf(rr, pp, w, levels=color_levels, cmap='RdBu_r', extend='both')
            available = divergence_levels[(divergence_levels > np.nanmin(div[:, shown])) & (divergence_levels < np.nanmax(div[:, shown]))]
            if len(available):
                ax.contour(rr, pp, div, levels=available, colors='#333333', linewidths=0.5, alpha=0.75, linestyles=['dashed' if level < 0 else 'solid' for level in available])
                clipped_contours += len(available)
            ax.set_ylim(levels.max(), 100)
            ax.set_xlim(0, radius_max)
            ax.set_xticks([0, 1, 2] if radius_max == 2.5 else np.linspace(0, radius_max, 3))
            ax.set_yticks([1000, 700, 400, 100])
            ax.tick_params(labelsize=7, length=2.5, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.set_title(f'({chr(97 + row * 4 + col)}) {label}', loc='left', fontsize=8, fontweight='bold', pad=5)
            if col == 0:
                ax.set_ylabel('Pressure (hPa)', fontsize=8)
            if row == len(groups) - 1:
                ax.set_xlabel('Radius ($^\\circ$)', fontsize=8)
        axes[row, 0].text(0, 1.18, group_label, transform=axes[row, 0].transAxes, fontsize=8.2, ha='left', va='bottom')
    cax = fig.add_axes([0.885, 0.2, 0.016, 0.57])
    cbar = fig.colorbar(cf, cax=cax, ticks=np.linspace(-limit, limit, 5), format='%.2f')
    cbar.set_label('$\\omega$ (Pa s$^{-1}$)', fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7, length=2)
    save_figure(fig, output_dir, 'tc_axisymmetric_vertical_motion', dpi, 'Pressure-conditioned axisymmetric vertical motion')
    return {'omega_color_limit_pa_s': limit, 'radius_max_deg': radius_max, 'pressure_display_hpa': [100, float(levels.max())], 'divergence_contours_s_inverse': divergence_levels.tolist(), 'available_contour_levels_total': clipped_contours, 'group_case_counts': {key: int(n) for key, _, n, _, _ in groups}}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker-dir', type=Path, default=OUTPUT_DIR.parent / 'tracker_multi')
    parser.add_argument('--structure-dir', type=Path, default=OUTPUT_DIR.parent / 'struc_multi')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--relationship-lead-hours', type=int, default=72)
    parser.add_argument('--vertical-lead-hours', type=int, default=72)
    parser.add_argument('--radius-max-deg', type=float, default=2.5)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    for lead in (args.relationship_lead_hours, args.vertical_lead_hours):
        if lead <= 0 or lead > 168 or lead % STEP_HOURS:
            parser.error('Leads must be positive multiples of 6 h, at most 168 h')
    if args.dpi <= 0 or not 0 < args.radius_max_deg <= 4:
        parser.error('DPI must be positive; radius maximum must lie in (0, 4] degrees')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_step = max(12, args.relationship_lead_hours // STEP_HOURS, args.vertical_lead_hours // STEP_HOURS)
    scalar_raw, source_audit = load_scalar_data(args.tracker_dir, max_step)
    audit, target_rows, metric_rows = ([], [], [])
    common_source = common_for_variables(scalar_raw, ALL_KEYS, audit, 'all_parameters')
    target_rows.extend(target_agreement(common_source, WIND_KEYS, strict=True))
    target_rows.extend(target_agreement(common_source, STRUCTURE_KEYS, strict=False))
    reference = common_source['MSE']
    target_columns = [VARS[key].target for key in ALL_KEYS]
    common = {}
    for label, source in common_source.items():
        df = source.copy()
        df[target_columns] = reference[target_columns].to_numpy()
        common[label] = df
    wind = common
    metric_rows.extend(summarize_variables(common, ALL_KEYS))
    summary = pd.DataFrame(metric_rows)
    for label, source in common_source.items():
        for key in ALL_KEYS:
            for step in TARGET_STEPS:
                sub = source.loc[source.step == step]
                errors = (sub[VARS[key].forecast] - sub[VARS[key].target]).to_numpy()
                mask = (summary.model == label) & (summary.variable == key) & (summary.lead_hours == step * STEP_HOURS)
                summary.loc[mask, 'original_csv_target_bias'] = float(np.mean(errors))
                summary.loc[mask, 'original_csv_target_rmse'] = float(np.sqrt(np.mean(errors ** 2)))
    profile_rows, selected_rows = ([], [])
    levels, radius, groups = load_vertical_composites(args.structure_dir, common, args.vertical_lead_hours // STEP_HOURS, audit, profile_rows, selected_rows)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False}):
        relation_rows = plot_wind_physics(wind, args.output_dir, args.relationship_lead_hours // STEP_HOURS, args.dpi)
        structure_relation_rows = plot_structure(common, args.output_dir, args.relationship_lead_hours // STEP_HOURS, args.dpi)
        vertical_config = plot_vertical(levels, radius, groups, args.output_dir, args.vertical_lead_hours // STEP_HOURS, args.dpi, args.radius_max_deg)
    exports = {'tc_physics_structure_summary.csv': summary, 'tc_relationship_distribution.csv': pd.DataFrame(relation_rows), 'tc_structure_relationship_distribution.csv': pd.DataFrame(structure_relation_rows), 'tc_physics_sample_audit.csv': pd.DataFrame(audit), 'tc_target_agreement.csv': pd.DataFrame(target_rows), 'tc_axisymmetric_composites.csv': pd.DataFrame(profile_rows), 'tc_axisymmetric_cases.csv': pd.DataFrame(selected_rows), 'tc_common_scalar_cases.csv': reference[CASE_KEY_COLS + target_columns]}
    for filename, frame in exports.items():
        frame.to_csv(args.output_dir / filename, index=False, float_format='%.10g')
    manifest = {'tracker_dir': str(args.tracker_dir.resolve()), 'structure_dir': str(args.structure_dir.resolve()), 'model_csvs': {s.label: s.relative_path for s in MODEL_SPECS}, 'profile_folders': NPZ_DIRS, 'source_audit': source_audit, 'verification_lead_hours': [24, 48, 72], 'relationship_lead_hours': args.relationship_lead_hours, 'vertical_lead_hours': args.vertical_lead_hours, 'scalar_target_policy': 'Shared MSE ERA5 targets after all-parameter four-model intersection. Original target discrepancies and original-target scores exported separately.', 'vertical_target_policy': 'MSE ERA5 profiles; equality checked across the three models.', 'sample_policy': 'Geographic eligibility then one four-model intersection requiring every scalar parameter finite. All scalar panels share these cases at each lead. Vertical panels use its subset with matched profiles for all three models.', 'geographic_bounds_exclusive': {'latitude': [5, 42.25], 'longitude': [107.5, 142.5]}, 'bootstrap': {'cluster': 'CYCLONE_NUM', 'samples': BOOTSTRAP_SAMPLES, 'seed': BOOTSTRAP_SEED, 'estimate_weighting': 'case-weighted', 'ci': '95% percentile'}, 'relationship_distribution': 'Two-dimensional Gaussian KDE with Scott bandwidth; ERA5-only black-to-gray contour lines (50% black, 75% dark gray, 90% light gray) enclose 50%, 75%, and 90% probability mass on a shared padded grid. Models use filled density bands at the same probability thresholds without outlines; ERA5 has no fill. Marginal KDEs use ERA5 lines and model filled areas. Model legend entries use coloured patches. Top and right one-dimensional Scott KDEs share the joint axes. Every model uses identical cases. Two relationship rows by four model columns, each comparing one forecast with ERA5; densities, grids, contour thresholds and marginal density scales are computed once across all models per relationship and reused in every column. Display limits are fixed: Vmax 7.5-27.5 m/s, MSL 960-1015 hPa, and SIM/GWB 0-140 m; all samples and the full padded KDE integration grid are retained.', 'relationship_statistics': {'layout': 'One bottom table per figure with four model rows and three metric groups: RMSE, KS D / p-value and actual forecast tail quantiles. Each group contains four individually named parameter columns; native units are shown for RMSE and tail quantiles, and Q05/Q95 are specified for tail quantiles. Tail-bias columns remain in CSV only. Correlation metrics remain in CSV only.', 'rmse': 'Root mean squared case-paired forecast minus shared ERA5 error in native units.', 'ks_d': 'Two-sided two-sample Kolmogorov-Smirnov statistic comparing each marginal forecast distribution with shared ERA5.', 'ks_pvalue': 'Raw p value from the same scipy.stats.ks_2samp call as D, alternative=two-sided, method=auto; no multiple-comparison correction or cyclone-cluster resampling. CSV columns ks_x_pvalue and ks_y_pvalue; ERA5 rows empty. Table cells show D on the first line and p on the second, with three decimals for p >= 0.001 and scientific notation for smaller positive values.', 'tail_bias': 'Forecast marginal quantile minus shared ERA5 marginal quantile in native units; numpy linear quantile interpolation.', 'tail_quantiles': {key: var.tail_quantile for key, var in RELATIONSHIP_VARIABLES.items()}, 'rho_error': 'Forecast x-y Spearman rho minus shared ERA5 x-y Spearman rho.', 'lead_hours': args.relationship_lead_hours, 'era5_error_columns': 'Empty; ERA5 tail values and Spearman rho are retained as references.'}, 'structure_distribution': 'Two relationship rows at relationship lead: Rmax-B and VTU-VTL, each faceted into four model-versus-ERA5 columns. Two-dimensional Scott KDE with ERA5-only 50%/75%/90% probability contour lines in black/dark gray/light gray, respectively and model filled bands without outlines, top/right one-dimensional Scott KDEs, and the same consolidated bottom statistics table as wind-pressure panels. Rmax KDE is reflected at zero in the joint and marginal densities and restricted to nonnegative radii; integration half-weights the zero-radius boundary. Original native signs of B/VTL/VTU are retained. Joint and marginal display limits are fixed for B (-25 to 20 m), VTU (-25 to 225 m), and VTL (-25 to 175 m). Rmax uses fixed display limits of 25-200 km; all samples and the full integration grid are retained.', 'structure_relationship_statistics': {'definitions': 'Same RMSE, KS D / p-value, marginal tail-bias and x-y Spearman-rho error definitions as relationship_statistics; shared MSE ERA5 targets and identical scalar cases.', 'tail_quantiles': {key: var.tail_quantile for key, var in STRUCTURE_VARIABLES.items()}, 'lead_hours': args.relationship_lead_hours, 'csv': 'tc_structure_relationship_distribution.csv'}, 'figure_text_policy': 'Distribution figures use row-level identifiers (a) and (b), plus (c) for statistics; model facets share model-name headings above the first row without separate identifiers. Row titles are 12 pt, model headings and axis labels 11 pt, ticks 9 pt, and table/legend text 10 pt; facet and row spacing is reduced. Variable titles, axes, units, model legends and pressure-group labels; wind-pressure and structure figures additionally show one consolidated bottom RMSE, KS D / p-value and actual tail-quantile table grouped by metric with four named parameters per metric. Spearman statistics remain in CSV only. Captions, sample counts, confidence intervals and methodological notes are external.', 'vertical': vertical_config, 'output_dpi': args.dpi, 'omega_units_source': 'https://confluence.ecmwf.int/spaces/S2S/pages/27399160/Vertical+velocity'}
    (args.output_dir / 'tc_physics_structure_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('All scalar parameters share these common cases:')
    print(summary.loc[(summary.model == 'MSE') & (summary.variable == 's10'), ['lead_hours', 'count', 'cyclone_count']].to_string(index=False))
    print(f'Saved summaries and sample audit to {args.output_dir}')
