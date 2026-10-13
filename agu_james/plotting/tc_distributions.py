# Derived from tc_verification/tc_distribution_plots.py
# See provenance/sources.json for original hash and modifications.
"""Distribution panels with marginal KDEs and compact relationship statistics."""
from agu_james import runtime
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde, ks_2samp, spearmanr
from agu_james.plotting.tc_verification import CASE_KEY_COLS, MODEL_SPECS, STEP_HOURS, TARGET_STEPS, style_axis

@dataclass(frozen=True)
class RelationshipVariable:
    label: str
    unit: str
    forecast: str
    target: str
    tail_quantile: float = 0.95
    nonnegative: bool = False

    @property
    def axis_label(self):
        unit = 'm s$^{-1}$' if self.unit == 'm/s' else self.unit
        return f'{self.label} ({unit})'
RELATIONSHIP_VARIABLES = {'s10': RelationshipVariable('S10', 'm/s', 'forecast_s10_max', 'target_s10_max'), 'msl': RelationshipVariable('MSL', 'hPa', 'forecast_msl_min_hPa', 'target_msl_min_hPa', 0.05), 'vmax': RelationshipVariable('Vmax', 'm/s', 'forecast_vmax', 'target_vmax'), 'sim100': RelationshipVariable('$\\Delta Z_{\\mathrm{SIM}}$', 'm', 'delta_z_sim_100km', 'target_delta_z_sim_100km'), 'gwb100': RelationshipVariable('$\\Delta Z_{\\mathrm{GWB}}$', 'm', 'delta_z_gwb_100km', 'target_delta_z_gwb_100km')}
RELATIONSHIPS = (('Vmax-MSL distribution', 'vmax', 'msl', False), ('100 km SIM-GWB distribution', 'sim100', 'gwb100', True))
STRUCTURE_VARIABLES = {'rmax': RelationshipVariable('Rmax', 'km', 'forecast_rmax_km', 'target_rmax_km', 0.05, True), 'b': RelationshipVariable('B', 'm', 'forecast_b_param', 'target_b_param', 0.05), 'vtl': RelationshipVariable('$\\!-V_T^L$', 'm', 'forecast_vtl', 'target_vtl'), 'vtu': RelationshipVariable('$\\!-V_T^U$', 'm', 'forecast_vtu', 'target_vtu')}
STRUCTURE_RELATIONSHIPS = (('Rmax-B distribution', 'rmax', 'b', False), ('$-V_T^U$–$-V_T^L$ distribution', 'vtu', 'vtl', False))
STRUCTURE_FIELDS = (('Rmax', 'km', 'forecast_rmax_km', 'target_rmax_km'), ('B', 'm', 'forecast_b_param', 'target_b_param'), ('$\\!-V_T^L$', 'm', 'forecast_vtl', 'target_vtl'), ('$\\!-V_T^U$', 'm', 'forecast_vtu', 'target_vtu'))

def handles(era5=True):
    result = [Patch(facecolor=s.color, edgecolor='none', alpha=0.5, label=s.label) for s in MODEL_SPECS]
    if era5:
        result.append(Line2D([0], [0], color='#333333', linewidth=1.7, label='ERA5'))
    return result

def series(datasets, forecast, target, step):
    reference = datasets['MSE'].loc[datasets['MSE'].step == step]
    result = [('ERA5', '#333333', '-', reference[target].to_numpy(dtype=float))]
    result += [(s.label, s.color, s.linestyle, datasets[s.label].loc[datasets[s.label].step == step, forecast].to_numpy(dtype=float)) for s in MODEL_SPECS]
    return result

def save(fig, output_dir, stem, dpi):
    fig.savefig(Path(output_dir) / f'{stem}.png', dpi=dpi, facecolor='white')
    fig.savefig(Path(output_dir) / f'{stem}.pdf')
    plt.close(fig)
    print(f'Saved {Path(output_dir) / stem}.png / .pdf')

def coloured_boxes(ax, values, positions, width, color):
    boxes = ax.boxplot(values, positions=positions, widths=width, manage_ticks=False, patch_artist=True, showfliers=False, whis=1.5, medianprops={'color': color, 'linewidth': 1.2}, whiskerprops={'color': color, 'linewidth': 0.65}, capprops={'color': color, 'linewidth': 0.65})
    for box in boxes['boxes']:
        box.set(facecolor=color, edgecolor=color, alpha=0.24, linewidth=0.7)

def relationship_samples(datasets, xvar, yvar, step):
    """Use aligned cases and one ERA5 reference for both plotting and scores."""
    reference = datasets['MSE'].loc[datasets['MSE'].step == step]
    if len(reference) < 3:
        raise ValueError(f'Step {step}: at least three common cases are required for joint KDE')
    target_x = reference[xvar.target].to_numpy(dtype=float)
    target_y = reference[yvar.target].to_numpy(dtype=float)
    if not np.isfinite(np.vstack([target_x, target_y])).all():
        raise ValueError(f'Step {step}: nonfinite shared ERA5 values')
    samples = [('ERA5', '#333333', '-', target_x, target_y)]
    for spec in MODEL_SPECS:
        sub = datasets[spec.label].loc[datasets[spec.label].step == step]
        if not sub[CASE_KEY_COLS].reset_index(drop=True).equals(reference[CASE_KEY_COLS].reset_index(drop=True)):
            raise ValueError(f'{spec.label}: joint distribution case alignment failed at step {step}')
        x = sub[xvar.forecast].to_numpy(dtype=float)
        y = sub[yvar.forecast].to_numpy(dtype=float)
        if not np.isfinite(np.vstack([x, y])).all():
            raise ValueError(f'{spec.label}: nonfinite relationship values at step {step}')
        samples.append((spec.label, spec.color, spec.linestyle, x, y))
    return samples

def relationship_statistics(samples, xvar, yvar):
    """Native-unit paired errors, marginal KS/tails, and x-y rank relation."""
    target_x, target_y = samples[0][3:]
    target_rho = float(spearmanr(target_x, target_y).statistic)
    target_qx = float(np.quantile(target_x, xvar.tail_quantile))
    target_qy = float(np.quantile(target_y, yvar.tail_quantile))
    rows = []
    for label, _, _, x, y in samples:
        rho = float(spearmanr(x, y).statistic)
        qx = float(np.quantile(x, xvar.tail_quantile))
        qy = float(np.quantile(y, yvar.tail_quantile))
        row = {'x_unit': xvar.unit, 'y_unit': yvar.unit, 'x_tail_quantile': xvar.tail_quantile, 'y_tail_quantile': yvar.tail_quantile, 'x_tail_value': qx, 'y_tail_value': qy, 'era5_x_tail_value': target_qx, 'era5_y_tail_value': target_qy, 'spearman_rho': rho, 'era5_spearman_rho': target_rho}
        if label == 'ERA5':
            row.update({key: np.nan for key in ('rmse_x', 'rmse_y', 'ks_x_d', 'ks_y_d', 'ks_x_pvalue', 'ks_y_pvalue', 'tail_bias_x', 'tail_bias_y', 'rho_error')})
        else:
            ks_x = ks_2samp(x, target_x, alternative='two-sided', method='auto')
            ks_y = ks_2samp(y, target_y, alternative='two-sided', method='auto')
            row.update({'rmse_x': float(np.sqrt(np.mean((x - target_x) ** 2))), 'rmse_y': float(np.sqrt(np.mean((y - target_y) ** 2))), 'ks_x_d': float(ks_x.statistic), 'ks_y_d': float(ks_y.statistic), 'ks_x_pvalue': float(ks_x.pvalue), 'ks_y_pvalue': float(ks_y.pvalue), 'tail_bias_x': qx - target_qx, 'tail_bias_y': qy - target_qy, 'rho_error': rho - target_rho})
        rows.append(row)
    return rows

def format_ks_pvalue(value):
    """Keep small positive p values visible rather than rounding to zero."""
    return f'{value:.1e}' if 0 < value < 0.001 else f'{value:.3f}'

def relationship_table(ax, groups):
    """Group four native-unit parameters under each metric."""
    ax.set_axis_off()
    parameters = [(rows, var, axis) for rows, xvar, yvar in groups for var, axis in ((xvar, 'x'), (yvar, 'y'))]
    widths = [0.13] + [0.87 / 12] * 12
    for center, title in ((0.275, 'RMSE'), (0.565, 'KS D / p-value'), (0.855, 'Tail quantiles')):
        ax.text(center, 0.98, title, transform=ax.transAxes, fontsize=10.5, fontweight='bold', ha='center', va='top')
    columns = ['Model']
    for metric in ('rmse', 'ks', 'tail'):
        for _, var, _ in parameters:
            if metric == 'ks':
                columns.append(var.label)
            elif metric == 'tail':
                columns.append(f'{var.label}\nQ{100 * var.tail_quantile:02.0f} ({var.unit})')
            else:
                columns.append(f'{var.label}\n({var.unit})')
    cells = []
    for model_index, spec in enumerate(MODEL_SPECS, start=1):
        values = [spec.label]
        for metric in ('rmse', 'ks', 'tail'):
            for rows, _, axis in parameters:
                row = rows[model_index]
                if metric == 'rmse':
                    values.append(f"{row[f'rmse_{axis}']:.1f}")
                elif metric == 'ks':
                    values.append(f"D={row[f'ks_{axis}_d']:.2f}\np={format_ks_pvalue(row[f'ks_{axis}_pvalue'])}")
                else:
                    value = row[f'{axis}_tail_value']
                    values.append('0.0' if round(value, 1) == 0 else f'{value:.1f}')
        cells.append(values)
    table = ax.table(cellText=cells, colLabels=columns, cellLoc='center', colWidths=widths, bbox=[0, 0, 1, 0.77])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        cell.set_linewidth(0.45)
        cell.PAD = 0.04
        if row == 0:
            cell.set_height(cell.get_height() * 1.45)
            cell.set_facecolor('#f3f3f3')
            cell.get_text().set_fontweight('bold')
        elif col == 0:
            cell.get_text().set_color(MODEL_SPECS[row - 1].color)
    for boundary in (0.13, 0.42, 0.71):
        ax.plot([boundary, boundary], [0, 0.77], color='#aaaaaa', linewidth=0.8, transform=ax.transAxes, clip_on=False)
    return table

def prepare_joint_density(samples, xvar, yvar, step):
    """Compute all densities once on one grid before model faceting."""
    kdes = []
    for label, _, _, x, y in samples:
        try:
            kdes.append(gaussian_kde(np.vstack([x, y]), bw_method='scott'))
        except np.linalg.LinAlgError as exc:
            raise ValueError(f'{label}: degenerate {xvar.label}-{yvar.label} KDE at step {step}') from exc
    xmin, xmax = (min((s[3].min() for s in samples)), max((s[3].max() for s in samples)))
    ymin, ymax = (min((s[4].min() for s in samples)), max((s[4].max() for s in samples)))
    xpad = 3 * max((np.sqrt(k.covariance[0, 0]) for k in kdes))
    ypad = 3 * max((np.sqrt(k.covariance[1, 1]) for k in kdes))
    if xvar.nonnegative and xmin < 0 or (yvar.nonnegative and ymin < 0):
        raise ValueError(f'Step {step}: negative values in a nonnegative relationship variable')
    gx = np.linspace(0 if xvar.nonnegative else xmin - xpad, xmax + xpad, 140)
    gy = np.linspace(0 if yvar.nonnegative else ymin - ypad, ymax + ypad, 140)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    cell_area = (gx[1] - gx[0]) * (gy[1] - gy[0])
    cell_weights = np.ones(xx.shape)
    if xvar.nonnegative:
        cell_weights[:, 0] *= 0.5
    if yvar.nonnegative:
        cell_weights[0, :] *= 0.5
    rows, entries = ([], [])
    statistics = relationship_statistics(samples, xvar, yvar)
    for (label, color, linestyle, x, y), kde, stats in zip(samples, kdes, statistics):
        density = kde(grid).reshape(xx.shape)
        if xvar.nonnegative:
            density += kde(np.vstack([-grid[0], grid[1]])).reshape(xx.shape)
        if yvar.nonnegative:
            density += kde(np.vstack([grid[0], -grid[1]])).reshape(xx.shape)
        if xvar.nonnegative and yvar.nonnegative:
            density += kde(-grid).reshape(xx.shape)
        order = np.argsort(density.ravel())[::-1]
        ordered = density.ravel()[order]
        if xvar.nonnegative or yvar.nonnegative:
            mass = np.cumsum(ordered * cell_weights.ravel()[order]) * cell_area
        else:
            mass = np.cumsum(ordered) * cell_area
        if mass[-1] < 0.98:
            raise ValueError(f'{label}: KDE grid covers insufficient probability mass')
        probabilities = (0.5, 0.75, 0.9, 0.95)
        threshold_by_probability = {p: float(ordered[min(np.searchsorted(mass, p), len(ordered) - 1)]) for p in (*probabilities, 0.8)}
        thresholds = np.unique([threshold_by_probability[p] for p in probabilities])
        rows.append({'relationship': f'{xvar.forecast}_vs_{yvar.forecast}', 'model': label, 'lead_hours': step * STEP_HOURS, 'count': len(x), 'hdr80_density_threshold': threshold_by_probability[0.8], **{f'hdr{int(p * 100)}_density_threshold': threshold_by_probability[p] for p in probabilities}, 'grid_probability_mass': float(mass[-1]), **stats})
        marginals = []
        for values, limits, var in ((x, gx, xvar), (y, gy, yvar)):
            marginal_grid = np.linspace(limits[0], limits[-1], 500)
            marginal_kde = gaussian_kde(values, bw_method='scott')
            marginal_values = marginal_kde(marginal_grid)
            if var.nonnegative:
                marginal_values += marginal_kde(-marginal_grid)
            marginals.append((marginal_grid, marginal_values))
        entries.append((label, color, linestyle, density, thresholds, marginals))
    return (gx, gy, entries, rows)
FIXED_DISPLAY_LIMITS = {'forecast_rmax_km': (0.0, 300.0), 'forecast_msl_min_hPa': (955.0, 1015.0), 'forecast_vmax': (7.5, 27.5), 'delta_z_sim_100km': (0.0, 150.0), 'delta_z_gwb_100km': (0.0, 150.0), 'forecast_b_param': (-30.0, 30.0), 'forecast_vtu': (-50.0, 250.0), 'forecast_vtl': (-25.0, 250.0)}

def display_limits(samples, xvar, yvar):
    """Use requested fixed bounds for displayed wind and structure parameters."""
    structure_fields = {var.forecast for var in STRUCTURE_VARIABLES.values()}
    limits = []
    for axis, var in ((3, xvar), (4, yvar)):
        if var.forecast in FIXED_DISPLAY_LIMITS:
            limits.append(FIXED_DISPLAY_LIMITS[var.forecast])
        elif var.forecast in structure_fields:
            limits.append((min((np.quantile(sample[axis], 0.025) for sample in samples)), max((np.quantile(sample[axis], 0.975) for sample in samples))))
        else:
            quantiles = (0.05, 1.0) if var == RELATIONSHIP_VARIABLES['msl'] else (0.0, 0.95)
            limits.append(tuple(np.quantile(samples[0][axis], quantiles)))
    return tuple(limits)

def joint_density(ax, ax_top, ax_right, samples, xvar, yvar, step, identity=False, prepared=None, model=None):
    gx, gy, entries, rows = prepare_joint_density(samples, xvar, yvar, step) if prepared is None else prepared
    selected_rows = []
    for (label, color, linestyle, density, thresholds, marginals), row in zip(entries, rows):
        if model is not None and label not in ('ERA5', model):
            continue
        if label == 'ERA5':
            gray_levels = [str(gray) for gray in np.linspace(0.6, 0.0, len(thresholds))]
            ax.contour(gx, gy, density, levels=thresholds, colors=gray_levels, linewidths=1.25, linestyles='-', zorder=3)
        else:
            rgb = np.array(to_rgb(color))
            shades = [tuple(1 - strength * (1 - rgb)) for strength in np.linspace(0.18, 0.82, len(thresholds))]
            ax.contourf(gx, gy, density, levels=[*thresholds, density.max() * 1.001], colors=shades, alpha=0.9, antialiased=True, zorder=1)
        selected_rows.append(row)
        for marginal, (grid, values), vertical in zip((ax_top, ax_right), marginals, (False, True)):
            if label == 'ERA5':
                if vertical:
                    marginal.plot(values, grid, color=color, linewidth=1.25, zorder=3)
                else:
                    marginal.plot(grid, values, color=color, linewidth=1.25, zorder=3)
            elif vertical:
                marginal.fill_betweenx(grid, 0, values, color=color, alpha=0.3, linewidth=0, edgecolor='none', zorder=1)
            else:
                marginal.fill_between(grid, 0, values, color=color, alpha=0.3, linewidth=0, edgecolor='none', zorder=1)
    xlimits, ylimits = display_limits(samples, xvar, yvar)
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax_top.set_xlim(*xlimits)
    ax_right.set_ylim(*ylimits)
    ax_top.set_ylim(0, 1.05 * max((e[5][0][1].max() for e in entries)))
    ax_right.set_xlim(0, 1.05 * max((e[5][1][1].max() for e in entries)))
    for marginal in (ax_top, ax_right):
        marginal.spines[:].set_visible(False)
        marginal.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    if identity:
        lo = max(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color='#aaaaaa', linewidth=0.7, linestyle=':', zorder=0)
    ax.set_xlabel(xvar.axis_label, fontsize=11)
    ax.set_ylabel(yvar.axis_label, fontsize=11)
    style_axis(ax)
    return selected_rows

def plot_relationship_panels(datasets, output_dir, step, dpi, relationships, variables, stem):
    fig = plt.figure(figsize=(14, 9))
    outer = fig.add_gridspec(2, 1, left=0.06, right=0.985, bottom=0.315, top=0.89, hspace=0.29)
    rows, groups = ([], [])
    for index, (title, xkey, ykey, identity) in enumerate(relationships):
        xvar, yvar = (variables[xkey], variables[ykey])
        samples = relationship_samples(datasets, xvar, yvar, step)
        prepared = prepare_joint_density(samples, xvar, yvar, step)
        panel_rows = prepared[3]
        groups.append((panel_rows, xvar, yvar))
        rows.extend(panel_rows)
        row_grid = outer[index, 0].subgridspec(1, 4, wspace=0.15)
        for col, spec in enumerate(MODEL_SPECS):
            kde_grid = row_grid[0, col].subgridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.03, wspace=0.03)
            ax = fig.add_subplot(kde_grid[1, 0])
            ax_top = fig.add_subplot(kde_grid[0, 0], sharex=ax)
            ax_right = fig.add_subplot(kde_grid[1, 1], sharey=ax)
            if index == 0:
                ax_top.set_title(spec.label, loc='left', fontsize=11, fontweight='bold', pad=5)
            joint_density(ax, ax_top, ax_right, samples, xvar, yvar, step, identity, prepared=prepared, model=spec.label)
            ax.tick_params(labelsize=9)
            if col > 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
        top = outer[index, 0].get_position(fig).y1
        fig.text(0.06, top + (0.037 if index == 0 else 0.002), f'({chr(97 + index)}) {title}', fontsize=12, fontweight='bold')
    fig.text(0.06, 0.245, '(c) Statistics', fontsize=12, fontweight='bold')
    ax_stats = fig.add_axes([0.06, 0.015, 0.925, 0.25])
    relationship_table(ax_stats, groups)
    fig.legend(handles=handles(), loc='upper center', bbox_to_anchor=(0.54, 0.995), ncol=5, frameon=False, fontsize=10, columnspacing=1.5, handlelength=2.1)
    save(fig, output_dir, stem, dpi)
    return rows

def plot_wind_physics(datasets, output_dir, step, dpi):
    return plot_relationship_panels(datasets, output_dir, step, dpi, RELATIONSHIPS, RELATIONSHIP_VARIABLES, 'tc_wind_pressure')

def marginal_density(ax, datasets, forecast, target, step, nonnegative=False):
    values = series(datasets, forecast, target, step)
    lo, hi = (min((s[3].min() for s in values)), max((s[3].max() for s in values)))
    kdes = [gaussian_kde(s[3]) for s in values]
    pad = 3 * max((float(np.sqrt(k.covariance[0, 0])) for k in kdes))
    grid = np.linspace(0 if nonnegative else lo - pad, hi + pad, 500)
    for (label, color, linestyle, _), kde in zip(values, kdes):
        density = kde(grid)
        if nonnegative:
            density += kde(-grid)
        if label == 'ERA5':
            ax.fill_between(grid, 0, density, color='#777777', alpha=0.13, linewidth=0)
        ax.plot(grid, density, color=color, linestyle=linestyle, linewidth=1.25)
    ax.set_ylim(bottom=0)
    ax.set_xlim(grid[0], grid[-1])

def plot_structure(datasets, output_dir, step, dpi):
    return plot_relationship_panels(datasets, output_dir, step, dpi, STRUCTURE_RELATIONSHIPS, STRUCTURE_VARIABLES, 'tc_structure_parameters')
