"""Redraw signed-B Q95 diagnostics using the existing native-level inputs."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agu_james.plotting import tc_distributions as distribution
from agu_james.plotting import tc_physics as physics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'artifacts/thermal_wind_native_levels'
    output = ROOT / 'artifacts/b_tail_q95'
    output.mkdir(exist_ok=True)
    inputs = [source / 'tracker_multi' / spec.relative_path for spec in physics.MODEL_SPECS]
    baseline = source / 'figure10/tc_structure_relationship_distribution.csv'
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in inputs + [baseline]}
    raw, _ = physics.load_scalar_data(source / 'tracker_multi', 12)
    audit = []
    common = physics.common_for_variables(raw, physics.ALL_KEYS, audit, 'all_parameters')
    reference = common['MSE']
    for frame in common.values():
        for var in physics.VARIABLES:
            frame[var.target] = reference[var.target].to_numpy()
    reference.loc[reference.step == 12, physics.CASE_KEY_COLS].to_csv(output / 'common_cases.csv', index=False)
    assert distribution.STRUCTURE_VARIABLES['b'].tail_quantile == .95
    rows = distribution.plot_structure(common, output, 12, 600)
    updated = pd.DataFrame(rows)
    updated.to_csv(output / baseline.name, index=False)
    previous = pd.read_csv(baseline)
    keys = ['relationship', 'model', 'lead_hours']
    before = previous.set_index(keys).sort_index()
    after = updated.set_index(keys).sort_index()
    assert before.index.equals(after.index)
    changes = []
    allowed = {'y_tail_quantile', 'y_tail_value', 'era5_y_tail_value', 'tail_bias_y'}
    for column in before.columns:
        a, b = before[column], after[column]
        if pd.api.types.is_numeric_dtype(a):
            same = np.isclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
        else:
            same = a.fillna('').to_numpy() == b.fillna('').to_numpy()
        for index in np.flatnonzero(~same):
            key = before.index[index]
            assert key[0] == 'forecast_rmax_km_vs_forecast_b_param' and column in allowed, (key, column)
            changes.append(dict(zip(keys, key), column=column, old=a.iloc[index], new=b.iloc[index]))
    pd.DataFrame(changes).to_csv(output / 'changed_cells.csv', index=False)
    for spec in physics.MODEL_SPECS:
        sub = common[spec.label].loc[common[spec.label].step == 12]
        row = updated.loc[(updated.model == spec.label) & (updated.relationship == 'forecast_rmax_km_vs_forecast_b_param')].iloc[0]
        assert len(sub) == row['count'] == 151
        np.testing.assert_allclose(row.y_tail_value, np.quantile(sub.forecast_b_param, .95))
    for path in inputs + [baseline]:
        assert sha(path) == fingerprints[str(path.relative_to(ROOT))]
    files = [output / name for name in ('common_cases.csv', baseline.name, 'changed_cells.csv',
             'tc_structure_parameters.png', 'tc_structure_parameters.pdf')]
    manifest = {
        'change': 'signed B tail Q05 -> Q95; no absolute-value transformation',
        'tail_quantiles': {key: var.tail_quantile for key, var in distribution.STRUCTURE_VARIABLES.items()},
        'lead_hours': 72, 'common_cases': 151,
        'inputs_sha256': fingerprints,
        'code_sha256': sha(ROOT / 'agu_james/plotting/tc_distributions.py'),
        'text': 'docs/TC_STRUCTURE_TEXT.md',
        'outputs_sha256': {str(path.relative_to(ROOT)): sha(path) for path in files},
        'verification': 'Only B quantile probability, forecast/reference tail values and tail bias changed; all other CSV fields reproduced within 1e-10 tolerance. Inputs and previous artifacts preserved.',
        'uncertainty': 'No new confidence intervals or dependence-adjusted p-values.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(updated.loc[updated.relationship == 'forecast_rmax_km_vs_forecast_b_param',
                      ['model', 'count', 'y_tail_quantile', 'y_tail_value', 'tail_bias_y']].to_string(index=False))


if __name__ == '__main__':
    main()
