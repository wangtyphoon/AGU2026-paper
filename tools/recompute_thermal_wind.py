"""Replay all four models with staged native-level code; preserve other CSV cells."""
import ast
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agu_james.analysis.gnn import tc_base, tc_phase
from agu_james.analysis.pangu import axisymmetric_b_vt_metrics_all as pg
OUT = ROOT/'artifacts/thermal_wind_native_levels'
LEVELS = [925, 850, 700, 600, 500, 400, 300]
COLS = ['forecast_vtl', 'forecast_vtu', 'target_vtl', 'target_vtu']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numerical_namespace(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    names = {'native_delta_z', 'compute_native_vt', 'regression_slope_ln_p', 'compute_vt_indicators'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 ('LOWER_PRESSURES', 'UPPER_PRESSURES') for t in n.targets)]
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def validate_candidates():
    results = []
    for entry in json.loads((OUT/'code_replacement/replacement_manifest.json').read_text()):
        if entry['kind'] == 'plot':
            continue
        ns = numerical_namespace(entry['staged'])
        levels = np.asarray(LEVELS)
        for slope in (-80., 0., 80.):
            profile = 1000+slope*np.log(levels)
            result = ns['compute_native_vt'](levels, profile)
            for layer in ('lower', 'upper'):
                assert np.isclose(result['vt_'+layer], slope, atol=1e-9)
                assert result[layer+'_valid_levels'] == 4
        missing = np.asarray([np.nan, np.nan, np.nan, 200., 190., 180., 170.])
        r = ns['compute_native_vt'](levels, missing)
        assert np.isnan(r['vt_lower']) and r['lower_valid_levels'] == 1
        partial = np.asarray([300., np.nan, np.nan, 200., 190., 180., 170.])
        r = ns['compute_native_vt'](levels, partial)
        assert r['lower_valid_levels'] == 2 and np.isfinite(r['vt_lower'])
        expected = 100/(np.log(925)-np.log(600))
        assert np.isclose(r['vt_lower'], expected)
        # Extraneous interpolated layers cannot affect either native regression.
        a = ns['compute_native_vt'](list(levels)+[900, 550], list(partial)+[1e9, -1e9])
        assert a == r
        if 'compute_vt_indicators' in ns:
            b = ns['compute_vt_indicators'](dict(zip(levels, partial)))
            assert b['vt_lower'] == r['vt_lower'] and b['vt_upper'] == r['vt_upper']
        results.append(dict(name=entry['name'], passed=True))
    (OUT/'candidate_checks.json').write_text(json.dumps(results, indent=2))
    print('Five analysis candidates passed signed/zero/missing/native-only checks', flush=True)

class Cache:
    def __init__(self, size=12):
        self.items = OrderedDict(); self.size = size; self.sources = {}
    def get(self, path, loader):
        path = Path(path)
        if path in self.items:
            self.items.move_to_end(path); return self.items[path]
        stat = path.stat()
        self.sources[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        value = loader(path)
        self.items[path] = value
        if len(self.items) > self.size: self.items.popitem(last=False)
        return value
    def verify(self):
        for path, before in self.sources.items():
            stat = Path(path).stat()
            assert before == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns), path

def run_model(m):
    source = ROOT/'artifacts/gwb_endpoint_corrected/tracker_multi'/m['relative_path']
    before_hash = sha(source)
    frame = pd.read_csv(source)
    with source.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f); headers = reader.fieldnames; records = list(reader)
    assert len(records) == len(frame)
    lat, lon = tc_base.to_1d_lat_lon(np.load(m['lat']), np.load(m['lon']))
    cache = Cache(); run_axes = {}; audit = []
    staged_name = 'agu_gnn' if m['kind'] == 'gnn' else 'agu_pangu'
    ns = numerical_namespace(OUT/f'code_replacement/{staged_name}.replacement.py.txt')
    if m['kind'] == 'gnn':
        channel_map = tc_base.get_channel_map(Path(m['features']), {f'z{p}':[f'{p}z'] for p in LEVELS})
        index = {}
        for path in Path(m['forecast_root']).rglob('*.npz'):
            if path.name in index: raise ValueError(f'Ambiguous source {path.name}')
            index[path.name] = path
        def load_npz(path, role):
            with np.load(path) as data:
                key = next((k for k in (('forecast','pred','y_pred') if role=='forecast' else ('target','label','y_true')) if k in data.files), None)
                if key is None: raise KeyError(f'{path}: missing {role}')
                field = tc_base.normalize_to_CHW(data[key], lat.size, lon.size)
                return {f'z{p}':field[channel_map[f'z{p}']].copy() for p in LEVELS}
    else:
        cfg = pg.Config(upper_nc_path=Path(m['upper_nc']))
        channel_map = pg.build_channel_map(LEVELS)
    values = np.full((len(frame), 4), np.nan)
    for i, row in frame.iterrows():
        for role in ('forecast', 'target'):
            center_lat = row['forecast_center_lat' if m['kind']=='gnn' else 'forecast_lat'] if role=='forecast' else row['bst_lat']
            center_lon = row['forecast_center_lon' if m['kind']=='gnn' else 'forecast_lon'] if role=='forecast' else row['bst_lon']
            path = None; reason = ''; count_l = count_u = 0; profile = {}
            if not np.isfinite([center_lat, center_lon]).all():
                reason = 'missing_center'
            else:
                if m['kind'] == 'gnn':
                    forecast_path = index[str(row.source_file)]
                    if role == 'forecast': path = forecast_path
                    else:
                        # Mirror load_forecast_and_target valid-time mapping.
                        step = tc_base.parse_step_from_filename(forecast_path) or 1
                        dt = tc_base.parse_timestamp_from_filename(forecast_path)
                        import re
                        name = forecast_path.name
                        if step > 1 and dt is not None:
                            name = re.sub(r'\d{10}', (dt+timedelta(hours=6*(step-1))).strftime('%Y%m%d%H'), name, count=1)
                            name = re.sub(r'_step\d+', '_step01', name, count=1)
                        path = Path(m['target_root'])/name
                    fields = cache.get(path, lambda p: load_npz(p, role))
                    mask = tc_phase.radius_mask(lat, lon, float(center_lat), tc_base.normalize_lon(lon, float(center_lon)), 500.)
                    for level in LEVELS:
                        v = fields[f'z{level}'][mask]/9.80665
                        v = v[np.isfinite(v)]
                        profile[level] = float(np.max(v)-np.min(v)) if v.size else np.nan
                else:
                    if role == 'forecast':
                        directory = Path(m['forecast_root'])/pg.dt_to_run_dir(pg.parse_ymdh(row.init_time))
                        path = directory/pg.dt_to_upper_name(pg.parse_ymdh(row.time))
                        if directory not in run_axes:
                            lp, op = directory/'latitudes_subset.npy', directory/'longitudes_subset.npy'
                            run_axes[directory] = tc_base.to_1d_lat_lon(np.load(lp), np.load(op)) if lp.exists() and op.exists() else (lat,lon)
                        axlat, axlon = run_axes[directory]
                    else:
                        path = Path(m['target_root'])/pg.dt_to_era5_dir(pg.parse_ymdh(row.time))/'input_upper.npy'
                        axlat, axlon = lat, lon
                    fields = cache.get(path, lambda p: pg.fields_from_upper(np.load(p, mmap_mode='r'), axlat, axlon, cfg, channel_map))
                    profile = pg.compute_delta_z_profile(fields, axlat, axlon, float(center_lat), float(center_lon), 500.)
                result = ns['compute_native_vt'](list(profile), list(profile.values()))
                count_l, count_u = result['lower_valid_levels'], result['upper_valid_levels']
                offset = 0 if role=='forecast' else 2
                values[i, offset:offset+2] = result['vt_lower'], result['vt_upper']
                reason = 'no_finite_native_profile' if not profile else 'insufficient_layers' if min(count_l,count_u)<2 else ''
            audit.append(dict(row=i, model=m['label'], role=role, time=row.time, step=row.step,
                source=str(path) if path else '', lower_valid_levels=count_l, upper_valid_levels=count_u,
                reason=reason, **{f'delta_z_core_{p}':profile.get(p, np.nan) for p in LEVELS}))
        if (i+1)%500 == 0: print(f"{m['id']}: {i+1}/{len(frame)}", flush=True)
    output = OUT/'tracker_multi'/m['relative_path']; output.parent.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        for j, col in enumerate(COLS): record[col] = repr(float(values[i,j])) if np.isfinite(values[i,j]) else ''
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader(); writer.writerows(records)
    with output.open(newline='', encoding='utf-8') as f: reread = list(csv.DictReader(f))
    with source.open(newline='', encoding='utf-8') as f: original = list(csv.DictReader(f))
    for old, new in zip(original, reread):
        assert all(old[k]==new[k] for k in headers if k not in COLS)
    loaded = pd.read_csv(output)
    np.testing.assert_allclose(loaded[COLS], values, rtol=1e-12, atol=1e-10, equal_nan=True)
    cache.verify(); assert sha(source)==before_hash
    pd.DataFrame(audit).to_csv(OUT/f"{m['id']}_profile_audit.csv", index=False)
    changes = frame[[k for k in ('time','forecast_init_time','init_time','step','CYCLONE_NUM','cyclone_num') if k in frame]].copy()
    changes['model'] = m['label']
    for j,col in enumerate(COLS):
        changes[col+'_old'] = frame[col]; changes[col+'_new'] = values[:,j]
        changes[col+'_difference'] = values[:,j]-frame[col]
    changes.to_csv(OUT/f"{m['id']}_case_changes.csv", index=False)
    manifest = dict(model=m['label'], rows=len(frame), input=str(source), input_sha256=before_hash,
        output=str(output), output_sha256=sha(output), non_thermal_cells_identical=True,
        data_fingerprints=cache.sources, finite_values=int(np.isfinite(values).sum()),
        audit_reason_counts=pd.Series([a['reason'] for a in audit]).value_counts().to_dict())
    (OUT/f"{m['id']}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"{m['id']}: complete, {np.isfinite(values).sum()} finite diagnostics", flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validate_candidates()
    config = json.loads((ROOT/'configs/gwb.local.json').read_text())
    ids = sys.argv[1:] or [m['id'] for m in config['models']]
    for m in config['models']:
        if m['id'] in ids: run_model(m)

if __name__ == '__main__': main()
