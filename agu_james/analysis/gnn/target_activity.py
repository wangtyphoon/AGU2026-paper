# Derived from gnn/ver_autoregressive/output/forecast_activity_by_step.py
from agu_james import runtime
import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def load_feature_names(csv_path):
    df = pd.read_csv(csv_path)
    if 'feature' not in df.columns:
        raise KeyError(f"{csv_path} missing 'feature' column")
    return df['feature'].astype(str).tolist()

def create_spatial_weights(height=192, width=192, boundary=8):
    lats_deg = np.linspace(0, 47.75, height)
    lat_w = np.cos(np.deg2rad(lats_deg))
    w2d = np.tile(lat_w[:, None], (1, width))
    w2d = np.flipud(w2d).copy()
    y_idx, x_idx = np.indices((height, width))
    dist_top = y_idx
    dist_bottom = height - 1 - y_idx
    dist_left = x_idx
    dist_right = width - 1 - x_idx
    min_dist = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])
    w2d[min_dist <= boundary] = 0.0
    mean_w = np.mean(w2d)
    if mean_w != 0:
        w2d = w2d / mean_w
    return w2d.astype(np.float64)

def parse_timestamp_from_filename(path, time_format='%Y%m%d%H'):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return datetime.strptime(name, time_format)
    except ValueError:
        pass
    digits = re.findall('(\\d{10})', name)
    for cand in digits:
        try:
            return datetime.strptime(cand, '%Y%m%d%H')
        except ValueError:
            continue
    return None

def parse_step_from_filename(path):
    name = os.path.basename(path)
    match = re.search('_step(\\d+)', name)
    if match:
        return int(match.group(1))
    match = re.search('step(\\d+)', name)
    return int(match.group(1)) if match else None

def load_pred_label_from_npz(npz_path, target_folder=None):
    data = np.load(npz_path)
    print(f'Loading npz: {npz_path}')
    keys = set(getattr(data, 'files', []))
    pred = None
    label = None
    for k_pred in ('forecast', 'pred', 'y_pred'):
        if k_pred in keys:
            pred = data[k_pred]
            break
    for k_label in ('target', 'label', 'y_true'):
        if k_label in keys:
            label = data[k_label]
            break
    if label is None and target_folder is not None:
        step_idx = parse_step_from_filename(npz_path)
        ts = parse_timestamp_from_filename(npz_path)
        candidate_name = os.path.basename(npz_path)
        if step_idx is not None and step_idx > 1 and (ts is not None):
            shifted_ts = ts + timedelta(hours=6 * (step_idx - 1))
            ts_str = shifted_ts.strftime('%Y%m%d%H')
            candidate_name = re.sub('\\d{10}', ts_str, candidate_name, count=1)
            candidate_name = re.sub('_step\\d+', '_step01', candidate_name, count=1)
        target_path = os.path.join(target_folder, candidate_name)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f'Missing target npz: {target_path}')
        tgt = np.load(target_path)
        tgt_keys = set(getattr(tgt, 'files', []))
        for k_label in ('target', 'label', 'y_true'):
            if k_label in tgt_keys:
                label = tgt[k_label]
                break
        if label is None:
            raise KeyError(f'{target_path} missing target/label/y_true')
    if pred is None or label is None:
        raise KeyError(f'{npz_path} missing forecast/pred or target/label')
    hw = None

    def reshape(arr, arr_name):
        nonlocal hw
        if arr.ndim == 4:
            batch, height, width, channels = arr.shape
            hw_here = (height, width)
            if hw is not None and hw != hw_here:
                raise ValueError(f'{npz_path} {arr_name} shape mismatch: {hw} vs {hw_here}')
            hw = hw_here
            return arr.reshape(batch * height * width, channels)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'{npz_path} {arr_name} unsupported shape: {arr.shape}')
    pred = reshape(pred, 'pred')
    label = reshape(label, 'label')
    if pred.shape != label.shape:
        raise ValueError(f'{npz_path} pred/label mismatch: {pred.shape} vs {label.shape}')
    return (pred, label, hw)

def build_time_file_map(clim_dir):
    mapping = {}
    for path in sorted(glob.glob(os.path.join(clim_dir, 'time_*.npz'))):
        data = np.load(path)
        if 'time' not in data:
            continue
        ts = pd.to_datetime(data['time']).to_pydatetime()
        if isinstance(ts, np.ndarray):
            ts = ts.item()
        key = (ts.timetuple().tm_yday, ts.hour)
        mapping[key] = path
    return mapping

def load_clim_npz_for_time(clim_dir, ts, time_file_map=None):
    dayofyear = ts.timetuple().tm_yday
    hour = ts.hour
    path = os.path.join(clim_dir, f'day{dayofyear:03d}_hour{hour:02d}.npz')
    if os.path.exists(path):
        return (np.load(path), time_file_map)
    if time_file_map is None:
        time_file_map = build_time_file_map(clim_dir)
    path = time_file_map.get((dayofyear, hour))
    if path and os.path.exists(path):
        return (np.load(path), time_file_map)
    return (None, time_file_map)

def _case_metrics(forecast, target, climatology, weights, epsilon=1e-12):
    forecast, target, climatology = np.broadcast_arrays(
        np.asarray(forecast,dtype=np.float64),np.asarray(target,dtype=np.float64),np.asarray(climatology,dtype=np.float64))
    weights=np.asarray(weights,dtype=np.float64)
    if forecast.ndim not in (1,2) or weights.ndim!=1 or len(weights)!=len(forecast):
        raise ValueError('One case must have shape (nodes,) or (nodes, channels)')
    if not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Weights must be finite and nonnegative')
    scalar=forecast.ndim==1
    keep=weights>0;w=weights[keep];sw=weights.sum()
    if sw<=epsilon:
        n=np.full(forecast.shape[1:] or (),np.nan)
        return n,n.copy(),n.copy(),n.copy()
    f=forecast[keep]-climatology[keep];t=target[keep]-climatology[keep]
    if scalar:f=f[:,None];t=t[:,None]
    vf=np.isfinite(f).all(axis=0);vt=np.isfinite(t).all(axis=0)
    def product(a,b):return np.einsum('ij,ij,i->j',a,b,w,optimize=False)
    ff=product(f,f);tt=product(t,t);ft=product(f,t)
    fm=np.einsum('ij,i->j',f,w,optimize=False)/sw
    tm=np.einsum('ij,i->j',t,w,optimize=False)/sw
    f-=fm;t-=tm
    fc=product(f,f);tc=product(t,t)
    with np.errstate(invalid='ignore',divide='ignore'):
        den=np.sqrt(ff*tt);old_den=np.sqrt(fc*tc)
        acc=np.where(vf & vt & (den>epsilon),ft/den,np.nan)
        activity=np.where(vf,np.sqrt(fc/sw),np.nan)
        target_activity=np.where(vt,np.sqrt(tc/sw),np.nan)
        legacy_acc=np.where(vf & vt & (old_den>epsilon),product(f,t)/old_den,np.nan)
    result=(acc,activity,target_activity,legacy_acc)
    return tuple(float(a[0]) for a in result) if scalar else result

def compute_acc_activity_weighted(forecast,target,climatology,weights,epsilon=1e-12):
    """One case: uncentered climatological ACC and centered spatial activity."""
    result=_case_metrics(forecast,target,climatology,weights,epsilon)
    return result[0],result[1]

def aggregate_activity(values,axis=0):
    return np.sqrt(np.nanmean(np.square(np.asarray(values,dtype=np.float64)),axis=axis))

def compute_activity_weighted(values, climatology, weights, epsilon=1e-12):
    return compute_acc_activity_weighted(values, values, climatology, weights, epsilon)[1]

def main(forecast_folder='.', target_folder=None, clim_folder='climate_meteology',
         feature_csv='next_feature_table_weight.csv', save_csv='per_channel_forecast_activity_summary.csv',
         per_file_csv='per_file_forecast_activity.csv', step_hours=6, boundary=8):
    from collections import OrderedDict
    from pathlib import Path
    from io import BytesIO
    import hashlib
    import json
    is_target = True
    names = load_feature_names(feature_csv)
    files = sorted(Path(forecast_folder).glob('*.npz'))
    if not files:
        raise FileNotFoundError(forecast_folder)
    caches = {'target': OrderedDict(), 'clim': OrderedDict()}
    fingerprints = {}
    rows, accs, acts, oldaccs, targetacts = [], [], [], [], []
    warnings = []
    def read(path, cache=None):
        path = Path(path)
        key = str(path.resolve())
        bucket = caches.get(cache)
        if bucket is not None and key in bucket:
            bucket.move_to_end(key)
            return bucket[key]
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Input changed during read: {path}')
        fingerprints[key] = {'size': len(raw), 'mtime_ns': after.st_mtime_ns,
                             'sha256': hashlib.sha256(raw).hexdigest()}
        with np.load(BytesIO(raw)) as archive:
            result = {k: archive[k] for k in archive.files}
        if bucket is not None:
            bucket[key] = result
            if len(bucket) > 48:
                bucket.popitem(last=False)
        return result
    def pick(data, keys):
        return next((data[k] for k in keys if k in data), None)
    def cases(array, h, w):
        array = np.asarray(array)
        if array.ndim == 4:
            if array.shape[1:3] != (h,w):
                raise ValueError('Spatial shape mismatch')
            return array.reshape(array.shape[0], h*w, array.shape[-1])
        if array.ndim == 2 and array.shape[0] % (h*w) == 0:
            return array.reshape(-1,h*w,array.shape[-1])
        raise ValueError(f'Unsupported case shape: {array.shape}')
    for index, path in enumerate(files):
        step = parse_step_from_filename(str(path))
        initial = parse_timestamp_from_filename(str(path))
        if initial is None or step is None:
            warnings.append({'file': path.name, 'reason': 'missing timestamp/step'})
            continue
        ts = initial + timedelta(hours=step_hours*step)
        clim_path = Path(clim_folder) / f'day{ts.timetuple().tm_yday:03d}_hour{ts.hour:02d}.npz'
        if not clim_path.exists():
            clim_data, _ = load_clim_npz_for_time(clim_folder, ts)
            if clim_data is None:
                warnings.append({'file':path.name,'reason':'missing climatology'})
                continue
            with clim_data:
                climate = {k:clim_data[k] for k in clim_data.files}
        else:
            climate = read(clim_path, 'clim')
        available = [i for i,name in enumerate(names) if name in climate]
        if not available:
            raise ValueError(f'No climatology channels: {path}')
        h,w = np.asarray(climate[names[available[0]]]).squeeze().shape
        c = np.full((h*w,len(names)),np.nan)
        for ch in available:
            field = np.asarray(climate[names[ch]]).squeeze()
            if field.shape != (h,w):
                raise ValueError(f'Climate shape mismatch: {names[ch]}')
            c[:,ch] = field.reshape(-1)
        # Target-only jobs need the same forecast roster, not the forecast values.
        data = {} if is_target and target_folder else read(path)
        target = pick(data, ('target','label','y_true'))
        if target is None:
            if target_folder is None:
                raise ValueError(f'No target: {path}')
            target_name = path.name
            if step > 1:
                target_time = initial + timedelta(hours=step_hours*(step-1))
                target_name = re.sub(r'\d{10}', target_time.strftime('%Y%m%d%H'), target_name, count=1)
                target_name = re.sub(r'_step\d+', '_step01', target_name, count=1)
            target = pick(read(Path(target_folder)/target_name,'target'),('target','label','y_true'))
        truth = cases(target,h,w)
        pred = truth if is_target else cases(pick(data,('forecast','pred','y_pred')),h,w)
        if pred.shape != truth.shape or pred.shape[-1] != len(names):
            raise ValueError(f'Forecast/target/channel mismatch: {path}')
        weights = create_spatial_weights(h,w,boundary).reshape(-1)
        for case in range(len(pred)):
            acc,activity,ta,oldacc = _case_metrics(pred[case],truth[case],c,weights)
            row = {'file':path.name,'timestamp':ts.strftime('%Y-%m-%d %H:%M:%S'),
                   'initial_time':initial.isoformat(),'step':step,'case_index':case}
            if is_target:
                row['target_activity_mean'] = float(np.nanmean(ta))
            else:
                row.update(acc_mean=float(np.nanmean(acc)),activity_mean=float(np.nanmean(activity)))
            rows.append(row);accs.append(acc);acts.append(activity);oldaccs.append(oldacc);targetacts.append(ta)
        if (index+1)%100 == 0:
            print(f'{Path(forecast_folder).name}: {index+1}/{len(files)}',flush=True)
    if not rows:
        raise RuntimeError('No valid cases processed')
    accs,acts,oldaccs,targetacts = map(np.asarray,(accs,acts,oldaccs,targetacts))
    frame = pd.DataFrame(rows)
    result = {'feature':names}
    values = targetacts if is_target else acts
    prefix = 'target_activity' if is_target else 'activity'
    result[prefix+'_overall'] = aggregate_activity(values)
    if not is_target:
        result['acc_overall'] = np.nanmean(accs,axis=0)
    counts = []
    for step in sorted(frame.step.unique()):
        mask = frame.step.to_numpy()==step
        result[f'{prefix}_step{step:02d}'] = aggregate_activity(values[mask])
        if not is_target:
            result[f'acc_step{step:02d}'] = np.nanmean(accs[mask],axis=0)
        counts.extend({'step':int(step),'feature':name,'activity_count':int(np.isfinite(values[mask,ch]).sum()),
                       'acc_count':int(np.isfinite(accs[mask,ch]).sum())} for ch,name in enumerate(names))
    summary = pd.DataFrame(result)
    valid = summary[[prefix+'_overall']].notna().any(axis=1)
    if not is_target:
        valid |= summary.acc_overall.notna()
    summary[valid].to_csv(save_csv,index=False)
    frame.to_csv(per_file_csv,index=False)
    audit = Path(per_file_csv).with_suffix('')
    np.savez_compressed(str(audit)+'_cases.npz', acc=accs,activity=acts,legacy_acc=oldaccs,
                        target_activity=targetacts,features=np.asarray(names),steps=frame.step.to_numpy())
    pd.DataFrame(counts).to_csv(str(audit)+'_counts.csv',index=False)
    Path(str(audit)+'_manifest.json').write_text(json.dumps({'definition':{'acc':'mean case uncentered climatological ACC',
        'activity':'sqrt(mean case centered spatial variance)'},'forecast_folder':str(forecast_folder),
        'target_folder':str(target_folder),'clim_folder':str(clim_folder),'boundary':boundary,
        'case_count':len(frame),'file_count':len(files),'warnings':warnings,'inputs':fingerprints},indent=2),encoding='utf-8')
    print(f'Saved {save_csv}; cases={len(frame)}; warnings={len(warnings)}',flush=True)
