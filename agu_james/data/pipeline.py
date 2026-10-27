"""Best-track selection, ERA5 conversion and training-only statistics.

All files produced here go to artifact_root. Original inputs are read-only.
Derived from the bwp/bst/era5/npz research workflows.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import csv
import importlib
import json
import shutil
import numpy as np
import pandas as pd
from agu_james import runtime
from agu_james.contracts import split_for_year, LEVELS, valid_initializations

def channel_config(kind="ar"):
    return json.loads((Path(__file__).parents[1] / "config" / f"channels_{kind}.json").read_text(encoding="utf-8"))

def best_track(source):
    """Read JTWC ATCF .dat/.txt or the repository's normalized CSV schema."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in {".dat", ".txt"})
        rows = []
        for file in files:
            for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = [v.strip() for v in line.split(",")]
                if len(fields) < 9 or fields[4] != "BEST":
                    continue
                rows.append(dict(BASIN=fields[0], CYCLONE_NUM=fields[1], DATE_TIME=fields[2], TECH=fields[4],
                                 LAT=fields[6], LON=fields[7], VMAX=fields[8]))
        frame = pd.DataFrame(rows)
    required = ["DATE_TIME", "CYCLONE_NUM", "LAT", "LON", "VMAX"]
    if any(c not in frame for c in required):
        raise ValueError(f"Best-track input must contain {required}")
    def coordinate(value):
        value = str(value).strip()
        if value[-1:] in "NSEW" and value[-1:]:
            return float(value[:-1]) / 10 * (-1 if value[-1] in "SW" else 1)
        return float(value)
    for c in ["LAT", "LON"]:
        frame[c] = frame[c].map(coordinate)
    frame["DATE_TIME"] = pd.to_datetime(frame.DATE_TIME.astype(str), format="%Y%m%d%H")
    frame["VMAX"] = pd.to_numeric(frame.VMAX, errors="raise")
    frame = frame[(frame.DATE_TIME.dt.year >= 1979) & (frame.DATE_TIME.dt.year <= 2024)]
    if 'BASIN' in frame:frame=frame[frame.BASIN.astype(str).str.upper()=='WP']
    keys = [c for c in ["BASIN", "CYCLONE_NUM", "DATE_TIME"] if c in frame]
    frame = frame.drop_duplicates(keys).sort_values("DATE_TIME")
    inside = frame.LAT.between(0, 47.75) & frame.LON.between(102.25, 150)
    days = sorted({(t + timedelta(days=d)).normalize() for t in frame.loc[inside, "DATE_TIME"] for d in range(-3, 4)
                   if 1979 <= (t + timedelta(days=d)).year <= 2024})
    if not days:
        raise ValueError("No in-domain best-track dates")
    out = runtime.artifact_root() / "data"
    raw = out / "raw_grib"
    raw.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "cyclone_data_all.csv", index=False, date_format="%Y%m%d%H")
    frame[frame.DATE_TIME.dt.year == 2024].to_csv(out / "cyclone_data_2024.csv", index=False, date_format="%Y%m%d%H")
    pd.DataFrame({"DATE": days}).to_csv(raw / "time_series.csv", index=False)
    runtime.write_json(out / "selection_manifest.json", dict(source=str(source.resolve()),
                       criterion="best track inside 0..47.75N,102.25..150E; date +/-3 days", days=len(days),
                       records=len(frame), time_sampling_hours=[0,6,12,18], vmax_unit="kt", split_policy="calendar year"))
    return out

def download():
    raw = runtime.artifact_root() / "data/raw_grib"
    for d in ["prs", "sfc"]:
        (raw / d).mkdir(parents=True, exist_ok=True)
    config = dict(runtime._CONFIG)
    overrides = config.setdefault("overrides", {})
    for name in ["download_prs", "download_sfc_regular", "download_sfc_average"]:
        module = "agu_james.data.requests." + name
        overrides[module + ".current_dir"] = str(raw)
    runtime.configure(config)
    for name in ["download_prs", "download_sfc_regular", "download_sfc_average"]:
        importlib.import_module("agu_james.data.requests." + name).main()

def time_features(t, lon):
    h = t.hour / 24 + lon / 360
    day = t.timetuple().tm_yday
    y = (day + h) / 365.2425
    return {"hour_ratio":h, "hour_ratio_sin":np.sin(2*np.pi*h), "hour_ratio_cos":np.cos(2*np.pi*h),
            "year_ratio":y, "year_ratio_sin":np.sin(2*np.pi*y), "year_ratio_cos":np.cos(2*np.pi*y)}

def update_ocean_mask(masks, name, array, split, stamp):
    """Keep the training frame with most missing cells, before background filling."""
    if split != 'train' or name not in {'sst','swh'}:
        return
    valid=np.isfinite(array)
    missing=int((~valid).sum())
    if name not in masks or missing > masks[name]['missing_cells']:
        masks[name]=dict(mask=valid,missing_cells=missing,reference_time=stamp)

def ocean_masks():
    masks={}
    for name in ['sst','swh']:
        file=runtime.path(name+'_mask',runtime.artifact_root()/f'data/masks/weight_mask_{name}.npz')
        if not file.is_file():
            raise FileNotFoundError(f'{file}: generate pre-fill ocean masks via convert, or set paths.{name}_mask for an existing dataset')
        with np.load(file) as z:mask=np.asarray(z['weight_mask'])
        if mask.shape != (192,192) or not np.isin(mask,[0,1]).all() or not mask.any():
            raise ValueError(f'Invalid binary 192x192 ocean mask: {file}')
        masks[name]=mask.astype(bool)
    return masks

def convert(raw_dir=None):
    """Convert cfgrib groups using valid time, including time+step accumulations."""
    import cfgrib
    from agu_james.data.nearest_fill import nearest_neighbor_fill
    from agu_james.data.wave_interpolation import resize_swh_bilinear
    raw = Path(raw_dir or runtime.artifact_root() / "data/raw_grib")
    root = runtime.artifact_root() / "data"
    root.mkdir(parents=True,exist_ok=True)
    files = sorted(raw.rglob("*.grib"))
    if not files:
        raise FileNotFoundError(f"No GRIB files under {raw}")
    payloads = defaultdict(dict)
    masks={}
    for name in ['sst','swh']:
        file=root/f'masks/weight_mask_{name}.npz'
        if file.is_file():
            with np.load(file) as z:
                masks[name]=dict(mask=z['weight_mask'].astype(bool),missing_cells=int(z['missing_cells']),reference_time=str(z['reference_time']))
    for file in files:
        group = "prs" if file.parent.name == "prs" else "sfc/average" if "_average" in file.stem else "sfc/regular"
        for ds in cfgrib.open_datasets(str(file)):
            try:
                dims = [d for d in ["time", "step"] if d in ds.dims]
                import itertools
                for indices in itertools.product(*(range(ds.sizes[d]) for d in dims)):
                    sub = ds.isel(dict(zip(dims, indices)))
                    valid = np.asarray(sub.valid_time.values if "valid_time" in sub.coords else sub.time.values).item()
                    t = pd.Timestamp(valid).to_pydatetime()
                    if t.hour not in (0,6,12,18) or t.minute:
                        continue
                    split = split_for_year(t.year)
                    lat = np.asarray(sub.latitude.values)
                    lon = np.asarray(sub.longitude.values)
                    for name, da in sub.data_vars.items():
                        if name == "meanSea":continue
                        lev_dim = "isobaricInhPa" if "isobaricInhPa" in da.dims else None
                        levels = np.asarray(da.isobaricInhPa.values).reshape(-1) if "isobaricInhPa" in da.coords else [None]
                        for level in levels:
                            arr = np.asarray(da.sel(isobaricInhPa=level).values if lev_dim else da.values).squeeze()
                            if name == "swh" and arr.shape != (192,192):
                                arr = resize_swh_bilinear(arr, (192,192))
                            if arr.shape != (192,192):
                                raise ValueError(f"{file}/{name}: expected 192x192, got {arr.shape}")
                            update_ocean_mask(masks,name,arr,split,t.strftime('%Y%m%d%H'))
                            # Retain original SST/SWH backgrounds. Other missing fields use nearest neighbor.
                            if name == "sst":arr=np.where(np.isnan(arr),270.0,arr)
                            elif name == "swh":arr=np.where(np.isnan(arr),0.0,arr)
                            elif np.isnan(arr).any():arr=nearest_neighbor_fill(arr)
                            if not np.isfinite(arr).all():raise ValueError(f"Nonfinite values remain: {file}/{name}")
                            key = f"{int(level)}{name}" if level is not None else name
                            payloads[(split, group, t.strftime("%Y%m%d%H"))][key] = arr.astype(np.float32)
                    if lat.size==192 and lon.size==192:
                        xx,yy=np.meshgrid(lon,lat)
                        np.save(root/"lon.npy",xx.astype(np.float32));np.save(root/"lat.npy",yy.astype(np.float32))
            finally:ds.close()
        # Flush each monthly file to avoid retaining the full archive in memory.
        for (split, group, stamp), arrays in payloads.items():
            dest=root/split/group/(stamp+".npz");dest.parent.mkdir(parents=True,exist_ok=True)
            existing={}
            if dest.exists():
                with np.load(dest) as z:existing={k:z[k] for k in z.files}
            existing.update(arrays);np.savez(dest,**existing)
        payloads.clear()
    mask_root=root/'masks';mask_root.mkdir(parents=True,exist_ok=True)
    for name,item in masks.items():
        np.savez(mask_root/f'weight_mask_{name}.npz',weight_mask=item['mask'].astype(np.float32),missing_cells=item['missing_cells'],reference_time=item['reference_time'])
    runtime.write_json(mask_root/'manifest.json',dict(policy='training frame with largest pre-fill missing-cell count; first encountered tie retained',
        source_policy='npz/create_weight_mask.py; explicit training-only selection replaces hardcoded snapshot counts',
        masks={k:{q:v for q,v in item.items() if q!='mask'} for k,item in masks.items()},
        equivalence='generated mask may differ from historical mask; use explicit external mask paths for historical checkpoints'))
    lon=np.load(root/"lon.npy")
    for split in ["train","val","test"]:
        for file in sorted((root/split/"prs").glob("*.npz")):
            t=datetime.strptime(file.stem,"%Y%m%d%H")
            dest=root/split/"time"/file.name;dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez(dest,**{k:np.asarray(v,dtype=np.float32) for k,v in time_features(t,lon).items()})
    return root

def statistics(train_root=None):
    """Population statistics and normalized 6-h increment scales, training years only."""
    root=Path(train_root or runtime.artifact_root()/"data/train")
    stats_root=runtime.artifact_root()/"statistics";stats_root.mkdir(parents=True,exist_ok=True)
    stats={};masks=ocean_masks()
    for group, filename in [("prs","prs_variable_stats.csv"),("sfc/regular","sfc_regular_variable_stats.csv"),("sfc/average","sfc_average_variable_stats.csv")]:
        acc={}
        files=sorted((root/group).glob("*.npz"))
        if not files:raise FileNotFoundError(root/group)
        for file in files:
            if split_for_year(int(file.stem[:4]))!="train":raise ValueError(f"Non-training file in statistics: {file}")
            with np.load(file) as z:
                for name in z.files:
                    a=z[name].astype(np.float64)
                    if name in masks:a=a[masks[name]]
                    a=a[np.isfinite(a)]
                    if not a.size:continue
                    d=acc.setdefault(name,[0,0.0,0.0,float('inf'),float('-inf')])
                    d[0]+=a.size;d[1]+=a.sum();d[2]+=(a*a).sum();d[3]=min(d[3],a.min());d[4]=max(d[4],a.max())
        rows=[]
        for name,(count,s,s2,mn,mx) in sorted(acc.items()):
            mean=s/count;std=np.sqrt(max(0,s2/count-mean*mean))
            row=dict(variable=name,mean=mean,std=std,min=mn,max=mx,count=count)
            stats[name]=row;rows.append(row)
        pd.DataFrame(rows).to_csv(stats_root/filename,index=False)
    channels=channel_config()
    times={datetime.strptime(p.stem,"%Y%m%d%H"):p.name for p in (root/'prs').glob('*.npz')}
    sums=np.zeros(116);squares=sums.copy();counts=sums.copy()
    def read(t):
        arrays={}
        for group in ['prs','sfc/regular','sfc/average']:
            with np.load(root/group/times[t]) as z:arrays.update({k:z[k] for k in z.files})
        return arrays
    for t in valid_initializations(times,1):
        a,b=read(t),read(t+timedelta(hours=6))
        for i,c in enumerate(channels):
            name=c['feature'];std=stats[name]['std']
            if std<=0:raise ValueError(f'Zero training std: {name}')
            delta=(b[name].astype(np.float64)-a[name])/std
            sums[i]+=delta.sum();squares[i]+=(delta*delta).sum();counts[i]+=delta.size
    if (counts==0).any():raise ValueError('No valid training increment pairs')
    means=sums/counts;scales=np.sqrt(np.maximum(0,squares/counts-means*means))
    if (scales<=0).any():raise ValueError('Nonpositive increment standard deviation')
    pd.DataFrame(dict(feature_idx=range(116),variable=[c['feature'] for c in channels],mean=means,std=scales)).to_csv(stats_root/'target_mean_std.csv',index=False)
    pd.DataFrame(channels).to_csv(stats_root/'next_feature_table_weight_ver3.csv',index=False)
    pd.DataFrame(channel_config('ar_revised')).to_csv(stats_root/'next_feature_table_weight_ver3_revised.csv',index=False)
    pd.DataFrame(channel_config('coarse')).to_csv(stats_root/'next_feature_table_weight_coarse.csv',index=False)
    physical=[dict(c,mean=stats[c['feature']]['mean'],std=stats[c['feature']]['std']) for c in channels]
    pd.DataFrame(physical).to_csv(stats_root/'next_feature_table_weight.csv',index=False)
    runtime.write_json(stats_root/'provenance.json',dict(population='paper training years only',train_root=str(root.resolve()),
                       normalization='population mean/std per feature',target_scale='std of 6-h increment / feature std',pairs=int(counts[0]/(192*192))))
    return stats_root

def pangu_inputs(times, output_root=None):
    """Download global ERA5 NetCDF inputs in the upstream Pangu directory format."""
    import cdsapi
    root=Path(output_root or runtime.artifact_root()/"pangu")
    client=cdsapi.Client()
    for t in times:
        folder=root/'forecasts'/t.strftime('%Y-%m-%d-%H-%M');folder.mkdir(parents=True,exist_ok=True)
        common=dict(product_type=['reanalysis'],year=[str(t.year)],month=[f'{t.month:02d}'],day=[f'{t.day:02d}'],time=[f'{t.hour:02d}:00'],area=[90,0,-90,359.75],data_format='netcdf',download_format='unarchived')
        for kind,variables in [('surface',['mean_sea_level_pressure','10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature']),('upper',['geopotential','specific_humidity','temperature','u_component_of_wind','v_component_of_wind'])]:
            dest=folder/(kind+'.nc')
            if dest.exists():continue
            request=dict(common,variable=variables)
            if kind=='upper':request['pressure_level']=[str(l) for l in LEVELS]
            client.retrieve('reanalysis-era5-pressure-levels' if kind=='upper' else 'reanalysis-era5-single-levels',request).download(str(dest))
        import xarray as xr
        for kind,codes in [('surface',['msl','u10','v10','t2m']),('upper',['z','q','t','u','v'])]:
            array=folder/f'input_{kind}.npy'
            if array.exists():continue
            with xr.open_dataset(folder/f'{kind}.nc') as ds:
                parts=[]
                for name in codes:
                    da=ds[name]
                    level_dim=next((d for d in ['pressure_level','level','isobaricInhPa'] if d in da.dims),None)
                    if level_dim:da=da.sel({level_dim:list(LEVELS)})
                    if 'latitude' in da.coords:da=da.sortby('latitude',ascending=False)
                    if 'longitude' in da.coords:da=da.sortby('longitude')
                    parts.append(np.asarray(da.values,dtype=np.float32).squeeze())
                values=np.stack(parts)
                expected=(4,721,1440) if kind=='surface' else (5,13,721,1440)
                if values.shape!=expected:raise ValueError(f'Pangu {kind} input shape {values.shape}, expected {expected}')
                np.save(array,values)
    return root
