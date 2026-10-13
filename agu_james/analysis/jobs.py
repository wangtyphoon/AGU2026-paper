"""Adapters around retained analysis, with publication input naming and safe outputs."""
import importlib
import inspect
from pathlib import Path
import shutil
import pandas as pd
from agu_james import runtime
from agu_james.data.pipeline import channel_config

SUFFIX={'mse':'auto','mae':'spectrum','spectral_mse':'spectrum_ver2'}

def table(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    p=directory/'next_feature_table_weight.csv'
    pd.DataFrame(channel_config()).to_csv(p,index=False)
    return p

def configure_module(name,overrides):
    config=dict(runtime._CONFIG);options=dict(config.get('overrides',{}))
    options.update({name+'.'+k:v for k,v in overrides.items()})
    config['overrides']=options;runtime.configure(config)
    module=importlib.import_module(name)
    module=importlib.reload(module)
    if hasattr(module,'initialize'):module.initialize()
    return module

def gnn(job,forecast_dir,target_dir,model='mse',rollout=28):
    if model not in SUFFIX:raise ValueError('GNN model must be mse, mae or spectral_mse')
    forecast=Path(forecast_dir).resolve();target=Path(target_dir).resolve();suffix=SUFFIX[model]
    paper=runtime.artifact_root()/'paper_inputs';stats=runtime.artifact_root()/'statistics'
    lat=runtime.path('lat',runtime.artifact_root()/'data/lat.npy');lon=runtime.path('lon',runtime.artifact_root()/'data/lon.npy')
    if job in ['rmse','acc_activity','target_activity','energy','coherence']:
        group={'rmse':'rmse_multi/ours','acc_activity':'acc_activity_multi/ours','target_activity':'acc_activity_multi/target','energy':'spectrum_multi/ours','coherence':'spectrum_multi/coherence'}[job]
        out=paper/group;feature=table(out)
        module=importlib.import_module('agu_james.analysis.gnn.'+job)
        common=dict(target_folder=str(target),boundary=8 if job in ['rmse','acc_activity','target_activity'] else 9)
        # Source metric masks use distance<=boundary, so 8 means exactly nine cells.
        if job=='rmse':
            kwargs=dict(common,folder=str(forecast),save_csv=str(out/f'per_channel_rmse_summary_macro_2024_bias_fix_attention_step{rollout}_{suffix}.csv'))
        elif job in ['acc_activity','target_activity']:
            name=f'per_channel_acc_activity_summary_step{rollout}_{"spectrumv2" if model=="spectral_mse" else suffix}.csv' if job=='acc_activity' else f'target_forecast_activity_summary_step{rollout}.csv'
            kwargs=dict(common,forecast_folder=str(forecast),clim_folder=str(runtime.path('climatology',runtime.artifact_root()/'climatology')),
                        feature_csv=str(feature),save_csv=str(out/name),per_file_csv=str(out/f'{job}_{suffix}_per_file.csv'))
        elif job=='energy':
            kwargs=dict(common,folder=str(forecast),amplitude=False,detrend_mean=False,apply_lat_weight=False,lat_path=str(lat),
                        source='all',save_npz=str(out/f'per_channel_energy_2024_fix_attention_step{rollout}_{suffix}.npz'))
        else:
            name={'mse':'autoregressive','mae':'spectrumv1','spectral_mse':'spectrumv2'}[model]
            kwargs=dict(common,folder=str(forecast),detrend_mean=False,apply_lat_weight=False,lat_path=str(lat),
                        save_npz=str(out/f'coherence_per_channel_step{rollout}_{name}.npz'))
        with runtime.workdir(out):module.main(**kwargs)
        return out
    if job!='tc':raise ValueError('Unknown GNN analysis job')
    out=paper/'tracker_multi/ours';feature=table(out);table(paper/'spectrum_multi')
    csv=out/f'tc_aurora_intensity35_step28_{suffix}.csv'
    options=dict(BASE_DIR=out,CSV_PATH=runtime.path('best_track',runtime.artifact_root()/'data/cyclone_data_2024.csv'),
                 NPZ_DIR=forecast,TARGET_DIR=target,FEATURE_TABLE=feature,LAT_NPY=lat,LON_NPY=lon,
                 LSM_NPY=runtime.path('lsm'),SURFACE_NC_PATH=runtime.path('surface_nc'),OUTPUT_CSV=csv)
    tracker=configure_module('agu_james.analysis.gnn.tc_tracker',options)
    with runtime.workdir(out):tracker.main()
    for name in ['tc_wind','tc_gwb','tc_phase','tc_ike','tc_vertical']:
        options=dict(base_dir=out,csv_path=csv,output_csv=csv,npz_root=forecast,target_root=target,
                     lat_path=lat,lon_path=lon,feature_table=feature)
        if name=='tc_vertical':
            ending='spectrumv2' if model=='spectral_mse' else suffix
            options['output_dir']=paper/f'struc_multi/axisym_wcc_step28_{ending}_npz'
        module=configure_module('agu_james.analysis.gnn.'+name,options)
        with runtime.workdir(out):module.main()
    structure=paper/'struc_multi';structure.mkdir(parents=True,exist_ok=True)
    ending='spectrumv2' if model=='spectral_mse' else suffix
    shutil.copy2(csv,structure/f'tc_aurora_intensity35_step28_{ending}.csv')
    runtime.write_json(out/f'tc_{suffix}_manifest.json',dict(forecast=str(forecast),target=str(target),rollout=rollout,
                       best_track=str(options.get('CSV_PATH',runtime.path('best_track',runtime.artifact_root()/'data/cyclone_data_2024.csv'))),
                       threshold_kt=35,boundary_margin_degrees=5,ike_radius_km=750,ike_thickness_m=1000))
    return csv

def pangu(job,rollout=28):
    root=runtime.path('pangu_root',runtime.artifact_root()/'pangu');paper=runtime.artifact_root()/'paper_inputs'
    lat=runtime.path('lat',runtime.artifact_root()/'data/lat.npy');lon=runtime.path('lon',runtime.artifact_root()/'data/lon.npy')
    upper=runtime.path('upper_nc');surface=runtime.path('surface_nc')
    if job in ['rmse','acc_activity','energy']:
        out=paper/({'rmse':'rmse_multi/pangu','acc_activity':'acc_activity_multi/pangu','energy':'spectrum_multi/Pangu'}[job]);out.mkdir(parents=True,exist_ok=True)
        for kind in ['prs','sfc']:
            name={'rmse':f'weight_rmse_{kind}_multistep','acc_activity':f'acc_metric_{kind}','energy':f'spectrum_multistep_{kind}'}[job]
            import numpy as np
            options=dict(BASE_DIR=root,BASE=root,ROOT=out,SCRIPT_DIR=out,results_root=root/'results',era5_root=root/'forecasts',
                         RESULTS_ROOT=root/'results',ERA5_ROOT=root/'forecasts',reference_nc=upper if kind=='prs' else surface,NC_PATH=str(upper if kind=='prs' else surface),
                         reference_path=runtime.path('test',runtime.artifact_root()/'data/test')/'prs',climate_dir=runtime.path('climatology',runtime.artifact_root()/'climatology'),
                         FORECAST_TOTAL_HOURS=rollout*6,FORECAST_STEPS=rollout,LAT_ALL=np.load(lat),LON_ALL=np.load(lon))
            config=dict(runtime._CONFIG);paths=dict(config.get('paths',{}));paths['agu_james.analysis.pangu.'+name+'.prs']=str(runtime.path('test',runtime.artifact_root()/'data/test')/'prs');config['paths']=paths;runtime.configure(config)
            module=configure_module('agu_james.analysis.pangu.'+name,options)
            # The spectra initialize their latitude arrays from the upstream root.
            with runtime.workdir(out):module.main()
        return out
    if job!='tc':raise ValueError('Unknown Pangu analysis job')
    out=paper/'tracker_multi/pangu';out.mkdir(parents=True,exist_ok=True);csv=out/'tc_aurora_metrics35_step28.csv'
    options=dict(BASE_DIR=out,CSV_PATH=runtime.path('best_track',runtime.artifact_root()/'data/cyclone_data_2024.csv'),
                 RESULTS_DIR=root/'results',ERA5_ROOT=root/'forecasts',LAT_NPY=lat,LON_NPY=lon,LSM_NPY=runtime.path('lsm'),
                 UPPER_NC_PATH=upper,SURFACE_NC_PATH=surface,OUTPUT_CSV=csv)
    module=configure_module('agu_james.analysis.pangu.tracker_ver_aurora',options)
    with runtime.workdir(out):module.main()
    for name in ['axisymmetric_wind_metrics_all','axisymmetric_gwb_metrics_all','axisymmetric_b_vt_metrics_all','ike_metric_700_all']:
        cfg=dict(csv_path=csv,output_path=csv,results_dir=root/'results',era5_dir=root/'forecasts',lat_path=lat,lon_path=lon)
        if name!='axisymmetric_wind_metrics_all':cfg['upper_nc_path']=upper
        module=configure_module('agu_james.analysis.pangu.'+name,{'config':cfg})
        with runtime.workdir(out):module.main()
    return csv

def export_gnn(forecast_dir,target_dir,model,initial_time,steps):
    import numpy as np
    # Retain the established map-file schema without duplicating source exporter variants.
    suffix={'mse':'ours','mae':'ours_spectrum','spectral_mse':'ours_spectrum_ver2'}[model]
    if max(steps)>28:suffix+='_monthly'
    out=runtime.artifact_root()/'paper_inputs/vis_multi_steps'/suffix/initial_time
    files=sorted(Path(forecast_dir).glob('*.npz'))
    for step in steps:
        candidates=[f for f in files if initial_time in f.name and f'_step{step:02d}' in f.name]
        if len(candidates)!=1:raise ValueError(f'Expected one file for {initial_time}/step{step}: {candidates}')
        # Use source exporter's tested batching selection and physical-value extraction.
        name='agu_james.analysis.export_gnn_maps'
        settings=dict(INPUT_DIR=Path(forecast_dir),TARGET_DIR=Path(target_dir),OUTPUT_DIR=out.parent,TIME_TAG=initial_time,STEPS_TO_SAVE=tuple(steps))
        config=dict(runtime._CONFIG);paths=dict(config.get('paths',{}));paths[name+'.lat.npy']=str(runtime.path('lat',runtime.artifact_root()/'data/lat.npy'));paths[name+'.lon.npy']=str(runtime.path('lon',runtime.artifact_root()/'data/lon.npy'));config['paths']=paths;runtime.configure(config)
        module=configure_module(name,settings)
        # Pure source functions; initialize has already loaded coordinates.
        module.INPUT_DIR=Path(forecast_dir);module.TARGET_DIR=Path(target_dir);module.OUTPUT_DIR=out.parent;module.TIME_TAG=initial_time;module.STEPS_TO_SAVE=list(steps)
        module.STEP_PATTERN=f'*{initial_time}*_step*.npz'
        module.main()
        break
    coords=out.parents[1]
    for key in ['lat','lon']:shutil.copy2(runtime.path(key,runtime.artifact_root()/f'data/{key}.npy'),coords/f'{key}.npy')
    return out
