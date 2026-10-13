from datetime import datetime,timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
from agu_james import runtime

def validate(checkpoint,rollout=28,split='test',initial_time=None,spectrum_output=None,device=None):
    import torch
    from torch_geometric.loader import DataLoader
    from agu_james.training.dataset import WeatherGraphNPZValDataset
    from agu_james.training.engine import make_model,configure_graph
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    stage=state.get('model_stage','3');configure_graph(stage)
    root=runtime.path(split,runtime.artifact_root()/f'data/{split}')
    ds=WeatherGraphNPZValDataset(root,fourcast_steps=rollout)
    if initial_time:
        stamp=datetime.strptime(initial_time,'%Y%m%d%H').strftime('%Y%m%d%H.npz')
        if stamp not in ds.samples:raise ValueError(f'Initialization missing required same-split frames: {initial_time}')
        ds.samples=[stamp]
    if not len(ds):raise ValueError('No complete sequences')
    current,_,_=ds[0]
    settings=state.get('training_settings',runtime._CONFIG.get('training',{}))
    model,_=make_model(stage,current,settings);model.load_state_dict(state['model_state_dict'],strict=True)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'));model=model.to(device)
    out=runtime.artifact_root()/f'validation/{split}/{state.get("loss_kind","mse")}_step{rollout:03d}'
    loader=DataLoader(ds,batch_size=runtime._CONFIG.get('validation',{}).get('batch_size',1),shuffle=False)
    if spectrum_output:
        from agu_james.validation.coherence import validate_autoregressive
        result=validate_autoregressive(model,loader,device,0,str(out),rollout,H=192,W=192,boundary=9,save_step_npz=False,
                     compute_spectrum=True,spectrum_boundary=9,spectrum_amplitude=False,spectrum_detrend_mean=False,
                     spectrum_apply_lat_weight=False,spectrum_save_npz=str(spectrum_output),
                     spectrum_metadata=dict(data_source=str(root),checkpoint=str(Path(checkpoint).resolve()),split=split,initial_hours='00,06,12,18'))
    else:
        from agu_james.validation.rollout import validate_autoregressive
        out.mkdir(parents=True,exist_ok=True)
        result=validate_autoregressive(model,loader,device,0,str(out),rollout,H=192,W=192,boundary=9,save_step_npz=True,save_every_n_steps=1)
    return result

def calibrate(input_npz,output_dir=None):
    from agu_james.contracts import cutoff_from_energy
    from agu_james.data.pipeline import channel_config
    out=Path(output_dir or runtime.artifact_root()/'calibration/cutoffs');out.mkdir(parents=True,exist_ok=True)
    names=[c['feature'] for c in channel_config()]
    with np.load(input_npz,allow_pickle=False) as z:
        if 'split' not in z or str(z['split'])!='train':raise ValueError('Calibration requires training-set spectrum metadata')
        if z['feature_names'].tolist()!=names:raise ValueError('Calibration channel order mismatch')
        steps=int(z['rollout_steps']);k=z['k']
        for step in range(1,steps+1):
            f=z[f'forecast_mean_step{step:02d}'];t=z[f'target_mean_step{step:02d}']
            fc=z[f'forecast_cnt_step{step:02d}'];tc=z[f'target_cnt_step{step:02d}']
            if not np.array_equal(fc,tc):raise ValueError('Forecast/target accepted sample counts differ')
            rows=[dict(channel=i,feature=name,k_energy_decay_10pct=cutoff_from_energy(k,f[i],t[i])) for i,name in enumerate(names)]
            pd.DataFrame(rows).to_csv(out/f'cutoff_step{step:02d}.csv',index=False)
    runtime.write_json(out/'manifest.json',dict(input=str(Path(input_npz).resolve()),population='train',threshold=.9,exclude_dc=True,rollout_steps=steps,
                         missing_cutoff='NaN; retained loss loader uses explicitly documented cap fallback'))
    return out

def pangu(initial_times,rollout=28,weights_dir=None,use_gpu=True):
    from agu_james.validation.pangu_inference import run_pangu_inference
    root=runtime.path('pangu_root',runtime.artifact_root()/'pangu');weights=Path(weights_dir or runtime.path('pangu_weights'))
    models={h:str(weights/f'pangu_weather_{h}.onnx') for h in [24,6]}
    for file in models.values():
        if not Path(file).is_file():raise FileNotFoundError(file)
    for stamp in initial_times:
        t=datetime.strptime(stamp,'%Y%m%d%H')
        run_pangu_inference(t,t+timedelta(hours=rollout*6),models,base_dir=str(root),use_gpu=use_gpu,
                            reference_nc=str(root/'forecasts'/t.strftime('%Y-%m-%d-%H-%M')/'upper.nc'),
                            save_every_hours=6,save_after_hours=0,direct_hours=tuple(range(24,rollout*6+1,24)),overwrite=False)
