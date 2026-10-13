"""A common paper trainer over retained GNN models and loss implementations.

New control flow replaces local demonstration configs, uses MSE in stages 1/2,
and counts successful optimizer updates. The original autoregressive feature
assembly and spectral formulas are retained. Full HPC reproduction is pending.
"""
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from agu_james import runtime
from agu_james.contracts import schedule, accumulation_steps, learning_rate
from agu_james.data.pipeline import channel_config

def configure_graph(stage):
    config=dict(runtime._CONFIG);paths=dict(config.get('paths',{}))
    mesh='mesh6' if str(stage)=='1' else 'mesh7'
    paths['graph']=paths.get(mesh+'_graph',str(runtime.artifact_root()/f'graphs/{mesh}/mesh_grid_data_unet{"" if mesh=="mesh6" else "7"}/hetero_graph_normalized.pt'))
    config['paths']=paths;runtime.configure(config)

def make_model(stage, sample, settings=None):
    settings=dict(settings or {})
    label='coarse' if str(stage)=='1' else 'ar'
    module='model_unet'
    cls=importlib.import_module(f'agu_james.training.models.{label}.{module}').HeteroGNN
    finest='mesh6' if label=='coarse' else 'mesh7'
    spec=dict(in_grid=sample['grid'].x.shape[1],in_mesh=sample[finest].x.shape[1],hidden_dim=settings.get('hidden_dim',512),
              out_grid=116,processor_layers=settings.get('processor_layers',[16,12,8,4] if label=='coarse' else [16,16,12,8,6]),
              ffn_ratio=settings.get('ffn_ratio',1.0 if label=='coarse' else 2.0),use_checkpoint=settings.get('use_checkpoint',True))
    return cls(**spec),spec

def transfer_weights(model, old):
    """The original mesh6 -> mesh7 same-shape/prefix transfer rules."""
    new=model.state_dict();copied=[]
    for k,v in old.items():
        if k in new and v.shape==new[k].shape:new[k]=v;copied.append(k)
    rules=[('encoders.enc6.','encoders.enc7.'),('pools.pool6_5.','pools.pool7_6.'),('processors.proc6.','processors.proc7.'),
           ('upsamples.up5_6.','upsamples.up6_7.'),('pool_norms.norm_pool6_5','pool_norms.norm_pool7_6'),
           ('up_norms.norm_up5_6','up_norms.norm_up6_7'),('decoders.dec6.','decoders.dec7.')]
    for a,b in rules:
        for k,v in old.items():
            dest=k.replace(a,b,1)
            if k.startswith(a) and dest in new and v.shape==new[dest].shape:new[dest]=v;copied.append(dest)
    if not copied:raise ValueError('No compatible weights found during transfer')
    model.load_state_dict(new)
    return sorted(set(copied))

def loss_for_batch(model,current,future,rollout,scale,channel_weights,spatial_weights,loss_kind='mse',cutoffs=None,settings=None):
    import torch
    from agu_james.training.dataset import process_data_step
    from agu_james.training.loss import weighted_mse_loss
    settings=settings or {}
    B=current['grid'].x.shape[0]//spatial_weights.numel()
    mask=(spatial_weights.reshape(-1)==0).repeat(B).unsqueeze(1)
    losses=[];components=[]
    for step in range(1,rollout+1):
        base=current['grid'].x[:,:116]
        truth=future[step-1]['grid'].x[:,:116]
        target=(truth-base)/scale
        outputs=model(current)
        if loss_kind=='mse':
            loss=weighted_mse_loss(outputs,target,channel_weights,spatial_weights);metrics={'mse':loss.detach()}
        else:
            module='spectral_mae' if loss_kind=='mae' else 'spectral_mse_ddp' if torch.distributed.is_initialized() else 'spectral_mse'
            fn=importlib.import_module('agu_james.training.'+module).spectral_composite_loss
            loss,metrics=fn(base,outputs,target,H=spatial_weights.shape[0],W=spatial_weights.shape[1],boundary=9,
                            cutoffs=cutoffs[step-1],spatial_w=spatial_weights,channel_weights=channel_weights,scale=scale,
                            alpha=settings.get('alpha',1.0),order=settings.get('order',12),
                            l_tilde=settings.get('l_tilde',87),l_max=settings.get('l_max',87),
                            weights=settings.get('loss_weights',(1,1,2) if loss_kind=='mae' else (1,8,1)))
        losses.append(loss);components.append(metrics)
        # Patch only the recent prediction before the NEXT call. Earlier state stays intact.
        if step<rollout:
            predicted=torch.where(mask,truth,outputs*scale+base)
            current=process_data_step(current,future,step=step,first116=predicted)
    return torch.stack(losses).mean(),{k:float(torch.stack([c[k] for c in components]).mean()) for k in components[0]}

def run(stage,rollout=1,loss_kind='mse',micro_batch=1,checkpoint=None,resume=None,device=None):
    import os
    import random
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler
    from torch_geometric.loader import DataLoader
    from agu_james.training.dataset import WeatherGraphNPZTrainDataset,WeatherGraphNPZValDataset
    from agu_james.training.loss import create_spatial_weights
    spec=schedule(stage,rollout)
    if str(stage)=='4' and loss_kind not in {'mae','spectral_mse'}:raise ValueError('Stage 4 requires mae or spectral_mse')
    if str(stage)!='4' and loss_kind!='mse':raise ValueError('Stages 1-3 use MSE')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1 and loss_kind=='mae':
        raise ValueError('MAE distributed batch-bias equivalence is not established; run one GPU with accumulation. MSE DDP uses retained collective loss.')
    if world>1:
        if not torch.cuda.is_available():raise RuntimeError('DDP training requires CUDA')
        torch.cuda.set_device(local)
        if not dist.is_initialized():dist.init_process_group('nccl')
    device=torch.device(device or (f'cuda:{local}' if torch.cuda.is_available() else 'cpu'))
    accumulation=accumulation_steps(spec['effective_batch'],micro_batch,world)
    settings=dict(runtime._CONFIG.get('training',{}));seed=settings.get('seed',3407)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    configure_graph(stage)
    train_root=runtime.path('train',runtime.artifact_root()/'data/train')
    val_root=runtime.path('val',runtime.artifact_root()/'data/val')
    train_ds=WeatherGraphNPZTrainDataset(train_root,fourcast_steps=rollout)
    val_ds=WeatherGraphNPZValDataset(val_root,fourcast_steps=rollout)
    if len(train_ds)<spec['effective_batch']//world or not len(val_ds):raise ValueError('Not enough train/validation sequences')
    sample,_=train_ds[0]
    expected=[c['feature'] for c in channel_config()]
    observed=[x for x in train_ds.prs_feature if x not in train_ds.remove_prs]+[x for x in train_ds.sfc_regular_feature if x not in ['z','lsm']]+[x for x in train_ds.sfc_average_feature if x!='tisr']
    if observed!=expected:raise ValueError('Dataset channel order differs from frozen 116-channel schema')
    model,model_spec=make_model(stage,sample,settings);model=model.to(device)
    transfer=[]
    if str(stage)!='1' and not (checkpoint or resume):raise ValueError('Supply previous-stage --checkpoint or same-stage --resume')
    if checkpoint:
        old=torch.load(checkpoint,map_location=device,weights_only=False)
        transfer=transfer_weights(model,old['model_state_dict']) if str(stage)=='2' else []
        if str(stage)!='2':model.load_state_dict(old['model_state_dict'],strict=True)
    if world>1:model=DistributedDataParallel(model,device_ids=[local])
    sampler=DistributedSampler(train_ds,shuffle=True,seed=seed) if world>1 else None
    loader=DataLoader(train_ds,batch_size=micro_batch,shuffle=sampler is None,sampler=sampler,drop_last=True,num_workers=settings.get('num_workers',0))
    updates_per_epoch=len(loader)//accumulation
    if not updates_per_epoch:raise ValueError('Not enough complete effective batches')
    total=spec['max_optimizer_updates'] or spec['epochs']*updates_per_epoch
    warmup=updates_per_epoch if str(stage) in {'1','2'} else max(1,int(total*.1))
    # Corrected coarse and AR sources both exempt bias/norm from weight decay.
    core=model.module if world>1 else model
    norm_ids={id(p) for m in core.modules() if isinstance(m,(torch.nn.LayerNorm,torch.nn.BatchNorm1d)) for p in m.parameters(recurse=False)}
    decay=[];no_decay=[]
    for name,p in core.named_parameters():
        (no_decay if name.endswith('.bias') or id(p) in norm_ids else decay).append(p)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':no_decay,'weight_decay':0.0}],lr=spec['peak_lr'],betas=(.9,.95))
    scales=pd.read_csv(runtime.legacy_path('target_mean_std.csv'))
    if scales.variable.tolist()!=expected:raise ValueError('Increment scale order mismatch')
    scale=torch.tensor(scales['std'].values,dtype=torch.float32,device=device)
    if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid increment scales')
    channels=channel_config('coarse' if str(stage)=='1' else 'ar')
    weights=torch.tensor([c['weight'] for c in channels],device=device)
    if str(stage)=='4':weights=torch.ones_like(weights)  # Source spectral entry points pass channel_weights=None.
    spatial=create_spatial_weights(boundary=9,device=device)
    cutoffs=None
    if str(stage)=='4':
        fn=importlib.import_module('agu_james.training.spectral_mae' if loss_kind=='mae' else 'agu_james.training.spectral_mse').load_cutoffs
        folder=runtime.path('cutoffs',runtime.artifact_root()/'calibration/cutoffs')
        cap=settings.get('cutoff_cap',46.0 if loss_kind=='mae' else 28.0)
        cutoffs=[fn(str(folder/f'cutoff_step{s:02d}.csv'),device=device,default_cutoff=cap) for s in range(1,rollout+1)]
    out=runtime.artifact_root()/f'training/stage{stage}_{loss_kind}_step{rollout:02d}'
    out.mkdir(parents=True,exist_ok=True)
    update=0;epoch=0;history=[]
    if resume:
        old=torch.load(resume,map_location=device,weights_only=False)
        if old.get('paper_schedule')!=spec:raise ValueError('Resume checkpoint schedule differs from requested stage')
        if old.get('loss_kind')!=loss_kind or old.get('training_settings')!=settings:
            raise ValueError('Resume loss/settings differ; use --checkpoint to start a new segment')
        core.load_state_dict(old['model_state_dict']);optimizer.load_state_dict(old['optimizer_state_dict'])
        update=old['optimizer_updates'];epoch=old['epoch'];history=old.get('history',[])
    while update<total:
        epoch+=1
        if sampler:sampler.set_epoch(epoch)
        model.train();optimizer.zero_grad(set_to_none=True)
        running=[]
        for i,(current,future) in enumerate(loader):
            if i>=updates_per_epoch*accumulation or update>=total:break
            current=current.to(device);future=[g.to(device) for g in future]
            loss,components=loss_for_batch(model,current,future,rollout,scale,weights,spatial,loss_kind,cutoffs,settings)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch={epoch},batch={i}')
            (loss/accumulation).backward();running.append(float(loss.detach()))
            if (i+1)%accumulation==0:
                lr=learning_rate(update,total,warmup,spec['peak_lr'])
                for group in optimizer.param_groups:group['lr']=lr
                torch.nn.utils.clip_grad_norm_(model.parameters(),32)
                optimizer.step();optimizer.zero_grad(set_to_none=True);update+=1
                if rank==0 and (update==1 or update%50==0 or update==total):print(f'epoch={epoch} optimizer_updates={update}/{total} lr={lr:.6g} objective={running[-1]:.6g}',flush=True)
        # Validation is counted independently; it never advances optimizer updates.
        core.eval();validation=[]
        with torch.no_grad():
            for current,future,*_ in DataLoader(val_ds,batch_size=micro_batch,num_workers=0):
                current=current.to(device);future=[g.to(device) for g in future]
                value,_=loss_for_batch(core,current,future,rollout,scale,weights,spatial,loss_kind,cutoffs,settings)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
                validation.append(float(value))
        history.append(dict(epoch=epoch,optimizer_updates=update,train_objective=float(np.mean(running)),val_objective=float(np.mean(validation))))
        if rank==0:
            state=dict(model_state_dict=core.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch,optimizer_updates=update,
                       history=history,paper_schedule=spec,model_spec=model_spec,model_stage=str(stage),channel_names=expected,
                       boundary_width=9,loss_kind=loss_kind,training_settings=settings,transfer_keys=transfer,
                       effective_batch=spec['effective_batch'],micro_batch=micro_batch,world_size=world,accumulation_steps=accumulation)
            torch.save(state,out/'last_model.pth')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            runtime.write_json(out/'manifest.json',{k:v for k,v in state.items() if k not in ['model_state_dict','optimizer_state_dict']})
        if world>1:dist.barrier()
    if world>1:dist.destroy_process_group()
    return out/'last_model.pth'
