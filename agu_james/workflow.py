"""Common CLI. --dry-run prints the requested operation without reading research data."""
import argparse
from datetime import datetime,timedelta
import importlib
import json
import logging
from pathlib import Path
from agu_james import runtime
from agu_james.contracts import schedule,accumulation_steps

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,help='JSON paths/settings; relative paths resolve beside this file')
    p.add_argument('--artifact-root',type=Path,help='Generated files only; defaults to ./artifacts')
    p.add_argument('--dry-run',action='store_true')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('prepare');d.add_argument('job',choices=['best-track','download','convert','statistics','climatology','pangu-inputs']);d.add_argument('--input',type=Path);d.add_argument('--initial-times',nargs='+');d.add_argument('--rollout',type=int,default=28)
    g=sub.add_parser('graph');g.add_argument('--mesh',choices=['6','7'],required=True)
    t=sub.add_parser('train');t.add_argument('--stage',choices=['1','2','3','4'],required=True);t.add_argument('--rollout',type=int,default=1);t.add_argument('--loss',choices=['mse','mae','spectral_mse'],default='mse');t.add_argument('--micro-batch',type=int,default=1);t.add_argument('--checkpoint',type=Path);t.add_argument('--resume',type=Path);t.add_argument('--device')
    v=sub.add_parser('validate');v.add_argument('--checkpoint',type=Path,required=True);v.add_argument('--rollout',type=int,default=28);v.add_argument('--split',choices=['train','val','test'],default='test');v.add_argument('--initial-time');v.add_argument('--spectrum-output',type=Path);v.add_argument('--device')
    c=sub.add_parser('calibrate');c.add_argument('--input',type=Path,required=True);c.add_argument('--output-dir',type=Path)
    f=sub.add_parser('pangu');f.add_argument('--initial-times',nargs='+',required=True);f.add_argument('--rollout',type=int,default=28);f.add_argument('--weights-dir',type=Path);f.add_argument('--cpu',action='store_true')
    a=sub.add_parser('analyze');a.add_argument('job',choices=['rmse','acc_activity','target_activity','energy','coherence','tc','export-gnn','export-pangu']);a.add_argument('--model',choices=['mse','mae','spectral_mse','pangu'],default='mse');a.add_argument('--forecast-dir',type=Path);a.add_argument('--target-dir',type=Path);a.add_argument('--rollout',type=int,default=28);a.add_argument('--initial-time');a.add_argument('--steps',nargs='+',type=int)
    z=sub.add_parser('plot');z.add_argument('group',choices=['metrics','energy','coherence','tc','physics','krathon-track','krathon-fields','monthly','graph']);z.add_argument('--input-root',type=Path);z.add_argument('--output-dir',type=Path)
    sub.add_parser('check')
    return p

def load_config(file=None,artifact_root=None):
    config=json.loads(Path(file).read_text(encoding='utf-8')) if file else {}
    base=Path(file).resolve().parent if file else Path.cwd()
    for key,value in list(config.get('paths',{}).items()):
        if value is None:config['paths'].pop(key)
        elif not str(value).startswith(('gs://','https://')):
            config['paths'][key]=str((base/Path(value)).resolve())
    root=artifact_root or config.get('artifact_root','artifacts')
    config['artifact_root']=str((base/Path(root)).resolve())
    runtime.configure(config)
    return config

def main(argv=None):
    p=parser();args,extra=p.parse_known_args(argv)
    if extra and args.command!='plot':p.error('Unknown options: '+' '.join(extra))
    config=load_config(args.config,args.artifact_root)
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    if args.dry_run:
        info={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        info['artifact_root']=config['artifact_root']
        if args.command=='train':
            info['paper_schedule']=schedule(args.stage,args.rollout)
            info['accumulation_steps']=accumulation_steps(info['paper_schedule']['effective_batch'],args.micro_batch)
        print(json.dumps(info,indent=2,ensure_ascii=False));return
    if args.command=='prepare':
        from agu_james.data import pipeline
        if args.job=='best-track':
            if not args.input:p.error('best-track requires --input (ATCF directory or normalized CSV)')
            result=pipeline.best_track(args.input)
        elif args.job=='download':result=pipeline.download()
        elif args.job=='convert':result=pipeline.convert(args.input)
        elif args.job=='statistics':result=pipeline.statistics(args.input)
        elif args.job=='pangu-inputs':
            if not args.initial_times:p.error('pangu-inputs requires --initial-times')
            times=sorted({datetime.strptime(t,'%Y%m%d%H')+timedelta(hours=6*k) for t in args.initial_times for k in range(args.rollout+1)})
            result=pipeline.pangu_inputs(times)
        else:
            from agu_james.analysis.jobs import configure_module,table
            root=runtime.artifact_root();work=root/'climatology';feature=table(work)
            work.mkdir(parents=True,exist_ok=True)
            import shutil
            for key in ['lat','lon']:shutil.copy2(runtime.path(key,root/f'data/{key}.npy'),work/f'{key}.npy')
            module=configure_module('agu_james.analysis.gnn.climatology',dict(FEATURE_CSV=feature,output_dir=work))
            with runtime.workdir(work):result=module.main()
    elif args.command=='graph':
        with runtime.workdir(runtime.artifact_root()/f'graphs/mesh{args.mesh}'):
            for index in range(1,7):importlib.import_module(f'agu_james.graphs.mesh{args.mesh}.step{index}').main()
        result=runtime.artifact_root()/f'graphs/mesh{args.mesh}'
    elif args.command=='train':
        from agu_james.training.engine import run
        result=run(args.stage,args.rollout,args.loss,args.micro_batch,args.checkpoint,args.resume,args.device)
    elif args.command=='validate':
        from agu_james.validation.runner import validate
        result=validate(args.checkpoint,args.rollout,args.split,args.initial_time,args.spectrum_output,args.device)
    elif args.command=='calibrate':
        from agu_james.validation.runner import calibrate
        result=calibrate(args.input,args.output_dir)
    elif args.command=='pangu':
        from agu_james.validation.runner import pangu
        result=pangu(args.initial_times,args.rollout,args.weights_dir,not args.cpu)
    elif args.command=='analyze':
        from agu_james.analysis import jobs
        if args.job=='export-pangu':
            name='agu_james.analysis.export_pangu_maps'
            if not args.initial_time:p.error('export-pangu requires --initial-time')
            root=runtime.path('pangu_root',runtime.artifact_root()/'pangu')
            run=root/'results'/datetime.strptime(args.initial_time,'%Y%m%d%H').strftime('%Y-%m-%d-%H-%M')
            module=jobs.configure_module(name,dict(ERA5_ROOT=root/'forecasts',REFERENCE_NC=runtime.path('upper_nc'),
                 OUTPUT_DIR=runtime.artifact_root()/'paper_inputs/vis_multi_steps/pangu',RUN_DIR=run))
            module.LEAD_HOURS=[s*6 for s in (args.steps or [4,8,12])]
            result=module.main()
        elif args.job=='export-gnn':
            if not args.initial_time or not args.steps:p.error('export-gnn requires --initial-time and --steps')
            result=jobs.export_gnn(args.forecast_dir,args.target_dir,args.model,args.initial_time,args.steps)
        elif args.model=='pangu':result=jobs.pangu(args.job,args.rollout)
        else:
            if not args.forecast_dir or not args.target_dir:p.error('GNN analysis requires --forecast-dir and --target-dir')
            result=jobs.gnn(args.job,args.forecast_dir,args.target_dir,args.model,args.rollout)
    elif args.command=='plot':
        from agu_james.plotting.runner import run
        result=run(args.group,args.input_root,args.output_dir,extra)
    else:
        from agu_james.verification import check
        result=check()
    if isinstance(result,Path):print(result)

if __name__=='__main__':main()
