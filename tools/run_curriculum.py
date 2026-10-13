"""Print the complete ordered curriculum; --execute runs it on supplied assets.

Data, graphs and statistics must already exist. No external data are bundled.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from agu_james.workflow import load_config
from agu_james import runtime

def commands(config,micro_batch=1):
    load_config(config);root=runtime.artifact_root()
    prefix=[sys.executable,'-m','agu_james.workflow','--config',str(Path(config).resolve())]
    previous=None
    for stage,steps in [(1,[1]),(2,[1]),(3,[2,4,8,12,20,28])]:
        for step in steps:
            args=['train','--stage',str(stage),'--rollout',str(step),'--micro-batch',str(micro_batch)]
            if previous:args+=['--checkpoint',str(previous)]
            yield prefix+args
            previous=root/f'training/stage{stage}_mse_step{step:02d}/last_model.pth'
    base=previous;spectrum=root/'calibration/mse_train_coherence_energy_rollout28.npz'
    yield prefix+['validate','--split','train','--rollout','28','--checkpoint',str(base),'--spectrum-output',str(spectrum)]
    yield prefix+['calibrate','--input',str(spectrum)]
    final=dict(mse=base)
    for loss in ['mae','spectral_mse']:
        previous=base
        for step in [1,2,4,8,12,20,28]:
            yield prefix+['train','--stage','4','--rollout',str(step),'--loss',loss,'--micro-batch',str(micro_batch),'--checkpoint',str(previous)]
            previous=root/f'training/stage4_{loss}_step{step:02d}/last_model.pth'
        final[loss]=previous
    for checkpoint in final.values():
        yield prefix+['validate','--split','test','--rollout','28','--checkpoint',str(checkpoint)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--micro-batch',type=int,default=1)
    parser.add_argument('--execute',action='store_true',help='Run the printed commands sequentially; long research computation')
    args=parser.parse_args()
    for command in commands(args.config,args.micro_batch):
        print(subprocess.list2cmdline(command),flush=True)
        if args.execute:subprocess.run(command,check=True)

if __name__=='__main__':main()
