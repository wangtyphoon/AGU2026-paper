"""Local release checks; no data downloads, real training or publication."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def run(args,cwd=ROOT,env=None):
    result=subprocess.run([sys.executable,*args],cwd=cwd,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError(f'Command failed: {args}\n{result.stdout}\n{result.stderr}')
    return result.stdout

def main():
    scratch=ROOT/'artifacts/verification';scratch.mkdir(parents=True,exist_ok=True)
    python_files=sorted(p for base in ['agu_james','tools','tests'] for p in (ROOT/base).rglob('*.py'))
    for p in python_files:ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
    help_commands=[['--help']]+[[cmd,'--help'] for cmd in ['prepare','graph','train','validate','calibrate','pangu','analyze','plot','check']]
    for args in help_commands:run(['-m','agu_james.workflow',*args])
    with tempfile.TemporaryDirectory(dir=scratch,prefix='relocation-') as name:
        moved=Path(name)
        shutil.copytree(ROOT/'agu_james',moved/'agu_james',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(ROOT/'configs',moved/'configs')
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        run(['-m','agu_james.workflow','--help'],moved,env)
        info=json.loads(run(['-m','agu_james.workflow','--config','configs/paper.example.json','--dry-run','train','--stage','4','--rollout','28','--loss','spectral_mse'],moved,env))
        if Path(info['artifact_root'])!=moved/'artifacts' or info['paper_schedule']['max_optimizer_updates']!=1000:
            raise AssertionError('Relocated configuration/schedule resolution failed')
    packages={}
    for name in ['numpy','pandas','scipy','matplotlib','torch','torch-geometric','torch-scatter','torch-sparse','cartopy','xarray','cfgrib','eccodes','netCDF4','gcsfs','zarr','cdsapi','onnxruntime']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    result=dict(python=sys.version,python_files_checked=len(python_files),minimum_syntax='3.10',help_commands_checked=len(help_commands),
        relocation='passed with PYTHONPATH removed; copied package/config outside original layout',packages=packages,
        scope='syntax/help/relocation only; separate numerical tests and dependency check required')
    (scratch/'release_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
