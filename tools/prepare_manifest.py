"""Record control-source evidence and release file hashes; no source edits."""
import ast
import hashlib
import json
from pathlib import Path
from agu_james.contracts import schedule

ROOT=Path(__file__).resolve().parents[1]
RDL=Path(r'D:\RDL_TC_MODEL')
PAPER=Path(r'D:\paper\paper_with_code')
PANGU=Path(r'C:\Users\王子祐\Documents\Pangu-Weather-ReadyToGo')

def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def record(category,setting,old,new,source,reason):
    return dict(category=category,setting=setting,original=old,curated=new,source=source,reason=reason)

def main():
    source_manifest=json.loads((ROOT/'provenance/sources.json').read_text(encoding='utf-8'))
    references=[Path(r'D:\master\James Paper V1.pdf')]
    references += [RDL/'gnn'/v/'train_script/auto_config.py' for v in ['ver_unetG','ver_unet7','ver_autoregressive','ver_spectrum_v2']]
    references += [RDL/'gnn'/v/f for v,f in [('ver_unetG','train_auto.py'),('ver_unet7','train_auto.py'),('ver_autoregressive','train_autoregressive.py'),('ver_autoregressive','train_autoregressive_spectrum.py'),('ver_spectrum_v2','train_autoregressive_spectrum_v2.py')]]
    references += [RDL/'npz'/f for f in ['create_weight_mask.py','stat_all_variables.py']]
    references += [PANGU/'README.md',PANGU/'func_inference.py']
    ref_records=[];config_snapshots={}
    for p in references:
        if not p.is_file():raise FileNotFoundError(p)
        ref_records.append(dict(source=str(p),source_sha256=sha(p),role='read-only scientific/control reference; not bundled'))
        if p.name=='auto_config.py':
            tree=ast.parse(p.read_text(encoding='utf-8-sig'))
            assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='params' for t in n.targets))
            config_snapshots[p.parents[1].name]=ast.literal_eval(assignment.value)
    changes=[]
    add=lambda *args:changes.append(record(*args))
    add('path_organization','imports/paths/import-time I/O','loose modules, absolute paths, implicit jobs','agu_james modules; config/CLI; main/initialize','per-file sources.json','portable, explicit workflow; all source checkouts preserved')
    add('path_organization','duplicate model layers','multiple copies','shared only for identical original SHA-256','sources.json records marked merged','avoid merging different scientific implementations')
    add('paper_settings','training schedule and effective batch',config_snapshots,{str(s):[schedule(s,r) for r in ([1] if s<3 else [2,4,8,12,20,28] if s==3 else [1,2,4,8,12,20,28])] for s in [1,2,3,4]},'manuscript Table3; approved user plan','match publication curriculum; 1000 means optimizer updates')
    add('paper_settings','calendar split','local folder/continuity assignment',dict(train='1979..2023 excluding val years',val=[1980,1990,2000,2010,2020],test=2024,no_cross_split=True),'manuscript data section; user plan','prevent cross-set sequence leakage')
    add('paper_settings','initialization hours',[0,12],[0,6,12,18],'ver_autoregressive/train_script/datasetM.py','publication validation population')
    add('paper_settings','boundary width','inclusive distance<=8 for loss; other routines use crop boundary','width9; latest-state patch before call2+; metrics inclusive8/crop9','loss.py; val_autoregressive.py; approved plan','one physical nine-cell mask; preserve earlier state and first call')
    add('paper_settings','climatology','1990-2019_6h_1440x721.zarr','1990-2017_6h_1440x721.zarr','output/download_clim.py; WeatherBench official guide','match paper reference period; recomputed ACC changes expected')
    add('paper_settings','Stage1 objective','weighted RMSE','weighted MSE','ver_unetG/train_script/training.py; approved MSE-stage specification','paper-setting synchronization; deliberately not source-equivalent training')
    add('paper_settings','training controller','source-specific warmup LR1e-8, early-stop/AMP and epoch/microstep controls','float32; full budget; one-epoch or10%-update warmup; cosine1%floor; successful optimizer count','original stage entries; user schedule','consistent update budget; stochastic/precision historical equivalence not established')
    add('paper_settings','spectral total-energy caption','V1 k5..87','k6..87 (lambda<1000km, effective5376km)','manuscript Figure5 caption; retained energy plot','calculation already uses requested wavelength criterion; caption correction needed')
    add('necessary_fix','ocean physical statistics','optional masks selected with hardcoded missing counts','mandatory pre-fill training masks or explicit historical masks; maximum-missing training reference','npz/create_weight_mask.py; stat_all_variables.py','exclude filled land backgrounds; report changed mask reference separately')
    add('necessary_fix','graph7 normalization output','mesh_grid_data_unet/norm_stats','mesh_grid_data_unet7/norm_stats','pre_process/pre_norm6_unet7.py','write normalization beside graph7')
    add('necessary_fix','graph object load','implicit torch.load policy','weights_only=False for generated PyG objects','selected graph preparation sources','compatible serialization of controlled graph inputs')
    add('necessary_fix','Pangu path/reference arguments','non-Windows _normalize_path returns None; region reference fixed upper.nc','normpath(os.fspath); caller base_dir preserved; supplied reference_nc used','func_inference.py','portable paths and explicit crop-coordinate input')
    add('necessary_fix','AST regeneration compatibility','Python3.12 ast.unparse can emit nested-quote fstrings','regenerate with3.10/3.11; verify3.10 syntax','tools/curate_sources.py','declared minimum syntax across retained files')
    add('user_supplement','Pangu integration','separate user-supplied checkout','retained preparation/inference/subset/metric/spectrum/TC/map workflow','Pangu-Weather-ReadyToGo','include explicitly supplied comparison code; no binaries copied')
    add('user_supplement','Pangu lead coverage',dict(save_after_hours=72,direct_hours=[24,48,72]),dict(save_after_hours=0,direct_hours='24-hour multiples through requested rollout',save_every_hours=6),'func_inference.py defaults/main; approved plan','six-hour outputs from first lead; retain global24/6hour evolution')
    add('preserved_local_reference','spectral branch controls',dict(mae_weights=[1,1,2],mse_weights=[1,8,1],alpha=1,order=12,entry_bounds=87,channel_weights=None,mae_cap=46,mse_cap=28),dict(mae_weights=[1,1,2],mse_weights=[1,8,1],alpha=1,order=12,entry_bounds=87,stage4_uniform_channel_weights=True,mae_cap=46,mse_cap=28),'local spectral entry points/loss loaders','preserve effective entry settings; helper validation56 is overridden; not verified HPC settings')
    add('unresolved_scientific_audit','MAE bias normalization','source unnormalized batch-mean error spectrum; manuscript Eq7 normalized','source formula retained; manuscript/HPC reconciliation pending','spectrum_loss.py; manuscript Eq7','avoid silent new method; explicit discrepancy')
    add('unresolved_scientific_audit','coarse attention','train_auto imports model_unetT block/k-hop processor','same source closure retained; coarse dependencies absent','ver_unetG/train_auto.py; manuscript model description','actual HPC Stage1 architecture needs confirmation')
    add('execution_limit','MAE DDP/global batch-bias','no established global-equivalent implementation','reject WORLD_SIZE>1 forMAE; source microbatch loss under accumulation retained','original spectral trainer and collective MSE loss','do not claim global full-effective-batch spectral-bias equivalence')
    add('path_organization','excluded material','full experiments, .vscode, caches, profiling, datasets, statistics, weights, outputs','necessary source closures/config/docs/tests only; generated artifacts ignored','all requested source trees','code-only local release; source folders retained')
    (ROOT/'provenance/reference_sources.json').write_text(json.dumps(ref_records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'provenance/changes.json').write_text(json.dumps(dict(status='local curation; not national HPC reproduction',changes=changes),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[]
    for base in ['agu_james','configs','docs','tools','tests','provenance']:
        files.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'] and p.name!='files.json')
    files.extend(ROOT/f for f in ['README.md','pyproject.toml','MANIFEST.in','.gitignore','THIRD_PARTY_NOTICES.md','CITATION_DRAFT.md','references.bib'])
    output=[]
    for p in sorted(set(files)):
        rel=p.relative_to(ROOT).as_posix()
        parents=[dict(source=r['source'],source_sha256=r['source_sha256']) for r in source_manifest if r['destination']==rel]
        output.append(dict(file=rel,sha256=sha(p),sources=parents or [dict(source='newly authored during approved curation; see user plan and changes.json',source_sha256=None)]))
    (ROOT/'provenance/files.json').write_text(json.dumps(dict(self_exclusion='files.json excluded to avoid recursive hash; archive checksum to be recorded at deposit',files=output),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Recorded {len(changes)} changes, {len(ref_records)} control references and {len(output)} release file hashes')

if __name__=='__main__':main()
