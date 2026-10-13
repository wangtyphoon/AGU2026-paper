"""Rebuild retained modules from the read-only research checkouts.

This utility writes only inside this repository. It never edits a source checkout.
All output hashes and original source paths are recorded in provenance/sources.json.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RDL = Path(r"D:\RDL_TC_MODEL")
PAPER = Path(r"D:\paper\paper_with_code")
PANGU = Path(r"C:\Users\王子祐\Documents\Pangu-Weather-ReadyToGo")
records = []

def write(rel, content):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p

def package(rel):
    p = ROOT / rel
    while p != ROOT:
        if p.name != "config":
            init = p / "__init__.py"
            if not init.exists():
                write(str(init.relative_to(ROOT)), '"""Curated paper workflow modules."""\n')
        p = p.parent

DEMO = {"test_forward_backward", "visualize_feature_map", "test_model", "MemoryProfiler", "peak_mem_trainlike", "analyze_no_decay_params"}

class Clean(ast.NodeTransformer):
    def __init__(self, imports, scope):
        self.imports, self.scope = imports, scope
    def visit_ImportFrom(self, n):
        if n.module in {"calflops", "torchviz"}:
            return None
        if n.module in self.imports:
            n.module = self.imports[n.module]
        if n.module == "sklearn" and any(a.name == "logger" for a in n.names):
            return ast.parse("from agu_james.logging_utils import logger").body[0]
        return n
    def visit_Import(self,n):
        if self.scope.endswith('pangu_inference'):
            n.names=[a for a in n.names if a.name!='onnx']
            if not n.names:return None
        for alias in n.names:
            if alias.name in self.imports:
                original=alias.name
                alias.name=self.imports[original]
                alias.asname=alias.asname or original.split('.')[0]
        return n
    def visit_FunctionDef(self, n):
        if n.name in DEMO:
            return None
        return self.generic_visit(n)
    def visit_ClassDef(self, n):
        if n.name in DEMO:
            return None
        return self.generic_visit(n)
    def visit_Constant(self, n):
        if isinstance(n.value, str) and re.match(r"^[CD]:[\\/]", n.value):
            key = self.scope + "." + Path(n.value.replace("\\", "/")).name
            return ast.copy_location(ast.parse(f"runtime.path({key!r})").body[0].value, n)
        return n

def is_main(n):
    return isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)

def curate(source, target, imports=None, transform=None, top_job=False, drop_main=True):
    original = source.read_text(encoding="utf-8-sig")
    tree = ast.parse(original)
    tree.body = [n for n in tree.body if not (drop_main and is_main(n))]
    tree = Clean(imports or {}, target.removesuffix(".py").replace("/", ".")).visit(tree)
    changes = ["package-qualified imports", "removed debug/demo entry points", "AST formatting; formulas retained"]
    if transform:
        tree = transform(tree)
        changes.append(transform.__name__)
    if top_job:
        declarations, work = [], []
        has_main = any(isinstance(n, ast.FunctionDef) and n.name == 'main' for n in tree.body)
        for n in tree.body:
            if is_main(n):
                work.extend(n.body)
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                if isinstance(n, ast.FunctionDef) and n.name == 'main':n.name = '_run'
                declarations.append(n)
            else:
                work.append(n)
        names = sorted({x.id for n in work for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)})
        if work:
            if has_main:work.append(ast.parse('_run()').body[0])
            declarations.append(ast.FunctionDef(name="main", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                                                  body=[ast.Global(names=names)] + work, decorator_list=[]))
        tree.body = declarations
        changes.append("top-level I/O moved into main()")
    scope = target.removesuffix('.py').replace('/', '.')
    class ConfigurePaths(ast.NodeTransformer):
        def visit_Assign(self, n):
            n = self.generic_visit(n)
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if target=='agu_james/validation/pangu_inference.py' and name=='base_dir':
                    if ast.unparse(n.value)=='os.getcwd()':
                        n.value=ast.parse("runtime.path('pangu_root', runtime.artifact_root() / 'pangu')").body[0].value
                    return n  # Preserve a caller-supplied base_dir through _normalize_path.
                if target=='agu_james/validation/pangu_inference.py' and name=='nc_for_slice':
                    n.value=ast.Name(id='reference_nc',ctx=ast.Load())
                    return n
                if name in {'BASE_DIR','BASE','ROOT','HERE','SCRIPT_DIR','OUTPUT_DIR','output_dir','base_dir','root_dir','current_dir','input_path','output_base','LAT_NPY','LON_NPY','LSM_NPY','UPPER_NC_PATH','SURFACE_NC_PATH','CSV_PATH','NPZ_DIR','INPUT_DIR','TARGET_DIR','OUTPUT_CSV','FEATURE_TABLE','FEATURE_CSV','LAT_PATH','LON_PATH','LAT_ALL','LON_ALL','DATA_DIR','DATA','MAP_BASE','GRAPH_PATH','graph_path','PANGU_ROOT','OURS_ROOT','OURS_SPECTRUM_ROOT','OURS_SPECTRUM_VER2_ROOT','OURS_SPECTRUM_MONTHLY_ROOT','OURS_SPECTRUM_VER2_MONTHLY_ROOT','MSE_TRAIN_NPZ','DEFAULT_MSE_TRAIN_NPZ','BEST_TRACK','csv_path','output_csv','lat_path','lon_path','npz_root','target_root','feature_table','npz_dir','output_npz','output_folder','forecast_folder','target_folder','clim_folder','climate_dir','results_root','era5_root','RESULTS_ROOT','RESULTS_DIR','ERA5_ROOT','RUN_DIR','REFERENCE_NC','reference_nc','NC_PATH','reference_path','tracker_dir','structure_dir','FORECAST_TOTAL_HOURS','FORECAST_STEPS'}:
                    if name in {'BASE_DIR','BASE','ROOT','HERE','base_dir','OUTPUT_DIR','output_dir','current_dir'}:
                        default = ast.parse(f"runtime.artifact_root() / {('jobs/' + scope)!r}").body[0].value
                    else:default=n.value
                    n.value = ast.Call(func=ast.Attribute(value=ast.Name(id='runtime',ctx=ast.Load()),attr='setting',ctx=ast.Load()),args=[ast.Constant(scope+'.'+name),ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=default)],keywords=[])
            return n
        def visit_Constant(self,n):
            if isinstance(n.value,str) and 'default: bundled training NPZ' in n.value:
                n.value=n.value.replace('default: bundled training NPZ','external training-set NPZ')
            if isinstance(n.value,str) and '1990-2019_6h_1440x721.zarr' in n.value:
                n.value=n.value.replace('1990-2019','1990-2017')
            if target=='agu_james/graphs/mesh7/step6.py' and n.value=='mesh_grid_data_unet/norm_stats/':
                n.value='mesh_grid_data_unet7/norm_stats/'
            return n
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if target.startswith('agu_james/graphs/') and ast.unparse(n.func)=='torch.load' and not any(k.arg=='weights_only' for k in n.keywords):
                n.keywords.append(ast.keyword(arg='weights_only',value=ast.Constant(False)))
            if isinstance(n.func,ast.Name) and n.func.id=='Config' and not n.args and not n.keywords:
                n.keywords=[ast.keyword(arg=None,value=ast.parse(f"runtime.setting({(scope+'.config')!r}, lambda: {{}})").body[0].value)]
            return n
    tree=ConfigurePaths().visit(tree)
    if target in {'agu_james/training/spectral_mse.py','agu_james/training/spectral_mse_ddp.py'}:
        tree.body.append(ast.parse('spectral_composite_loss = spectral_composite_loss_v2').body[0])
        changes.append('public spectral_composite_loss alias; same v2 function')
    if target=='agu_james/validation/pangu_inference.py':
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name=='_normalize_path':
                n.body=ast.parse('return None if path is None else os.path.normpath(os.fspath(path))').body
    if target in {'agu_james/data/nearest_fill.py','agu_james/data/wave_interpolation.py'}:
        helper='nearest_neighbor_fill' if target.endswith('nearest_fill.py') else 'resize_swh_bilinear'
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name==helper]
        changes.append('retain only conversion helper; remove original in-place bulk repair/demo utilities')
    # Monthly composites import retained helpers through package names.
    if target=='agu_james/plotting/monthly.py':
        for i,n in enumerate(tree.body):
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'maps','energy'}:
                name=n.targets[0].id
                tree.body[i]=ast.parse('import agu_james.plotting.' + ('monthly_maps' if name=='maps' else 'energy') + ' as ' + name).body[0]
    deferred=[]; deferred_names=set()
    for n in list(tree.body):
        if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):continue
        if (isinstance(n,ast.Assign) and any(isinstance(c,ast.Call) and ast.unparse(c.func) in {'np.load','pd.read_csv'} for c in ast.walk(n))) or any(isinstance(c,ast.Name) and isinstance(c.ctx,ast.Load) and c.id in deferred_names for c in ast.walk(n)):
            deferred.append(n);tree.body.remove(n)
            deferred_names.update(x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store))
    if deferred:
        names=sorted(deferred_names)
        tree.body.append(ast.parse('def initialize():\n    pass').body[0])
        tree.body[-1].body=[ast.Global(names=names)]+deferred
    changes.append('configurable artifact paths; deferred import-time I/O')
    # Keep future imports before the runtime import.
    at = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
    while at < len(tree.body) and isinstance(tree.body[at], ast.ImportFrom) and tree.body[at].module == "__future__":
        at += 1
    tree.body.insert(at, ast.parse("from agu_james import runtime").body[0])
    ast.fix_missing_locations(tree)
    text = '# Derived from ' + source.relative_to(RDL if source.is_relative_to(RDL) else PAPER if source.is_relative_to(PAPER) else PANGU).as_posix() + '\n# See provenance/sources.json for original hash and modifications.\n' + ast.unparse(tree) + '\n'
    p = write(target, text)
    package(str(p.parent.relative_to(ROOT)))
    records.append(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        destination=target, output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), modifications=changes))

def model_closure(version, entry, label):
    base = RDL / "gnn" / version
    pending, found = [entry], set()
    while pending:
        rel = pending.pop()
        if rel in found:
            continue
        found.add(rel)
        tree = ast.parse((base / rel).read_text(encoding="utf-8-sig"))
        tree.body=[n for n in tree.body if not is_main(n) and not (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in DEMO)]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                candidate = node.module.replace(".", "/") + ".py"
                if (base / candidate).exists() and not candidate.startswith("train_script/"):
                    pending.append(candidate)
    mapping = {rel.removesuffix(".py").replace("/", "."): "agu_james.training.models." + label + "." + rel.removesuffix(".py").replace("/", ".") for rel in found}
    for rel in sorted(found):
        curate(base / rel, "agu_james/training/models/" + label + "/" + rel, mapping)

def dataset_paths(tree):
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str) and n.value in ("../../era5/lat.npy", "../../era5/lon.npy"):
                return ast.parse(f"runtime.legacy_path({n.value!r})").body[0].value
            return n
        def visit_Call(self, n):
            n = self.generic_visit(n)
            code = ast.unparse(n)
            if code.startswith("torch.load(Path('pre_process/"):
                n.args[0] = ast.parse("runtime.path('graph', runtime.artifact_root() / 'graphs/mesh7/mesh_grid_data_unet7/hetero_graph_normalized.pt')").body[0].value
            if code.startswith("pd.read_csv(") and any(k in code for k in ['prs_variable_stats', 'sfc_regular_variable_stats', 'sfc_average_variable_stats']):
                name = next(k for k in ['prs_variable_stats', 'sfc_regular_variable_stats', 'sfc_average_variable_stats'] if k in code)
                n.args[0] = ast.parse(f"runtime.legacy_path('../../npz/{name}.csv')").body[0].value
            return n
        def visit_FunctionDef(self, n):
            if n.name == "_get_valid_samples":
                return ast.parse('''def _get_valid_samples(self):
    from agu_james.contracts import valid_initializations
    times = [datetime.strptime(f[:10], "%Y%m%d%H") for f in self.all_samples]
    return [t.strftime("%Y%m%d%H.npz") for t in valid_initializations(times, self.fourcast_steps)]
''').body[0]
            return self.generic_visit(n)
    return Adapt().visit(tree)

def training_asset_paths(tree):
    class Adapt(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            if isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == "pd.read_csv" and n.args and isinstance(n.args[0], ast.Constant):
                if n.args[0].value in ['target_mean_std.csv', 'next_feature_table_weight.csv', 'next_feature_table_weight_ver3.csv']:
                    n.args[0] = ast.parse(f"runtime.legacy_path({n.args[0].value!r})").body[0].value
            return n
    return Adapt().visit(tree)

def torch_spatial_width(tree):
    tree = training_asset_paths(tree)
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name == 'create_spatial_weights':
            tree.body[i] = ast.parse('''def create_spatial_weights(H=192, W=192, boundary=9, device=None):
    from agu_james.contracts import spatial_weights
    lat = np.linspace(47.75, 0.0, H)
    return torch.tensor(spatial_weights(lat, W, boundary), dtype=torch.float32, device=device)
''').body[0]
    return tree

def validation_width(tree):
    tree = training_asset_paths(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == 'validate_autoregressive':
            for i,arg in enumerate(n.args.args):
                if arg.arg == 'boundary':
                    j = i - (len(n.args.args) - len(n.args.defaults))
                    if j >= 0:n.args.defaults[j] = ast.Constant(9)
    return tree

def pangu_functions_only(tree):
    return ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))], type_ignores=[])

def main():
    if sys.version_info >= (3,12):
        raise RuntimeError('Regenerate with Python 3.10/3.11: newer ast.unparse can emit f-strings incompatible with the declared minimum Python version')
    model_closure('ver_unet_scale','model_unet.py','coarse')
    model_closure('ver_autoregressive','model_unet.py','ar')
    shared = {'train_script.datasetM':'agu_james.training.dataset', 'train_script.loss':'agu_james.training.loss', 'train_script.loggers':'agu_james.logging_utils'}
    base=RDL/'gnn/ver_autoregressive/train_script'
    curate(base/'datasetM.py','agu_james/training/dataset.py',shared,dataset_paths)
    curate(base/'loss.py','agu_james/training/loss.py',shared,torch_spatial_width)
    curate(base/'spectrum_loss.py','agu_james/training/spectral_mae.py',shared,training_asset_paths)
    curate(RDL/'gnn/ver_spectrum_v2/train_script/spectrum_loss_v2.py','agu_james/training/spectral_mse.py',shared,training_asset_paths)
    curate(RDL/'gnn/ver_spectrum_v2/train_script/spectrum_loss_v2_ddp.py','agu_james/training/spectral_mse_ddp.py',shared,training_asset_paths)
    curate(base/'val_autoregressive.py','agu_james/validation/rollout.py',shared,validation_width)
    curate(base/'val_autoregressive_with_spectrum.py','agu_james/validation/coherence.py',shared,validation_width)
    graph_names=['create_hetero_mesh_poolk','edge_feature4_unet','node_feature2_unet','PE+SE3_unet','self_loop5_unet','pre_norm6_unet']
    for label,suffix in [('mesh6',''),('mesh7','7')]:
        for index,name in enumerate(graph_names):
            filename = name + suffix + '.py'
            if name=='create_hetero_mesh_poolk':filename=name+suffix+'.py'
            curate(RDL/'gnn/ver_unet7/pre_process'/filename, f'agu_james/graphs/{label}/step{index+1}.py',top_job=True,drop_main=False)
    # Data requests and exact fill/interpolation policies retained as library functions.
    for name in ['download_prs','download_sfc_regular','download_sfc_average']:
        curate(RDL/'era5'/f'{name}.py',f'agu_james/data/requests/{name}.py',top_job=True)
    curate(RDL/'npz/fill_nan_nearest.py','agu_james/data/nearest_fill.py')
    curate(RDL/'npz/resize_swh.py','agu_james/data/wave_interpolation.py')
    gnn_files={'rmse':'rmse','acc_forecast_acitivity':'acc_activity','forecast_activity_by_step':'target_activity',
               'spectrum_all_stat':'energy','coherence_spectrum_all_stat':'coherence','spectral_threshold_summary':'cutoff',
               'tc_bst_msl_intensity':'tc_base','tracker_ver_aurora':'tc_tracker',
               'tc_axisymmetric_metrics_to_csv':'tc_wind','tc_axisymmetric_wcc_to_npz':'tc_vertical',
               'tc_phase_core_to_csv':'tc_phase','tc_gwb_700z_to_csv':'tc_gwb','tc_ike_calculation_to_csv':'tc_ike',
               'download_clim':'climatology'}
    aliases={name:'agu_james.analysis.gnn.'+alias for name,alias in gnn_files.items()}
    for name,alias in gnn_files.items():
        curate(RDL/'gnn/ver_autoregressive/output'/f'{name}.py',f'agu_james/analysis/gnn/{alias}.py',aliases)
    pg_files=['subset','loss','tc_tracker_bst_msl','tracker_ver_aurora','weight_rmse_prs_multistep','weight_rmse_sfc_multistep',
              'acc_metric_prs','acc_metric_sfc','spectrum_multistep_prs','spectrum_multistep_sfc','axisymmetric_wind_metrics_all',
              'axisymmetric_gwb_metrics_all','axisymmetric_b_vt_metrics_all','ike_metric_700_all']
    aliases={name:'agu_james.analysis.pangu.'+name for name in pg_files}
    for name in pg_files:
        curate(PANGU/'validation'/f'{name}.py',f'agu_james/analysis/pangu/{name}.py',aliases,top_job=name.startswith(('weight_rmse','acc_metric')))
    curate(PANGU/'func_inference.py','agu_james/validation/pangu_inference.py',transform=pangu_functions_only)
    plots={'evaluation_metric/plot_james_metrics.py':'metrics','spectrum analysis/plot_spectral_energy_summary.py':'energy',
           'spectrum analysis/coherence_energy_ratio_scatter.py':'coherence',
           'tc_verification/plot_tc_verification.py':'tc_verification','tc_verification/plot_tc_physics_structure.py':'tc_physics',
           'tc_verification/tc_distribution_plots.py':'tc_distributions',
           'case_study and monthly simulation/plot_krathon_track_intensity.py':'krathon_track',
           'case_study and monthly simulation/plot_krathon_wind850_q700.py':'krathon_fields',
           'case_study and monthly simulation/plot_monthly_spatial_spectrum.py':'monthly',
           'vis_multi_steps/vis_multi_model_forecast_compare_monthly_all.py':'monthly_maps',
           'pre_process/chapter2_vis_composite.py':'graph','pre_process/chapter2_vis.py':'graph_helpers'}
    aliases={Path(s).stem:'agu_james.plotting.'+t for s,t in plots.items()}
    for source,target in plots.items():
        curate(PAPER/source,f'agu_james/plotting/{target}.py',aliases)
    curate(PAPER/'vis_multi_steps/vis_muliti_step_ours_save.py','agu_james/analysis/export_gnn_maps.py')
    curate(PANGU/'vis_multi_steps/vis_muliti_step_combined.py','agu_james/analysis/export_pangu_maps.py')
    # Fixed channel order and weights are configuration, not copied data/statistics.
    for label,ver,name in [('ar','ver_autoregressive','next_feature_table_weight_ver3.csv'),('coarse','ver_unet_scale','next_feature_table_weight.csv')]:
        source=RDL/'gnn'/ver/name
        rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
        rows=[{'feature':r['feature'],'weight':float(r['weight'])} for r in rows][:116]
        p=write(f'agu_james/config/channels_{label}.json',json.dumps(rows,indent=2)+'\n')
        records.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), destination=p.relative_to(ROOT).as_posix(),output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),modifications=['extract fixed channel order/weights as configuration; exclude measured statistics']))
    # Merge model dependencies whose parsed source is identical (comments/format ignored).
    groups=defaultdict(list)
    for record in records:
        if record['destination'].startswith('agu_james/training/models/'):
            parsed=ast.parse(Path(record['source']).read_text(encoding='utf-8-sig'))
            parsed.body=[n for n in parsed.body if not is_main(n) and not (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in DEMO)]
            semantic=hashlib.sha256(ast.dump(parsed,include_attributes=False).encode()).hexdigest()
            record['source_ast_sha256']=semantic
            groups[semantic].append(record)
    redirects={}
    for group in groups.values():
        if len(group)<2:continue
        old_paths=[g['destination'] for g in group]
        canonical='agu_james/training/models/shared/'+Path(old_paths[0]).name
        source=ROOT/old_paths[0]
        write(canonical,source.read_text(encoding='utf-8'));package('agu_james/training/models/shared')
        for record in group:
            redirects[record['destination'].removesuffix('.py').replace('/','.')]=canonical.removesuffix('.py').replace('/','.')
            record['destination']=canonical
            record['modifications'].append('merged AST-identical original sources into shared model library; original byte hashes remain recorded')
        for old in old_paths:
            target=(ROOT/old).resolve()
            if not target.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe deduplication target')
            target.unlink()
    for p in (ROOT/'agu_james').rglob('*.py'):
        original=p.read_text(encoding='utf-8');tree=ast.parse(original);changed=False
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and n.module in redirects:
                n.module=redirects[n.module];changed=True
        if changed:
            header='\n'.join(original.splitlines()[:2])+'\n' if original.startswith('# Derived from') else ''
            ast.fix_missing_locations(tree);p.write_text(header+ast.unparse(tree)+'\n',encoding='utf-8')
    for record in records:record['output_sha256']=hashlib.sha256((ROOT/record['destination']).read_bytes()).hexdigest()
    write('provenance/sources.json',json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(f'Curated {len(records)} sources; no data or weights copied.')

if __name__=='__main__':main()
