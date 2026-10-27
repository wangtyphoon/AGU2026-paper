import importlib
from pathlib import Path
import sys
from agu_james import runtime

GROUPS={'metrics':'metrics','energy':'energy','coherence':'coherence','tc':'tc_verification','physics':'tc_physics','krathon-track':'krathon_track','krathon-fields':'krathon_fields','monthly':'monthly','graph':'graph'}

def input_paths(input_root=None):
    """Resolve a portable figure bundle; an explicit root overrides asset paths."""
    bundled = input_root is not None or 'figure_inputs' in runtime._CONFIG.get('paths', {})
    root = Path(input_root).resolve() if input_root is not None else runtime.path(
        'figure_inputs', runtime.artifact_root() / 'paper_inputs')
    graph = root / 'graphs/mesh7/hetero_graph.pt' if bundled else runtime.artifact_root() / 'graphs/mesh7/mesh_grid_data_unet7/hetero_graph.pt'
    train = (root if bundled else runtime.artifact_root()) / 'calibration/mse_train_coherence_energy_rollout28.npz'
    ibtracs = root / 'case_study and monthly simulation/krathon_ibtracs_source.csv'
    if input_root is None:
        graph = runtime.path('mesh7_raw_graph', graph)
        train = runtime.path('mse_train_spectrum', train)
        ibtracs = runtime.path('krathon_ibtracs', ibtracs)
    return root, graph, train, ibtracs

def run(group,input_root=None,output_dir=None,extra=None):
    root,graph,train,ibtracs=input_paths(input_root)
    out=Path(output_dir or runtime.artifact_root()/'figures'/group).resolve();out.mkdir(parents=True,exist_ok=True)
    config=dict(runtime._CONFIG);overrides=dict(config.get('overrides',{}))
    settings={
        'metrics':dict(ROOT=root),
        'energy':dict(DATA_DIR=root/'spectrum_multi'),
        'coherence':dict(BASE_DIR=out,DATA_DIR=root/'spectrum_multi',OUTPUT_DIR=out,output_dir=out,DEFAULT_MSE_TRAIN_NPZ=train),
        'krathon_track':dict(HERE=out),
        'krathon_fields':dict(HERE=out,ROOT=root,DATA=root/'vis_multi_steps',BEST_TRACK=root/'tracker_multi/cyclone_data_2024.csv'),
        'monthly':dict(BASE=out,MAP_BASE=root/'vis_multi_steps'),
        'monthly_maps':dict(LAT_PATH=root/'vis_multi_steps/lat.npy',LON_PATH=root/'vis_multi_steps/lon.npy',
                            OURS_SPECTRUM_MONTHLY_ROOT=root/'vis_multi_steps/ours_spectrum_monthly',OURS_SPECTRUM_VER2_MONTHLY_ROOT=root/'vis_multi_steps/ours_spectrum_ver2_monthly'),
        'tc_verification':dict(output_dir=out,tracker_dir=root/'tracker_multi'),
        'tc_physics':dict(OUTPUT_DIR=out),
        'graph':dict(GRAPH_PATH=graph,OUTPUT_DIR=out),
    }
    for name,values in settings.items():overrides.update({'agu_james.plotting.'+name+'.'+k:v for k,v in values.items()})
    config['overrides']=overrides;runtime.configure(config)
    # Import helpers with the same configured paths before loading composites.
    for name in ['energy','monthly_maps']:importlib.reload(importlib.import_module('agu_james.plotting.'+name))
    module=importlib.reload(importlib.import_module('agu_james.plotting.'+GROUPS[group]))
    old=sys.argv
    args=list(extra or [])
    if group in ['metrics','energy','monthly']:args=['--output-dir',str(out)]+args
    if group=='physics':args=['--tracker-dir',str(root/'tracker_multi'),'--structure-dir',str(root/'struc_multi'),'--output-dir',str(out)]+args
    if group=='krathon-track':args=['--tracker-root',str(root/'tracker_multi'),'--output-dir',str(out),'--ibtracs-path',str(ibtracs)]+args
    try:
        sys.argv=[module.__name__]+args
        status=module.main()
        if isinstance(status,int) and status != 0:
            raise SystemExit(status)
    finally:sys.argv=old
    runtime.write_json(out/'workflow_manifest.json',dict(group=group,input_root=str(root),output_dir=str(out),parameters=args,
                       variant='paper settings; not verified against HPC execution',figure2='manual draw.io source, no generated-code equivalent'))
    return out
