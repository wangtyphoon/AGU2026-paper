"""Verify complete replay, generate paired differences, and redraw Figure 10."""
import importlib.util
import importlib.machinery
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.recompute_thermal_wind import OUT, LEVELS, numerical_namespace, sha
from agu_james.plotting import tc_physics as physics
from agu_james.analysis.gnn import tc_phase

def staged(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; loader.exec_module(module)
    return module

def main():
    config = json.loads((ROOT/'configs/gwb.local.json').read_text())
    checks, summaries, baseline_checks = [], [], []
    for m in config['models']:
        manifest = json.loads((OUT/f"{m['id']}_manifest.json").read_text())
        assert sha(manifest['input']) == manifest['input_sha256']
        assert sha(manifest['output']) == manifest['output_sha256']
        for path, fp in manifest['data_fingerprints'].items():
            stat = Path(path).stat()
            assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == fp
        checks.append({k:v for k,v in manifest.items() if k!='data_fingerprints'})
        changes = pd.read_csv(OUT/f"{m['id']}_case_changes.csv")
        audit = pd.read_csv(OUT/f"{m['id']}_profile_audit.csv")
        # Reconstruct the OLD interpolated calculation from every cached profile.
        for role in ('forecast','target'):
            rows = audit[audit.role==role]
            reconstructed = {'vtl':[], 'vtu':[]}
            for values in rows[[f'delta_z_core_{p}' for p in LEVELS]].to_numpy():
                for key, grid in [('vtl', np.arange(900,550,-50)), ('vtu', np.arange(600,250,-50))]:
                    interp = tc_phase.interpolate_delta_z(np.array(LEVELS), values, grid)
                    reconstructed[key].append(tc_phase.regression_slope_ln_p(grid, interp))
            for var in ('vtl','vtu'):
                col = f'{role}_{var}'
                old, new = changes[col+'_old'].to_numpy(), changes[col+'_new'].to_numpy()
                baseline = np.asarray(reconstructed[var]); finite = np.isfinite(old)&np.isfinite(baseline)
                delta = baseline[finite]-old[finite]
                baseline_checks.append(dict(model=m['label'],role=role,variable=var,count=int(finite.sum()),
                    max_abs_old_reproduction_error=float(np.max(np.abs(delta))) if finite.any() else None))
                assert np.allclose(baseline[finite],old[finite],rtol=1e-10,atol=1e-8), (m['id'], col, np.max(np.abs(delta)))
                for step in (0,4,8,12):
                    sel = np.ones(len(old),bool) if step==0 else changes.step.to_numpy()==step
                    paired = sel&np.isfinite(old)&np.isfinite(new)
                    d = new[paired]-old[paired]
                    summaries.append(dict(model=m['label'],role=role,variable=var,lead_hours=step*6,
                        rows=int(sel.sum()),paired_count=int(paired.sum()),
                        mean_difference=float(np.mean(d)),mean_abs_difference=float(np.mean(np.abs(d))),
                        max_abs_difference=float(np.max(np.abs(d))),
                        difference_q05=float(np.quantile(d,.05)),difference_q50=float(np.quantile(d,.5)),difference_q95=float(np.quantile(d,.95)),
                        sign_flips=int(((old[paired]*new[paired])<0).sum()),
                        finite_added=int((sel&~np.isfinite(old)&np.isfinite(new)).sum()),
                        finite_lost=int((sel&np.isfinite(old)&~np.isfinite(new)).sum())))
    pd.DataFrame(checks).to_csv(OUT/'output_verification.csv',index=False)
    pd.DataFrame(baseline_checks).to_csv(OUT/'old_algorithm_reproduction.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'paired_change_summary.csv',index=False)
    audit_rows, oldaudit, newaudit = [], [], []
    oldraw,_ = physics.load_scalar_data(ROOT/'artifacts/gwb_endpoint_corrected/tracker_multi',12)
    newraw,_ = physics.load_scalar_data(OUT/'tracker_multi',12)
    oldcommon = physics.common_for_variables(oldraw,physics.ALL_KEYS,oldaudit,'old')
    newcommon = physics.common_for_variables(newraw,physics.ALL_KEYS,newaudit,'new')
    pd.DataFrame(oldaudit+newaudit).to_csv(OUT/'figure_sample_audit.csv',index=False)
    keys=physics.CASE_KEY_COLS
    intersection = oldcommon['MSE'][keys].merge(newcommon['MSE'][keys],on=keys,validate='one_to_one')
    pd.DataFrame(physics.target_agreement(newcommon,('vtl','vtu'),strict=False)).to_csv(OUT/'target_agreement.csv',index=False)
    metrics=[]; common_changes=[]
    for label in oldcommon:
        old = intersection.merge(oldcommon[label],on=keys,validate='one_to_one').sort_values(keys)
        new = intersection.merge(newcommon[label],on=keys,validate='one_to_one').sort_values(keys)
        oldref = intersection.merge(oldcommon['MSE'],on=keys,validate='one_to_one').sort_values(keys)
        newref = intersection.merge(newcommon['MSE'],on=keys,validate='one_to_one').sort_values(keys)
        for step in (4,8,12):
            for var in ('vtl','vtu'):
                mask = old.step==step
                for role in ('forecast','target'):
                    col=role+'_'+var
                    a,b=old.loc[mask,col].to_numpy(),new.loc[mask,col].to_numpy()
                    d=b-a
                    common_changes.append(dict(model=label,lead_hours=step*6,role=role,variable=var,count=len(d),
                        mean_difference=float(np.mean(d)),mean_abs_difference=float(np.mean(np.abs(d))),
                        max_abs_difference=float(np.max(np.abs(d))),difference_q05=float(np.quantile(d,.05)),
                        difference_q50=float(np.quantile(d,.5)),difference_q95=float(np.quantile(d,.95)),
                        sign_flips=int(((a*b)<0).sum())))
                for version,df,ref in [('old',old,oldref),('new',new,newref)]:
                    f,t = df.loc[mask,'forecast_'+var].to_numpy(),ref.loc[mask,'target_'+var].to_numpy()
                    ks=ks_2samp(f,t)
                    metrics.append(dict(model=label,lead_hours=step*6,variable=var,version=version,count=len(f),
                        rmse=float(np.sqrt(np.mean((f-t)**2))),ks_d=float(ks.statistic),ks_pvalue=float(ks.pvalue),
                        forecast_q95=float(np.quantile(f,.95)),era5_q95=float(np.quantile(t,.95))))
    pd.DataFrame(metrics).to_csv(OUT/'lead_paired_metrics.csv',index=False)
    pd.DataFrame(common_changes).to_csv(OUT/'lead_common_change_summary.csv',index=False)
    intersection.to_csv(OUT/'old_new_common_cases.csv',index=False)
    targets=[physics.VARS[k].target for k in physics.ALL_KEYS]
    newcommon['MSE'][keys+targets].to_csv(OUT/'new_common_scalar_cases.csv',index=False)
    common={}
    for label,df in newcommon.items():
        df=df.copy(); df[targets]=newcommon['MSE'][targets].to_numpy(); common[label]=df
    directory=OUT/'figure10'; directory.mkdir(exist_ok=True)
    distribution=staged('thermal_staged_distributions',OUT/'code_replacement/agu_distributions.replacement.py.txt')
    with distribution.plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                                     'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False}):
        rows=distribution.plot_structure(common,directory,12,600)
    pd.DataFrame(rows).to_csv(directory/'tc_structure_relationship_distribution.csv',index=False)
    manifest=dict(status='recomputed_pending_installation',method='signed-native-pressure-levels-v1',
        lower_pressures_hpa=[925,850,700,600],upper_pressures_hpa=[600,500,400,300],
        radius_km=500,p0_hpa=1,regression='equal-weight OLS',interpolation=False,
        missing_policy='at least two finite native layers; no filling',units='m',positive='warm core',negative='cold core',
        delta_z_core='maximum minus minimum geopotential height within 500 km on each pressure layer',
        delta_z_sim='Z(R)-Z(0), unchanged',source_pdf=r'D:\master\James Paper V1.pdf',
        original_total_rows=sum(c['rows'] for c in checks),
        finite_diagnostics=sum(c['finite_values'] for c in checks),
        common_cases={str(s*6):int((newcommon['MSE'].step==s).sum()) for s in (4,8,12)},
        old_common_cases={str(s*6):int((oldcommon['MSE'].step==s).sum()) for s in (4,8,12)},
        fixed_paired_cases={str(s*6):int((intersection.step==s).sum()) for s in (4,8,12)},
        target_policy='shared MSE ERA5 targets; original target discrepancies exported',
        figure_contours_probability=[.5,.75,.9])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('original_total_rows','finite_diagnostics','common_cases')},indent=2))

if __name__=='__main__': main()
