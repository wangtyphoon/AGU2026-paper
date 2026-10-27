"""Numerical and entrypoint checks for the signed native-pressure policy."""
import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
LEVELS = [925, 850, 700, 600, 500, 400, 300]

def candidate(name):
    modules = {'agu_gnn': 'agu_james.analysis.gnn.tc_phase',
               'agu_pangu': 'agu_james.analysis.pangu.axisymmetric_b_vt_metrics_all'}
    return importlib.import_module(modules[name])

class ThermalWind(unittest.TestCase):
    def test_analytic_sign_and_cross_implementation(self):
        for name in ['agu_gnn', 'agu_pangu']:
            module=candidate(name)
            for slope in (-75.,0.,75.):
                result=module.compute_native_vt(LEVELS,1000+slope*np.log(LEVELS))
                self.assertAlmostEqual(result['vt_lower'],slope,places=8)
                self.assertAlmostEqual(result['vt_upper'],slope,places=8)

    def test_missing_layers_without_interpolation(self):
        module=candidate('agu_gnn')
        result=module.compute_native_vt([925,600],[300.,200.])
        self.assertEqual(result['lower_valid_levels'],2)
        self.assertEqual(result['upper_valid_levels'],1)
        self.assertTrue(np.isnan(result['vt_upper']))
        self.assertAlmostEqual(result['vt_lower'],100/np.log(925/600))
        other=module.compute_native_vt([925,600,900,550],[300.,200.,1e8,-1e8])
        self.assertEqual(other['vt_lower'],result['vt_lower'])
        self.assertEqual(other['upper_valid_levels'],1)

    def test_main_no_interpolation_calls(self):
        for name in ('agu_gnn','agu_pangu'):
            tree=ast.parse(Path(candidate(name).__file__).read_text(encoding='utf-8'))
            for function in [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('main','compute_vt_indicators','compute_native_vt')]:
                calls={n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
                self.assertFalse(calls & {'interpolate_delta_z','interpolate_profile','build_pressure_grid'})

    def test_gnn_main_outputs_and_counts(self):
        module=candidate('agu_gnn')
        scratch=ROOT/'artifacts/verification'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            base=Path(directory); csv=base/'input.csv'; output=base/'output.csv'
            lat=np.linspace(19,21,3);lon=np.linspace(119,121,3)
            np.save(base/'lat.npy',lat);np.save(base/'lon.npy',lon)
            pd.DataFrame([dict(CYCLONE_NUM=1,forecast_init_time=2024010100,step=i+1,
                forecast_center_lat=20,forecast_center_lon=120,bst_lat=20,bst_lon=120,source_file='sample.npz') for i in range(2)]).to_csv(csv,index=False)
            channels={f'z{p}':i for i,p in enumerate(LEVELS)}
            field=np.zeros((7,3,3),dtype=float)
            for i,p in enumerate(LEVELS): field[i,0,0]=(1000+80*np.log(p))*9.80665
            settings={'base_dir':base,'csv_path':csv,'output_csv':output,'lat_path':base/'lat.npy','lon_path':base/'lon.npy'}
            with patch.object(module.runtime,'setting',side_effect=lambda key,default:settings.get(key.rsplit('.',1)[-1],default())), \
                 patch.object(module,'get_channel_map',return_value=channels), \
                 patch.object(module,'find_npz',return_value=base/'sample.npz'), \
                 patch.object(module,'load_forecast_and_target',return_value=(field,field)):
                module.main()
            result=pd.read_csv(output)
            np.testing.assert_allclose(result[['forecast_vtl','forecast_vtu','target_vtl','target_vtu']],80,atol=1e-8)
            counts=pd.read_csv(output.with_suffix('.thermal_wind_counts.csv'))
            self.assertTrue((counts.drop(columns='row')==4).all().all())
            method=json.loads(output.with_suffix('.thermal_wind_method.json').read_text())
            self.assertFalse(method['interpolation'])

    def test_pangu_main_outputs_and_counts(self):
        module=candidate('agu_pangu')
        scratch=ROOT/'artifacts/verification'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            base=Path(directory);csv=base/'input.csv';output=base/'output.csv'
            lat=np.linspace(19,21,3);lon=np.linspace(119,121,3)
            np.save(base/'lat.npy',lat);np.save(base/'lon.npy',lon)
            pd.DataFrame([dict(init_time=2024010100,cyclone_num=1,time=2024010106,step=1,
                forecast_lat=20,forecast_lon=120,bst_lat=20,bst_lon=120)]).to_csv(csv,index=False)
            field=np.zeros((5,13,3,3))
            for p in LEVELS: field[0,module.PANGU_PRESSURE_LEVELS[p],0,0]=(1000+80*np.log(p))*9.80665
            forecast=base/'results/2024-01-01-00-00';forecast.mkdir(parents=True)
            target=base/'forecasts/2024-01-01-06-00';target.mkdir(parents=True)
            np.save(forecast/'output_upper_2024-01-01-06-00.npy',field)
            np.save(target/'input_upper.npy',field)
            cfg=dict(csv_path=csv,output_path=output,results_dir=base/'results',era5_dir=base/'forecasts',
                lat_path=base/'lat.npy',lon_path=base/'lon.npy')
            with patch.object(module.runtime,'setting',side_effect=lambda key,default:cfg if key.endswith('.config') else default()):
                module.main()
            result=pd.read_csv(output)
            np.testing.assert_allclose(result[['forecast_vtl','forecast_vtu','target_vtl','target_vtu']],80,atol=1e-8)
            counts=pd.read_csv(output.with_suffix('.thermal_wind_counts.csv'))
            self.assertTrue((counts.drop(columns='row')==4).all().all())

if __name__=='__main__': unittest.main()
