"""Inference control contract with fake operators; no real ONNX weights used."""
from datetime import datetime,timedelta
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from agu_james import runtime

@unittest.skipUnless(importlib.util.find_spec('onnxruntime'),'ONNX Runtime environment needed')
class PanguControls(unittest.TestCase):
    def test_six_hour_outputs_global_state_and_24h_anchors(self):
        from agu_james.validation import pangu_inference as source
        calls=[]
        class Session:
            def __init__(self,file,**kwargs):self.hours=int(Path(file).stem.split('_')[-1])
            def run(self,unused,values):
                u,s=values['input'],values['input_surface']
                calls.append((self.hours,float(u.mean()),u.shape))
                return u+self.hours,s+self.hours
        scratch=Path(__file__).resolve().parents[1]/'artifacts/verification';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='pangu-test-') as directory:
            root=Path(directory);runtime.configure(dict(artifact_root=str(root/'other-root')))
            t=datetime(2024,9,29,12);stamp=t.strftime('%Y-%m-%d-%H-%M')
            inputs=root/'forecasts'/stamp;inputs.mkdir(parents=True)
            np.save(inputs/'input_upper.npy',np.zeros((5,13,4,4),dtype=np.float32))
            np.save(inputs/'input_surface.npy',np.zeros((4,4,4),dtype=np.float32))
            selector=((slice(1,3),slice(1,3)),np.array([1,0]),np.array([103,104]))
            with patch.object(source.ort,'InferenceSession',Session),patch.object(source,'_build_region_selector',return_value=selector) as crop:
                source.run_pangu_inference(t,t+timedelta(hours=54),{6:'model_6.onnx',24:'model_24.onnx'},
                    base_dir=root,use_gpu=False,reference_nc='external-coordinate.nc',save_every_hours=6,save_after_hours=0,direct_hours=(24,48))
            crop.assert_called_once_with('external-coordinate.nc',lat_range=(0.,47.75),lon_range=(102.25,150.))
            out=root/'results'/stamp
            for hour in range(6,55,6):
                file=out/f'output_upper_{(t+timedelta(hours=hour)).strftime("%Y-%m-%d-%H-%M")}.npy'
                value=np.load(file);self.assertEqual(value.shape,(5,13,2,2));np.testing.assert_equal(value,hour)
            self.assertEqual([base for interval,base,shape in calls if interval==24],[0.,24.])
            self.assertTrue(all(shape==(5,13,4,4) for _,_,shape in calls))

if __name__=='__main__':unittest.main()
