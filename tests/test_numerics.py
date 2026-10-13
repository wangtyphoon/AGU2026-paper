import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from typing import Dict,Optional,Sequence,Tuple
import numpy as np
import pandas as pd
import torch
from agu_james import runtime
from agu_james.training import spectral_mae,spectral_mse
from agu_james.validation.coherence import _build_radial_index,_radial_coherence_torch

torch.set_num_threads(2)

def original_functions(source,names):
    tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
    namespace=dict(torch=torch,np=np,pd=pd,Optional=Optional,Tuple=Tuple,Dict=Dict,Sequence=Sequence,RANDOM_EPS=1e-12)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace

class NumericalContracts(unittest.TestCase):
    def setUp(self):runtime.configure({})

    def test_boundary_reuse_preserves_previous_state(self):
        from torch_geometric.data import HeteroData
        from torch_geometric.data import Batch
        from agu_james.training.engine import loss_for_batch
        from agu_james.training.loss import create_spatial_weights
        current=HeteroData();current['grid'].x=torch.zeros(192*192,256)
        future=[]
        for step in [1,2]:
            g=HeteroData();g['grid'].x=torch.ones(192*192,126);g['grid'].x[:,-1]=step;future.append(g)
        seen=[]
        class Model(torch.nn.Module):
            def forward(self,g):
                seen.append(g['grid'].x.clone())
                return torch.full((192*192,116),2.0)
        loss,_=loss_for_batch(Model(),current,future,2,torch.ones(116),torch.ones(116),create_spatial_weights(boundary=9))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((seen[0][:,:116]==0).all())
        image=seen[1][:,0].reshape(192,192)
        self.assertEqual(image[8,9],1);self.assertEqual(image[9,9],2)
        self.assertTrue((seen[1][:,125:247]==0).all())

    def test_loss_gradients_finite(self):
        torch.manual_seed(3407)
        base=torch.randn(2*24*24,3)
        pred=torch.randn_like(base,requires_grad=True);target=torch.randn_like(base)
        for module in [spectral_mae,spectral_mse]:
            value,components=module.spectral_composite_loss(base,pred,target,H=24,W=24,boundary=2,
                              cutoffs=torch.full((3,),4.0),channel_weights=torch.ones(3),scale=torch.ones(3),l_tilde=6,l_max=6)
            self.assertTrue(torch.isfinite(value));value.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(pred.grad).all());pred.grad.zero_()

    def test_ar_model_forward_backward(self):
        from torch_geometric.data import HeteroData
        from agu_james.training.models.ar.model_unet import HeteroGNN
        g=HeteroData();g['grid'].x=torch.randn(16,256)
        for level in range(3,8):g[f'mesh{level}'].x=torch.randn(4,7)
        def edge(a,rel,b,source_nodes,target_nodes):
            index=torch.tensor([[i%source_nodes for i in range(max(source_nodes,target_nodes))],[i%target_nodes for i in range(max(source_nodes,target_nodes))]])
            g[a,rel,b].edge_index=index;g[a,rel,b].edge_attr=torch.randn(index.shape[1],6)
        edge('grid','encoder7','mesh7',16,4)
        for level in range(3,8):
            edge(f'mesh{level}','processor',f'mesh{level}',4,4)
            edge(f'mesh{level}',f'decoder{level}','grid',4,16)
        for level in range(4,8):
            edge(f'mesh{level}',f'pool{level}_to_{level-1}',f'mesh{level-1}',4,4)
            edge(f'mesh{level-1}',f'upsample{level-1}_to_{level}',f'mesh{level}',4,4)
        model=HeteroGNN(256,7,8,116,processor_layers=[1]*5,use_checkpoint=False,ffn_ratio=2)
        from torch_geometric.data import Batch
        out=model(Batch.from_data_list([g]))
        self.assertEqual(tuple(out.shape),(16,116));self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_optimizer_counter_counts_accumulated_updates(self):
        import tempfile
        from unittest.mock import patch
        from torch_geometric.data import HeteroData
        from agu_james.training import engine,dataset
        from agu_james.data.pipeline import channel_config
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        model=Model()
        class Train(torch.utils.data.Dataset):
            prs_feature=[c['feature'] for c in channel_config()]
            sfc_regular_feature=[];sfc_average_feature=[];remove_prs=[]
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 4
            def __getitem__(self,i):
                g=HeteroData();g['grid'].x=torch.zeros(1,256)
                return g,[g.clone()]
        class Val(Train):
            def __len__(self):return 1
            def __getitem__(self,i):
                a,b=super().__getitem__(i);return a,b,'1979010106.npz'
        def objective(m,*args,**kwargs):return m.weight.square(),[]
        scratch=Path(__file__).resolve().parents[1]/'artifacts/verification'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='optimizer-test-') as directory:
            root=Path(directory);(root/'statistics').mkdir()
            pd.DataFrame(dict(variable=Train.prs_feature,std=np.ones(116))).to_csv(root/'statistics/target_mean_std.csv',index=False)
            runtime.configure(dict(artifact_root=str(root),training=dict(hidden_dim=8)))
            spec=dict(stage='1',rollout_steps=1,effective_batch=4,epochs=None,max_optimizer_updates=3,peak_lr=1e-3)
            with patch.object(engine,'schedule',return_value=spec),patch.object(engine,'make_model',return_value=(model,{})),patch.object(engine,'loss_for_batch',side_effect=objective),patch.object(dataset,'WeatherGraphNPZTrainDataset',Train),patch.object(dataset,'WeatherGraphNPZValDataset',Val):
                output=engine.run('1',1,micro_batch=1,device='cpu')
            state=torch.load(output,map_location='cpu',weights_only=False)
            self.assertEqual(state['optimizer_updates'],3)
            self.assertEqual(state['epoch'],3)
            steps=[int(s['step']) for s in state['optimizer_state_dict']['state'].values()]
            self.assertEqual(steps,[3])
            self.assertLess(float(model.weight.detach()),1.)

    @unittest.skipUnless(os.environ.get('AGU_RDL_REFERENCE'),'read-only research source path not configured')
    def test_source_coherence_equivalence(self):
        source=Path(os.environ['AGU_RDL_REFERENCE'])/'gnn/ver_autoregressive/train_script/val_autoregressive_with_spectrum.py'
        original=original_functions(source,['_build_radial_index','_radial_coherence_torch'])
        torch.manual_seed(7);a=torch.randn(3,2,24,24);b=torch.randn_like(a)
        args=_build_radial_index(24,24,torch.device('cpu'))
        np.testing.assert_allclose(_radial_coherence_torch(a,b,*args).numpy(),original['_radial_coherence_torch'](a,b,*args).numpy(),rtol=1e-13,atol=1e-13)

    @unittest.skipUnless(os.environ.get('AGU_RDL_REFERENCE'),'read-only research source path not configured')
    def test_source_spectral_loss_equivalence(self):
        root=Path(os.environ['AGU_RDL_REFERENCE'])/'gnn'
        torch.manual_seed(4);base=torch.randn(2*24*24,3);pred=torch.randn_like(base);target=torch.randn_like(base)
        for module,source in [(spectral_mae,root/'ver_autoregressive/train_script/spectrum_loss.py'),(spectral_mse,root/'ver_spectrum_v2/train_script/spectrum_loss_v2.py')]:
            original=original_functions(source,None)
            kwargs=dict(H=24,W=24,boundary=2,scale=torch.ones(3),cutoffs=torch.full((3,),4.),channel_weights=torch.ones(3),l_tilde=6,l_max=6)
            a=module.spectral_composite_loss(base,pred,target,**kwargs)[0]
            fn=original.get('spectral_composite_loss',original.get('spectral_composite_loss_v2'))
            b=fn(base,pred,target,**kwargs)[0]
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    @unittest.skipUnless(os.environ.get('AGU_PAPER_REFERENCE'),'read-only paper input path not configured')
    def test_common_cases_and_summary(self):
        from agu_james.plotting import tc_verification as revised
        root=Path(os.environ['AGU_PAPER_REFERENCE'])
        source=root/'tc_verification/plot_tc_verification.py'
        spec=importlib.util.spec_from_file_location('original_tc_verification',source)
        original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
        a,aw=revised.load_common_data(root/'tracker_multi');b,bw=original.load_common_data(root/'tracker_multi')
        for model in a:
            pd.testing.assert_frame_equal(a[model],b[model]);pd.testing.assert_frame_equal(aw[model],bw[model])
        pd.testing.assert_frame_equal(revised.summarize(a,aw),original.summarize(b,bw))

    @unittest.skipUnless(os.environ.get('AGU_PAPER_REFERENCE'),'read-only paper input path not configured')
    def test_metric_and_energy_inputs(self):
        from agu_james.plotting import metrics,energy
        root=Path(os.environ['AGU_PAPER_REFERENCE'])
        for name,module,relative in [('original_metrics',metrics,'evaluation_metric/plot_james_metrics.py'),('original_energy',energy,'spectrum analysis/plot_spectral_energy_summary.py')]:
            spec=importlib.util.spec_from_file_location(name,root/relative);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
            if module is metrics:
                a=module.load_data(root,28);b=old.load_data(root,28)
                # Compare the complete normalized metric table.
                pd.testing.assert_frame_equal(a,b)
            else:
                module.DATA_DIR=root/'spectrum_multi'
                a=module.load_period('7d');b=old.load_period('7d')
                for x,y in zip(a,b):
                    for field in ['k','forecast','target','ratio','total_ratio']:np.testing.assert_allclose(getattr(x,field),getattr(y,field),rtol=1e-13,atol=1e-13,equal_nan=True)

if __name__=='__main__':unittest.main()
