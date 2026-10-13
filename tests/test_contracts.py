from datetime import datetime,timedelta
from pathlib import Path
import ast
import json
import tempfile
import unittest
import numpy as np
from agu_james.contracts import *
from agu_james.data.pipeline import channel_config,time_features

class PaperContracts(unittest.TestCase):
    def test_calendar_split(self):
        self.assertEqual(split_for_year(1979),'train')
        for year in VALIDATION_YEARS:self.assertEqual(split_for_year(year),'val')
        self.assertEqual(split_for_year(2024),'test')
        with self.assertRaises(ValueError):split_for_year(2025)

    def test_ocean_mask_is_training_only_before_fill(self):
        from agu_james.data.pipeline import update_ocean_mask
        masks={};array=np.array([[1.,np.nan],[2.,3.]])
        update_ocean_mask(masks,'sst',array,'test','2024010100')
        self.assertFalse(masks)
        update_ocean_mask(masks,'sst',array,'train','1979010100')
        self.assertEqual(masks['sst']['missing_cells'],1)
        self.assertFalse(masks['sst']['mask'][0,1])
        update_ocean_mask(masks,'sst',np.ones((2,2)),'train','1979010106')
        self.assertEqual(masks['sst']['reference_time'],'1979010100')

    def test_jtwc_conversion_dedup_basin_and_sampling(self):
        from agu_james import runtime
        from agu_james.data.pipeline import best_track
        import pandas as pd
        scratch=Path(__file__).resolve().parents[1]/'artifacts/verification';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='best-track-test-') as directory:
            root=Path(directory);source=root/'input.dat'
            rows=['WP, 01, 2024071012, 00, BEST, 0, 200N, 1200E, 50',
                  'WP, 01, 2024071012, 00, BEST, 0, 200N, 1200E, 50',
                  'WP, 02, 2024072012, 00, BEST, 0, 200N, 1600E, 50',
                  'IO, 03, 2024072512, 00, BEST, 0, 200N, 1200E, 50']
            source.write_text('\n'.join(rows),encoding='utf-8')
            runtime.configure(dict(artifact_root=str(root/'output')))
            out=best_track(source)
            frame=pd.read_csv(out/'cyclone_data_2024.csv')
            self.assertEqual(len(frame),2);self.assertEqual(frame.LAT.iloc[0],20.)
            days=pd.read_csv(out/'raw_grib/time_series.csv')
            self.assertEqual(len(days),7)
            self.assertEqual(days.DATE.iloc[0][:10],'2024-07-07')
            self.assertEqual(days.DATE.iloc[-1][:10],'2024-07-13')
            self.assertEqual(source.read_text(encoding='utf-8'),'\n'.join(rows))
        runtime.configure({})

    def test_sequence_gap_and_split(self):
        start=datetime(2023,12,31,0)
        times=[start+timedelta(hours=6*i) for i in range(10)]
        valid=valid_initializations(times,2)
        self.assertIn(datetime(2023,12,31,6),valid)
        self.assertNotIn(datetime(2023,12,31,18),valid)
        self.assertNotIn(datetime(2024,1,1,0),valid)
        self.assertIn(datetime(2024,1,1,6),valid)
        self.assertFalse(valid_initializations([times[0],times[2]],1))
        self.assertEqual({t.hour for t in valid_initializations([datetime(2024,7,1)+timedelta(hours=6*i) for i in range(12)],1)},{0,6,12,18})

    def test_nine_cells(self):
        mask=boundary_mask()
        self.assertEqual(int((~mask).sum()),174*174)
        self.assertTrue(mask[:9].all());self.assertTrue(mask[:,:9].all())
        self.assertFalse(mask[9,9]);self.assertTrue(mask[8,9])
        w=spatial_weights(np.linspace(47.75,0,192),192)
        np.testing.assert_allclose(w[mask],0)
        self.assertAlmostEqual(w.mean(),1)

    def test_effective_batch(self):
        self.assertEqual(accumulation_steps(16,1,8),2)
        self.assertEqual(accumulation_steps(8,1,8),1)
        with self.assertRaises(ValueError):accumulation_steps(16,3,1)

    def test_stage_budgets(self):
        self.assertEqual(schedule(1)['epochs'],150)
        self.assertEqual(schedule(2)['peak_lr'],1e-4)
        with self.assertRaises(ValueError):schedule(3,1)
        self.assertEqual(schedule(4,12)['effective_batch'],16)
        for stage in [3,4]:
            for step in [20,28]:
                self.assertEqual(schedule(stage,step)['max_optimizer_updates'],1000)
                self.assertEqual(schedule(stage,step)['peak_lr'],1e-6)
        self.assertAlmostEqual(learning_rate(0,1000,100,1e-6),1e-8)
        self.assertAlmostEqual(learning_rate(99,1000,100,1e-6),1e-6)

    def test_cutoff_uses_power_not_amplitude(self):
        self.assertEqual(cutoff_from_energy([0,1,2,3],[100,1,.89,.8],[1,1,1,1]),2)
        self.assertTrue(np.isnan(cutoff_from_energy([0,1],[0,2],[0,1])))
        self.assertEqual(np.flatnonzero(short_wavelength_mask(np.arange(88)))[0],5)

    def test_channels(self):
        channels=channel_config()
        self.assertEqual(len(channels),116)
        names=[c['feature'] for c in channels]
        self.assertEqual(len(set(names)),116)
        self.assertEqual(names[76],'700q')
        self.assertEqual(names[86:88],['850u','850v'])
        self.assertTrue(all(c['weight']>0 for c in channels))

    def test_local_time_features(self):
        t=datetime(2024,9,29,12);lon=np.full((2,2),120.0)
        f=time_features(t,lon)
        np.testing.assert_allclose(f['hour_ratio'],.5+120/360)
        np.testing.assert_allclose(f['hour_ratio_sin']**2+f['hour_ratio_cos']**2,1)

    def test_no_bundled_data(self):
        root=Path(__file__).parents[1]
        disallowed={'.npz','.npy','.pt','.pth','.onnx','.grib','.nc','.png','.pdf','.csv'}
        for folder in ['agu_james','configs','docs','provenance','tests','tools']:
            for p in (root/folder).rglob('*'):
                self.assertNotIn(p.suffix.lower(),disallowed,str(p))

if __name__=='__main__':unittest.main()
