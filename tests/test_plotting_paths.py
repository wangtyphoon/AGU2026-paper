"""Check relocation, explicit input selection, and legacy path compatibility."""
from pathlib import Path
import tempfile
import unittest

from agu_james import runtime
from agu_james.plotting.runner import input_paths
from agu_james.workflow import load_config


class PlottingPaths(unittest.TestCase):
    def setUp(self):
        self.config = dict(runtime._CONFIG)

    def tearDown(self):
        runtime.configure(self.config)

    def test_relocated_config_and_explicit_root(self):
        scratch = Path(__file__).resolve().parents[1] / 'artifacts/verification'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            moved = Path(directory)
            config = moved / 'figures.example.json'
            config.write_text('{"paths":{"figure_inputs":"inputs"}}', encoding='utf-8')
            load_config(config)
            root, graph, train, ibtracs = input_paths()
            self.assertEqual(root, moved / 'inputs')
            self.assertEqual(graph, root / 'graphs/mesh7/hetero_graph.pt')
            self.assertEqual(train, root / 'calibration/mse_train_coherence_energy_rollout28.npz')
            self.assertEqual(ibtracs, root / 'case_study and monthly simulation/krathon_ibtracs_source.csv')
            runtime._CONFIG['paths'].update(mesh7_raw_graph='old-graph.pt',
                mse_train_spectrum='old-train.npz', krathon_ibtracs='old-ibtracs.csv')
            selected = moved / 'selected'
            paths = input_paths(selected)
            self.assertEqual(paths[0], selected)
            for path in paths[1:]:
                self.assertTrue(path.is_relative_to(selected))

    def test_legacy_defaults_and_explicit_asset_paths(self):
        artifacts = Path('artifacts').resolve()
        runtime.configure({'artifact_root': str(artifacts)})
        root, graph, train, _ = input_paths()
        self.assertEqual(root, artifacts / 'paper_inputs')
        self.assertEqual(graph, artifacts / 'graphs/mesh7/mesh_grid_data_unet7/hetero_graph.pt')
        self.assertEqual(train, artifacts / 'calibration/mse_train_coherence_energy_rollout28.npz')
        runtime.configure({'paths': {'mesh7_raw_graph': 'external.pt'}})
        self.assertEqual(input_paths()[1], Path('external.pt').resolve())


if __name__ == '__main__':
    unittest.main()
