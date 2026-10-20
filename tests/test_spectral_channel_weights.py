"""Regression for original entry-point None -> branch-specific CSV weights."""
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import torch
from agu_james import runtime
from agu_james.data.pipeline import channel_config
from agu_james.training.engine import spectral_channel_weights
from agu_james.training import spectral_mae, spectral_mse


class SpectralChannelWeights(unittest.TestCase):
    def test_all_three_components_use_nonuniform_weights(self):
        torch.manual_seed(7)
        base = torch.randn(2 * 12 * 12, 3)
        pred = torch.randn_like(base)
        target = torch.randn_like(base)
        weights = torch.tensor([1., 2., 4.])
        for module in (spectral_mae, spectral_mse):
            kwargs = dict(H=12, W=12, boundary=1, weights=(1, 1, 1), l_tilde=4, l_max=4)
            _, actual = module.spectral_composite_loss(base, pred, target,
                channel_weights=weights, scale=torch.tensor([1., 2., 3.]),
                cutoffs=torch.tensor([2., 3., 4.]), **kwargs)
            parts = []
            for c in range(3):
                _, metrics = module.spectral_composite_loss(base[:, c:c+1], pred[:, c:c+1], target[:, c:c+1],
                    channel_weights=torch.ones(1), scale=torch.tensor([float(c+1)]),
                    cutoffs=torch.tensor([float(c+2)]), **kwargs)
                parts.append(metrics)
            for key in actual:
                expected = sum(weights[c] * parts[c][key] for c in range(3)) / weights.sum()
                torch.testing.assert_close(actual[key], expected)

    def test_branch_csv_fallback_and_path_override(self):
        scratch = Path(__file__).resolve().parents[1] / 'artifacts/verification'
        scratch.mkdir(parents=True, exist_ok=True)
        previous = runtime._CONFIG
        try:
            with tempfile.TemporaryDirectory(dir=scratch) as folder:
                root = Path(folder)
                stats = root / 'statistics'
                stats.mkdir()
                for kind, config, filename in [
                    ('mae', 'ar', 'next_feature_table_weight_ver3.csv'),
                    ('spectral_mse', 'ar_revised', 'next_feature_table_weight_ver3_revised.csv'),
                ]:
                    frame = pd.DataFrame(channel_config(config))
                    path = stats / filename
                    frame.to_csv(path, index=False)
                    runtime.configure(dict(artifact_root=str(root)))
                    actual = spectral_channel_weights(kind, 'cpu')
                    module = spectral_mae if kind == 'mae' else spectral_mse
                    expected = module.load_channel_weights(path, normalize=True)
                    torch.testing.assert_close(actual, expected)
                    self.assertAlmostEqual(actual.mean().item(), 1, places=6)
                    self.assertGreater(actual.max().item() - actual.min().item(), 1)
                    override = root / 'override.csv'
                    frame['weight'] = frame['weight'] * torch.linspace(1, 2, 116).numpy()
                    frame.to_csv(override, index=False)
                    runtime.configure(dict(artifact_root=str(root), paths={f'statistics/{filename}': str(override)}))
                    torch.testing.assert_close(spectral_channel_weights(kind, 'cpu'), module.load_channel_weights(override, normalize=True))
                    frame.iloc[::-1].to_csv(override, index=False)
                    with self.assertRaisesRegex(ValueError, 'order'):
                        spectral_channel_weights(kind, 'cpu')
                runtime.configure(dict(artifact_root=str(root / 'missing')))
                with self.assertRaises(FileNotFoundError):
                    spectral_channel_weights('spectral_mse', 'cpu')
        finally:
            runtime.configure(previous)


if __name__ == '__main__':
    unittest.main()
