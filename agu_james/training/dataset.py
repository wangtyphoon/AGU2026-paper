# Derived from gnn/ver_autoregressive/train_script/datasetM.py
from agu_james import runtime
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _BaseWeatherGraphNPZDataset(Dataset):
    """
    Shared multi-step dataset implementation. Sub-classes toggle whether
    metadata (e.g. date) should be returned to the caller.
    """

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1, *, return_date: bool=False):
        super().__init__(None, transform, None)
        self.root_dir = Path(root_dir)
        self.width, self.height = grid_size
        self.graph_structure = torch.load(runtime.path('graph', runtime.artifact_root() / 'graphs/mesh7/mesh_grid_data_unet7/hetero_graph_normalized.pt'), weights_only=False)
        self.prs_feature = self._get_feature_dirs(sub='prs')
        self.sfc_regular_feature = self._get_feature_dirs(sub='sfc/regular')
        self.sfc_average_feature = self._get_feature_dirs(sub='sfc/average')
        self.time_feature = self._get_feature_dirs(sub='time')
        self.all_samples = self._get_all_samples()
        self.fourcast_steps = fourcast_steps
        self.samples = self._get_valid_samples()
        self.stats = self._load_stats()
        self.sfc_stats = self._load_sfc_stats()
        self.lon = np.load(runtime.legacy_path('../../era5/lon.npy'))
        self.lat = np.load(runtime.legacy_path('../../era5/lat.npy'))
        self.return_date = return_date
        self.remove_prs = ['50cc', '100cc', '1000cc', '50w', '100w', '1000w']
        self.remove_sfc_regular = ['lsm', 'z']
        self.remove_sfc_average = ['tisr']

    def _get_feature_dirs(self, sub=None):
        feature_dirs = set()
        folder = self.root_dir / sub
        if folder.exists() and any(folder.iterdir()):
            first_npz = next((f for f in folder.iterdir() if f.suffix == '.npz'), None)
            if first_npz:
                data = np.load(first_npz)
                feature_dirs.update(data.keys())
        return sorted(feature_dirs)

    def _get_all_samples(self):
        reference_path = runtime.setting('agu_james.training.dataset.reference_path', lambda: self.root_dir / 'prs')
        all_samples = [f for f in os.listdir(reference_path) if f.endswith('.npz')]
        return sorted(all_samples)

    def _get_valid_samples(self):
        from agu_james.contracts import valid_initializations
        times = [datetime.strptime(f[:10], '%Y%m%d%H') for f in self.all_samples]
        return [t.strftime('%Y%m%d%H.npz') for t in valid_initializations(times, self.fourcast_steps)]

    def _load_stats(self):
        stats = {}
        df = pd.read_csv(runtime.legacy_path('../../npz/prs_variable_stats.csv'))
        for _, row in df.iterrows():
            stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std'])}
        return stats

    def _load_sfc_stats(self):
        sfc_stats = {}
        df_reg = pd.read_csv(runtime.legacy_path('../../npz/sfc_regular_variable_stats.csv'))
        for _, row in df_reg.iterrows():
            sfc_stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std']), 'max': float(row['max']), 'min': float(row['min'])}
        df_avg = pd.read_csv(runtime.legacy_path('../../npz/sfc_average_variable_stats.csv'))
        for _, row in df_avg.iterrows():
            sfc_stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std']), 'max': float(row['max']), 'min': float(row['min'])}
        return sfc_stats

    def __len__(self):
        return len(self.samples)

    def _load_features(self, sample_name, remove_prs=None, remove_sfc_regular=None, remove_sfc_average=None, time=False):
        prs_npz_path = self.root_dir / 'prs' / sample_name
        prs_data = np.load(prs_npz_path)
        prs_feature = [var for var in self.prs_feature if var not in (remove_prs or [])]
        features = []
        static = []
        for var in prs_feature:
            arr = prs_data[var]
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))
        sfc_regular_npz_path = self.root_dir / 'sfc/regular' / sample_name
        sfc_regular_feature = [var for var in self.sfc_regular_feature if var not in (remove_sfc_regular or [])]
        sfc_regular_data = np.load(sfc_regular_npz_path)
        for var in sfc_regular_feature:
            if var == 'z':
                arr = sfc_regular_data[var]
                max = self.sfc_stats[var]['max']
                min = self.sfc_stats[var]['min']
                arr = (arr - min) / (max - min)
                static.append(arr.reshape(-1))
                continue
            elif var == 'lsm':
                arr = sfc_regular_data[var]
                static.append(arr.reshape(-1))
                continue
            arr = sfc_regular_data[var]
            mean = self.sfc_stats[var]['mean']
            std = self.sfc_stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))
        sfc_average_npz_path = self.root_dir / 'sfc/average' / sample_name
        sfc_average_feature = [var for var in self.sfc_average_feature if var not in (remove_sfc_average or [])]
        sfc_average_data = np.load(sfc_average_npz_path)
        for var in sfc_average_feature:
            if var == 'tisr':
                arr = sfc_average_data[var]
                max = self.sfc_stats[var]['max']
                min = self.sfc_stats[var]['min']
                arr = (arr - min) / (max - min)
                static.append(arr.reshape(-1))
                continue
            arr = sfc_average_data[var]
            mean = self.sfc_stats[var]['mean']
            std = self.sfc_stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))
        if time:
            time_feature = np.load(self.root_dir / 'time' / sample_name)
            for var in self.time_feature:
                arr = time_feature[var]
                features.append(arr.reshape(-1))
        features = features + static
        return np.column_stack(features)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        logger.info(f'Loading sample: {sample_name}')
        current_idx = self.all_samples.index(sample_name)
        current_graph = self.graph_structure.clone()
        current_features = self._load_features(sample_name, remove_prs=self.remove_prs, time=True)
        prev_sample = self.all_samples[current_idx - 1]
        prev_features = self._load_features(prev_sample, remove_prs=self.remove_prs, remove_sfc_regular=self.remove_sfc_regular, time=True)
        additional_feats = self.graph_structure['grid'].x
        curr_combined = torch.cat([torch.tensor(current_features, dtype=torch.float32), torch.tensor(prev_features, dtype=torch.float32), additional_feats], dim=1)
        zero_step = torch.zeros((curr_combined.shape[0], 1), dtype=torch.float32)
        current_graph['grid'].x = torch.cat([curr_combined, zero_step], dim=1)
        next_graphs = []
        for step in range(1, self.fourcast_steps + 1):
            future_name = self.all_samples[current_idx + step]
            future_feats = self._load_features(future_name, remove_prs=self.remove_prs, time=True)
            g = self.graph_structure.clone()
            x = torch.tensor(future_feats, dtype=torch.float32)
            step_feat = torch.full((x.shape[0], 1), float(step), dtype=torch.float32)
            g['grid'].x = torch.cat([x, step_feat], dim=1)
            next_graphs.append(g)
        if self.return_date:
            return (current_graph, next_graphs, sample_name[:-4])
        return (current_graph, next_graphs)

class WeatherGraphNPZTrainDataset(_BaseWeatherGraphNPZDataset):
    """Multi-step dataset used for training; does not return timestamps."""

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1):
        super().__init__(root_dir, grid_size, transform, fourcast_steps, return_date=False)

class WeatherGraphNPZValDataset(_BaseWeatherGraphNPZDataset):
    """Validation variant that surfaces the sample date for bookkeeping."""

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1):
        super().__init__(root_dir, grid_size, transform, fourcast_steps, return_date=True)
WeatherGraphNPZDataset = WeatherGraphNPZValDataset

def process_data_step(current_data, next_data, step=0, first116: torch.Tensor=None):
    """
    使用『已覆寫邊界後』的預報 first116 (Bn, 116) 更新 current_data['grid'].x。
    其他欄位（prev_features, tisr, graph_feature, step_feat）沿用現有設計。
    """
    assert first116 is not None, 'process_data_step 需要傳入 first116（已覆寫邊界的預報）'
    prev_step_idx = step - 1
    prev_features = current_data['grid'].x[:, :122]
    current_features = next_data[prev_step_idx]['grid'].x[:, -10:-1]
    tisr = current_data['grid'].x[:, 124]
    graph_feature = current_data['grid'].x[:, -8:-1]
    step_feat = next_data[prev_step_idx]['grid'].x[:, -1].unsqueeze(1)
    current_data['grid'].x = torch.cat([first116, current_features, prev_features, tisr.unsqueeze(1), graph_feature, step_feat], dim=1)
    return current_data

def move_to_device(data, device):
    return data.to(device)
