# Derived from gnn/ver_unetG/assign_graph_data_npz_unet.py
# See provenance/sources.json for original hash and modifications.
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

class WeatherGraphNPZDataset(Dataset):

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1):
        super().__init__(None, transform, None)
        self.root_dir = Path(root_dir)
        self.width, self.height = grid_size
        self.graph_structure = torch.load(Path('pre_process/mesh_grid_data_unet7') / 'hetero_graph_normalized.pt', weights_only=False)
        self.prs_feature = self._get_feature_dirs(sub='prs')
        self.sfc_regular_feature = self._get_feature_dirs(sub='sfc/regular')
        self.sfc_average_feature = self._get_feature_dirs(sub='sfc/average')
        self.time_feature = self._get_feature_dirs(sub='time')
        self.all_samples = self._get_all_samples()
        self.fourcast_steps = fourcast_steps
        self.samples = self._get_valid_samples()
        self.stats = self._load_stats()
        self.sfc_stats = self._load_sfc_stats()
        self.lon = np.load('../../era5/lon.npy')
        self.lat = np.load('../../era5/lat.npy')

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
        reference_path = runtime.setting('agu_james.training.models.coarse.assign_graph_data_npz_unet.reference_path', lambda: self.root_dir / 'prs')
        all_samples = [f for f in os.listdir(reference_path) if f.endswith('.npz')]
        return sorted(all_samples)

    def _get_valid_samples(self):

        def consecutive_6h_steps(files, ascending=True):
            files_sorted = sorted(files, reverse=not ascending)
            times = [datetime.strptime(f[:10], '%Y%m%d%H') for f in files_sorted]
            step_list = [0]
            for i in range(1, len(times)):
                delta = times[i] - times[i - 1]
                target = timedelta(hours=6) if ascending else timedelta(hours=-6)
                if delta == target:
                    step_list.append(step_list[-1] + 1)
                else:
                    step_list.append(0)
            return step_list if ascending else step_list[::-1]
        asc_steps = consecutive_6h_steps(self.all_samples, ascending=True)
        desc_steps = consecutive_6h_steps(self.all_samples, ascending=False)
        df = pd.DataFrame({'filename': self.all_samples, 'asc_steps': asc_steps, 'desc_steps': desc_steps})
        keep_mask = (df['asc_steps'] >= self.fourcast_steps) & (df['desc_steps'] >= self.fourcast_steps)
        valid_samples = df.loc[keep_mask, 'filename'].tolist()
        return sorted(valid_samples)

    def _load_stats(self):
        stats = {}
        df = pd.read_csv(self.root_dir.parent / '../npz/prs_variable_stats.csv')
        for _, row in df.iterrows():
            stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std'])}
        return stats

    def _load_sfc_stats(self):
        sfc_stats = {}
        df_reg = pd.read_csv(self.root_dir.parent / '../npz/sfc_regular_variable_stats.csv')
        for _, row in df_reg.iterrows():
            sfc_stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std']), 'max': float(row['max']), 'min': float(row['min'])}
        df_avg = pd.read_csv(self.root_dir.parent / '../npz/sfc_average_variable_stats.csv')
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
        print(f'Loading sample: {sample_name}')
        try:
            current_sample_index_in_all = self.all_samples.index(sample_name)
        except ValueError:
            logger.error(f'Current sample {sample_name} not found in all samples.')
            raise
        next_sample = self.all_samples[current_sample_index_in_all + 1]
        prev_sample = self.all_samples[current_sample_index_in_all - 1]
        remove_prs = ['50cc', '100cc', '1000cc', '50w', '100w', '1000w']
        remove_sfc_regular = ['lsm', 'z']
        remove_sfc_average = ['tisr']
        current_features = self._load_features(sample_name, remove_prs=remove_prs, time=True)
        next_features = self._load_features(next_sample, remove_prs=remove_prs, remove_sfc_regular=remove_sfc_regular, remove_sfc_average=remove_sfc_average, time=False)
        prev_features = self._load_features(prev_sample, remove_prs=remove_prs, remove_sfc_regular=remove_sfc_regular, time=True)
        current_graph = self.graph_structure.clone()
        next_graph = self.graph_structure.clone()
        additional_features = self.graph_structure['grid'].x
        current_combined = torch.cat([torch.tensor(current_features, dtype=torch.float32), torch.tensor(prev_features, dtype=torch.float32), additional_features], dim=1)
        relative_time_step = torch.full((current_combined.shape[0], 1), 0, dtype=torch.float32)
        current_combined = torch.cat([current_combined, relative_time_step], dim=1)
        current_graph['grid'].x = current_combined
        next_graph['grid'].x = torch.tensor(next_features, dtype=torch.float32)
        return (current_graph, next_graph)

def compute_target_std(dataset, batch_size=4, csv_path='target_std.csv'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    n = 0
    total_sum = None
    total_sum_sq = None
    for current_data, next_data in dataloader:
        target = next_data['grid'].x - current_data['grid'].x[:, :116]
        target = target.to(torch.float64)
        if total_sum is None:
            total_sum = torch.zeros(target.shape[1], dtype=torch.float64)
            total_sum_sq = torch.zeros(target.shape[1], dtype=torch.float64)
        total_sum += target.sum(dim=0)
        total_sum_sq += (target ** 2).sum(dim=0)
        n += target.shape[0]
    mean = total_sum / n
    var = total_sum_sq / n - mean ** 2
    std = torch.sqrt(var)
    global_mean = total_sum.sum() / (n * target.shape[1])
    global_var = total_sum_sq.sum() / (n * target.shape[1]) - global_mean ** 2
    global_std = torch.sqrt(global_var)
    df = pd.DataFrame({'feature_idx': range(len(std)), 'std': std.cpu().numpy()})
    df.to_csv(csv_path, index=False)
    print(f'逐特徵 std 已儲存到 {csv_path}')
    print('全域 std:', global_std.item())
    return (std.cpu().numpy(), global_std.item())
import time
