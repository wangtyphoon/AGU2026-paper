# Derived from era5/download_sfc_regular.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import cdsapi
import pandas as pd

def main():
    global client, current_dir, dataset, days, file_path, group, month, request, time_series, time_series_path, year, year_month
    current_dir = runtime.setting('agu_james.data.requests.download_sfc_regular.current_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.data.requests.download_sfc_regular')
    time_series_path = os.path.join(current_dir, 'time_series.csv')
    time_series = pd.read_csv(time_series_path)
    time_series['DATE'] = pd.to_datetime(time_series['DATE'])
    time_series['YEAR_MONTH'] = time_series['DATE'].dt.to_period('M')
    for year_month, group in time_series.groupby('YEAR_MONTH'):
        year = year_month.year
        month = year_month.month
        days = group['DATE'].dt.day.tolist()
        file_path = os.path.join(current_dir, f'sfc/{year_month}_regular.grib')
        if os.path.exists(file_path):
            print(f'{file_path} already exists, skipping download.')
            continue
        dataset = 'reanalysis-era5-single-levels'
        request = {'product_type': ['reanalysis'], 'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'mean_sea_level_pressure', 'sea_surface_temperature', 'forecast_logarithm_of_surface_roughness_for_heat', 'significant_height_of_combined_wind_waves_and_swell', 'boundary_layer_height', 'geopotential', 'land_sea_mask'], 'year': [year], 'month': [month], 'day': days, 'time': ['00:00', '06:00', '12:00', '18:00'], 'data_format': 'grib', 'download_format': 'unarchived', 'area': [47.75, 102.25, 0, 150]}
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(file_path)
        print(f'{file_path} downloaded successfully.')
