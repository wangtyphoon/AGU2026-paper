# Derived from era5/download_prs.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import cdsapi
import pandas as pd

def main():
    global client, current_dir, dataset, days, file_path, group, month, request, time_series, time_series_path, year, year_month
    current_dir = runtime.setting('agu_james.data.requests.download_prs.current_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.data.requests.download_prs')
    time_series_path = os.path.join(current_dir, 'time_series.csv')
    time_series = pd.read_csv(time_series_path)
    time_series['DATE'] = pd.to_datetime(time_series['DATE'])
    time_series['YEAR_MONTH'] = time_series['DATE'].dt.to_period('M')
    for year_month, group in time_series.groupby('YEAR_MONTH'):
        year = year_month.year
        month = year_month.month
        days = group['DATE'].dt.day.tolist()
        file_path = os.path.join(current_dir, f'prs/{year_month}.grib')
        if os.path.exists(file_path):
            print(f'{file_path} already exists, skipping download.')
            continue
        dataset = 'reanalysis-era5-pressure-levels'
        request = {'product_type': ['reanalysis'], 'variable': ['fraction_of_cloud_cover', 'geopotential', 'potential_vorticity', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'], 'year': [year], 'month': [month], 'day': days, 'time': ['00:00', '06:00', '12:00', '18:00'], 'pressure_level': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'data_format': 'grib', 'download_format': 'unarchived', 'area': [47.75, 102.25, 0, 150]}
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(file_path)
        print(f'{file_path} downloaded successfully.')
