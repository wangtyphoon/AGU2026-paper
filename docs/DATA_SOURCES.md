# Data sources and input contracts

The [GitHub software](https://github.com/wangtyphoon/AGU2026-paper), version
**0.1.1**, uses the weights and figure inputs archived at
[Zenodo, DOI 10.5281/zenodo.23710090](https://doi.org/10.5281/zenodo.23710090):

| Attachment | Contents / extraction |
|---|---|
| [model_weights.zip](https://zenodo.org/records/23710090/files/model_weights.zip?download=1) | Extract at the code root to create `WEIGHTING/`: final MSE, Spectrum-MAE and Spectrum-MSE weights, shared normalized graph, statistics and coordinates |
| [plotting_data.zip](https://zenodo.org/records/23710090/files/plotting_data.zip?download=1) | Extract at the code root to create `plotting_data/`: existing numerical inputs for Figures 1–14, Figure 2 editable schematic/image, plotting configuration and simple file index |

## External and generated inputs

| Asset | Obtain / generate | Format and downstream consumer |
|---|---|---|
| ERA5 pressure fields | [CDS pressure-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview), DOI 10.24381/cds.bd0915c6; `prepare download` | Monthly GRIB → `prepare convert` → pressure-level NPZ fields, training and ERA5 verification |
| ERA5 single-level fields | [CDS single-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview), DOI 10.24381/cds.adbb2d47 | Regular/static and accumulated/mean fields in separate requests; valid-time alignment before six-hour sampling |
| JTWC best tracks | [JTWC best-track portal](https://www.metoc.navy.mil/jtwc/jtwc.html?best-tracks); original historical ATCF snapshot remains author-supplied | BEST records, knots, tenths-of-degree N/S/E/W; deduplicate basin/storm/time, select in-domain dates ±3 days |
| IBTrACS Krathon | [NCEI v04r01 dataset](https://www.ncei.noaa.gov/products/international-best-track-archive), DOI 10.25921/82ty-9e16 | External original CSV or source-selected Krathon CSV; USA/JTWC agency fields for track/intensity case study |
| Climatology | [WeatherBench 2 official data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5-climatology), exact 1990–2017 six-hourly 1440×721 Zarr | `prepare climatology` → `dayDDD_hourHH.npz` cropped to regional coordinates; ACC/Activity |
| Pangu parameters | [Official Pangu-Weather downloads](https://github.com/198808xc/Pangu-Weather#downloading-trained-models) | `pangu_weather_6.onnx`, `pangu_weather_24.onnx`; external `pangu_weights` path |
| Pangu inputs/targets | `prepare pangu-inputs` using global ERA5 | Float32 surface `(4,721,1440)` in msl,u10,v10,t2m order; upper `(5,13,721,1440)` in z,q,t,u,v order, pressure levels 1000→50 hPa; native descending latitude |
| Regional coordinates | `prepare convert` | `lat.npy`/`lon.npy`, `(192,192)`, lat 47.75→0, lon 102.25→150 at 0.25°; dataset, diagnostics, plots |
| Land-sea mask / reference metadata | ERA5 single-level `lsm`, external reference surface/upper NetCDF | Set `lsm`, `surface_nc`, `upper_nc`; mask shape matches regional coordinates; TC diagnostics need real coordinate/level metadata |
| Ocean validity masks | `prepare convert` **before** SST/SWH background filling, training-only maximum missing-cell reference | `data/masks/weight_mask_{sst,swh}.npz`, binary `weight_mask` 192×192 plus reference time/count; physical statistics. Legacy filled NPZ input requires explicit `sst_mask`/`swh_mask` |
| Physical statistics | `prepare statistics` | Per-variable population mean/std/min/max/count CSVs, training years only; SST/SWH masks exclude background. `next_feature_table_weight.csv` adds physical mean/std to fixed configuration |
| Increment scales | `prepare statistics` | `target_mean_std.csv`, ordered feature_idx/variable/mean/std; normalized six-hour increments over complete training sequences, source all-cell convention |
| Graphs | `graph --mesh 6` / `7` | PyG heterogeneous `.pt`, edge/node/encoding/self-loop/normalization assets; normalized mesh7 graph supplied in the inference archive; raw mesh7 geometry in the plotting archive |
| Cutoffs | Training-set `validate --spectrum-output`, then `calibrate` | One CSV per lead, `channel,feature,k_energy_decay_10pct`; train-only metadata, matching counts/order |
| GNN weights | Four training stages and two spectral branches | Training writes full `last_model.pth`; `model_weights.zip` separately supplies the three final inference-only `.pth` files |
| Evaluation fields | `validate` and `pangu` | GNN forecast/target NPZ per lead; Pangu native-global state → regional saved NPY each 6 hours; input arrays remain global |
| Analysis products | `analyze` jobs | Physical metric CSV, energy/coherence NPZ, tracking/structure CSV, axisymmetric profile NPZ; original numerical conventions retained |
| Figure exports | `plot` groups | Generated PNG/vector PDF and figure CSV/manifests; published figure inputs are supplied in `plotting_data.zip` |

ERA5 pressure levels used by the package are `(1000,925,850,700,600,500,400,300,250,200,150,100,50)` hPa. The frozen **116 prognostic channels and weights** are in `agu_james/config/channels_{ar,coarse}.json`; these are configuration, not measured statistics. Stage 4 MAE reads `statistics/next_feature_table_weight_ver3.csv`; Stage 4 MSE reads `statistics/next_feature_table_weight_ver3_revised.csv`, frozen from the original MSE-v2 CSV in `channels_ar_revised.json`. Both are generated by `prepare statistics`, support explicit runtime path overrides, and are normalized to mean one before use. Missing CSVs fail rather than falling back to uniform weights. Channel 76 is 700q, 86 is 850u and 87 is 850v. Static inputs and six time features supplement the prognostic states. Physical units follow ERA5 (e.g. q kg/kg, wind m/s, z m²/s², pressure Pa); source TC output conversions are retained and CSV labels must be used rather than inferred.

Samples are training years 1979–2023 excluding 1980/1990/2000/2010/2020, validation those five years, and test 2024. Every initialization requires t−6 h and each t+6 h lead in the same set. Regional evaluation uses the nine-cell border exclusion (174×174 interior). TC tracking additionally retains its source physical-domain and intensity criteria; common cases are the intersection across all four evaluated models.

SST background is 270 K and SWH background is zero. They are finite model inputs, not observations over land. A newly generated training mask may differ from a historical mask chosen from a fixed missing-count snapshot. For historical checkpoint evaluation, configure external masks and **the matching historical normalization/target scales**; do not silently regenerate its normalization. JSON path keys such as `statistics/prs_variable_stats.csv`, `statistics/sfc_regular_variable_stats.csv`, `statistics/sfc_average_variable_stats.csv`, `statistics/target_mean_std.csv` and `statistics/next_feature_table_weight.csv` override the generated defaults. Statistics require both mask files and fail clearly if they are unavailable.


## Sample populations and units

Monthly spatial fields are the **single 2024102100 initialization**; monthly
power curves use each model's existing aggregate population. Do not relabel
those curves as the spectrum of the map case. The plotting file index identifies
the figure associated with each input; calculation and filtering definitions
are in [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md).

## Availability and citations

Data availability: ERA5 pressure-level and single-level data are obtained from
the Copernicus Climate Data Store (DOIs 10.24381/cds.bd0915c6 and
10.24381/cds.adbb2d47); JTWC supplies the best-track records; the Krathon USA/JTWC
subset is from IBTrACS v04r01 (DOI 10.25921/82ty-9e16). The 1990–2017 six-hourly
climatology is obtained from WeatherBench 2. Model weights and figure inputs
are available as the two attachments above at
[Zenodo, DOI 10.5281/zenodo.23710090](https://doi.org/10.5281/zenodo.23710090).
The record was published on 27 October 2026 and lists CC BY 4.0. Full ERA5
meteorological training inputs and Pangu ONNX weights are obtained separately.

Archive citation: WANG, TZUYU (2026). *Pretrained Model Weights for
Spectral-Loss Training of a Regional Graph Neural Weather Forecast Model*.
Zenodo. https://doi.org/10.5281/zenodo.23710090.

Software availability: the companion code is maintained at the GitHub URL
above; a separate software archive is identified by
[DOI 10.5281/zenodo.23682380](https://doi.org/10.5281/zenodo.23682380).
The weights/data record does not specify a matching software release/tag;
the commands in this repository document package version 0.1.1.

Dataset access links and DOIs are listed above. Method references:

- Hersbach et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049. [DOI 10.1002/qj.3803](https://doi.org/10.1002/qj.3803).
- Gahtan et al. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01*. NOAA National Centers for Environmental Information. [DOI 10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16).
- Knapp et al. (2010). *The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data*. Bulletin of the American Meteorological Society, 91, 363–376. [DOI 10.1175/2009BAMS2755.1](https://doi.org/10.1175/2009BAMS2755.1).
- Bi et al. (2023). *Accurate medium-range global weather forecasting with 3D neural networks*. Nature, 619, 533–538. [DOI 10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3).

Pangu workflow utilities also retain attribution to
[HaxyMoly/Pangu-Weather-ReadyToGo](https://github.com/HaxyMoly/Pangu-Weather-ReadyToGo);
see [third-party notices](../THIRD_PARTY_NOTICES.md) for licensing and attribution.
