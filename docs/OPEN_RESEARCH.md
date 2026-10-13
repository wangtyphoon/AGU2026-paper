# Open Research statements — author review draft

Prepared for manuscript V1 on 2026-10-13. These statements describe upstream access and the **current local curation status**. Bracketed `PENDING` fields must be resolved before submission. No archive has been created by this task.

## Data Availability Statement

ERA5 reanalysis used for model development and evaluation is available from the Copernicus Climate Change Service Climate Data Store through the pressure-level dataset (https://doi.org/10.24381/cds.bd0915c6) and single-level dataset (https://doi.org/10.24381/cds.adbb2d47). Downloading through the CDS API requires a registered account and acceptance of the applicable dataset terms. Western North Pacific tropical cyclone best tracks used to select training and evaluation periods were obtained from the Joint Typhoon Warning Center best-track archive (https://www.metoc.navy.mil/jtwc/jtwc.html?best-tracks). The precise source-file snapshot, access date and archive citation remain [PENDING: JTWC SNAPSHOT/CITATION]. The Krathon case-study agency fields are from NOAA NCEI's International Best Track Archive for Climate Stewardship, version 4r01 (https://doi.org/10.25921/82ty-9e16), using the USA/JTWC fields for the selected storm [PENDING: ORIGINAL SNAPSHOT ACCESS DATE AND STORM IDENTIFIER]. The 1990–2017 ERA5 six-hourly climatology is obtained from WeatherBench 2 at `gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr` (https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5-climatology).

Processed ERA5 subsets, sample-selection and split manifests, normalization statistics and ocean masks, graph assets, spectral calibration tables, model evaluation outputs, tropical cyclone diagnostics and the numerical data underlying Figures 1–14 have not yet been deposited in a public archive. Their planned repository, version, persistent identifier and redistribution terms are [PENDING: DERIVED-DATA ARCHIVE/DOI/VERSION/LICENSE]. The final deposit must identify the original source snapshots, variable ordering, units, initialization and lead times, processing configuration, missing-value treatment and checksums. The present code repository excludes these data files. Scripts for generating the derived inputs and analysis products are documented in the ordered workflow, but the paper-setting variant has not yet been rerun on the national HPC system.

## Software Availability Statement

The curated Python workflows for data processing, graph construction, four-stage GNN training, spectral calibration, autoregressive validation, physical and tropical cyclone diagnostics, and manuscript plotting are currently maintained in a local code repository, version 0.1.0. The public development URL, immutable release archive and software citation are [PENDING: PUBLIC REPOSITORY URL], [PENDING: SOFTWARE ARCHIVE DOI] and [PENDING: AUTHORS/YEAR/RELEASE VERSION]. The license for author-owned software is [PENDING: AUTHOR-APPROVED LICENSE]; third-party notices and applicable restrictions are recorded in `THIRD_PARTY_NOTICES.md`. No public release or archive availability is asserted by this draft.

Pangu-Weather pretrained 6-hour and 24-hour ONNX models are obtained separately from the official upstream repository (https://github.com/198808xc/Pangu-Weather), with attribution to Bi et al. (2023, https://doi.org/10.1038/s41586-023-06185-3). The upstream trained parameters are provided under CC BY-NC-SA 4.0. The adapted preparation/inference/validation utilities also retain attribution to the unofficial Pangu-Weather-ReadyToGo project (https://github.com/HaxyMoly/Pangu-Weather-ReadyToGo) and its stated terms. GNN checkpoints are not included; their archive, provenance and license are [PENDING: GNN WEIGHT ARCHIVE/DOI/LICENSE]. Local regression checks validate selected calculation contracts and existing aggregate inputs. Full retraining, distributed execution and national HPC reproduction are outside this curation release and remain pending.

## Complete before manuscript submission

| Pending item | Required concrete information |
|---|---|
| Author-owned code | Authors/ORCIDs, approved license, public URL, exact commit/release and archive DOI |
| GNN weights | Three final branch checkpoints and prerequisite/transfer checkpoints needed for reproduction; provenance, checksums and terms |
| Supporting results | Figure numerical inputs, accepted-sample masks/counts, common-case keys, diagnostics, aggregate spectra/coherence, case/monthly fields; archive DOI |
| Processing inputs | Exact source request/snapshot inventory, split/selection manifests, masks/statistics, coordinates, graphs and cutoff tables, or archived deterministic generation manifests plus retrievable inputs |
| Third-party snapshots | JTWC/IBTrACS access dates/files/storm ID; Pangu 6/24-h file checksums and upstream versions |
| Scientific audit | Reconcile MAE spectral-bias normalization and implementation controls; corrected model sources are documented against the actual HPC experiment; label regenerated results separately |
| References | Add both dataset/software citations and relevant method papers to the manuscript bibliography; replace all PENDING fields |

AGU's [official data/software sharing guidance](https://data.agu.org/resources/agu-data-software-sharing-guidance) is the basis for the separate statements and archive placeholders. A development repository alone does not establish that the supporting data and release have been durably archived. See [citation drafts](../CITATION_DRAFT.md) and [data contracts](DATA_SOURCES.md).
