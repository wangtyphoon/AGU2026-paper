# AGU JAMES research workflows

This is a code-only, curated companion to **Physically Consistent Typhoon Intensity Forecasting in a Graph Neural Network Framework through Spectral Loss Training** (manuscript V1). It contains necessary data preparation, graph construction, four-stage training, validation, Pangu comparison, analysis and figure code.

**Status:** local curation and numerical regression checks; the paper-setting training variant has not been rerun on the national HPC system. No research data, statistics, graphs, model weights or pre-existing figures are bundled. License and archive DOI are pending author confirmation. This repository has not been pushed or published by this task.

## Start here

1. Read [the execution guide](docs/WORKFLOW.md), in order. Stage-3 **training-set** spectral calibration comes before Stage 4.
2. Install the package in an appropriate environment, then adapt `configs/paper.example.json`. All generated artifacts default to the ignored `artifacts/` directory; inputs can remain external.
3. Consult [Figure 1–14 mapping](docs/FIGURE_MAP.md), [input/data contracts](docs/DATA_SOURCES.md), and [recorded changes](docs/MODIFICATIONS.md).
4. Use [the English availability-statement draft](docs/OPEN_RESEARCH.md). Complete its marked archive/license fields before submission.

```powershell
python -m pip install -e ".[gnn,maps,data]"
python -m agu_james.workflow --help
python -m agu_james.workflow --config configs/paper.example.json --dry-run train --stage 4 --rollout 28 --loss spectral_mse
python -m unittest discover -s tests -v
```

Stage-1 graph attention also needs `torch-scatter` and `torch-sparse` matching the installed PyTorch/CUDA combination. Pangu needs ONNX Runtime; select **either** CPU `onnxruntime` or GPU `onnxruntime-gpu`. See [environment notes](docs/ENVIRONMENTS.md). Commands shown as `python` refer to the environment selected by the researcher.

## Organization

| Package | Function |
|---|---|
| `agu_james.data` | JTWC selection, ERA5 requests/conversion, time features, training statistics |
| `agu_james.graphs` | mesh 3–6 and 3–7 construction and graph feature preparation |
| `agu_james.training` | retained stage-specific models, shared identical layers, losses, common training controls |
| `agu_james.validation` | autoregressive forecasts, true cross-spectrum coherence, calibration, Pangu inference |
| `agu_james.analysis` | physical metrics, spectra, TC tracking and structural diagnostics, map exports |
| `agu_james.plotting` | compact manuscript figures and required plotting helpers |
| `provenance` | original source paths/SHA-256 and change categories |

The original research checkouts remain read-only. `tools/curate_sources.py` is a maintainer-only regeneration utility for the original author paths; it is not needed to install or run the curated package elsewhere. All ordinary workflows are independent of those checkout paths.

## Verification and remaining work

Local checks cover syntax/import safety, 116-channel order, calendar splits and complete sequences, nine-cell boundaries, effective batches, optimizer-update schedules, spectral cutoff semantics, retained loss/coherence numerical equivalence, model forward/backward, common-case TC tables, and existing metric/energy inputs. Representative figures were rendered and visually inspected; generated QA data/figures are ignored by Git.

`docs/VALIDATION.md` distinguishes executed checks from full ERA5 downloads, full graph construction, four-stage retraining, distributed/HPC verification and real ONNX inference. In particular, consult the unresolved **MAE spectral-bias normalization and coarse attention** audit before claiming exact agreement with the national HPC run.

The source archive described by [AGU guidance](https://data.agu.org/resources/agu-data-software-sharing-guidance) is separate from a GitHub development repository. No public data/archive availability is asserted here.
