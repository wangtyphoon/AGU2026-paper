# Environment and installation notes

Python ≥3.10 is declared. Local numerical checks used Python 3.12.5 in the existing `atm312` environment. Dependencies are capability groups rather than a claim of a locked HPC environment.

```bash
python -m pip install -e ".[gnn,maps,data]"
python -m agu_james.workflow --help
python -m agu_james.workflow check
```

| Capability | Required dependencies | Local result |
|---|---|---|
| Contracts / aggregate plots | NumPy, Pandas, SciPy, Matplotlib, tqdm | Available; comparisons/plots executed |
| Fine/AR GNN | PyTorch ≥2.3, torch-geometric, scikit-learn | Available; synthetic forward/backward executed |
| Coarse Stage 1 | Above + torch-scatter + torch-sparse | Both extensions absent locally; coarse model import/execution needs installation |
| Maps | Cartopy / projection resources | Cartopy available; map plotting depends on supplied arrays and local Natural Earth cache |
| ERA5 | cdsapi, xarray, cfgrib, eccodes, netCDF4 | Need CDS credentials/terms and real downloads; no full preparation executed |
| Climatology | gcsfs, zarr, xarray | Imports available, official input path verified; download not executed |
| Pangu CPU | ONNX Runtime CPU, NumPy, netCDF4, cdsapi | `pangu` environment imports ONNX inference; real weighted inference not executed |
| Pangu GPU | Compatible ONNX Runtime GPU/CUDA/cuDNN | Not validated in this task |

Install torch-scatter/torch-sparse for the **actual PyTorch and CUDA build**, using the appropriate [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). A generic source build via `[coarse]` may require compiler tooling. Coarse dependencies are required for Stage 1; they are not optional if running that stage.

For CPU Pangu, use `pip install -e ".[pangu]"`. For GPU, install the base package plus netCDF4/cdsapi and compatible `onnxruntime-gpu`, selecting **one** runtime package. Do not install CPU and GPU ONNX Runtime together. `onnx` is declared for upstream interchange support; the retained inference module directly imports `onnxruntime`. ONNX input/field/level ordering is checked by preparation, but real ONNX operators and providers require the external models.

The existing local environments used for verification were `C:\miniconda\envs\atm312\python.exe` and `C:\miniconda\envs\pangu\python.exe`. These paths are recorded evidence, not runtime requirements. No package installation or modification of either environment was performed. An environment/version inventory is generated in `artifacts/verification/` by `tools/verify_release.py`; it is a local snapshot, not the historical HPC dependency lock.

Maintainer source regeneration needs Python3.10/3.11 because newer `ast.unparse` can emit Python3.12-only f-string quoting. Ordinary packaged code was checked against the Python3.10 grammar and runs in the two local3.11/3.12 environments.

Configuration JSON contains shared paths and a `training` object. Relative `paths` entries resolve beside the configuration file. The `overrides` object is a low-level escape hatch keyed by full module plus constant name; public CLI adapters supply ordinary job settings. Do not copy credentials or author-specific data directories into a public release. New outputs are confined to the selected artifact/output paths; choose a fresh artifact root for a different experiment.
