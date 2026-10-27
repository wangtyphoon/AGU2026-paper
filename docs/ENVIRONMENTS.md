# Environment and installation

Python >=3.10 is required. Choose dependencies for the task:

| Capability | Install / requirements |
|---|---|
| Basic utilities and scalar plots | `python -m pip install -e .` (NumPy, Pandas, SciPy, Matplotlib, tqdm) |
| All archived figures and GNN inference/training | `python -m pip install -e ".[gnn,maps]"` (PyTorch >=2.3, PyG, scikit-learn, Cartopy) |
| ERA5 conversion, requests and climatology | Add the `data` extra; configure CDS access separately |
| Pangu CPU | Install the `pangu` extra |
| Pangu GPU | Install compatible `onnxruntime-gpu`, ONNX, netCDF4 and cdsapi in place of the CPU runtime |

Select either CPU or GPU ONNX Runtime for Pangu. CUDA/cuDNN compatibility depends
on the selected runtime and device. External Pangu model files are required for
real inference. The GAT/SG model modules do not directly require torch-scatter
or torch-sparse. Cartopy may obtain Natural Earth basemap assets on first use.

```bash
python -m agu_james.workflow --help
python -m agu_james.workflow check
python -m unittest discover -s tests -v
```

`check` lists missing optional dependencies separately from other import errors;
it does not download inputs or execute training. Use an environment with
PyTorch/PyG for the complete numerical tests. Historical-source comparison
tests skip unless `AGU_RDL_REFERENCE` / `AGU_PAPER_REFERENCE` are configured.
This is not a lockfile for the historical HPC experiment.

## Configuration

Pass `--config PATH` before the subcommand. Copy an example to a `.local.json`
file and adapt its paths. Relative `paths` entries and `artifact_root` resolve
beside that JSON file. Command-line checkpoint paths resolve from the working
directory. Run the documented examples from the repository root.

Use `configs/paper.example.json` for the full pipeline, and the configurations
inside the extracted Zenodo archives for archived figures or model inference.
The `training` object contains explicit model/training controls; `overrides`
provides module-specific advanced settings. Store credentials outside the
repository. Generated outputs default to the ignored `artifacts/` directory.
