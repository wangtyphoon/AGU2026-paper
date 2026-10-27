# AGU JAMES research workflows

Software companion to **Physically Consistent Typhoon Intensity Forecasting in
a Graph Neural Network Framework through Spectral Loss Training**.

The release has two parts:

| Location | Contents |
|---|---|
| [GitHub](https://github.com/wangtyphoon/AGU2026-paper) | Python code, installation/configuration, reproduction instructions and numerical tests |
| Zenodo — DOI **PENDING** | `model_weights.zip` (three final models and shared inference assets) and `plotting_data.zip` (Figure 1–14 inputs) |

Compatible software version: **0.1.1**. The two archives are prepared locally;
the Zenodo DOI and matching GitHub release/tag must be filled in after deposit.
The author-owned license remains pending; third-party terms are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Install and obtain the archived inputs

Use Python 3.10 or newer in a suitable environment:

```bash
python -m pip install -e ".[gnn,maps]"
python -m agu_james.workflow --help
```

Download the required Zenodo attachment(s) and extract them **at the repository
root**. Each archive already contains its top-level directory:

```text
AGU2026-paper/
  pyproject.toml
  agu_james/
  configs/
  docs/
  WEIGHTING/       # from model_weights.zip
  plotting_data/   # from plotting_data.zip
```

Git and Python distributions exclude both data directories. The archive READMEs
contain their file descriptions and portable settings. Outputs go to
`artifacts/`, outside the archived inputs.

## Reproduce the figures

After extracting `plotting_data.zip`, run from the repository root:

```bash
python -m agu_james.workflow --config plotting_data/figures.example.json plot metrics --step 28 --style lines
```

The [figure guide](docs/FIGURE_MAP.md) lists all ten plotting commands for
Figures 1–14. Figure 2 is supplied as an editable draw.io schematic and image.
The data archive is sufficient for these plots; model weights are not needed
to redraw existing results.

## Run models or train

For inference, also extract `model_weights.zip`, adapt its
`WEIGHTING/inference.example.json` and supply the required meteorological
inputs. Follow the [execution guide](docs/WORKFLOW.md#10-model-architecture-and-checkpoint-compatibility).
The three checkpoints are inference weights, without full training-resume state.

For data preparation and training, install the additional data dependencies and
follow the ordered guide from the beginning:

```bash
python -m pip install -e ".[gnn,maps,data]"
python -m agu_james.workflow --config configs/paper.example.json --dry-run train --stage 4 --rollout 28 --loss spectral_mse
python -m tools.run_curriculum --config configs/paper.example.json
```

Stage-3 **training-set** spectra must be calibrated before Stage 4. Full ERA5
training/weather inputs and Pangu ONNX parameters are obtained separately.

## Documentation and checks

| Guide | Purpose |
|---|---|
| [Workflow](docs/WORKFLOW.md) | Data → graphs → training → validation → diagnostics; architecture and reproduction limits |
| [Environments](docs/ENVIRONMENTS.md) | Dependencies and portable configuration |
| [Data sources](docs/DATA_SOURCES.md) | Archive contents, external inputs, formats and availability/citations |
| [Metric definitions](docs/METRIC_DEFINITIONS.md) | ACC/Activity, GWB, spectra and TC structure conventions |
| [Figure map](docs/FIGURE_MAP.md) | Figure inputs and commands |

```bash
python -m agu_james.workflow check
python -m unittest discover -s tests -v
```

`check` reports syntax/import problems and unavailable optional dependencies.
Numerical tests requiring external historical sources skip when those sources
are not configured. These checks and direct plotting do not establish full
historical training reproduction. The MAE bias-expression and validation-border
differences are explained in the workflow guide.

## Upload contents

Publish the code, configuration examples, five guides, tests, curriculum tool,
references and third-party notices to GitHub. Deposit only `model_weights.zip`
and `plotting_data.zip` in the associated Zenodo record. Set the record's
related GitHub release and replace the DOI placeholder after publication.
Manuscripts, internal audits, checksums, backup archives, caches and generated
outputs are not part of either upload.
