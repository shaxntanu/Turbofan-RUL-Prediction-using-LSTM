# Turbofan Remaining Useful Life Observatory

An end-to-end NASA C-MAPSS benchmark for aircraft-engine remaining useful life (RUL), with an LSTM and Ridge baseline, a reproducible experiment pipeline, research analyses, Matplotlib figures, a Streamlit application, and a React results dashboard.

See [measured results and research analyses](docs/RESULTS.md) for the completed FD001-FD004 comparison, transfer study, uncertainty coverage, and evaluation caveats.

## Project Contents

- `Ref/CMAPSSData/`: included C-MAPSS FD001-FD004 train/test trajectories and test RUL labels.
- `src/`: modular loading, validation, labels, engine-level splitting, preprocessing, models, metrics, uncertainty, and research analyses.
- `outputs/`: generated metrics, predictions, reports, and Matplotlib figures (created by a run; not tracked).
- `models/`: generated model and scaler files (not tracked).
- `web/`: Vite/React benchmark and research dashboard, with a committed static results bundle.
- `streamlit_app.py`: interactive data exploration and prediction interface.
- `docs/turbofan-pipeline.html`: interactive Archify workflow diagram.
- `Ref/*.pdf`: supplied research papers used to scope methodological adaptations.

## Setup

Use Python 3.12 for the TensorFlow training workflow. Create an environment and install the Python dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pipeline discovers the bundled data under `Ref/CMAPSSData/`. Set `CMAPSS_DATA_DIR` to use a different C-MAPSS directory.

## Run Experiments

Run all four subsets with the LSTM and Ridge comparison (early stopping is enabled):

```powershell
python -m src.train_all
```

Choose subsets, cap training time, or run the fast Ridge-only baseline:

```powershell
python -m src.train_all --datasets FD001 FD003 --epochs 30
python -m src.train_all --baseline-only
```

Add the source-to-target and pooled transfer study:

```powershell
python -m src.train_all --baseline-only --include-transfer
```

Each run selects the model by validation MAE, then evaluates it on the official test trajectories and RUL labels. Artifacts are written beneath `outputs/<subset>/` and `models/<subset>/`. For the transfer experiment, see `outputs/transfer/`.

Regenerate the static dashboard bundle from the latest run:

```powershell
python scripts/export_web_data.py
```

## Applications

Start the Streamlit interface:

```powershell
streamlit run streamlit_app.py
```

Start the React dashboard:

```powershell
cd web
npm ci
npm run dev
```

The dashboard presents measured per-subset scores, transfer errors, figures, and methodology notes. After retraining, rerun the export command above to refresh its data bundle.

## Evaluation Protocol

- Compute capped RUL labels per engine, with a 125-cycle cap and 30-cycle input windows.
- Split training engines into training and validation sets before fitting preprocessing.
- Fit sensor filtering, scaling, and model selection without the official test labels.
- Compare window-summary Ridge regression with the stacked many-to-one LSTM.
- Report held-out MAE/RMSE, engine-level robustness, validation-calibrated split-conformal intervals, sensor/PCA screening, fractional-life stages, and cross-subset transfer.

The supplied papers study turbine gas temperature and proprietary operational or fleet records. This repository instead predicts RUL on simulated C-MAPSS data. Fractional-life stages are a proxy for maintenance phases because C-MAPSS does not contain overhaul histories; FD001-FD004 are related simulated domains, not independent real-world fleets. These experiments are methodological adaptations, not numerical reproductions of the papers.

## Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit, React, Vite, Chart.js, and Archify.
