# Turbofan Remaining Useful Life Observatory

[![Vercel dashboard](https://img.shields.io/badge/Vercel-Dashboard-000000?logo=vercel&logoColor=white)](https://turbofan-rul-prediction-using-lstm.vercel.app)
[![Streamlit app](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://turbofan-rul-prediction-lstm.streamlit.app)
[![GitHub Pages Archify pipeline](https://img.shields.io/badge/GitHub%20Pages-Archify%20Pipeline-222222?logo=githubpages&logoColor=white)](https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/turbofan-pipeline.html)

An end-to-end NASA C-MAPSS benchmark for aircraft-engine remaining useful life (RUL), with an LSTM and Ridge baseline, a reproducible experiment pipeline, research analyses, Matplotlib figures, a Streamlit application, and a React results dashboard.

See [measured results and research analyses](docs/RESULTS.md) for the completed FD001-FD004 comparison, transfer study, uncertainty coverage, and evaluation caveats.

## Project Contents

- `Ref/CMAPSSData/`: included C-MAPSS FD001-FD004 train/test trajectories and test RUL labels.
- `src/`: modular loading, validation, labels, engine-level splitting, preprocessing, models, metrics, uncertainty, and research analyses.
- `outputs/`: generated metrics, predictions, reports, and Matplotlib figures (created by a run; not tracked).
- `models/`: generated model and scaler files (not tracked).
- `web/`: Vite/React benchmark and research dashboard, with a committed static results bundle.
- `streamlit_app.py`: interactive data exploration and prediction interface.
- `docs/index.html`: GitHub Pages landing page linking to both interactive research pipelines.
- `docs/turbofan-pipeline.html`: the original baseline Archify diagram, also [published on GitHub Pages](https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/turbofan-pipeline.html).
- `docs/reliability-pipeline.html`: a separate Archify workflow for validation-controlled uncertainty, held-out-subset, short-history, and sensor-stress evaluation, [published on GitHub Pages](https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/reliability-pipeline.html).
- `docs/reliability-pipeline.workflow.json`: the source workflow for the reliability diagram.
- [Captioned RUL method animation](https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/rul-method-animation.html): a ManimGx explainer driven by the real FD001 engine trace. The bundled MP4 also appears in the Streamlit Fleet overview and All-subset benchmark; playback does not require ManimGx. Source is in `animations/rul_method.py` and the rendered video is in `docs/media/`.
- `Ref/*.pdf`: supplied research papers used to scope methodological adaptations.

## Setup

Use Python 3.12 for the complete training environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

`requirements.txt` is the Streamlit deployment manifest. It installs TensorFlow 2.21 on supported Python versions and skips TensorFlow on Python 3.14, so Ridge can run there while LSTM training remains unavailable. The supplied Streamlit logs showed Python 3.14.8, which has no stable TensorFlow 2.21 wheel; use Python 3.12 for the complete training app. On Streamlit Community Cloud, changing an existing app's Python version requires deleting and redeploying it; record its repository, branch, entrypoint, custom subdomain, and any secrets first, then select Python 3.12 in Advanced settings and re-enter secrets privately. See [Streamlit's Python-version instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python). For local experiments, `requirements-training.txt` installs the training dependencies.

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
python -c "from src.config import ProjectConfig; from src.transfer_analysis import run_transfer_analysis; run_transfer_analysis(config=ProjectConfig(dataset='FD001'), train_lstm=False)"
```

Each run selects the model by validation MAE, then evaluates it on the official test trajectories and RUL labels. Artifacts are written beneath `outputs/<subset>/` and `models/<subset>/`. For the transfer experiment, see `outputs/transfer/`.

Run the reliability and robustness study with repeated seeds, an LSTM comparison on seed 42, optional model challengers, and leave-one-subset-out evaluation:

```powershell
python -m src.reliability_analysis --datasets FD001 FD002 FD003 FD004 --seeds 42 43 44 --lstm-seeds 42 --extended-models --with-cnn --epochs 20
```

The study writes terminal-engine and overlapping-window metrics, sensor-stress and risk-coverage tables, progressive-prefix onset results, short-history protocols, RUL-cap sensitivity, and cross-subset holdout results under `outputs/reliability/`. XGBoost is optional; install `xgboost` separately before adding `--with-xgboost`. Protocol definitions and limitations are in [Reliability and robustness](docs/RELIABILITY_AND_ROBUSTNESS.md).

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

Both apps include an all-subset view for combined FD001-FD004 selected-model scores, candidate-model comparisons, and cross-subset results. The Streamlit Reliability tab and web Reliability page read the committed aggregate study snapshot, so those views work in hosted deployments without ignored local `outputs/` files. The dashboard presents measured per-subset scores, transfer errors, figures, and methodology notes. After retraining or rerunning the reliability study, rerun the export command above to refresh its data bundle.

## Optional ManimGx Animation

The static, captioned [methods explainer](https://shaxntanu.github.io/Turbofan-RUL-Prediction-using-LSTM/rul-method-animation.html) shows a measured FD001 sensor trace, its moving 30-cycle input window, a true training-label example, and the validation-calibrated conformal interval method. The model output is explicitly schematic; no sample-specific prediction is claimed. ManimGx is only needed to regenerate this video, not to run either app or the training pipeline. Its pinned optional dependency requires Python 3.13 or newer and stays separate from `requirements.txt` and `requirements-training.txt`.

On Windows with Python 3.14 installed, render into a temporary directory so inspection storyboards do not clutter the repository:

```powershell
py -3.14 -m venv "$env:TEMP\turbofan-animation-env"
& "$env:TEMP\turbofan-animation-env\Scripts\python.exe" -m pip install -r requirements-animation.txt
& "$env:TEMP\turbofan-animation-env\Scripts\manimgx.exe" inspect animations\rul_method.py --output "$env:TEMP\rul-method-storyboard.png"
& "$env:TEMP\turbofan-animation-env\Scripts\manimgx.exe" inspect animations\rul_method.py --time 3.0 --output "$env:TEMP\rul-method-poster.png"
& "$env:TEMP\turbofan-animation-env\Scripts\manimgx.exe" render animations\rul_method.py --output "$env:TEMP\rul-method.mp4" --resolution 1280x720 --fps 30 --preset veryfast --crf 27
Copy-Item "$env:TEMP\rul-method.mp4" docs\media\rul-method.mp4
Copy-Item "$env:TEMP\rul-method-poster.png" docs\media\rul-method-poster.png
```

## Optional C++ Sensor Kernels

`cpp_engine/` contains optional C++17 PCA and K-Means kernels exposed to Python through pybind11. Build tools with a C++17 compiler are required:

```powershell
python -m pip install ./cpp_engine
python -m src.run_cpp_benchmark --datasets FD001 FD002 FD003 FD004 --max-rows 10000
python scripts/export_web_data.py
```

On Windows, stop Python/Streamlit processes that have already imported `turbofan_cpp` before reinstalling it; a loaded extension binary is locked by the process.

The benchmark uses training-engine sensor rows and compares numerical parity and local timings with scikit-learn. PCA timings include fit plus score projection on both sides; K-Means consumes shared sklearn PCA scores. Results depend on the machine/compiler. These exploratory kernels do not affect RUL model selection, prediction, or LSTM inference. The Streamlit Research outputs tab can run the benchmark when the extension is installed.

## Evaluation Protocol

- Compute capped RUL labels per engine, with a 125-cycle cap and 30-cycle input windows.
- Split training engines into training and validation sets before fitting preprocessing.
- Fit sensor filtering, scaling, and model selection without the official test labels.
- Compare window-summary Ridge regression with the stacked many-to-one LSTM.
- Report held-out MAE/RMSE, engine-level robustness, validation-calibrated split-conformal intervals, sensor/PCA screening, fractional-life stages, and cross-subset transfer.

The supplied papers study turbine gas temperature and proprietary operational or fleet records. This repository instead predicts RUL on simulated C-MAPSS data. Fractional-life stages are a proxy for maintenance phases because C-MAPSS does not contain overhaul histories; FD001-FD004 are related simulated domains, not independent real-world fleets. These experiments are methodological adaptations, not numerical reproductions of the papers.

## Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit, React, Vite, Chart.js, Archify, and optional ManimGx.
