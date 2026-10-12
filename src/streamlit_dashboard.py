from __future__ import annotations

import io
import json
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, ProjectConfig
from src.cpp_bridge import module_status
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.normalization import Normalizer
from src.pipeline import run_pipeline
from src.run_cpp_benchmark import OUTPUT_PATH as CPP_BENCHMARK_PATH, run_benchmark
from src.sequence_generator import SequenceGenerator
from src.target_construction import TargetConstructor

RAW_COLUMNS = DataLoader(ProjectConfig()).columns
RESULTS_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "web" / "public" / "results" / "benchmark.json"
RELIABILITY_TABLE_KEYS = {
    "repeated_seed_summary.csv": "repeated_seed_summary",
    "sensor_stress_summary.csv": "sensor_stress_summary",
    "risk_coverage.csv": "risk_coverage",
    "history_protocol_metrics.csv": "history_protocol_metrics",
    "online_onset_metrics.csv": "online_onset_metrics",
    "rul_cap_sensitivity.csv": "rul_cap_sensitivity",
    "leave_one_subset_out_metrics.csv": "leave_one_subset_out_metrics",
}
SUBSET_CONDITIONS = {
    "FD001": "1 operating condition / 1 fault mode",
    "FD002": "6 operating conditions / 1 fault mode",
    "FD003": "1 operating condition / 2 fault modes",
    "FD004": "6 operating conditions / 2 fault modes",
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_dashboard_bundle() -> dict:
    return _read_json(RESULTS_BUNDLE_PATH) or {}


def _published_result(dataset: str, bundle: dict) -> dict | None:
    published = bundle.get("datasets", {}).get(dataset)
    if not published:
        return None
    figures = {
        name: str(RESULTS_BUNDLE_PATH.parent / Path(url).name)
        for name, url in published.get("figures", {}).items()
    }
    try:
        tensorflow_version = package_version("tensorflow")
    except PackageNotFoundError:
        tensorflow_version = "not-installed"
    return {
        "run_id": published.get("run_id", "published benchmark"),
        "selected_model": published.get("selected_model", "ridge"),
        "test_metrics": published.get("model_metrics", {}),
        "validation_metrics": published.get("validation_metrics", {}),
        "uncertainty": published.get("uncertainty", {}),
        "package_versions": {"tensorflow": tensorflow_version},
        "dataset_summary": published.get("dataset_summary", {}),
        "sensor_screening": published.get("sensor_screening", {}),
        "engine_robustness": published.get("engine_robustness", {}),
        "fractional_life_analysis": {
            "metrics": published.get("life_stage_metrics", []),
            "limitation": (
                "Fractional engine life is a proxy for maintenance phases; C-MAPSS does not contain overhaul records."
            ),
        },
        "figures": figures,
    }


def _combined_subset_rows(bundle: dict) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_NAMES:
        result = bundle.get("datasets", {}).get(dataset)
        if not result:
            continue
        metrics = result.get("test_metrics", {})
        summary = result.get("dataset_summary", {})
        rows.append(
            {
                "Subset": dataset,
                "Conditions / faults": SUBSET_CONDITIONS[dataset],
                "Selected model": result.get("selected_model", "--"),
                "Test engines": summary.get("test_engines"),
                "Test windows": metrics.get("n_samples"),
                "MAE (cycles)": metrics.get("mae"),
                "RMSE (cycles)": metrics.get("rmse"),
                "Interval coverage": result.get("uncertainty", {}).get("picp"),
            }
        )
    return pd.DataFrame(rows)


def _combined_model_rows(bundle: dict) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_NAMES:
        result = bundle.get("datasets", {}).get(dataset)
        if not result:
            continue
        for model, metrics in result.get("model_metrics", {}).items():
            validation = result.get("validation_metrics", {}).get(model, {})
            rows.append(
                {
                    "Subset": dataset,
                    "Model": model,
                    "Selected on validation": model == result.get("selected_model"),
                    "Validation MAE (cycles)": validation.get("mae"),
                    "Test MAE (cycles)": metrics.get("mae"),
                    "Test RMSE (cycles)": metrics.get("rmse"),
                }
            )
    return pd.DataFrame(rows)


def _reliability_snapshot(metrics_dir: Path, bundle: dict) -> tuple[dict | None, dict]:
    reliability = bundle.get("reliability", {})
    study = _read_json(metrics_dir / "reliability_study.json") or reliability.get("study")
    return study, reliability.get("tables", {})


def _render_all_subsets() -> None:
    bundle = _read_dashboard_bundle()
    rows = _combined_subset_rows(bundle)
    if rows.empty:
        st.title("All-subset benchmark")
        st.info("The committed benchmark bundle does not contain all four subset results yet.")
        return

    st.title("All-subset benchmark")
    st.caption(
        "FD001-FD004 results together. Each subset selects its model on validation data; "
        "the reported aggregate is an unweighted macro-average across subsets."
    )
    summary = st.columns(3)
    summary[0].metric("Macro-average MAE", f"{rows['MAE (cycles)'].mean():.2f} cycles")
    summary[1].metric("Macro-average RMSE", f"{rows['RMSE (cycles)'].mean():.2f} cycles")
    summary[2].metric("Test engines", f"{int(rows['Test engines'].sum()):,} across {len(rows)} subsets")

    st.subheader("Selected model by subset")
    fig, axis = plt.subplots(figsize=(10, 4.2))
    positions = np.arange(len(rows))
    axis.bar(positions - 0.18, rows["MAE (cycles)"], width=0.36, color="#17877f", label="MAE")
    axis.bar(positions + 0.18, rows["RMSE (cycles)"], width=0.36, color="#e68a4b", label="RMSE")
    axis.set_xticks(positions, rows["Subset"])
    axis.set_ylabel("Error (cycles)")
    axis.set_title("Held-out test error across all C-MAPSS subsets")
    axis.set_axisbelow(True)
    axis.yaxis.grid(color="#e8ece9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(rows, hide_index=True, width="stretch")
    st.download_button(
        "Download all-subset scores",
        rows.to_csv(index=False).encode("utf-8"),
        file_name="cmapss_all_subset_scores.csv",
        mime="text/csv",
    )

    model_rows = _combined_model_rows(bundle)
    if not model_rows.empty:
        st.subheader("All candidate models")
        st.dataframe(model_rows, hide_index=True, width="stretch")
    transfer_rows = bundle.get("transfer", {}).get("rows", [])
    if transfer_rows:
        st.subheader("Cross-subset transfer results")
        st.dataframe(pd.DataFrame(transfer_rows), hide_index=True, width="stretch")
    st.caption(
        "Macro-averages weight each subset equally; they are not pooled-window metrics. "
        "These four datasets are related simulator conditions, not independent fleets."
    )


@st.cache_data(show_spinner=False)
def _load_training_data(dataset: str) -> pd.DataFrame:
    config = ProjectConfig(dataset=dataset)
    frame = DataLoader(config).load_train_data()
    return TargetConstructor.compute_training_rul(frame, config.rul_clip_value)


def _read_upload(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    first_line = content.decode("utf-8-sig", errors="replace").splitlines()[0]
    extension = Path(uploaded_file.name).suffix.lower()
    has_header = "unit" in first_line.lower() or "sensor_" in first_line.lower()
    if extension == ".txt" or ("," not in first_line and not has_header):
        frame = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None, names=RAW_COLUMNS)
    else:
        frame = pd.read_csv(io.BytesIO(content), header=0 if has_header else None)
        if not has_header and frame.shape[1] == len(RAW_COLUMNS):
            frame.columns = RAW_COLUMNS

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "unit" not in frame.columns:
        frame.insert(0, "unit", 1)
    if "cycle" not in frame.columns:
        frame.insert(1, "cycle", frame.groupby("unit").cumcount() + 1)
    for column in (
        "unit",
        "cycle",
        *[f"setting_{index}" for index in range(1, 4)],
        *[f"sensor_{index}" for index in range(1, 22)],
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def _run_experiment(dataset: str, *, train_lstm: bool) -> dict:
    return run_pipeline(dataset, ProjectConfig(dataset=dataset), train_lstm=train_lstm)


def _render_signals(dataset: str) -> None:
    try:
        train_frame = _load_training_data(dataset)
        sensors = [column for column in train_frame if column.startswith("sensor_")]
        engine_col, sensor_col = st.columns([1, 2])
        engine_ids = sorted(train_frame["unit"].unique().tolist())
        engine = engine_col.selectbox("Training engine", engine_ids)
        ranked_sensors = train_frame[sensors].var().sort_values(ascending=False).index.tolist()
        selected_sensors = sensor_col.multiselect("Sensor channels", sensors, default=ranked_sensors[:4])
        engine_frame = train_frame.loc[train_frame["unit"] == engine].sort_values("cycle")

        if selected_sensors:
            fig, axes = plt.subplots(
                len(selected_sensors),
                1,
                figsize=(10, max(2.4, 2.0 * len(selected_sensors))),
                sharex=True,
            )
            axes = np.atleast_1d(axes)
            for axis, sensor in zip(axes, selected_sensors, strict=True):
                axis.plot(engine_frame["cycle"], engine_frame[sensor], color="#138a8a", linewidth=1.5)
                axis.set_ylabel(sensor)
                axis.spines[["top", "right"]].set_visible(False)
            axes[-1].set_xlabel("Cycle")
            fig.suptitle(f"{dataset} · engine {engine}", y=1.01)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        left, right = st.columns(2)
        with left:
            fig, axis = plt.subplots(figsize=(7, 4))
            sns.histplot(train_frame["RUL"], bins=40, color="#4169a1", ax=axis)
            axis.set(title="Training target distribution", xlabel="RUL (cycles)", ylabel="Cycle count")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            fig, axis = plt.subplots(figsize=(7, 4))
            sns.heatmap(train_frame[sensors].corr(), cmap="vlag", center=0, ax=axis, cbar=False)
            axis.set(title="Sensor correlation", xlabel="Sensor", ylabel="Sensor")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        st.dataframe(engine_frame, hide_index=True, width="stretch")
    except Exception as error:
        st.error(f"Could not load {dataset} training signals: {error}")


def _render_forecast(dataset: str, config: ProjectConfig, model_dir: Path, result: dict | None) -> None:
    st.subheader(f"Upload · {dataset}")
    uploaded_file = st.file_uploader("Engine sensor history", type=["csv", "txt"], key="forecast_upload")
    required_artifacts = ("ridge_baseline.joblib", "scaler.joblib", "feature_columns.json")
    model_ready = all((model_dir / name).is_file() for name in required_artifacts)
    if not model_ready:
        st.warning("No Ridge artifact exists for this subset. Run its baseline experiment from the sidebar first.")
        return
    if uploaded_file is None:
        return

    try:
        uploaded = _read_upload(uploaded_file)
        required_columns = DataValidator.REQUIRED_COLUMNS
        missing_columns = [column for column in required_columns if column not in uploaded]
        if missing_columns:
            st.error(f"Missing NASA predictor columns: {', '.join(missing_columns)}")
            return
        valid, issues = DataValidator.validate(uploaded[required_columns], dataset)
        if not valid:
            st.error(" · ".join(issues))
            return

        st.caption(f"{len(uploaded):,} rows · {uploaded['unit'].nunique()} engines · {uploaded_file.name}")
        if not st.button("Generate RUL forecast", type="primary"):
            return

        feature_data = _read_json(model_dir / "feature_columns.json")
        feature_columns = feature_data["features"]
        DataCleaner.align_features(uploaded, feature_columns)
        normalizer = Normalizer(config).load(model_dir / "scaler.joblib")
        scaled = normalizer.transform(uploaded, feature_columns)
        sequences, targets, units, cycles = SequenceGenerator(config).generate_with_metadata(
            scaled,
            feature_columns,
        )
        if not len(sequences):
            st.warning(f"Each engine needs at least {config.window_size} cycles.")
            return

        model = WindowRidgeBaseline.load(model_dir / "ridge_baseline.joblib")
        predictions = model.predict(sequences)
        ridge_uncertainty = (result or {}).get("uncertainty_by_model", {}).get("ridge", {})
        if not ridge_uncertainty and (result or {}).get("selected_model") == "ridge":
            ridge_uncertainty = (result or {}).get("uncertainty", {})
        radius = float(ridge_uncertainty.get("interval_radius_cycles", 0.0))
        lower = np.maximum(0, predictions - radius)
        upper = np.minimum(config.rul_clip_value, predictions + radius)
        prediction_frame = pd.DataFrame(
            {
                "unit": units,
                "cycle": cycles,
                "rul_predicted": predictions,
                "interval_lower": lower,
                "interval_upper": np.maximum(lower, upper),
            }
        )
        if "RUL" in uploaded:
            actual_values = uploaded.sort_values(["unit", "cycle"])["RUL"].to_numpy()
            labeled = scaled.assign(RUL=actual_values)
            _, observed, _, _ = SequenceGenerator(config).generate_with_metadata(labeled, feature_columns)
            if len(observed) == len(prediction_frame):
                prediction_frame["rul_observed"] = observed
                st.write(Evaluator.compute_metrics(observed, predictions))

        latest = prediction_frame.groupby("unit", sort=False).tail(1)
        st.dataframe(latest, hide_index=True, width="stretch")
        selected_engine = st.selectbox("Trajectory engine", latest["unit"].tolist())
        trajectory = prediction_frame[prediction_frame["unit"] == selected_engine]
        fig, axis = plt.subplots(figsize=(9, 4.2))
        axis.fill_between(
            trajectory["cycle"],
            trajectory["interval_lower"],
            trajectory["interval_upper"],
            color="#138a8a",
            alpha=0.18,
            label="Validation-calibrated interval",
        )
        axis.plot(trajectory["cycle"], trajectory["rul_predicted"], color="#4169a1", label="Predicted RUL")
        if "rul_observed" in trajectory:
            axis.plot(trajectory["cycle"], trajectory["rul_observed"], color="#dd6b55", label="Observed RUL")
        axis.set(xlabel="Cycle", ylabel="RUL (cycles)", title=f"Engine {selected_engine}")
        axis.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(
            "Intervals use split-conformal residuals from held-out validation engines; they are not per-prediction "
            "probabilities and may under-cover after domain shift."
        )
        st.download_button(
            "Download forecast CSV",
            prediction_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset.lower()}_rul_forecast.csv",
            mime="text/csv",
        )
        with st.expander("Uploaded data"):
            st.dataframe(uploaded, hide_index=True, width="stretch")
    except Exception as error:
        st.error(f"Could not process the uploaded history: {error}")


def _render_reliability(dataset: str, config: ProjectConfig) -> None:
    metrics_dir = config.output_dir / "reliability" / "metrics"
    study, published_tables = _reliability_snapshot(metrics_dir, _read_dashboard_bundle())
    if not study:
        st.info("No exported reliability study is available in this deployment.")
        st.code(
            "python -m src.reliability_analysis --datasets FD001 FD002 FD003 FD004 "
            "--seeds 42 43 44 --lstm-seeds 42 --extended-models --with-cnn --epochs 20",
            language="bash",
        )
        return
    if dataset not in study.get("datasets", []):
        st.warning(f"{dataset} is not included in reliability run {study.get('run_id', '--')}.")
        st.caption(f"Included subsets: {', '.join(study.get('datasets', [])) or 'none'}")
        return

    st.caption(
        f"Run {study.get('run_id', '--')} | seeds {', '.join(map(str, study.get('repeated_seeds', [])))} | "
        f"window {study.get('window_size', '--')} | RUL cap {study.get('rul_cap', '--')} cycles"
    )
    if study.get("domain_note"):
        st.info(study["domain_note"])

    def read_table(name: str) -> pd.DataFrame:
        path = metrics_dir / name
        if path.is_file() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                pass
        key = RELIABILITY_TABLE_KEYS.get(name)
        return pd.DataFrame(published_tables.get(key, [])) if key else pd.DataFrame()

    repeated = read_table("repeated_seed_summary.csv")
    if not repeated.empty:
        repeated = repeated.loc[repeated["dataset"] == dataset]
    if not repeated.empty:
        st.subheader("Repeated-seed terminal performance")
        columns = [
            name for name in (
                "model", "seed_count", "terminal_mae_mean", "terminal_mae_sd",
                "terminal_rmse_mean", "terminal_rmse_sd", "phm08_sum_mean",
                "interval_coverage_mean", "interval_coverage_sd",
            ) if name in repeated.columns
        ]
        st.dataframe(repeated[columns], hide_index=True, width="stretch")
        fig, axis = plt.subplots(figsize=(10, max(3.2, 0.35 * len(repeated))))
        axis.barh(repeated["model"], repeated["terminal_mae_mean"], color="#17877f")
        axis.set(xlabel="Terminal-engine MAE (cycles)", title=f"{dataset} mean across recorded seeds")
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("The LSTM challenger uses one configured seed; a single run does not estimate between-seed variance.")

    stress = read_table("sensor_stress_summary.csv")
    if not stress.empty:
        stress = stress.loc[stress["dataset"] == dataset]
    if not stress.empty:
        st.subheader("Controlled sensor stress")
        models = sorted(stress["model"].dropna().unique().tolist())
        model = st.selectbox("Stress model", models, key=f"reliability-stress-{dataset}")
        selected = stress.loc[(stress["model"] == model) & (stress["corruption_type"] != "clean")].copy()
        if not selected.empty:
            selected["condition"] = selected["corruption_type"] + " / " + selected["severity"]
            colors = [
                "#c85b54" if severity == "severe" else "#e68a4b" if severity == "moderate" else "#17877f"
                for severity in selected["severity"]
            ]
            fig, axis = plt.subplots(figsize=(10, max(3.5, 0.29 * len(selected))))
            axis.barh(selected["condition"], selected["terminal_engine_mae_mean"], color=colors)
            axis.set(xlabel="Terminal-engine MAE (cycles)", title=f"{dataset} / {model}")
            axis.invert_yaxis()
            axis.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.dataframe(selected, hide_index=True, width="stretch")
        st.caption("Corruptions are controlled synthetic stressors, not an empirical sensor-failure distribution.")

    risk = read_table("risk_coverage.csv")
    if not risk.empty:
        if "record_type" in risk.columns:
            risk = risk.loc[risk["record_type"] == "risk_coverage"]
        if "dataset" in risk.columns:
            risk = risk.loc[risk["dataset"] == dataset]
    if not risk.empty:
        st.subheader("Selective risk and coverage")
        left, right = st.columns(2)
        models = sorted(risk["model"].dropna().unique().tolist())
        model = left.selectbox("Risk-coverage model", models, key=f"reliability-risk-model-{dataset}")
        scenarios = sorted(risk.loc[risk["model"] == model, "scenario"].dropna().unique().tolist())
        scenario = right.selectbox("Sensor scenario", scenarios, key=f"reliability-risk-scenario-{dataset}")
        curve = risk.loc[(risk["model"] == model) & (risk["scenario"] == scenario)].sort_values("coverage_fraction")
        fig, axis = plt.subplots(figsize=(9, 3.8))
        axis.plot(curve["coverage_fraction"], curve["retained_mae"], marker="o", color="#17877f")
        axis.set(xlabel="Retained prediction fraction", ylabel="MAE (cycles)", title=f"{dataset} / {model} / {scenario}")
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    for title, filename, subset_column in (
        ("Online onset and progressive-prefix results", "online_onset_metrics.csv", "dataset"),
        ("Short-history protocols", "history_protocol_metrics.csv", "dataset"),
        ("RUL-cap sensitivity", "rul_cap_sensitivity.csv", "dataset"),
        ("Leave-one-subset-out transfer", "leave_one_subset_out_metrics.csv", "target_dataset"),
    ):
        frame = read_table(filename)
        if not frame.empty and subset_column in frame:
            frame = frame.loc[frame[subset_column] == dataset]
        if not frame.empty:
            st.subheader(title)
            st.dataframe(frame, hide_index=True, width="stretch")

    if study.get("limitations"):
        with st.expander("Protocol limits"):
            for item in study["limitations"]:
                st.write(f"- {item}")


def main() -> None:
    st.set_page_config(page_title="Turbofan RUL Lab", page_icon="✈", layout="wide")
    selection = st.sidebar.selectbox("C-MAPSS subset", ("All subsets", *DATASET_NAMES), key="dataset")
    if selection == "All subsets":
        _render_all_subsets()
        return

    dataset = selection
    config = ProjectConfig(dataset=dataset)
    results_dir = config.output_dir / dataset
    model_dir = config.model_dir / dataset
    result = _read_json(results_dir / "metrics" / "results.json") or _published_result(
        dataset, _read_dashboard_bundle()
    )
    tensorflow_available = find_spec("tensorflow") is not None

    st.title("Turbofan RUL Lab")
    st.caption("NASA C-MAPSS run-to-failure trajectories · Remaining useful life")
    with st.sidebar:
        st.divider()
        st.caption(f"Window: {config.window_size} cycles · RUL cap: {config.rul_clip_value} cycles")
        if st.button("Run Ridge baseline", width="stretch", type="primary"):
            try:
                with st.spinner(f"Running {dataset} through the research pipeline…"):
                    _run_experiment(dataset, train_lstm=False)
                st.success(f"{dataset} Ridge baseline complete")
                st.rerun()
            except Exception as error:
                st.error(f"Experiment failed: {error}")
        if st.button("Train LSTM + compare", width="stretch", disabled=not tensorflow_available):
            try:
                with st.spinner(f"Training and comparing models for {dataset}…"):
                    _run_experiment(dataset, train_lstm=True)
                st.success(f"{dataset} LSTM comparison complete")
                st.rerun()
            except Exception as error:
                st.error(f"Experiment failed: {error}")
        if not tensorflow_available:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            st.caption(
                f"LSTM training is disabled because TensorFlow is not installed for Python {version}. "
                "On Streamlit Cloud, set the app Python version to 3.12 and redeploy; Ridge runs without TensorFlow. "
                "Locally, install requirements-training.txt."
            )

    overview_tab, signals_tab, outputs_tab, reliability_tab, forecast_tab = st.tabs(
        ["Fleet overview", "Signals", "Research outputs", "Reliability", "Forecast"]
    )
    with overview_tab:
        if result:
            summary = result["dataset_summary"]
            metrics = result["test_metrics"].get(result["selected_model"], {})
            metric_cols = st.columns(4)
            metric_cols[0].metric("Test MAE (cycles)", f"{metrics['mae']:.2f}")
            metric_cols[1].metric("Test RMSE (cycles)", f"{metrics['rmse']:.2f}")
            metric_cols[2].metric("Test engines", summary["test_engines"])
            metric_cols[3].metric("Selected model", result["selected_model"].upper())
            tf_version = result["package_versions"].get("tensorflow", "not-installed")
            tf_note = "TensorFlow not installed" if tf_version == "not-installed" else f"TensorFlow {tf_version}"
            st.caption(
                f"Run {result['run_id']} · selected on validation MAE · {tf_note} · "
                f"{summary['retained_sensors']} sensor channels retained"
            )
        else:
            st.info(f"No saved experiment is available for {dataset} yet.")

        fleet_rows = []
        for subset in DATASET_NAMES:
            subset_result = _read_json(config.output_dir / subset / "metrics" / "results.json")
            if subset_result:
                selected = subset_result["selected_model"]
                subset_metrics = subset_result["test_metrics"][selected]
                fleet_rows.append(
                    {
                        "Subset": subset,
                        "Model": selected,
                        "MAE (cycles)": subset_metrics["mae"],
                        "RMSE (cycles)": subset_metrics["rmse"],
                        "Engines": subset_result["dataset_summary"]["test_engines"],
                        "Run": subset_result["run_id"],
                    }
                )
        if fleet_rows:
            st.subheader("Saved subset benchmarks")
            st.dataframe(pd.DataFrame(fleet_rows), hide_index=True, width="stretch")
        else:
            st.subheader("C-MAPSS benchmark")
            st.write("FD001–FD004 vary operating conditions and fault modes. Each subset has separate train, test, and RUL-label files.")

        if result:
            left, right = st.columns(2)
            uncertainty = result["uncertainty"]
            left.metric("Nominal interval coverage", f"{uncertainty['nominal_coverage']:.0%}")
            right.metric("Observed test coverage", f"{uncertainty['picp']:.1%}")
            st.caption(
                f"Split-conformal radius: {uncertainty['interval_radius_cycles']:.1f} cycles, calibrated on held-out validation engines. "
                "Coverage can change under operating-condition or fleet shift."
            )

    with signals_tab:
        _render_signals(dataset)

    with outputs_tab:
        st.subheader("Native PCA / K-Means kernels")
        native_status = module_status()
        if native_status["available"]:
            st.caption(
                "Optional C++17 sensor-analysis kernels. Timings are local microbenchmarks, "
                "not end-to-end RUL or LSTM speedups."
            )
            if st.button("Run native parity benchmark", key="run-cpp-benchmark", width="stretch"):
                try:
                    with st.spinner(f"Benchmarking native kernels on {dataset} training sensors…"):
                        run_benchmark((dataset,), max_rows=10000)
                    st.rerun()
                except Exception as error:
                    st.error(f"Native benchmark failed: {error}")

            native_result = _read_json(CPP_BENCHMARK_PATH)
            native_row = next(
                (
                    item for item in (native_result or {}).get("datasets", [])
                    if item.get("dataset") == dataset and "pca" in item and "kmeans" in item
                ),
                None,
            )
            if native_row:
                pca = native_row["pca"]
                kmeans = native_row["kmeans"]
                st.dataframe(
                    pd.DataFrame(
                        [{
                            "Subset": dataset,
                            "Rows": native_row["n_samples"],
                            "Sensors": native_row["n_features"],
                            "PCA sklearn (ms)": round(pca["python_ms"], 3),
                            "PCA C++ (ms)": round(pca["cpp_ms"], 3),
                            "PCA subspace angle (deg)": round(
                                pca["maximum_principal_angle_degrees"], 8
                            ),
                            "K-Means sklearn (ms)": round(kmeans["python_ms"], 3),
                            "K-Means C++ (ms)": round(kmeans["cpp_ms"], 3),
                            "K-Means speedup": round(kmeans["speedup_x"], 2),
                            "ARI": round(kmeans["adjusted_rand_index"], 4),
                        }]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Both PCA implementations use the same standardized training rows. "
                    "K-Means consumes shared sklearn PCA scores; ARI compares cluster labels. "
                    "Results vary by machine and compiler."
                )
            elif native_result and native_result.get("status") == "not_executed":
                st.info(native_result.get("reason") or "Run the native benchmark to populate results.")
        else:
            st.info(
                "The optional native module is not built. From the repository root, run "
                "`python -m pip install ./cpp_engine` to enable the C++ benchmark."
            )

        if not result:
            st.info("Run the selected subset from the sidebar to populate these results.")
        else:
            metrics_path = results_dir / "metrics" / "model_metrics.csv"
            if metrics_path.is_file():
                st.subheader("Model comparison")
                st.dataframe(pd.read_csv(metrics_path), hide_index=True, width="stretch")

            per_engine_path = results_dir / "metrics" / "per_engine_metrics.csv"
            if per_engine_path.is_file():
                per_engine = pd.read_csv(per_engine_path)
                st.subheader("Per-engine error")
                fig, axis = plt.subplots(figsize=(10, 3.5))
                axis.bar(per_engine["unit"], per_engine["mae"], color="#138a8a", width=0.85)
                axis.axhline(per_engine["mae"].median(), color="#dd6b55", linestyle="--", label="Median")
                axis.set(xlabel="Test engine", ylabel="MAE (cycles)", title="Per-engine test error")
                axis.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            figure_names = ["predicted_vs_actual", "fractional_life_metrics", "prediction_intervals", "sensor_correlation"]
            figure_columns = st.columns(2)
            for index, name in enumerate(figure_names):
                relative = result.get("figures", {}).get(name)
                image_path = config.output_dir / relative if relative else None
                if image_path and image_path.is_file():
                    with figure_columns[index % 2]:
                        st.image(str(image_path), caption=name.replace("_", " ").title(), width="stretch")

            stage_result = result.get("fractional_life_analysis", {})
            if stage_result.get("metrics"):
                st.subheader("Fractional-life evaluation")
                st.dataframe(pd.DataFrame(stage_result["metrics"]), hide_index=True, width="stretch")
                st.caption(stage_result["limitation"])

            screening = result.get("sensor_screening", {})
            if screening:
                st.subheader("Sensor screening and PCA")
                st.write(
                    f"{screening['components_to_threshold']} components explain "
                    f"{screening['cumulative_variance_at_threshold']:.1%} of standardized sensor variance."
                )
                st.dataframe(pd.DataFrame(screening["top_sensors_by_absolute_pearson"]), hide_index=True, width="stretch")

            transfer_path = config.output_dir / "transfer" / "transfer_metrics.csv"
            if transfer_path.is_file():
                st.subheader("Cross-subset transfer")
                st.dataframe(pd.read_csv(transfer_path), hide_index=True, width="stretch")
                transfer_meta = _read_json(transfer_path.with_name("transfer_results.json"))
                if transfer_meta:
                    st.caption(transfer_meta["domain_caveat"])

    with reliability_tab:
        _render_reliability(dataset, config)

    with forecast_tab:
        _render_forecast(dataset, config, model_dir, result)
