from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, ProjectConfig
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.normalization import Normalizer
from src.pipeline import run_pipeline
from src.sequence_generator import SequenceGenerator
from src.target_construction import TargetConstructor

RAW_COLUMNS = DataLoader(ProjectConfig()).columns


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        left, right = st.columns(2)
        with left:
            fig, axis = plt.subplots(figsize=(7, 4))
            sns.histplot(train_frame["RUL"], bins=40, color="#4169a1", ax=axis)
            axis.set(title="Training target distribution", xlabel="RUL (cycles)", ylabel="Cycle count")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            fig, axis = plt.subplots(figsize=(7, 4))
            sns.heatmap(train_frame[sensors].corr(), cmap="vlag", center=0, ax=axis, cbar=False)
            axis.set(title="Sensor correlation", xlabel="Sensor", ylabel="Sensor")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.dataframe(engine_frame, hide_index=True, use_container_width=True)
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
        st.dataframe(latest, hide_index=True, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
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
            st.dataframe(uploaded, hide_index=True, use_container_width=True)
    except Exception as error:
        st.error(f"Could not process the uploaded history: {error}")


def main() -> None:
    st.set_page_config(page_title="Turbofan RUL Lab", page_icon="✈", layout="wide")
    st.sidebar.selectbox("C-MAPSS subset", DATASET_NAMES, key="dataset")
    dataset = st.session_state.dataset
    config = ProjectConfig(dataset=dataset)
    results_dir = config.output_dir / dataset
    model_dir = config.model_dir / dataset
    result = _read_json(results_dir / "metrics" / "results.json")

    st.title("Turbofan RUL Lab")
    st.caption("NASA C-MAPSS run-to-failure trajectories · Remaining useful life")
    with st.sidebar:
        st.divider()
        st.caption(f"Window: {config.window_size} cycles · RUL cap: {config.rul_clip_value} cycles")
        if st.button("Run Ridge baseline", use_container_width=True, type="primary"):
            try:
                with st.spinner(f"Running {dataset} through the research pipeline…"):
                    _run_experiment(dataset, train_lstm=False)
                st.success(f"{dataset} Ridge baseline complete")
                st.rerun()
            except Exception as error:
                st.error(f"Experiment failed: {error}")
        if st.button("Train LSTM + compare", use_container_width=True):
            try:
                with st.spinner(f"Training and comparing models for {dataset}…"):
                    _run_experiment(dataset, train_lstm=True)
                st.success(f"{dataset} LSTM comparison complete")
                st.rerun()
            except Exception as error:
                st.error(f"Experiment failed: {error}")

    overview_tab, signals_tab, outputs_tab, forecast_tab = st.tabs(
        ["Fleet overview", "Signals", "Research outputs", "Forecast"]
    )
    with overview_tab:
        if result:
            summary = result["dataset_summary"]
            metrics = result["test_metrics"].get(result["selected_model"], {})
            metric_cols = st.columns(5)
            metric_cols[0].metric("Test MAE", f"{metrics['mae']:.2f} cycles")
            metric_cols[1].metric("Test RMSE", f"{metrics['rmse']:.2f} cycles")
            metric_cols[2].metric("Test engines", summary["test_engines"])
            metric_cols[3].metric("Sensors retained", summary["retained_sensors"])
            metric_cols[4].metric("Selected model", result["selected_model"].upper())
            tf_version = result["package_versions"].get("tensorflow", "not-installed")
            tf_note = "TensorFlow not installed" if tf_version == "not-installed" else f"TensorFlow {tf_version}"
            st.caption(f"Run {result['run_id']} · selected on validation MAE · {tf_note}")
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
            st.dataframe(pd.DataFrame(fleet_rows), hide_index=True, use_container_width=True)
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
        if not result:
            st.info("Run the selected subset from the sidebar to populate these results.")
        else:
            metrics_path = results_dir / "metrics" / "model_metrics.csv"
            if metrics_path.is_file():
                st.subheader("Model comparison")
                st.dataframe(pd.read_csv(metrics_path), hide_index=True, use_container_width=True)

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
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            figure_names = ["predicted_vs_actual", "fractional_life_metrics", "prediction_intervals", "sensor_correlation"]
            figure_columns = st.columns(2)
            for index, name in enumerate(figure_names):
                relative = result.get("figures", {}).get(name)
                image_path = config.output_dir / relative if relative else None
                if image_path and image_path.is_file():
                    with figure_columns[index % 2]:
                        st.image(str(image_path), caption=name.replace("_", " ").title(), use_container_width=True)

            stage_result = result.get("fractional_life_analysis", {})
            if stage_result.get("metrics"):
                st.subheader("Fractional-life evaluation")
                st.dataframe(pd.DataFrame(stage_result["metrics"]), hide_index=True, use_container_width=True)
                st.caption(stage_result["limitation"])

            screening = result.get("sensor_screening", {})
            if screening:
                st.subheader("Sensor screening and PCA")
                st.write(
                    f"{screening['components_to_threshold']} components explain "
                    f"{screening['cumulative_variance_at_threshold']:.1%} of standardized sensor variance."
                )
                st.dataframe(pd.DataFrame(screening["top_sensors_by_absolute_pearson"]), hide_index=True, use_container_width=True)

            transfer_path = config.output_dir / "transfer" / "transfer_metrics.csv"
            if transfer_path.is_file():
                st.subheader("Cross-subset transfer")
                st.dataframe(pd.read_csv(transfer_path), hide_index=True, use_container_width=True)
                transfer_meta = _read_json(transfer_path.with_name("transfer_results.json"))
                if transfer_meta:
                    st.caption(transfer_meta["domain_caveat"])

    with forecast_tab:
        _render_forecast(dataset, config, model_dir, result)
