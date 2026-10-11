"""Compare optional C++ sensor-analysis kernels with sklearn on C-MAPSS."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_NAMES, PROJECT_ROOT, ProjectConfig
from src.cpp_bridge import AVAILABLE, COMPILE_INFO, REASON, kmeans_fit_numpy, module_status, pca_fit_numpy
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter
from src.target_construction import TargetConstructor

logger = logging.getLogger(__name__)
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "cpp_engine_benchmark.json"


def _timed_best(function: Callable, repeats: int = 3) -> tuple[object, float]:
    function()
    best_result = None
    best_seconds = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        if elapsed < best_seconds:
            best_result, best_seconds = result, elapsed
    return best_result, best_seconds * 1000.0


def _sensor_matrix(dataset: str, max_rows: int, seed: int) -> tuple[np.ndarray, list[str]]:
    config = ProjectConfig(dataset=dataset)
    raw = DataLoader(config).load_train_data()
    labeled = TargetConstructor.compute_training_rul(
        raw, config.rul_clip_value if config.clip_rul else None
    )
    training_engines, _ = DataSplitter(config).split_engines(labeled)
    training_engines, _ = DataCleaner(config).remove_constant_sensors(training_engines)
    sensors = [name for name in training_engines if name.startswith("sensor_")]
    values = training_engines[sensors].to_numpy(dtype=np.float64, copy=True)
    if len(values) > max_rows:
        indices = np.random.default_rng(seed).choice(len(values), max_rows, replace=False)
        values = values[np.sort(indices)]
    standardized = StandardScaler().fit_transform(values)
    return np.ascontiguousarray(standardized, dtype=np.float64), sensors


def _benchmark_subset(dataset: str, max_rows: int, seed: int) -> dict:
    matrix, sensors = _sensor_matrix(dataset, max_rows, seed)

    def fit_python_pca() -> tuple[PCA, np.ndarray]:
        model = PCA(n_components=0.95, svd_solver="full", copy=True)
        return model, model.fit_transform(matrix)

    (python_pca, python_scores), python_pca_ms = _timed_best(fit_python_pca)

    cpp_pca, cpp_pca_ms = _timed_best(
        lambda: pca_fit_numpy(matrix, variance_threshold=0.95)
    )
    components_match = python_pca.components_.shape == cpp_pca["components"].shape
    if components_match:
        signed_alignment = np.sum(
            python_pca.components_ * cpp_pca["components"], axis=1
        )
        aligned_cpp_components = cpp_pca["components"] * np.where(
            signed_alignment[:, None] < 0, -1.0, 1.0
        )
        component_diff = float(
            np.max(np.abs(python_pca.components_ - aligned_cpp_components))
        )
        principal_cosines = np.linalg.svd(
            python_pca.components_ @ cpp_pca["components"].T,
            compute_uv=False,
        )
        minimum_principal_cosine = float(np.clip(principal_cosines.min(), 0.0, 1.0))
        maximum_principal_angle_degrees = float(
            np.degrees(np.arccos(minimum_principal_cosine))
        )
    else:
        component_diff = None
        minimum_principal_cosine = None
        maximum_principal_angle_degrees = None
    variance_diff = float(
        np.max(
            np.abs(
                python_pca.explained_variance_ratio_ -
                cpp_pca["explained_variance_ratio"][: python_pca.n_components_]
            )
        )
    ) if components_match else None

    cluster_count = min(4, len(matrix))
    python_kmeans, python_kmeans_ms = _timed_best(
        lambda: KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=seed,
            algorithm="lloyd",
        ).fit(python_scores)
    )
    cpp_kmeans, cpp_kmeans_ms = _timed_best(
        lambda: kmeans_fit_numpy(
            python_scores,
            k=cluster_count,
            max_iter=300,
            tolerance=1e-4,
            n_init=10,
            init="kmeanspp",
            seed=seed,
        )
    )

    return {
        "dataset": dataset,
        "source": "training engines only; seeded sample; train-sample standardization",
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "pca": {
            "python_ms": python_pca_ms,
            "cpp_ms": cpp_pca_ms,
            "speedup_x": python_pca_ms / cpp_pca_ms if cpp_pca_ms else None,
            "python_components": int(python_pca.n_components_),
            "cpp_components": int(cpp_pca["n_components"]),
            "component_count_match": components_match,
            "max_abs_component_difference_sign_aligned": component_diff,
            "minimum_principal_subspace_cosine": minimum_principal_cosine,
            "maximum_principal_angle_degrees": maximum_principal_angle_degrees,
            "max_explained_variance_ratio_difference": variance_diff,
            "variance_retained_python": float(python_pca.explained_variance_ratio_.sum()),
            "variance_retained_cpp": float(
                cpp_pca["cumulative_variance"][cpp_pca["n_components"] - 1]
            ),
            "cpp_jacobi_sweeps": cpp_pca["jacobi_sweeps"],
        },
        "kmeans": {
            "input": "shared sklearn PCA scores",
            "k": cluster_count,
            "python_ms": python_kmeans_ms,
            "cpp_ms": cpp_kmeans_ms,
            "speedup_x": python_kmeans_ms / cpp_kmeans_ms if cpp_kmeans_ms else None,
            "adjusted_rand_index": float(
                adjusted_rand_score(python_kmeans.labels_, cpp_kmeans["labels"])
            ),
            "adjusted_mutual_info": float(
                adjusted_mutual_info_score(python_kmeans.labels_, cpp_kmeans["labels"])
            ),
            "python_inertia": float(python_kmeans.inertia_),
            "cpp_inertia": cpp_kmeans["inertia"],
            "inertia_relative_difference": float(
                abs(python_kmeans.inertia_ - cpp_kmeans["inertia"])
                / max(abs(python_kmeans.inertia_), 1e-12)
            ),
            "python_iterations": int(python_kmeans.n_iter_),
            "cpp_iterations": cpp_kmeans["n_iterations"],
            "cpp_converged": cpp_kmeans["converged"],
        },
        "sensor_columns": sensors,
    }


def run_benchmark(
    datasets: tuple[str, ...] = DATASET_NAMES,
    *,
    max_rows: int = 10000,
    seed: int = 42,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    payload = {
        "schema_version": 1,
        "status": "not_executed",
        "reason": None,
        "generated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": module_status(),
        "protocol": {
            "pca": "sklearn full-SVD fit_transform vs C++ covariance/Jacobi fit plus score projection on identical standardized rows",
            "kmeans": "same sklearn PCA scores, k-means++, seed 42 by default, n_init=10",
            "timing": "one warm-up and best-of-three wall-clock measurements",
            "selection": "sensor screening and clustering are diagnostics; they do not affect RUL model selection",
            "limitation": "microbenchmarks are hardware/compiler dependent and are not end-to-end LSTM speedups",
        },
        "datasets": [],
        "notes": [],
    }
    if not AVAILABLE:
        payload["reason"] = REASON
        payload["notes"].append(
            "No Python-vs-C++ timings were fabricated. Build the optional extension with "
            "python -m pip install ./cpp_engine."
        )
    else:
        payload["status"] = "executed"
        payload["compile_info"] = COMPILE_INFO
        for dataset in datasets:
            try:
                payload["datasets"].append(_benchmark_subset(dataset, max_rows, seed))
            except Exception as error:
                logger.exception("Native benchmark failed for %s", dataset)
                payload["datasets"].append(
                    {"dataset": dataset, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                )
        if any(row.get("status") == "failed" for row in payload["datasets"]):
            payload["status"] = "partial"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.max_rows < 2:
        parser.error("--max-rows must be at least 2")
    payload = run_benchmark(tuple(args.datasets), max_rows=args.max_rows,
                            seed=args.seed, output_path=args.output)
    print(args.output)
    print(f"status: {payload['status']}")


if __name__ == "__main__":
    main()
