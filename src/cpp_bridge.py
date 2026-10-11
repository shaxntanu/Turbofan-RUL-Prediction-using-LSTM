"""Optional bridge to the native C++ sensor-analysis kernels."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import turbofan_cpp as _native

    AVAILABLE = True
    REASON = None
    COMPILE_INFO = dict(_native.compile_info())
except Exception as error:  # Optional extension can be absent or ABI-incompatible.
    _native = None
    AVAILABLE = False
    REASON = f"{type(error).__name__}: {error}"
    COMPILE_INFO = {}
    logger.info("Optional turbofan_cpp module unavailable: %s", REASON)


class EngineUnavailable(RuntimeError):
    """Raised when the C++ engine is explicitly requested but not built."""


def module_status() -> dict:
    return {
        "available": AVAILABLE,
        "reason": REASON,
        "compile_info": COMPILE_INFO,
        "build_command": "python -m pip install ./cpp_engine" if not AVAILABLE else None,
    }


def resolve_engine(engine: str = "auto") -> str:
    if engine in {"python", "reference", "sklearn"}:
        return "python"
    if engine in {"cpp", "c++", "native"}:
        if not AVAILABLE:
            raise EngineUnavailable(REASON or "turbofan_cpp is not importable")
        return "cpp"
    if engine in {"", "auto", "best"}:
        return "cpp" if AVAILABLE else "python"
    raise ValueError(f"Unknown engine selector: {engine!r}")


def _matrix(values: np.ndarray, function_name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{function_name} expects a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{function_name} input must contain only finite values")
    return np.ascontiguousarray(matrix)


def pca_fit_numpy(
    values: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 0,
) -> dict:
    if not AVAILABLE:
        raise EngineUnavailable(REASON or "turbofan_cpp is not importable")
    matrix = _matrix(values, "pca_fit_numpy")
    rows, cols = matrix.shape
    result = _native.pca_fit(
        matrix.ravel().tolist(), rows, cols,
        float(variance_threshold), int(max_components),
    )
    count = int(result["n_components"])
    return {
        "n_components": count,
        "mean": np.asarray(result["mean"], dtype=np.float64),
        "components": np.asarray(result["components"], dtype=np.float64).reshape(count, cols),
        "eigenvalues": np.asarray(result["eigenvalues"], dtype=np.float64),
        "explained_variance_ratio": np.asarray(
            result["explained_variance_ratio"], dtype=np.float64
        ),
        "cumulative_variance": np.asarray(result["cumulative_variance"], dtype=np.float64),
        "scores": np.asarray(result["scores"], dtype=np.float64).reshape(rows, count),
        "jacobi_sweeps": int(result["jacobi_sweeps"]),
    }


def kmeans_fit_numpy(
    values: np.ndarray,
    k: int,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    init: str = "kmeanspp",
    seed: int = 42,
) -> dict:
    if not AVAILABLE:
        raise EngineUnavailable(REASON or "turbofan_cpp is not importable")
    matrix = _matrix(values, "kmeans_fit_numpy")
    rows, cols = matrix.shape
    result = _native.kmeans_fit(
        matrix.ravel().tolist(), rows, cols, int(k), int(max_iter),
        float(tolerance), int(n_init), str(init), int(seed),
    )
    return {
        "labels": np.asarray(result["labels"], dtype=np.int64),
        "centroids": np.asarray(result["centroids"], dtype=np.float64).reshape(int(k), cols),
        "inertia": float(result["inertia"]),
        "n_iterations": int(result["n_iterations"]),
        "best_init": int(result["best_init"]),
        "converged": bool(result["converged"]),
        "k": int(k),
        "seed": int(seed),
        "n_init": int(n_init),
    }
