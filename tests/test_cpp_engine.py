import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.cpp_bridge import AVAILABLE, kmeans_fit_numpy, pca_fit_numpy

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="optional turbofan_cpp module is not built")


def test_cpp_pca_matches_sklearn_subspace():
    values = np.random.default_rng(12).normal(size=(500, 8))
    reference = PCA(n_components=0.95, svd_solver="full").fit(values)
    native = pca_fit_numpy(values, variance_threshold=0.95)

    assert native["n_components"] == reference.n_components_
    np.testing.assert_allclose(
        native["explained_variance_ratio"],
        reference.explained_variance_ratio_,
        rtol=1e-9,
        atol=1e-11,
    )
    principal_cosines = np.linalg.svd(
        reference.components_ @ native["components"].T,
        compute_uv=False,
    )
    assert principal_cosines.min() > 1.0 - 1e-8


def test_cpp_kmeans_agrees_with_sklearn_on_separated_clusters():
    rng = np.random.default_rng(24)
    centers = np.array([[-6.0, -5.0], [0.0, 6.0], [6.0, -5.0]])
    values = np.vstack(
        [center + rng.normal(scale=0.25, size=(80, 2)) for center in centers]
    )
    reference = KMeans(n_clusters=3, n_init=10, random_state=42, algorithm="lloyd").fit(values)
    native = kmeans_fit_numpy(values, k=3, n_init=10, seed=42)

    assert native["converged"]
    assert adjusted_rand_score(reference.labels_, native["labels"]) > 0.99
    assert native["inertia"] == pytest.approx(reference.inertia_, rel=1e-5)
