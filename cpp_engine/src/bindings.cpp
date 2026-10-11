#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kmeans.hpp"
#include "pca.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

turbofan::Matrix make_matrix(const std::vector<double>& values,
                             std::size_t rows, std::size_t cols,
                             const char* function_name) {
  if (rows == 0 || cols == 0 || values.size() != rows * cols) {
    throw std::invalid_argument(std::string(function_name) +
                                ": flat input size must equal rows * cols");
  }
  turbofan::Matrix matrix(rows, cols);
  matrix.data = values;
  return matrix;
}

py::dict pca_fit_py(const std::vector<double>& values, std::size_t rows,
                    std::size_t cols, double variance_threshold,
                    int max_components) {
  const auto result = turbofan::pca_fit(
      make_matrix(values, rows, cols, "pca_fit"), variance_threshold,
      max_components);
  py::dict output;
  output["n_components"] = result.n_components;
  output["mean"] = result.mean;
  output["components"] = result.components;
  output["eigenvalues"] = result.eigenvalues;
  output["explained_variance_ratio"] = result.explained_variance_ratio;
  output["cumulative_variance"] = result.cumulative_variance;
  output["scores"] = result.scores;
  output["jacobi_sweeps"] = result.jacobi_sweeps;
  return output;
}

py::dict kmeans_fit_py(const std::vector<double>& values, std::size_t rows,
                       std::size_t cols, int k, int max_iter, double tolerance,
                       int n_init, const std::string& init, std::uint64_t seed) {
  bool plus_plus = false;
  if (init == "kmeanspp" || init == "k-means++" || init == "kmeans++") {
    plus_plus = true;
  } else if (init != "random") {
    throw std::invalid_argument("kmeans_fit: init must be 'kmeanspp' or 'random'");
  }
  const auto result = turbofan::kmeans_fit(
      make_matrix(values, rows, cols, "kmeans_fit"), k, max_iter, tolerance,
      n_init, plus_plus, seed);
  py::dict output;
  output["labels"] = result.labels;
  output["centroids"] = result.centroids;
  output["inertia"] = result.inertia;
  output["n_iterations"] = result.n_iterations;
  output["best_init"] = result.best_init;
  output["converged"] = result.converged;
  return output;
}

py::dict compile_info_py() {
  py::dict info;
  info["library"] = "turbofan_cpp";
  info["description"] =
      "Optional C++17 PCA and K-Means kernels for C-MAPSS sensor analysis; "
      "the Python/scikit-learn path remains the scientific reference.";
#ifdef _MSC_VER
  info["compiler"] = "MSVC";
  info["compiler_version"] = std::to_string(_MSC_VER);
#elif defined(__clang__)
  info["compiler"] = "clang";
  info["compiler_version"] = __clang_version__;
#elif defined(__GNUC__)
  info["compiler"] = "gcc";
  info["compiler_version"] = __VERSION__;
#else
  info["compiler"] = "unknown";
  info["compiler_version"] = "unknown";
#endif
#ifdef _OPENMP
  info["openmp"] = true;
  info["openmp_version"] = _OPENMP;
#else
  info["openmp"] = false;
#endif
  info["cxx_standard"] = std::to_string(__cplusplus);
  return info;
}

}  // namespace

PYBIND11_MODULE(turbofan_cpp, module) {
  module.doc() = "Optional C++ PCA and K-Means kernels for C-MAPSS analysis.";
  module.def("pca_fit", &pca_fit_py, py::arg("values"), py::arg("rows"),
             py::arg("cols"), py::arg("variance_threshold") = 0.95,
             py::arg("max_components") = 0);
  module.def("kmeans_fit", &kmeans_fit_py, py::arg("values"), py::arg("rows"),
             py::arg("cols"), py::arg("k"), py::arg("max_iter") = 300,
             py::arg("tolerance") = 1e-4, py::arg("n_init") = 10,
             py::arg("init") = "kmeanspp", py::arg("seed") = 42);
  module.def("compile_info", &compile_info_py);
}
