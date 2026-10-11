#ifndef TURBOFAN_CPP_PCA_HPP
#define TURBOFAN_CPP_PCA_HPP

#include "utilities.hpp"

namespace turbofan {

struct PcaResult {
  int n_components = 0;
  std::vector<double> mean;
  std::vector<double> components;
  std::vector<double> eigenvalues;
  std::vector<double> explained_variance_ratio;
  std::vector<double> cumulative_variance;
  std::vector<double> scores;
  std::size_t jacobi_sweeps = 0;
};

PcaResult pca_fit(const Matrix& values, double variance_threshold = 0.95,
                  int max_components = 0);

Matrix pca_transform(const std::vector<double>& mean,
                     const std::vector<double>& components,
                     int n_components, const Matrix& values);

}  // namespace turbofan

#endif  // TURBOFAN_CPP_PCA_HPP
