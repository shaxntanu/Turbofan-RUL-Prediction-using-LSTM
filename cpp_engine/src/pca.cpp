#include "pca.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace turbofan {
namespace {

void jacobi_eigen(std::vector<double>& matrix, std::vector<double>& vectors,
                  std::size_t size, std::size_t max_sweeps,
                  std::size_t& sweeps_used) {
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    double off_diagonal = 0.0;
    for (std::size_t row = 0; row < size; ++row) {
      for (std::size_t col = row + 1; col < size; ++col) {
        const double value = matrix[row * size + col];
        off_diagonal += value * value;
      }
    }
    if (std::sqrt(off_diagonal) <= 1e-14) {
      sweeps_used = sweep;
      return;
    }

    for (std::size_t p = 0; p + 1 < size; ++p) {
      for (std::size_t q = p + 1; q < size; ++q) {
        const double apq = matrix[p * size + q];
        if (std::abs(apq) < 1e-300) continue;
        const double app = matrix[p * size + p];
        const double aqq = matrix[q * size + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + t * t);
        const double sine = t * cosine;

        for (std::size_t k = 0; k < size; ++k) {
          if (k == p || k == q) continue;
          const double akp = matrix[k * size + p];
          const double akq = matrix[k * size + q];
          const double next_kp = cosine * akp - sine * akq;
          const double next_kq = sine * akp + cosine * akq;
          matrix[k * size + p] = next_kp;
          matrix[p * size + k] = next_kp;
          matrix[k * size + q] = next_kq;
          matrix[q * size + k] = next_kq;
        }
        matrix[p * size + p] = cosine * cosine * app -
                               2.0 * sine * cosine * apq +
                               sine * sine * aqq;
        matrix[q * size + q] = sine * sine * app +
                               2.0 * sine * cosine * apq +
                               cosine * cosine * aqq;
        matrix[p * size + q] = 0.0;
        matrix[q * size + p] = 0.0;

        for (std::size_t k = 0; k < size; ++k) {
          const double vkp = vectors[k * size + p];
          const double vkq = vectors[k * size + q];
          vectors[k * size + p] = cosine * vkp - sine * vkq;
          vectors[k * size + q] = sine * vkp + cosine * vkq;
        }
      }
    }
  }
  sweeps_used = max_sweeps;
}

}  // namespace

PcaResult pca_fit(const Matrix& values, double variance_threshold,
                  int max_components) {
  if (values.rows < 2 || values.cols == 0 ||
      values.data.size() != values.rows * values.cols) {
    throw std::invalid_argument("pca_fit: expected a non-empty matrix with at least two rows");
  }
  if (!(variance_threshold > 0.0 && variance_threshold <= 1.0)) {
    throw std::invalid_argument("pca_fit: variance threshold must be in (0, 1]");
  }
  if (max_components < 0) {
    throw std::invalid_argument("pca_fit: max_components cannot be negative");
  }

  const std::size_t n = values.rows;
  const std::size_t d = values.cols;
  PcaResult result;
  result.mean.assign(d, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) result.mean[j] += values(i, j);
  }
  for (double& value : result.mean) value /= static_cast<double>(n);

  std::vector<double> covariance(d * d, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < d; ++p) {
      const double xp = values(i, p) - result.mean[p];
      for (std::size_t q = p; q < d; ++q) {
        covariance[p * d + q] += xp * (values(i, q) - result.mean[q]);
      }
    }
  }
  for (std::size_t p = 0; p < d; ++p) {
    for (std::size_t q = p; q < d; ++q) {
      covariance[p * d + q] /= static_cast<double>(n - 1);
      covariance[q * d + p] = covariance[p * d + q];
    }
  }

  std::vector<double> eigenvectors(d * d, 0.0);
  for (std::size_t j = 0; j < d; ++j) eigenvectors[j * d + j] = 1.0;
  jacobi_eigen(covariance, eigenvectors, d, 200, result.jacobi_sweeps);

  std::vector<std::size_t> order(d);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&covariance, d](std::size_t a, std::size_t b) {
    return covariance[a * d + a] > covariance[b * d + b];
  });

  result.eigenvalues.resize(d);
  result.explained_variance_ratio.resize(d);
  result.cumulative_variance.resize(d);
  std::vector<double> sorted_vectors(d * d);
  double total_variance = 0.0;
  for (std::size_t j = 0; j < d; ++j) {
    result.eigenvalues[j] = std::max(0.0, covariance[order[j] * d + order[j]]);
    total_variance += result.eigenvalues[j];
    for (std::size_t i = 0; i < d; ++i) {
      sorted_vectors[i * d + j] = eigenvectors[i * d + order[j]];
    }
  }
  if (!(total_variance > 0.0)) {
    throw std::runtime_error("pca_fit: input has zero total variance");
  }

  double cumulative = 0.0;
  for (std::size_t j = 0; j < d; ++j) {
    result.explained_variance_ratio[j] = result.eigenvalues[j] / total_variance;
    cumulative += result.explained_variance_ratio[j];
    result.cumulative_variance[j] = cumulative;
  }

  // Align each direction with sklearn's deterministic svd_flip convention.
  for (std::size_t component = 0; component < d; ++component) {
    std::size_t best_row = 0;
    double largest_projection = -1.0;
    double signed_projection = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double projection = 0.0;
      for (std::size_t j = 0; j < d; ++j) {
        projection += (values(i, j) - result.mean[j]) * sorted_vectors[j * d + component];
      }
      if (std::abs(projection) > largest_projection) {
        largest_projection = std::abs(projection);
        best_row = i;
        signed_projection = projection;
      }
    }
    (void)best_row;
    if (signed_projection < 0.0) {
      for (std::size_t j = 0; j < d; ++j) sorted_vectors[j * d + component] *= -1.0;
    }
  }

  int retained = static_cast<int>(d);
  for (std::size_t j = 0; j < d; ++j) {
    if (result.cumulative_variance[j] >= variance_threshold) {
      retained = static_cast<int>(j + 1);
      break;
    }
  }
  if (max_components > 0) retained = std::min(retained, max_components);
  result.n_components = retained;
  result.components.resize(static_cast<std::size_t>(retained) * d);
  result.scores.resize(n * static_cast<std::size_t>(retained));
  for (int component = 0; component < retained; ++component) {
    for (std::size_t j = 0; j < d; ++j) {
      result.components[static_cast<std::size_t>(component) * d + j] =
          sorted_vectors[j * d + static_cast<std::size_t>(component)];
    }
    for (std::size_t i = 0; i < n; ++i) {
      double projection = 0.0;
      for (std::size_t j = 0; j < d; ++j) {
        projection += (values(i, j) - result.mean[j]) *
                      sorted_vectors[j * d + static_cast<std::size_t>(component)];
      }
      result.scores[i * static_cast<std::size_t>(retained) +
                    static_cast<std::size_t>(component)] = projection;
    }
  }
  return result;
}

Matrix pca_transform(const std::vector<double>& mean,
                     const std::vector<double>& components,
                     int n_components, const Matrix& values) {
  if (n_components < 1 || mean.size() != values.cols ||
      components.size() != static_cast<std::size_t>(n_components) * values.cols) {
    throw std::invalid_argument("pca_transform: incompatible fit and input dimensions");
  }
  Matrix scores(values.rows, static_cast<std::size_t>(n_components));
  for (std::size_t i = 0; i < values.rows; ++i) {
    for (int component = 0; component < n_components; ++component) {
      double projection = 0.0;
      for (std::size_t j = 0; j < values.cols; ++j) {
        projection += (values(i, j) - mean[j]) *
                      components[static_cast<std::size_t>(component) * values.cols + j];
      }
      scores(i, static_cast<std::size_t>(component)) = projection;
    }
  }
  return scores;
}

}  // namespace turbofan
