#include "kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace turbofan {
namespace {

double squared_distance(const double* left, const double* right,
                        std::size_t dimensions) {
  double result = 0.0;
  for (std::size_t j = 0; j < dimensions; ++j) {
    const double delta = left[j] - right[j];
    result += delta * delta;
  }
  return result;
}

void assign_all(const Matrix& values, const std::vector<double>& centroids,
                int k, std::vector<int>& labels, double& inertia) {
  labels.resize(values.rows);
  inertia = 0.0;
  for (std::size_t i = 0; i < values.rows; ++i) {
    int best = 0;
    double best_distance = squared_distance(
        &values.data[i * values.cols], centroids.data(), values.cols);
    for (int cluster = 1; cluster < k; ++cluster) {
      const double distance = squared_distance(
          &values.data[i * values.cols],
          &centroids[static_cast<std::size_t>(cluster) * values.cols],
          values.cols);
      if (distance < best_distance) {
        best_distance = distance;
        best = cluster;
      }
    }
    labels[i] = best;
    inertia += best_distance;
  }
}

std::vector<double> initialize_centroids(const Matrix& values, int k,
                                         bool plus_plus, Rng& rng) {
  const std::size_t n = values.rows;
  const std::size_t d = values.cols;
  std::vector<double> centers(static_cast<std::size_t>(k) * d);
  const std::size_t first = rng.uniform_int(n);
  std::copy_n(&values.data[first * d], d, centers.begin());
  if (!plus_plus) {
    for (int c = 1; c < k; ++c) {
      const std::size_t row = rng.uniform_int(n);
      std::copy_n(&values.data[row * d], d,
                  centers.begin() + static_cast<std::size_t>(c) * d);
    }
    return centers;
  }

  std::vector<double> min_distance(n, std::numeric_limits<double>::infinity());
  for (int c = 1; c < k; ++c) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double distance = squared_distance(
          &values.data[i * d], &centers[static_cast<std::size_t>(c - 1) * d], d);
      min_distance[i] = std::min(min_distance[i], distance);
      total += min_distance[i];
    }
    std::size_t selected = rng.uniform_int(n);
    if (total > std::numeric_limits<double>::epsilon()) {
      double target = rng.uniform01() * total;
      for (std::size_t i = 0; i < n; ++i) {
        target -= min_distance[i];
        if (target <= 0.0) {
          selected = i;
          break;
        }
      }
    }
    std::copy_n(&values.data[selected * d], d,
                centers.begin() + static_cast<std::size_t>(c) * d);
  }
  return centers;
}

KMeansResult lloyd(const Matrix& values, int k, std::vector<double> centers,
                   int max_iter, double tolerance) {
  const std::size_t n = values.rows;
  const std::size_t d = values.cols;
  KMeansResult result;
  result.labels.assign(n, 0);
  std::vector<double> sums(static_cast<std::size_t>(k) * d);
  std::vector<std::size_t> counts(static_cast<std::size_t>(k));

  for (int iteration = 0; iteration < max_iter; ++iteration) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
#ifdef _OPENMP
#pragma omp parallel
    {
      std::vector<double> local_sums(static_cast<std::size_t>(k) * d, 0.0);
      std::vector<std::size_t> local_counts(static_cast<std::size_t>(k), 0);
#pragma omp for schedule(static)
      for (std::int64_t row = 0; row < static_cast<std::int64_t>(n); ++row) {
        const std::size_t i = static_cast<std::size_t>(row);
        int best = 0;
        double best_distance = squared_distance(
            &values.data[i * d], centers.data(), d);
        for (int c = 1; c < k; ++c) {
          const double distance = squared_distance(
              &values.data[i * d], &centers[static_cast<std::size_t>(c) * d], d);
          if (distance < best_distance) {
            best_distance = distance;
            best = c;
          }
        }
        result.labels[i] = best;
        ++local_counts[static_cast<std::size_t>(best)];
        for (std::size_t j = 0; j < d; ++j) {
          local_sums[static_cast<std::size_t>(best) * d + j] += values(i, j);
        }
      }
#pragma omp critical
      {
        for (std::size_t j = 0; j < sums.size(); ++j) sums[j] += local_sums[j];
        for (int c = 0; c < k; ++c) {
          counts[static_cast<std::size_t>(c)] += local_counts[static_cast<std::size_t>(c)];
        }
      }
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
      int best = 0;
      double best_distance = squared_distance(
          &values.data[i * d], centers.data(), d);
      for (int c = 1; c < k; ++c) {
        const double distance = squared_distance(
            &values.data[i * d], &centers[static_cast<std::size_t>(c) * d], d);
        if (distance < best_distance) {
          best_distance = distance;
          best = c;
        }
      }
      ++counts[static_cast<std::size_t>(best)];
      for (std::size_t j = 0; j < d; ++j) {
        sums[static_cast<std::size_t>(best) * d + j] += values(i, j);
      }
    }
#endif

    double max_shift = 0.0;
    for (int c = 0; c < k; ++c) {
      const std::size_t count = counts[static_cast<std::size_t>(c)];
      if (count == 0) continue;
      for (std::size_t j = 0; j < d; ++j) {
        const std::size_t index = static_cast<std::size_t>(c) * d + j;
        const double next = sums[index] / static_cast<double>(count);
        max_shift = std::max(max_shift, std::abs(next - centers[index]));
        centers[index] = next;
      }
    }
    result.n_iterations = iteration + 1;
    if (max_shift <= tolerance) {
      result.converged = true;
      break;
    }
  }

  result.centroids = std::move(centers);
  assign_all(values, result.centroids, k, result.labels, result.inertia);
  return result;
}

}  // namespace

KMeansResult kmeans_fit(const Matrix& values, int k, int max_iter,
                        double tolerance, int n_init, bool kmeans_plus_plus,
                        std::uint64_t seed) {
  if (values.rows == 0 || values.cols == 0 ||
      values.data.size() != values.rows * values.cols) {
    throw std::invalid_argument("kmeans_fit: expected a non-empty matrix");
  }
  if (k < 2 || static_cast<std::size_t>(k) > values.rows) {
    throw std::invalid_argument("kmeans_fit: k must be between 2 and sample count");
  }
  if (max_iter < 1 || n_init < 1 || tolerance < 0.0) {
    throw std::invalid_argument("kmeans_fit: invalid iteration, restart, or tolerance value");
  }

  KMeansResult best;
  best.inertia = std::numeric_limits<double>::infinity();
  Rng master(seed);
  for (int init = 0; init < n_init; ++init) {
    Rng rng(master.next_u64());
    auto centers = initialize_centroids(values, k, kmeans_plus_plus, rng);
    KMeansResult current = lloyd(values, k, std::move(centers), max_iter, tolerance);
    current.best_init = init;
    if (current.inertia < best.inertia) best = std::move(current);
  }
  return best;
}

std::vector<int> kmeans_assign(const std::vector<double>& centroids, int k,
                               const Matrix& values) {
  if (k < 1 || values.cols == 0 ||
      centroids.size() != static_cast<std::size_t>(k) * values.cols) {
    throw std::invalid_argument("kmeans_assign: centroid and input dimensions differ");
  }
  std::vector<int> labels;
  double inertia = 0.0;
  assign_all(values, centroids, k, labels, inertia);
  return labels;
}

}  // namespace turbofan
