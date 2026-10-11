#ifndef TURBOFAN_CPP_KMEANS_HPP
#define TURBOFAN_CPP_KMEANS_HPP

#include "utilities.hpp"

namespace turbofan {

struct KMeansResult {
  std::vector<int> labels;
  std::vector<double> centroids;
  double inertia = 0.0;
  int n_iterations = 0;
  int best_init = 0;
  bool converged = false;
};

KMeansResult kmeans_fit(const Matrix& values, int k, int max_iter = 300,
                        double tolerance = 1e-4, int n_init = 10,
                        bool kmeans_plus_plus = true, std::uint64_t seed = 42);

std::vector<int> kmeans_assign(const std::vector<double>& centroids, int k,
                              const Matrix& values);

}  // namespace turbofan

#endif  // TURBOFAN_CPP_KMEANS_HPP
