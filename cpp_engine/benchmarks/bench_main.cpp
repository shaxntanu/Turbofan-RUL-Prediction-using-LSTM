#include "kmeans.hpp"
#include "pca.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

turbofan::Matrix synthetic_sensors(std::size_t rows, std::size_t cols,
                                   std::uint64_t seed) {
  turbofan::Rng rng(seed);
  turbofan::Matrix values(rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      double u1 = std::max(rng.uniform01(), 1e-12);
      const double u2 = rng.uniform01();
      const double normal = std::sqrt(-2.0 * std::log(u1)) *
                            std::cos(6.283185307179586 * u2);
      values(i, j) = normal + 0.2 * std::sin(static_cast<double>(i) / 37.0 + j);
    }
  }
  return values;
}

template <typename Function>
double best_of_three(Function&& function) {
  double best = 1e300;
  for (int repeat = 0; repeat < 3; ++repeat) {
    turbofan::Stopwatch timer;
    timer.start();
    function();
    best = std::min(best, timer.elapsed_ms());
  }
  return best;
}

}  // namespace

int main() {
  std::printf("{\n  \"engine\": \"turbofan_cpp (C++17)\",\n  \"rows\": [\n");
  const std::vector<std::pair<const char*, std::size_t>> cases = {
      {"small", 500}, {"large", 10000}};
  bool first = true;
  for (const auto& item : cases) {
    const auto values = synthetic_sensors(item.second, 21, 42);
    const auto pca = turbofan::pca_fit(values, 0.95);
    const double pca_ms = best_of_three([&]() { turbofan::pca_fit(values, 0.95); });
    turbofan::Matrix scores(values.rows, static_cast<std::size_t>(pca.n_components));
    scores.data = pca.scores;
    const auto clusters = turbofan::kmeans_fit(scores, 4, 100, 1e-4, 5, true, 42);
    const double kmeans_ms = best_of_three(
        [&]() { turbofan::kmeans_fit(scores, 4, 100, 1e-4, 5, true, 42); });
    if (!first) std::printf(",\n");
    first = false;
    std::printf(
        "    {\"dataset\": \"%s\", \"n_samples\": %zu, \"n_features\": 21, "
        "\"pca_ms\": %.4f, \"pca_components\": %d, \"variance_retained\": %.6f, "
        "\"kmeans_ms\": %.4f, \"k\": 4, \"inertia\": %.6f, "
        "\"iterations\": %d, \"converged\": %s}",
        item.first, item.second, pca_ms, pca.n_components,
        pca.cumulative_variance[static_cast<std::size_t>(pca.n_components) - 1],
        kmeans_ms, clusters.inertia, clusters.n_iterations,
        clusters.converged ? "true" : "false");
  }
  std::printf("\n  ]\n}\n");
  return 0;
}
