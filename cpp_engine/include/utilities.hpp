#ifndef TURBOFAN_CPP_UTILITIES_HPP
#define TURBOFAN_CPP_UTILITIES_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace turbofan {

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> data;

  Matrix() = default;
  Matrix(std::size_t n_rows, std::size_t n_cols, double fill = 0.0)
      : rows(n_rows), cols(n_cols), data(n_rows * n_cols, fill) {}

  double& operator()(std::size_t row, std::size_t col) {
    return data[row * cols + col];
  }
  const double& operator()(std::size_t row, std::size_t col) const {
    return data[row * cols + col];
  }
};

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next_u64() {
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
  }

  double uniform01() {
    return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
  }

  std::size_t uniform_int(std::size_t size) {
    if (size == 0) throw std::invalid_argument("Rng: size must be positive");
    const auto value = static_cast<std::size_t>(uniform01() * size);
    return value < size ? value : size - 1;
  }

 private:
  std::uint64_t state_;
};

class Stopwatch {
 public:
  void start() { start_ = std::chrono::steady_clock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace turbofan

#endif  // TURBOFAN_CPP_UTILITIES_HPP
