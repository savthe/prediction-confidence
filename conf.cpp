/*
 * This file is released into the public domain.
 * 
 * No copyright is claimed. You may use, modify, and distribute this file
 * without restriction for any purpose, commercial or non-commercial.
 * 
 */

#include <array>
#include <iostream>

template<typename Params>
struct NormalProbabilityDensity {
  static constexpr float kMean = Params::kMean;
  static constexpr float kStdev = Params::kStdev;

  // Computes x to the non-negative integer power n in O(log(n)).
  static constexpr float static_int_pow(float x, int n) {
    float p = 1;
    while (n > 0) {
      if (n & 1) {
        p *= x;
      }
      x *= x;
      n >>= 1;
    }
    return p;
  }

  // Computex exp(x) as constexpr.
  static constexpr float static_exp(float x) {
    constexpr float kAccuracy = 1e-6;

    const bool positive = x >= 0;
    x = x < 0 ? -x : x;

    // Compute exp(x) = exp(u) * exp(v), where u and v are the integer and fractional parts of x.
    int integer_part = static_cast<int>(x);
    x -= integer_part;

    // sum is exp(v), where v is a fractional part of x. Use Taylor series expension.
    float sum = 0;
    float term = 1;
    int i = 1;
    while (term > kAccuracy) {
      sum += term;
      term *= x / i;
      ++i;
    }
    if (!positive) sum = 1. / sum;

    constexpr float e = 2.718281828459045;
    const float base = positive ? e : (1. / e);

    return static_int_pow(base, integer_part) * sum;
  }

  // Computes the probability density.
  static constexpr float eval(float x) {
    constexpr float kRootOf2Pi = 2.50662827463;
    const float power = -0.5 * (x - kMean) * (x - kMean) / kStdev / kStdev;
    return (1. / (kStdev * kRootOf2Pi)) * static_exp(power);
  }
};

template<size_t kPoints, typename Limits, typename DistributionFunction>
class Confidence {
  constexpr static float kLower = Limits::kLower;
  constexpr static float kUpper = Limits::kUpper;
  constexpr static float kDelta = (kUpper - kLower) / kPoints;

public:
  static constexpr float evaluate(float x) {
    if (x <= kLower || x >= kUpper) return 0; 

    const int i = int((x - kLower) / kDelta);
    return 2 * std::min(cdf_[i], 1 - cdf_[i]);
  }
private:
  using Cdf = std::array<float, kPoints + 1>;

  inline static constexpr Cdf cdf_ = []() {
    Cdf v{};
    v[0] = 0;
    float f_i = DistributionFunction::eval(Limits::kLower);
    float sum = f_i / 2;
    for (int i = 1; i <= kPoints; ++i) {
      f_i = DistributionFunction::eval(kLower + kDelta * i);
      v[i] = kDelta * (sum + f_i / 2);
      sum += f_i;
    }
    return v;
  }();
};

struct NormalDistributionParams {
  static constexpr float kMean = 0.043;
  static constexpr float kStdev = 0.026;
};

struct Limits {
  static constexpr float kMean = NormalDistributionParams::kMean;
  static constexpr float kStdev = NormalDistributionParams::kStdev;
  static constexpr float kLower = kMean - 6 * kStdev;
  static constexpr float kUpper = kMean + 6 * kStdev;
};

using Metric = Confidence<10000, Limits, NormalProbabilityDensity<NormalDistributionParams>>;

namespace tests {
  static constexpr bool test_equal(float x, float y) {
    constexpr float accuracy = 0.001;
    return y - accuracy < x && x < y + accuracy;
  }
  static_assert(test_equal(Metric::evaluate(NormalDistributionParams::kMean), 1.0));
}  // namespace tests
 
int main() {
  float x;
  std::cin >> x;
  std::cout << Metric::evaluate(x) << std::endl;
}


