#ifndef ACTIVATION_FUNCTIONS_CPP
#define ACTIVATION_FUNCTIONS_CPP 

#include <vector>
#include <cmath>


class ActivationFunctions {
public:
  static std::vector<std::vector<double>> ReLU(std::vector<std::vector<double>>& x) {
    for(auto& row: x) {
      for(double& value: row) {
        value = std::max(0.0, value);
      }
    }
    return x;
  }

  static std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>>& x) {
    for(auto& row: x) {
      for(double& value: row) {
        value = 1.0 / (1.0 + std::exp(-value));
      }
    }
    return x;
  }

  static std::vector<std::vector<double>> no_activation(std::vector<std::vector<double>>& x) {
    return x;
  }
};

#endif
