#include <vector>
#include <iostream>
#include "../utils/Exception.cpp"
#include <random>
#include "../utils/activation_functions.cpp"
#include <functional>

class FullyConnectedLayer {
private:
  std::mt19937 gen; // Mersenne Twister random number generator
  std::normal_distribution<> d;
  std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>&)> activation_function;
public:
  std::vector<std::vector<double>> weights;
  int height, width;

  FullyConnectedLayer(int height, int width,
                      std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>&)> activation_func):
    height(height), width(width),
    activation_function(activation_func),
    gen(std::random_device{}()),
    d(0.0, std::sqrt(2.0 / (height + width))) // He initialization
  {
    this->weights.resize(height, std::vector<double>(width, 0));
    initialize_weights();
  }

  void test_matrix() {
    for(int i = 0; i < this->height; i++) {
      for(int j = 0; j < this->width; j++) {
        weights[i][j] = i + j;
      }
    }
  }

  void initialize_weights() {
    for(int i = 0; i < this->height; i++) {
      for(int j = 0; j < this->width; j++) {
        weights[i][j] = d(gen);
      }
    }
  }

  double get_value(int i, int j) {
    return weights[i][j];
  }
  

  std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) {
    int x_height = x.size();
    int x_width = x[0].size();
    
    if(x_width != this->height) {
      throw Exception("Size mismatch between layers");
    }

    std::vector<std::vector<double>> y(x_height, std::vector<double>(this->width, 0));

    for(int y_i = 0; y_i < x_height; y_i++) {
      for(int y_j = 0; y_j < this->width; y_j++) {
        double sum = 0;

        for(int z = 0; z < x_width; z++) {
          sum += x[y_i][z] * this->weights[z][y_j];
        }
        
        y[y_i][y_j] = sum;
      }
    }

    return activation_function(y);
  }

  std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradient) {
    std::vector<std::vector<double>> prev_gradient(gradient.size(), std::vector<double>(height, 0.0));
    for (size_t i = 0; i < gradient.size(); ++i) {
        for (size_t j = 0; j < height; ++j) {
            for (size_t k = 0; k < width; ++k) {
                prev_gradient[i][j] += gradient[i][k] * weights[j][k];
            }
        }
    }
    return prev_gradient;
}
};
