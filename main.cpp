#include <iostream>
#include <cstdint>
#include "layers/FullyConnected.cpp"
#include "utils/Exception.cpp"
#include "utils/activation_functions.cpp"
#include "utils/output_functions.cpp"
#include "utils/utils.cpp"
#include "utils/MNISTReader.cpp"
#include "utils/loss_functions.cpp"

// Add these new functions for backpropagation
std::vector<std::vector<double>> softmax_derivative(const std::vector<std::vector<double>>& output, const std::vector<uint8_t>& labels) {
    std::vector<std::vector<double>> gradient = output;
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i][labels[i]] -= 1.0;
    }
    return gradient;
}

std::vector<std::vector<double>> relu_derivative(const std::vector<std::vector<double>>& input) {
    std::vector<std::vector<double>> gradient = input;
    for (auto& row : gradient) {
        for (auto& val : row) {
            val = (val > 0) ? 1.0 : 0.0;
        }
    }
    return gradient;
}

void update_weights(FullyConnectedLayer& layer, const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& gradient, double learning_rate) {
    for (size_t i = 0; i < layer.weights.size(); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
            double update = 0.0;
            for (size_t k = 0; k < input.size(); ++k) {
                update += input[k][i] * gradient[k][j];
            }
            layer.weights[i][j] -= learning_rate * update / input.size();
        }
    }
}

int main() {
  int batch_index = 0;
  int batch_size = 16;
  int num_epochs = 10;
  double learning_rate = 0.01;
  std::string filename_images = "/home/ichim/Downloads/mnist/train-images.idx3-ubyte";
  std::string filename_labels = "/home/ichim/Downloads/mnist/train-labels.idx1-ubyte";
 
  // std::vector<std::vector<double>> inputs(batch_size, std::vector<double>(784, 0));
  // inputs[0] = MNISTReader::readSingleImage("/home/ichim/Downloads/mnist/train-images.idx3-ubyte", 0);
  // std::cout << inputs.size() << " " << inputs[0].size() << "\n";
  // auto single_label = MNISTReader::readSingleLabel("/home/ichim/Downloads/mnist/train-labels.idx1-ubyte", 0);
  // std::cout << single_label << "\n";
  
  // std::vector<std::vector<double>> input_images = MNISTReader::readBatchImages(filename_images, batch_size, batch_index);
  // std::vector<uint8_t> input_labels = MNISTReader::readBatchLabels(filename_labels, batch_size, batch_index);
  // std::cout << "input labels = " << input_labels.size() << "\n";
 
  int input_shape = 784;
  int number_of_classes = 10;

  FullyConnectedLayer fc1 = FullyConnectedLayer(input_shape, 300, ActivationFunctions::ReLU);
  FullyConnectedLayer fc2 = FullyConnectedLayer(300, 100, ActivationFunctions::ReLU);
  FullyConnectedLayer fc3 = FullyConnectedLayer(100, number_of_classes, OutputFunctions::softmax);

  // std::vector<std::vector<double>> x(3, std::vector<double>(4, 0));
  // for(int i = 0; i < 3; i++) {
  //   for(int j = 0; j < 4; j++) {
  //     x[i][j] = (i + j + 3) / 12.0;
  //   }
  // }

  // std::cout << x.size() << "\n";
  try {
    // auto train_images = MNISTReader::readImages("~/Downloads/mnist/train-images.idx3-ubyte");
    // auto train_labels = MNISTReader::readLabels("~/Downloads/mnist/train-labels.idx1-ubyte");
    // // Read MNIST test images
    // auto test_images = MNISTReader::readImages("~/Downloads/mnist/t10k-images.idx3-ubyte");
    // auto test_labels = MNISTReader::readLabels("~/Downloads/mnist/t10k-labels.idx1-ubyte");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      double total_loss = 0.0;
      int num_batches = 60000 / batch_size;  // Total MNIST training images / batch_size

      for (int batch_index = 0; batch_index < num_batches; ++batch_index) {
        std::vector<std::vector<double>> input_images = MNISTReader::readBatchImages(filename_images, batch_size, batch_index);
        std::vector<uint8_t> input_labels = MNISTReader::readBatchLabels(filename_labels, batch_size, batch_index);

        // Forward pass
        auto h1 = fc1.forward(input_images);
        auto h2 = fc2.forward(h1);
        auto output_probabilities = fc3.forward(h2);

        // Compute loss
        double loss = crossEntropyLoss(output_probabilities, input_labels);
        total_loss += loss;

        // Backward pass
        auto d_output = softmax_derivative(output_probabilities, input_labels);
        auto d_h2 = fc3.backward(d_output);
        auto d_h1 = fc2.backward(d_h2);
        fc1.backward(d_h1);

        // Update weights
        update_weights(fc3, h2, d_output, learning_rate);
        update_weights(fc2, h1, d_h2, learning_rate);
        update_weights(fc1, input_images, d_h1, learning_rate);
      }

      std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << ", Average Loss: " << total_loss / num_batches << std::endl;
    }

    std::cout << "Training completed." << std::endl;
  }
  catch (Exception e) {
    std::cout << e.get_message() << "\n";
  }

  return 0;
}
