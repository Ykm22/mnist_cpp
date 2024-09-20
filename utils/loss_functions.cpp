#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

double crossEntropyLoss(const std::vector<std::vector<double>>& predictions, 
                        const std::vector<uint8_t>& labels) {
    if (predictions.size() != labels.size()) {
    std::cout << predictions.size() << " " << labels.size() << "\n";
        throw std::invalid_argument("Number of predictions must match number of labels");
    }

    double loss = 0.0;
    const int numClasses = predictions[0].size();
    const int numSamples = predictions.size();

    for (size_t i = 0; i < numSamples; ++i) {
        if (labels[i] >= numClasses) {
            throw std::invalid_argument("Invalid label encountered");
        }

        // Add a small epsilon to avoid log(0)
        const double epsilon = 1e-15;
        loss -= std::log(std::max(predictions[i][labels[i]], epsilon));
    }

    return loss / numSamples;
}
