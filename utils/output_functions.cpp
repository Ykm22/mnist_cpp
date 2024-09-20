#ifndef OUTPUT_FUNCTIONS_CPP
#define OUTPUT_FUNCTIONS_CPP 

#include <vector>
#include <cmath>

class OutputFunctions {
public:
    static std::vector<std::vector<double>> softmax(std::vector<std::vector<double>>& x) {
        for (auto& row : x) {
            // max_val
            double max_val = row[0];
            for (const double& val : row) {
                max_val = std::max(max_val, val);
            }

            // sum += exp(x - max_val)
            double sum = 0.0;
            for (double& val : row) {
                val = std::exp(val - max_val);
                sum += val;
            }

            // x / sum
            for (double& val : row) {
                val /= sum;
            }
        }
        return x;
    }
};

#endif
