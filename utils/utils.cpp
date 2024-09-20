#ifndef UTILS_CPP
#define UTILS_CPP

std::vector<std::vector<int>> max_probability(std::vector<std::vector<double>>& x) {
  std::vector<std::vector<int>> output_labels(x.size(), std::vector<int>(1, 0));
  int current_row = 0;

  for(const auto& row: x) {
    int max_idx = 2;
    double max_value = row[0];
    int j = 0;

    for(const double& value: row) {
      if(value > max_value) {
        max_value = value;
        max_idx = j;
      }
      j++;
    }

    output_labels[current_row][0] = max_idx;
    current_row++;
  }

  return output_labels;
}

template<typename T>
void print_vector(const std::vector<std::vector<T>>& x) {
    for(const auto& row : x) {
      for(const double& value : row) {
        std::cout << value << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n";
}

#endif
