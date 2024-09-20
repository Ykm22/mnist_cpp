#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>

class MNISTReader {
private:
  class FileHandler {
  private:
    std::ifstream file;
    int32_t num_items, num_rows, num_cols;
    long data_start;
    bool is_image_file;

    void readHeader() {
      int32_t magic_number;
      file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
      file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
      
      magic_number = __builtin_bswap32(magic_number);
      num_items = __builtin_bswap32(num_items);

      if (magic_number == 2051) {
        is_image_file = true;
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);
      } else if (magic_number == 2049) {
        is_image_file = false;
        num_rows = num_cols = 1;
      } else {
        throw std::runtime_error("Invalid MNIST file format");
      }

      data_start = file.tellg();
    }

  public:
    FileHandler(const std::string& filename) : file(filename, std::ios::binary) {
      if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
      }
      readHeader();
    }

    std::vector<double> readItem(int index) {
      if (index < 0 || index >= num_items) {
        throw std::out_of_range("Item index out of range");
      }

      long pos = data_start + index * num_rows * num_cols;
      file.seekg(pos);

      std::vector<double> item(num_rows * num_cols);
      for (int j = 0; j < num_rows * num_cols; ++j) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        item[j] = is_image_file ? static_cast<double>(pixel) / 255.0 : static_cast<double>(pixel);
      }

      return item;
    }

    int getNumItems() const { return num_items; }
    int getNumRows() const { return num_rows; }
    int getNumCols() const { return num_cols; }
    bool isImageFile() const { return is_image_file; }
  };

public:
  static std::vector<std::vector<double>> readImages(const std::string& filename) {
    FileHandler handler(filename);
    std::vector<std::vector<double>> images;
    images.reserve(handler.getNumItems());
    for (int i = 0; i < handler.getNumItems(); ++i) {
      images.push_back(handler.readItem(i));
    }
    return images;
  }

  static std::vector<uint8_t> readLabels(const std::string& filename) {
    FileHandler handler(filename);
    std::vector<uint8_t> labels(handler.getNumItems());
    for (int i = 0; i < handler.getNumItems(); ++i) {
      labels[i] = static_cast<uint8_t>(handler.readItem(i)[0]);
    }
    return labels;
  }

  static std::vector<double> readSingleImage(const std::string& filename, int index) {
    FileHandler handler(filename);
    if (!handler.isImageFile()) {
      throw std::runtime_error("File is not an image file");
    }
    return handler.readItem(index);
  }

  static uint8_t readSingleLabel(const std::string& filename, int index) {
    FileHandler handler(filename);
    if (handler.isImageFile()) {
      throw std::runtime_error("File is not a label file");
    }
    return static_cast<uint8_t>(handler.readItem(index)[0]);
  }

  static std::vector<std::vector<double>> readBatchImages(const std::string& filename, int batch_size, int batch_index) {
    FileHandler handler(filename);
    if (!handler.isImageFile()) {
      throw std::runtime_error("File is not an image file");
    }
    std::vector<std::vector<double>> batch_images;
    batch_images.reserve(batch_size);
    int start_of_batch = batch_size * batch_index;
    for (int i = start_of_batch; i < start_of_batch + batch_size && i < handler.getNumItems(); ++i) {
      batch_images.push_back(handler.readItem(i));
    }
    return batch_images;
  }

  static std::vector<uint8_t> readBatchLabels(const std::string& filename, int batch_size, int batch_index) {
    FileHandler handler(filename);
    // if (!handler.isImageFile()) {
    //   throw std::runtime_error("File is not an image file");
    // }

    std::vector<uint8_t> batch_labels;
    batch_labels.reserve(batch_size);
    int start_of_batch = batch_size * batch_index;
    for (int i = start_of_batch; i < start_of_batch + batch_size && i < handler.getNumItems(); ++i) {
      batch_labels.push_back(static_cast<uint8_t>(handler.readItem(i)[0]));
    }
    return batch_labels;
  }
};
