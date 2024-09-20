#ifndef EXCEPTION_CPP
#define EXCEPTION_CPP

#include <string>

class Exception {
private:
  std::string message;
public:
  Exception(const std::string& message) {
    this->message = message;
  }
  const std::string& get_message() {
    return this->message;
  }
};

#endif
