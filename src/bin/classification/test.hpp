#ifndef EONEURAL_TEST_HPP
#define EONEURAL_TEST_HPP
#include <cstdint>
#include <eoneural/hpp/Network.hpp>
#include <fmt/core.h>
#include <string_view>
#include <vector>
#include <sstream>

enum class DatasetType {
   Simple = 0,
   ThreeGauss = 1,
};

constexpr std::string_view dataset_type_to_string(DatasetType kind) {
   switch (kind) {
   case DatasetType::Simple:
      return "simple";
   case DatasetType::ThreeGauss:
      return "three_gauss";
   }
   return "";
}

constexpr int dataset_type_to_output_count(DatasetType kind) {
   switch (kind) {
   case DatasetType::Simple:
      return 2;
   case DatasetType::ThreeGauss:
      return 3;
   }
   return 1;
}

inline std::string format_hidden_layers(const std::vector<int> &values) {
   if (values.empty())
      return "none";

   std::stringstream ss;
   auto it = values.begin();
   ss << *it;
   ++it;
   while (it != values.end()) {
      ss << '-' << *it;
      ++it;
   }

   return ss.str();
}

struct TestConfig {
   int observation_count = 100;
   DatasetType type = DatasetType::Simple;
   std::uint32_t seed = 737;
   eoneural::ActivationFunc func = eoneural::ActivationFunc::Sigmoid;
   std::vector<int> hidden_layers{};
   double target = 0.99;
   int epoch_limit = -1;

   [[nodiscard]] inline std::string train_dataset_filename() const {
      return fmt::format("data.{}.train.{}.csv", dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string test_dataset_filename() const {
      return fmt::format("data.{}.test.{}.csv", dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string objective_log_filename() const {
      return fmt::format("result/objective/{}.{}.{}.{}.csv", eoneural::activation_func_to_string(func), format_hidden_layers(hidden_layers), dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string train_log_filename() const {
      return fmt::format("result/training/{}.{}.{}.{}.csv", eoneural::activation_func_to_string(func), format_hidden_layers(hidden_layers), dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string network_filename() const {
      return fmt::format("result/network/{}.{}.{}.{}.net", eoneural::activation_func_to_string(func), format_hidden_layers(hidden_layers), dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string test_classification_filename() const {
      return fmt::format("result/points/test.{}.{}.{}.{}.net", eoneural::activation_func_to_string(func), format_hidden_layers(hidden_layers), dataset_type_to_string(type), observation_count);
   }

   [[nodiscard]] inline std::string train_classification_filename() const {
      return fmt::format("result/points/train.{}.{}.{}.{}.net", eoneural::activation_func_to_string(func), format_hidden_layers(hidden_layers), dataset_type_to_string(type), observation_count);
   }
};

void run_test(const TestConfig &cfg);

#endif//EONEURAL_TEST_HPP
