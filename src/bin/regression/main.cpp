#include "test.hpp"
#include <array>
#include <fmt/color.h>
#include <fmt/core.h>

static std::vector<TestConfig> prepare_config(std::uint32_t seed);

auto main() -> int {
   auto test_cases = prepare_config(737);

   int reached_target_count = 0;
   int i = 1;
   for (auto &test_case : test_cases) {
      fmt::print(fmt::emphasis::bold, "RUNNING TRAINING {}/{}\n", i, test_cases.size());
      fmt::print("observation count: {}\n", test_case.observation_count);
      fmt::print("dataset type: {}\n", dataset_type_to_string(test_case.type));
      fmt::print("target: {}\n", test_case.target);
      fmt::print("activation function: {}\n", eoneural::activation_func_to_string(test_case.func));
      fmt::print("hidden layers: {}\n", format_hidden_layers(test_case.hidden_layers));
      fmt::print("epoch limit: {}\n", test_case.epoch_limit);
      auto result = run_test(test_case);
      ++i;
      if (result.reached_objective) {
         ++reached_target_count;
         fmt::print(fmt::fg(fmt::color::green), "\n[OK]\n");
      }
      else
         fmt::print(fmt::fg(fmt::color::red), "\n[FAIL]\n");
   }

   fmt::print(fmt::emphasis::bold, "\nDONE {}/{} networks reached objective\n", reached_target_count, test_cases.size());
}

static double choose_target(DatasetType type, eoneural::ActivationFunc func, int hidden_layer_count) {
   return 0.0001;
}

std::array<std::vector<int>, 5> g_hidden_layers{
        std::vector<int>{},
        std::vector<int>{6},
        std::vector<int>{6, 3},
        std::vector<int>{6, 3, 3},
        std::vector<int>{6, 3, 3, 3},
};

static std::vector<TestConfig> prepare_config(std::uint32_t seed) {
   std::vector<TestConfig> result;
   for (auto type : {DatasetType::Activation, DatasetType::Cube}) {
      for (auto observation_count : {100, 500, 1000, 10000}) {
         for (auto func : {eoneural::ActivationFunc::Sigmoid, eoneural::ActivationFunc::Tanh, eoneural::ActivationFunc::Atan, eoneural::ActivationFunc::Gaussian})
            for (const auto& hidden_layers : g_hidden_layers) {
               result.push_back(TestConfig{
                       .observation_count = observation_count,
                       .type = type,
                       .seed = seed,
                       .func = func,
                       .hidden_layers = hidden_layers,
                       .target = choose_target(type, func, hidden_layers.size()),
                       .epoch_limit = static_cast<int>((20000 / observation_count) * (hidden_layers.size() + 1)),
               });
            }
      }
   }
   return result;
}
