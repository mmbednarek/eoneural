#ifndef EONEURAL_OBJECTIVE_HPP
#define EONEURAL_OBJECTIVE_HPP
#include "Network.hpp"
#include <cassert>

namespace eoneural {

template<typename T>
concept MSELog = requires(T v) {
   {v.log_mse(0.0)};
};

template<MSELog TL>
class MSEObjective {
   double m_mse = 0.01;
   TL m_log;

 public:
   constexpr explicit MSEObjective(double mse, TL log = TL()) : m_mse(mse), m_log(std::move(log)) {}

   [[nodiscard]] constexpr bool has_reached_objective(const Network &net, const TrainResult &res) {
      m_log.log_mse(res.mse);
      return res.mse <= m_mse;
   }
};

constexpr double test_categorisation(const Network &net, std::span<double> test_data) {
   auto correct_category_count = 0;
   auto category_count = 0;
   for (auto input_it = test_data.begin(); input_it != test_data.end(); input_it += net.output_count()) {
      auto cat = net.pass_category(input_it);
      input_it += net.input_count();

      auto expected_cat = find_category(input_it, input_it + net.output_count());

      if (cat == expected_cat)
         ++correct_category_count;

      ++category_count;
   }

   return static_cast<double>(correct_category_count) / static_cast<double>(category_count);
}

template<typename T>
concept CategorisationLog = requires(T v) {
   {v.log_categorisation(0.0, 0.0, 0.0)};
};

template<CategorisationLog TL>
class ClassificationObjective {
   std::span<double> m_training;
   std::span<double> m_test;
   double m_target;
   TL m_logger;

 public:
   inline ClassificationObjective(std::span<double> training, std::span<double> test, double target, TL logger = TL()) : m_training(training),
                                                                                                                         m_test(test),
                                                                                                                         m_target(target),
                                                                                                                         m_logger(std::move(logger)) {}


   [[nodiscard]] constexpr bool has_reached_objective(const Network &net, const TrainResult &res) {
      auto train_level = test_categorisation(net, m_training);
      auto test_level = test_categorisation(net, m_test);

      m_logger.log_categorisation(res.mse, train_level, test_level);
      return test_level >= m_target;
   }
};

}// namespace eoneural

#endif//EONEURAL_OBJECTIVE_HPP
