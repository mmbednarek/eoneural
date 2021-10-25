#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Network.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <gtest/gtest.h>
#define SPDLOG_FMT_EXTERNAL
#include <spdlog/spdlog.h>

class SpdMseLog {
 public:
   constexpr void log_mse(double mse) const {
      spdlog::info("MSE: {}", mse);
   }
};

class SpdCatLog {
 public:
   constexpr void log_categorisation(double mse, double train, double test) const {
      spdlog::info("MSE: {}, Train: {}, Test: {}", mse, train, test);
   }
};

TEST(Train, Xor) {
   eoneural::RandomWeightGenerator weight_gen(69);

   auto net = eoneural::NetworkBuilder()
                      .with_activation_func(eoneural::ActivationFunc::Sigmoid)
                      .with_input_count(2)
                      .with_layer(2)
                      .with_layer(1)
                      .with_weight_generator(weight_gen)
                      .build();

   std::array<double, 12> train_data{
           0, 0, /* -> */ 0,
           0, 1, /* -> */ 1,
           1, 0, /* -> */ 1,
           1, 1, /* -> */ 0,
   };

   eoneural::BasicContext ctx(net, train_data, 0.8, 0.1);
   eoneural::MSEObjective<SpdMseLog> objective(0.001);
   net.train(ctx, objective);

   auto do_xor = [&net](double x, double y) -> double {
      std::array<double, 2> val{x, y};
      double out;
      net.pass(val.begin(), &out);
      return out;
   };

   ASSERT_LE(do_xor(0, 0), 0.1);
   ASSERT_GE(do_xor(0, 1), 0.9);
   ASSERT_GE(do_xor(1, 0), 0.9);
   ASSERT_LE(do_xor(1, 1), 0.1);
}

TEST(Train, CategTrain) {
   std::array<double, 24> train_data{
           0.25, 0.3, /* -> */ 1, 0,
           0,    0.9, /* -> */ 1, 0,
           0.1,  0.7, /* -> */ 1, 0,
           0.8,  0.2, /* -> */ 0, 1,
           0.3,  0.1, /* -> */ 0, 1,
           0.5,  0.3, /* -> */ 0, 1,
   };

   std::array<double, 24> test_data{
           0.3,  0.5, /* -> */ 1, 0,
           0,    0.5, /* -> */ 1, 0,
           0.3,  0.8, /* -> */ 1, 0,
           0.8,  0.3, /* -> */ 0, 1,
           0.5,  0.1, /* -> */ 0, 1,
           0.8,  0.5, /* -> */ 0, 1,
      };

   eoneural::RandomWeightGenerator weight_gen(69);

   auto net = eoneural::NetworkBuilder()
           .with_activation_func(eoneural::ActivationFunc::Sigmoid)
           .with_input_count(2)
           .with_layer(2)
           .with_layer(2)
           .with_weight_generator(weight_gen)
           .build();

   eoneural::BasicContext ctx(net, train_data, 0.8, 0.1);
   eoneural::ClassificationObjective<SpdCatLog> objective(train_data, test_data, 0.9);
   net.train(ctx, objective);
}

TEST(Train, CategTrain3) {
   std::array<double, 30> train_data{
           0.25, 0.3, /* -> */ 0, 0, 1,
           0,    0.9, /* -> */ 0, 0, 1,
           0.1,  0.7, /* -> */ 0, 0, 1,
           0.8,  0.2, /* -> */ 0, 1, 0,
           0.3,  0.1, /* -> */ 0, 1, 0,
           0.5,  0.3, /* -> */ 0, 1, 0,
   };

   std::array<double, 30> test_data{
           0.3,  0.5, /* -> */ 0, 0, 1,
           0,    0.5, /* -> */ 0, 0, 1,
           0.3,  0.8, /* -> */ 0, 0, 1,
           0.8,  0.3, /* -> */ 0, 1, 0,
           0.5,  0.1, /* -> */ 0, 1, 0,
           0.8,  0.5, /* -> */ 0, 1, 0,
   };

   eoneural::RandomWeightGenerator weight_gen(69);

   auto net = eoneural::NetworkBuilder()
           .with_activation_func(eoneural::ActivationFunc::Sigmoid)
           .with_input_count(2)
           .with_layer(2)
           .with_layer(3)
           .with_weight_generator(weight_gen)
           .build();

   eoneural::BasicContext ctx(net, train_data, 0.8, 0.1);
   eoneural::ClassificationObjective<SpdCatLog> objective(train_data, test_data, 0.9);
   net.train(ctx, objective);
}
