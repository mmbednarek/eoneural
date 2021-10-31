#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Batch.hpp>
#include <eoneural/hpp/Network.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <gtest/gtest.h>
#define SPDLOG_FMT_EXTERNAL
#include <spdlog/spdlog.h>

class SpdMseLog {
 public:
   inline void log_mse(double mse) const {
      spdlog::info("MSE: {}", mse);
   }
};

class SpdCatLog {
 public:
   inline void log_categorisation(double mse, double train, double test) const {
      spdlog::info("MSE: {}, Train: {}, Test: {}", mse, train, test);
   }
};

TEST(TrainC, BatchTrainXor) {
   srand(69);
   std::array<int, 2> hidden_layers{2, 1};

   auto net = network_create(2, 2, hidden_layers.data(), 1);

   std::array<double, 2> input0{0, 0};
   std::array<double, 1> output0{0};
   std::array<double, 2> input1{0, 1};
   std::array<double, 1> output1{1};
   std::array<double, 2> input2{1, 0};
   std::array<double, 1> output2{1};
   std::array<double, 2> input3{1, 1};
   std::array<double, 1> output3{0};

   auto batch = batch_create(net);

   for (size_t i = 0; i < 100000; ++i) {
      batch_begin(batch, 0.3);

      batch_put(batch, input0.data(), output0.data(), 0.8);
      batch_put(batch, input3.data(), output3.data(), 0.8);
      batch_put(batch, input2.data(), output2.data(), 0.8);
      batch_put(batch, input1.data(), output1.data(), 0.8);

      batch_end(batch);
   }

   batch_destroy(batch);

   fmt::print("0 0: {}\n", *network_perform(net, input0.data()));
   fmt::print("0 1: {}\n", *network_perform(net, input1.data()));
   fmt::print("1 0: {}\n", *network_perform(net, input2.data()));
   fmt::print("1 1: {}\n", *network_perform(net, input3.data()));

   network_destroy(net);
}

TEST(Train, BatchTrainXor) {
   eoneural::RandomWeightGenerator weight_gen(345345);

   auto net = eoneural::NetworkBuilder()
                      .with_activation_func(eoneural::ActivationFunc::Sigmoid)
                      .with_input_count(2)
                      .with_layer(2)
                      .with_layer(1)
                      .with_weight_generator(weight_gen)
                      .build();

   std::array<double, 2> input0{0, 0};
   std::array<double, 1> output0{0};
   std::array<double, 2> input1{0, 1};
   std::array<double, 1> output1{1};
   std::array<double, 2> input2{1, 0};
   std::array<double, 1> output2{1};
   std::array<double, 2> input3{1, 1};
   std::array<double, 1> output3{0};

   eoneural::BatchTrain train(net);

   for (size_t i = 0; i < 100000; ++i) {
      eoneural::Batch b(train, 0.2);
      b.put(input0.data(), output0.data(), 0.8);
      b.put(input1.data(), output1.data(), 0.8);
      b.put(input2.data(), output2.data(), 0.8);
      b.put(input3.data(), output3.data(), 0.8);
   }

   auto do_xor = [&net](double x, double y) -> double {
      std::array<double, 2> val{x, y};
      double out;
      net.pass(val.begin(), &out);
      return out;
   };

   fmt::print("0 0: {}\n", do_xor(0, 0));
   fmt::print("0 1: {}\n", do_xor(0, 1));
   fmt::print("1 0: {}\n", do_xor(1, 0));
   fmt::print("1 1: {}\n", do_xor(1, 1));
}

TEST(Train, BatchTrainXor2) {
   eoneural::RandomWeightGenerator weight_gen(345345);

   auto net = eoneural::NetworkBuilder()
                      .with_activation_func(eoneural::ActivationFunc::Sigmoid)
                      .with_input_count(2)
                      .with_layer(2)
                      .with_layer(1)
                      .with_weight_generator(weight_gen)
                      .build();

   std::array<double, 12> train_data{
           0,
           0,
           /* -> */ 0,
           0,
           1,
           /* -> */ 1,
           1,
           0,
           /* -> */ 1,
           1,
           1,
           /* -> */ 0,
   };

   eoneural::MSEObjective<SpdMseLog> objective(0.001);
   net.batch_train(objective, train_data, 4, 0.8, 0.1);

   auto do_xor = [&net](double x, double y) -> double {
      std::array<double, 2> val{x, y};
      double out;
      net.pass(val.begin(), &out);
      return out;
   };

   fmt::print("0 0: {}\n", do_xor(0, 0));
   fmt::print("0 1: {}\n", do_xor(0, 1));
   fmt::print("1 0: {}\n", do_xor(1, 0));
   fmt::print("1 1: {}\n", do_xor(1, 1));
}

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
           0,
           0,
           /* -> */ 0,
           0,
           1,
           /* -> */ 1,
           1,
           0,
           /* -> */ 1,
           1,
           1,
           /* -> */ 0,
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
           0.25,
           0.3,
           /* -> */ 1,
           0,
           0,
           0.9,
           /* -> */ 1,
           0,
           0.1,
           0.7,
           /* -> */ 1,
           0,
           0.8,
           0.2,
           /* -> */ 0,
           1,
           0.3,
           0.1,
           /* -> */ 0,
           1,
           0.5,
           0.3,
           /* -> */ 0,
           1,
   };

   std::array<double, 24> test_data{
           0.3,
           0.5,
           /* -> */ 1,
           0,
           0,
           0.5,
           /* -> */ 1,
           0,
           0.3,
           0.8,
           /* -> */ 1,
           0,
           0.8,
           0.3,
           /* -> */ 0,
           1,
           0.5,
           0.1,
           /* -> */ 0,
           1,
           0.8,
           0.5,
           /* -> */ 0,
           1,
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
           0.25,
           0.3,
           /* -> */ 0,
           0,
           1,
           0,
           0.9,
           /* -> */ 0,
           0,
           1,
           0.1,
           0.7,
           /* -> */ 0,
           0,
           1,
           0.8,
           0.2,
           /* -> */ 0,
           1,
           0,
           0.3,
           0.1,
           /* -> */ 0,
           1,
           0,
           0.5,
           0.3,
           /* -> */ 0,
           1,
           0,
   };

   std::array<double, 30> test_data{
           0.3,
           0.5,
           /* -> */ 0,
           0,
           1,
           0,
           0.5,
           /* -> */ 0,
           0,
           1,
           0.3,
           0.8,
           /* -> */ 0,
           0,
           1,
           0.8,
           0.3,
           /* -> */ 0,
           1,
           0,
           0.5,
           0.1,
           /* -> */ 0,
           1,
           0,
           0.8,
           0.5,
           /* -> */ 0,
           1,
           0,
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
