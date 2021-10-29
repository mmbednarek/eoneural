#include <eoneural/hpp/Network.hpp>
#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Logger.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <rapidcsv.h>
#include <fstream>
#include <fmt/core.h>

class CSVLogger {
   eoneural::CSVWriter m_writer;
   std::size_t m_epoch_id = 0;

 public:
   explicit CSVLogger(std::ostream &stream) : m_writer(stream) {
      m_writer.write("epoch", "mse");
   }

   inline void log_mse(double mse) {
      fmt::print("{}\r", mse);
      fflush(stdout);
      m_writer.write(m_epoch_id, mse);
      ++m_epoch_id;
   }
};

std::vector<double> prepare_training_data(std::string_view filename, double min, double max) {
   rapidcsv::Document doc(filename.data());
   auto col_count = doc.GetColumnCount();
   auto row_count = doc.GetRowCount();
   std::vector<double> result(col_count * row_count);

   auto it = result.begin();
   for (std::size_t i = 0; i < row_count; ++i) {
      auto row = doc.GetRow<double>(i);
      *it = row[0];
      *(it + 1) = (row[1] - min) / (max - min);
      it += col_count;
   }
   return result;
}

static void write_result(std::ostream &stream, eoneural::Network &net, std::span<double> test_data) {
   eoneural::CSVWriter writer(stream);
   writer.write("x", "y", "ey");
   auto test_it = test_data.begin();
   while (test_it != test_data.end()) {
      auto x = *test_it;
      auto exp_y = *(test_it + 1);
      double y;
      net.pass(&x, &y);
      writer.write(x, y, exp_y);
      test_it += 2;
   }
}

auto main() -> int {
   eoneural::RandomWeightGenerator weight_gen(800);

   auto net = eoneural::NetworkBuilder()
                      .with_activation_func(eoneural::ActivationFunc::Sigmoid)
                      .with_input_count(1)
                      .with_layer(10)
                      .with_layer(10)
                      .with_layer(10)
                      .with_layer(1)
                      .with_weight_generator(weight_gen)
                      .build();

    auto training_data = prepare_training_data("data.multimodal.train.100.csv", -380, 102);
    auto test_data = prepare_training_data("data.multimodal.test.100.csv", -380, 102);

   eoneural::BasicContext ctx(net, training_data, 0.8, 0.1);
   eoneural::MSEObjective<CSVLogger> objective(0.0001, CSVLogger(std::cout));
   auto result = net.train(ctx, objective);

   fmt::print("\nresult: {}", result.reached_objective);

   std::ofstream os_test("result_test.csv");
   write_result(os_test, net, test_data);
   std::ofstream os_train("result_train.csv");
   write_result(os_train, net, training_data);
}