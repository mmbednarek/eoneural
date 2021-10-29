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
      m_writer.write("epoch", "mse", "train", "test");
   }

   inline void log_categorisation(double mse, double train, double test) {
      fmt::print("mse: {} train: {} test: {}\r", mse, train, test);
      fflush(stdout);
      m_writer.write(m_epoch_id, mse, train, test);
      ++m_epoch_id;
   }
};

std::vector<double> prepare_training_data(std::string_view filename, int category_count) {
   rapidcsv::Document doc(filename.data());
   auto col_count = doc.GetColumnCount();
   auto row_count = doc.GetRowCount();
   std::vector<double> result(((col_count - 1) + category_count) * row_count);

   auto it = result.begin();
   for (std::size_t i = 0; i < row_count; ++i) {
      auto row = doc.GetRow<double>(i);
      std::copy(row.begin(), row.end() - 1, it);
      it += col_count - 1;
      auto cat = static_cast<int>(row[col_count - 1]) - 1;
      for (std::size_t j = 0; j < category_count; ++j) {
         if (j == cat) {
            *it = 1;
            ++it;
            continue;
         }
         *it = 0;
         ++it;
      }
   }

   return result;
}

std::vector<double> prepare_and_mix_training_data(std::string_view filename, int category_count) {
   rapidcsv::Document doc(filename.data());
   auto col_count = doc.GetColumnCount();
   auto row_count = doc.GetRowCount();
   std::vector<double> result(((col_count - 1) + category_count) * row_count);

   auto it = result.begin();
   auto single_type = row_count / category_count;
   for (auto t = 0; t < single_type; ++t) {
      for (auto c = 0; c < category_count; ++c) {
         auto i = c * single_type + t;
         auto row = doc.GetRow<double>(i);
         std::copy(row.begin(), row.end() - 1, it);
         it += col_count - 1;
         auto cat = static_cast<int>(row[col_count - 1]) - 1;
         for (std::size_t j = 0; j < category_count; ++j) {
            if (j == cat) {
               *it = 1;
               ++it;
               continue;
            }
            *it = 0;
            ++it;
         }
      }
   }

   return result;
}


void write_result(std::ostream &os, const eoneural::Network &net, std::span<double> test_data) {
   eoneural::CSVWriter writer(os);

   writer.write("x", "y", "result", "expected");

   for (auto input_it = test_data.begin(); input_it != test_data.end(); input_it += net.output_count()) {
      auto cat = net.pass_category(input_it);
      auto x = *input_it;
      auto y = *(input_it+1);
      input_it += net.input_count();

      auto expected_cat = eoneural::find_category(input_it, input_it + net.output_count());

      writer.write(x, y, cat, expected_cat);
   }
}

auto main() -> int {
   eoneural::RandomWeightGenerator weight_gen(10000);

   auto net = eoneural::NetworkBuilder()
                      .with_activation_func(eoneural::ActivationFunc::Sigmoid)
                      .with_input_count(2)
                      .with_layer(4)
                      .with_weight_generator(weight_gen)
                      .build();

    auto training_data = prepare_training_data("data.circles.train.1000.csv", 4);
    auto test_data = prepare_training_data("data.circles.test.1000.csv", 4);

   eoneural::BasicContext ctx(net, training_data, 0.9, 0.2);
   eoneural::ClassificationObjective<CSVLogger> objective(training_data, test_data, 0.92, CSVLogger(std::cout));

   std::ofstream training_log("training.csv");
   eoneural::TrainLogger train_logger(net, training_log);

   auto result = net.train(ctx, objective, train_logger, 500);

   fmt::print("\nresult: {}", result.reached_objective);

   std::ofstream os_test("result_test.csv");
   write_result(os_test, net, test_data);
   std::ofstream os_train("result_train.csv");
   write_result(os_train, net, training_data);
}