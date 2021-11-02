#include "test.hpp"
#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Logger.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <rapidcsv.h>
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

static std::vector<double> prepare_training_data(std::string_view filename, double min, double max);
static void write_result(std::ostream &stream, eoneural::Network &net, std::span<double> test_data);

eoneural::TrainResult run_test(const TestConfig &cfg) {
   auto output_count = 1;

   double min = -300, max = 0;
   if (cfg.type == DatasetType::Cube) {
      min = -1253;
      max = 475;
   }

   auto train_data = prepare_training_data(cfg.train_dataset_filename(), min, max);
   auto test_data = prepare_training_data(cfg.test_dataset_filename(), min, max);

   eoneural::RandomWeightGenerator weight_gen(cfg.seed);

   auto builder = eoneural::NetworkBuilder().with_activation_func(cfg.func).with_input_count(1);
   for (auto neuron_count : cfg.hidden_layers) {
      builder.with_layer(neuron_count);
   }

   auto net = builder.with_layer(output_count).with_weight_generator(weight_gen).build();

   std::ofstream objective_log(cfg.objective_log_filename());

   eoneural::MSEObjective<CSVLogger> objective(cfg.target, CSVLogger(objective_log));

   std::ofstream training_log(cfg.train_log_filename());
   eoneural::TrainLogger train_logger(net, training_log);

   auto result = net.batch_train(objective, train_data, 10, 0.8, 0.1, cfg.epoch_limit);

   std::ofstream points_train(cfg.train_classification_filename());
   write_result(points_train, net, train_data);
   std::ofstream points_test(cfg.test_classification_filename());
   write_result(points_test, net, test_data);

   net.save(cfg.network_filename());
   return result;
}

void write_train_pass(std::ostream &os, const eoneural::Network &net, std::span<double> test_data) {
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