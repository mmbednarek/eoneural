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
      m_writer.write("epoch", "mse", "mae", "train", "test", "train_cross", "test_cross");
   }

   inline void log_categorisation(double mse, double mae, double train, double test, double train_c, double test_c) {
      fmt::print("mse: {} mae: {} train: {} test: {} trainc: {}, testc: {}\r", mse, mae, train, test, train_c, test_c);
      fflush(stdout);
      m_writer.write(m_epoch_id, mse, mae, train, test, train_c, test_c);
      ++m_epoch_id;
   }
};

static void write_train_pass(std::ostream &os, const eoneural::Network &net, std::span<double> test_data);
static std::vector<double> prepare_training_data(std::string_view filename, int category_count);
static std::vector<double> prepare_and_mix_training_data(std::string_view filename, int category_count);

eoneural::TrainResult run_test(const TestConfig &cfg) {
   std::vector<double> train_data;
   auto output_count = dataset_type_to_output_count(cfg.type);

   if (cfg.type == DatasetType::ThreeGauss) {
      train_data = prepare_and_mix_training_data(cfg.train_dataset_filename(), output_count);
   } else {
      train_data = prepare_training_data(cfg.train_dataset_filename(), output_count);
   }

   auto test_data = prepare_training_data(cfg.test_dataset_filename(), output_count);

   eoneural::RandomWeightGenerator weight_gen(cfg.seed);

   auto builder = eoneural::NetworkBuilder().with_activation_func(cfg.func).with_input_count(2);
   for (auto neuron_count : cfg.hidden_layers) {
      builder.with_layer(neuron_count);
   }

   auto net = builder.with_layer(output_count).with_weight_generator(weight_gen).build();

   std::ofstream objective_log(cfg.objective_log_filename());

   eoneural::ClassificationObjective<CSVLogger> objective(train_data, test_data, cfg.target, CSVLogger(objective_log));

   std::ofstream training_log(cfg.train_log_filename());
   eoneural::TrainLogger train_logger(net, training_log);

   auto result = net.batch_train(objective, train_data, 2, 0.8, 0.2, cfg.epoch_limit);

   std::ofstream points_train(cfg.train_classification_filename());
   write_train_pass(points_train, net, train_data);
   std::ofstream points_test(cfg.test_classification_filename());
   write_train_pass(points_test, net, test_data);

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
