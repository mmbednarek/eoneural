#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Network.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <eoneural/hpp/Logger.hpp>
#include <rapidcsv.h>
#include <string_view>
#include <fmt/core.h>

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

class CSVLogger {
   eoneural::CSVWriter m_writer;
   std::size_t m_iteration_id = 0;

 public:
   explicit CSVLogger(std::ostream &stream) : m_writer(stream) {
      m_writer.write("iteration", "mse", "train", "test");
   }

   constexpr void log_categorisation(double mse, double train, double test) {
      m_writer.write(m_iteration_id, mse, train, test);
      ++m_iteration_id;
   }
};

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

static inline std::string make_input_filename(bool simple, std::string_view tp, int observation_count) {
   return fmt::format("data.{}.{}.{}.csv", simple ? "simple" : "three_gauss", tp, observation_count);
}

static inline std::string make_output_filename(bool simple, int observation_count) {
   return fmt::format("result/log.{}.{}.csv", simple ? "simple" : "three_gauss", observation_count);
}

static inline std::string make_training_filename(bool simple, int observation_count) {
   return fmt::format("result/training.{}.{}.csv", simple ? "simple" : "three_gauss", observation_count);
}

static inline std::string make_network_filename(bool simple, int observation_count) {
   return fmt::format("network/trained.{}.{}.net", simple ? "simple" : "three_gauss", observation_count);
}


static inline std::string make_points_filename(bool simple, std::string_view kind, int observation_count) {
   return fmt::format("result/points.{}.{}.{}.csv", simple ? "simple" : "three_gauss", kind, observation_count);
}

void run_test(int observation_count, bool simple, bool mix, eoneural::RandomWeightGenerator::SeedType seed, double target, std::ostream &log_output) {
   auto train_filename = make_input_filename(simple, "train", observation_count);
   auto test_filename = make_input_filename(simple, "test", observation_count);

   const auto output_count = simple ? 2 : 3;

   std::vector<double> train_data;
   if (mix) {
      train_data = prepare_and_mix_training_data(train_filename, output_count);
   } else {
      train_data = prepare_training_data(train_filename, output_count);
   }
   auto test_data = prepare_training_data(test_filename, output_count);

   eoneural::RandomWeightGenerator weight_gen(seed);

   auto net = eoneural::NetworkBuilder()
           .with_activation_func(eoneural::ActivationFunc::Sigmoid)
           .with_input_count(2)
           .with_layer(2)
           .with_layer(output_count)
           .with_weight_generator(weight_gen)
           .build();

   eoneural::BasicContext ctx(net, train_data, 0.8, 0.1);
   eoneural::ClassificationObjective<CSVLogger> objective(train_data, test_data, target, CSVLogger(log_output));

   std::ofstream training_fs(make_training_filename(simple, observation_count));
   eoneural::TrainLogger train_logger(net, training_fs);

   net.train(ctx, objective, train_logger);

   std::ofstream points_train(make_points_filename(simple, "train", observation_count));
   write_train_pass(points_train, net, train_data);
   std::ofstream points_test(make_points_filename(simple, "test", observation_count));
   write_train_pass(points_test, net, test_data);

   net.save(make_network_filename(simple, observation_count));
}

void run_test(int observation_count, bool simple, bool mix, eoneural::RandomWeightGenerator::SeedType seed, double target) {
   std::ofstream fs(make_output_filename(simple, observation_count));
   run_test(observation_count, simple, mix, seed, target, fs);
}

void run_test_stdout(int observation_count, bool simple, bool mix, eoneural::RandomWeightGenerator::SeedType seed, double target) {
   run_test(observation_count, simple, mix, seed, target, std::cout);
}

auto main() -> int {
   run_test(100, true, false, 737, 0.99);
   run_test(500, true, false, 737, 0.99);
   run_test(1000, true, false, 737, 0.99);
   run_test(10000, true, false, 737, 0.99);

   run_test(100, false, true, 737, 0.93);
   run_test(500, false, true, 737, 0.94);
   run_test(1000, false, true, 888, 0.93);
   run_test(10000, false, true, 737, 0.93);
}