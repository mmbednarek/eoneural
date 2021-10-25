#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Network.hpp>
#include <eoneural/hpp/Objective.hpp>
#include <eoneural/hpp/Util.hpp>
#include <fmt/core.h>
#include <rapidcsv.h>
#include <string_view>

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

class CSVLogger {
   eoneural::CSVWriter m_writer;
   std::size_t m_iteration_id = 0;

 public:
   explicit CSVLogger(std::ostream &stream) : m_writer(stream) {
      m_writer.write("iteration", "mse");
   }

   constexpr void log_mse(double mse) {
      m_writer.write(m_iteration_id, mse);
      ++m_iteration_id;
   }
};

static inline std::string make_input_filename(std::string_view kind, std::string_view train, int observation_count) {
   return fmt::format("data.{}.{}.{}.csv", kind, train, observation_count);
}

static inline std::string make_log_filename(std::string_view kind, int observation_count) {
   return fmt::format("result/train.{}.{}.csv", kind, observation_count);
}

static inline std::string make_points_filename(std::string_view kind, std::string_view train, int observation_count) {
   return fmt::format("result/points.{}.{}.{}.csv", kind, train, observation_count);
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

eoneural::Network get_network(std::string_view kind, eoneural::RandomWeightGenerator::SeedType seed) {
   eoneural::RandomWeightGenerator weight_gen(seed);

   if (kind == "cube") {
      return eoneural::NetworkBuilder()
                         .with_activation_func(eoneural::ActivationFunc::Tanh)
                         .with_input_count(1)
                         .with_layer(3)
                         .with_layer(1)
                         .with_weight_generator(weight_gen)
                         .build();
   }

   return eoneural::NetworkBuilder()
           .with_activation_func(eoneural::ActivationFunc::Sigmoid)
           .with_input_count(1)
           .with_layer(1)
           .with_weight_generator(weight_gen)
           .build();
}

void run_test(std::string_view kind, eoneural::RandomWeightGenerator::SeedType seed, int observation_count, double target, bool to_stdout = false) {
   double min = -300, max = 0;
   if (kind == "cube") {
      min = -1253;
      max = 475;
   }

   auto train_data = prepare_training_data(make_input_filename(kind, "train", observation_count), min, max);
   auto test_data = prepare_training_data(make_input_filename(kind, "test", observation_count), min, max);

   auto net = get_network(kind, seed);

   std::ofstream fs_log;
   std::ostream *s_log = &std::cout;

   if (!to_stdout) {
      fs_log.open(make_log_filename(kind, observation_count));
      s_log = &fs_log;
   }

   eoneural::BasicContext ctx(net, train_data, 0.8, 0.1);
   eoneural::MSEObjective<CSVLogger> objective(target, CSVLogger(*s_log));
   net.train(ctx, objective);

   std::ofstream fs_train(make_points_filename(kind, "train", observation_count));
   write_result(fs_train, net, train_data);

   std::ofstream fs_test(make_points_filename(kind, "test", observation_count));
   write_result(fs_test, net, test_data);
}

auto main() -> int {
   auto seed = 737;
   run_test("activation", seed, 100, 0.1e-5);
   run_test("activation", seed, 500, 0.1e-5);
   run_test("activation", seed, 1000, 0.1e-5);
   run_test("activation", seed, 10000, 0.1e-5);

   run_test("cube", seed, 100, 0.1e-5);
   run_test("cube", seed, 500, 0.1e-5);
   run_test("cube", seed, 1000, 0.1e-5);
   run_test("cube", seed, 10000, 0.1e-5);
}