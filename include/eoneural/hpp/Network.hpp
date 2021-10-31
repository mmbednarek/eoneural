#ifndef EONEURAL_NETWORK_HPP
#define EONEURAL_NETWORK_HPP
extern "C" {
#include "../network.h"
};
#include "Batch.hpp"
#include "BatchContext.hpp"
#include <numeric>
#include <string_view>
#include <vector>

namespace eoneural {

enum class ActivationFunc {
   Identity = 0,
   Sigmoid = 1,
   Tanh = 2,
   Atan = 3,
   Gaussian = 4,
};

constexpr std::string_view activation_func_to_string(ActivationFunc func) {
   switch (func) {
   case ActivationFunc::Identity:
      return "identity";
   case ActivationFunc::Sigmoid:
      return "sigmoid";
   case ActivationFunc::Tanh:
      return "tanh";
   case ActivationFunc::Atan:
      return "atan";
   case ActivationFunc::Gaussian:
      return "gaussian";
   }
   return "";
}

struct TrainResult {
   double mse;
   double mae;
   bool reached_objective{};
};

template<typename T>
concept TrainContext = requires(T v) {
   { v.momentum() }
   ->std::convertible_to<double>;
   { v.learn() }
   ->std::convertible_to<double>;
   { v.next_input() }
   ->std::convertible_to<const double *>;
   { v.next_target() }
   ->std::convertible_to<const double *>;
   { v.partial_output() }
   ->std::convertible_to<double *>;
   { v.output() }
   ->std::convertible_to<double *>;
   { v.errors() }
   ->std::convertible_to<double *>;
   {v.on_iteration_finished()};
};

class NetworkBuilder;
class Network;

template<typename IT>
[[nodiscard]] constexpr auto find_category(IT beg, IT end) {
   return std::max_element(beg, end) - beg;
}

template<typename TO>
concept TrainObjective = requires(TO to, const Network &net, const TrainResult &res) {
   { to.has_reached_objective(net, res) }
   ->std::convertible_to<bool>;
};

template<typename T>
concept Logger = requires(T v, int epoch, int iter, const Network &net) {
   {v.log_train_iteration(epoch, iter, net)};
};

class NoLogger {
 public:
   constexpr void log_train_iteration(int epoch, int iter, const Network &net) const {}
};

extern NoLogger g_no_logger;

class Network {
   friend NetworkBuilder;

   network_t m_net;

   explicit constexpr Network(network_t net) : m_net(net) {}

 public:
   ~Network();

   Network(const Network &net);
   Network &operator=(const Network &net);

   Network(Network &&net) noexcept;
   Network &operator=(Network &&net) noexcept;

   void save(std::string_view filename) const;

   [[nodiscard]] constexpr int input_count() const {
      return m_net->num_in;
   }

   [[nodiscard]] constexpr int output_count() const {
      return m_net->num_out;
   }

   [[nodiscard]] constexpr int weight_count() const {
      return m_net->total_weights;
   }

   [[nodiscard]] constexpr double weight(std::size_t index) const {
      return m_net->neurons[index];
   }

   [[nodiscard]] constexpr double delta(std::size_t index) const {
      return m_net->deltas[index];
   }

   [[nodiscard]] constexpr network_t raw() const {
      return m_net;
   }

   TrainResult train_result(std::span<double> data);

   template<TrainObjective TO, Logger TL = NoLogger>
   inline eoneural::TrainResult batch_train(TO &to, std::span<double> train_data, std::size_t batch_size, double learning, double momentum, long max_epoch = -1) {
      BatchContext ctx(*this, train_data);
      BatchTrain batch_train(*this);
      eoneural::TrainResult result{};
      long epoch = 0;

      do {
         ctx.start_epoch();
         int iteration = 0;
         while (!ctx.has_finished_epoch()) {
            Batch batch(batch_train, momentum);
            for (std::size_t i = 0; i < batch_size; ++i) {
               auto obs = ctx.next();
               batch.put(obs.input, obs.output, learning);
               ++iteration;
            }
         }
         ++epoch;
         result = train_result(train_data);

         if (max_epoch != -1 && epoch > max_epoch)
            return result;

      } while (!to.has_reached_objective(*this, result));

      result.reached_objective = true;
      return result;
   }

   double do_train_iteration(TrainContext auto &ctx) {
      auto mse = network_train(m_net, ctx.partial_output(), ctx.output(), ctx.errors(), ctx.next_input(), ctx.next_target(), ctx.learn(), ctx.momentum());
      ctx.on_iteration_finished();
      return mse;
   }

   template<TrainContext TC, TrainObjective TO, Logger TL = NoLogger>
   TrainResult train(TC &ctx, TO &to, TL &logger = g_no_logger, int max_epoch = -1) {
      eoneural::TrainResult result{};
      int epoch = 0;

      do {
         ctx.start_epoch();
         double total_mse = 0;
         int iteration = 0;
         while (!ctx.has_finished_epoch()) {
            total_mse += do_train_iteration(ctx);
            logger.log_train_iteration(epoch, iteration, *this);
            ++iteration;
         }
         ++epoch;
         result.mse = total_mse / iteration;

         if (max_epoch != -1 && epoch > max_epoch)
            return result;
      } while (!to.has_reached_objective(*this, result));

      result.reached_objective = true;
      return result;
   }

   template<typename II, typename OI>
   void pass(II input_it, OI output_it) const {
      double input[m_net->num_in];
      std::copy(input_it, input_it + m_net->num_in, input);
      auto output = network_perform(m_net, input);
      std::copy(output, output + m_net->num_out, output_it);
   }

   template<typename II>
   int pass_category(II input_it) const {
      double result[m_net->num_out];
      pass(input_it, result);
      return static_cast<int>(find_category(result, result + m_net->num_out));
   }
};

class NetworkBuilder {
   int m_input_count = 1;
   std::vector<int> m_neuron_count;
   ActivationFunc m_activation_func = ActivationFunc::Sigmoid;
   std::vector<double> m_weights;

   [[nodiscard]] inline std::size_t calculate_total_weight_count() const {
      return std::accumulate(m_neuron_count.begin(), m_neuron_count.end(), 0, [prev_weight_count = static_cast<std::size_t>(m_input_count)](std::size_t acc, std::size_t next) mutable {
         auto result = acc + next * (prev_weight_count + 1);
         prev_weight_count = next;
         return result;
      });
   }

 public:
   constexpr NetworkBuilder &with_input_count(int count) {
      m_input_count = count;
      return *this;
   };

   constexpr NetworkBuilder &with_activation_func(ActivationFunc func) {
      m_activation_func = func;
      return *this;
   };

   inline NetworkBuilder &with_layer(int layer_count) {
      m_neuron_count.push_back(layer_count);
      return *this;
   };


   template<typename G>
   constexpr NetworkBuilder &with_weight_generator(G gen) {
      m_weights.resize(calculate_total_weight_count());
      std::generate(m_weights.begin(), m_weights.end(), gen);
      return *this;
   }

   Network build();
};

}// namespace eoneural

#endif//EONEURAL_NETWORK_HPP
