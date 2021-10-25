#ifndef EONEURAL_LOGGER_H
#define EONEURAL_LOGGER_H
#include "Network.hpp"
#include "Util.hpp"
#include <ostream>
#include <fmt/format.h>

namespace eoneural {

class TrainLogger {
   CSVWriter m_writer;
 public:
   inline TrainLogger(const Network &net, std::ostream &out) : m_writer(out) {
      std::vector<std::string> weights(net.weight_count() * 2);
      std::generate(weights.begin(), weights.begin() + net.weight_count(), [i = 0]() mutable {
         ++i;
         return fmt::format("w{}", i);
      });
      std::generate(weights.begin() + net.weight_count(), weights.end(), [i = 0]() mutable {
        ++i;
        return fmt::format("e{}", i);
      });

      m_writer.write_trailing<std::string>(weights, "epoch", "iter");
   }

   inline void log_train_iteration(int epoch, int iter, const Network &net) {
      std::vector<double> weights(net.weight_count() * 2);
      for (int i = 0; i < net.weight_count(); ++i) {
         weights[i] = net.weight(i);
         weights[net.weight_count() + i] = net.delta(i);
      }

      m_writer.write_trailing<double>(weights, epoch, iter);
   }
};

}

#endif//EONEURAL_LOGGER_H
