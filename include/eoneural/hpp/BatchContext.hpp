#ifndef HPP_EONEURAL_BATCH_CONTEXT
#define HPP_EONEURAL_BATCH_CONTEXT

#pragma once

#include "Observation.hpp"
#include <span>

namespace eoneural {

class Network;

class BatchContext {
   std::span<double> m_train_data;

   std::span<double>::iterator m_at;

   std::size_t m_input_count;
   std::size_t m_output_count;

   bool m_epoch_done = false;


 public:
   BatchContext(const Network &net, std::span<double> train_data);

   [[nodiscard]] constexpr bool has_finished_epoch() const {
      return m_epoch_done;
   }

   constexpr void start_epoch() {
      m_epoch_done = false;
   }

   [[nodiscard]] Observation next();
};

}// namespace eoneural

#endif// HPP_EONEURAL_BATCH_CONTEXT