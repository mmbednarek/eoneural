#ifndef HPP_EONEURAL_BATCH
#define HPP_EONEURAL_BATCH

#pragma once

extern "C" {
#include "../batch.h"
}
#include <span>

namespace eoneural {

class Network;
class BatchTrain;

class Batch {
   BatchTrain &m_batch;

 public:
   explicit Batch(BatchTrain &batch, double momentum);
   ~Batch();

   Batch(const Batch &) = delete;
   Batch &operator=(const Batch &) = delete;

   Batch(Batch &&) noexcept = delete;
   Batch &operator=(Batch &&) noexcept = delete;

   void put(const double *in, const double *out, double learning);
};

class BatchTrain {
   batch_t m_batch;

 public:
   explicit BatchTrain(const Network &net);
   ~BatchTrain();

   BatchTrain(const BatchTrain &) = delete;
   BatchTrain &operator=(const BatchTrain &) = delete;

   BatchTrain(BatchTrain &&other) noexcept;
   BatchTrain &operator=(BatchTrain &&other) noexcept;

   [[nodiscard]] constexpr const batch_t &raw() const {
      return m_batch;
   }

   void begin(double momentum);
   void end();
};


}// namespace eoneural

#endif// HPP_EONEURAL_BATCH