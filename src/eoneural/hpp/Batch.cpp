#include <eoneural/hpp/Batch.hpp>
#include <eoneural/hpp/Network.hpp>
#include <utility>

namespace eoneural {

Batch::Batch(BatchTrain &batch, double momentum) : m_batch(batch) {
   batch_begin(m_batch.raw(), momentum);
}

Batch::~Batch() {
   batch_end(m_batch.raw());
}

void Batch::put(const std::span<double> in, const std::span<double> out, double learning) {
   batch_put(m_batch.raw(), in.data(), out.data(), learning);
}

BatchTrain::BatchTrain(const Network &net) : m_batch(batch_create(net.raw())) {}

BatchTrain::~BatchTrain() {
   batch_destroy(m_batch);
}

BatchTrain::BatchTrain(BatchTrain &&other) noexcept : m_batch{
                                                              .net = std::exchange(other.m_batch.net, nullptr),
                                                              .errors = std::exchange(other.m_batch.errors, nullptr),
                                                              .output = std::exchange(other.m_batch.output, nullptr),
                                                              .partial = std::exchange(other.m_batch.partial, nullptr),
                                                      } {}

BatchTrain &BatchTrain::operator=(BatchTrain &&other) noexcept {
   m_batch.net = std::exchange(other.m_batch.net, nullptr);
   m_batch.errors = std::exchange(other.m_batch.errors, nullptr);
   m_batch.output = std::exchange(other.m_batch.output, nullptr);
   m_batch.partial = std::exchange(other.m_batch.partial, nullptr);
   return *this;
}

void BatchTrain::begin(double momentum) {
   batch_begin(m_batch, momentum);
}

void BatchTrain::end() {
   batch_end(m_batch);
}

}// namespace eoneural