#include <eoneural/hpp/BatchContext.hpp>
#include <eoneural/hpp/Network.hpp>

namespace eoneural {

BatchContext::BatchContext(const Network &net, std::span<double> train_data) : m_train_data(train_data),
                                                                               m_input_count(net.input_count()),
                                                                               m_output_count(net.output_count()),
                                                                               m_at(m_train_data.begin()) {}

[[nodiscard]] Observation BatchContext::next() {
   auto *input = &(*m_at);
   auto *output = &(*(m_at + m_input_count));

   m_at += (m_input_count + m_output_count);
   if (m_at == m_train_data.end()) {
      m_at = m_train_data.begin();
      m_epoch_done = true;
   }

   return Observation{
           .input = input,
           .output = output,
   };
}

}// namespace eoneural