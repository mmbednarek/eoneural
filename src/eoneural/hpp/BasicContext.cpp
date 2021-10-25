#include <eoneural/hpp/BasicContext.hpp>
#include <eoneural/hpp/Network.hpp>

namespace eoneural {

BasicContext::BasicContext(const Network &net, std::span<double> train_data, double learn, double momentum) : m_train_data(train_data),
                                                                                                              m_learn(learn),
                                                                                                              m_momentum(momentum),
                                                                                                              m_input_count(net.input_count()),
                                                                                                              m_output_count(net.output_count()),
                                                                                                              m_at(m_train_data.begin()),
                                                                                                              m_output(network_alloc_area(net.raw())),
                                                                                                              m_partial_output(network_alloc_area(net.raw())),
                                                                                                              m_errors(network_alloc_area(net.raw())) {}

void BasicContext::on_iteration_finished() {
   m_at += (m_input_count + m_output_count);
   if (m_at == m_train_data.end()) {
      m_at = m_train_data.begin();
      m_epoch_done = true;
   }
}

BasicContext::~BasicContext() {
   if (m_output != nullptr)
      free(m_output);
   if (m_partial_output != nullptr)
      free(m_partial_output);
   if (m_errors != nullptr)
      free(m_errors);
}

}// namespace eoneural