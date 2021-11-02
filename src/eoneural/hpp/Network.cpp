#include <eoneural/hpp/Network.hpp>
#include <stdexcept>
#include <utility>
#include <cmath>

namespace eoneural {

Network::~Network() {
   if (m_net != nullptr) {
      network_destroy(m_net);
   }
}

Network::Network(const Network &net) : m_net(network_copy(net.m_net)) {}


Network &Network::operator=(const Network &net) {
   if (m_net != nullptr) {
      network_destroy(m_net);
   }
   m_net = network_copy(net.m_net);
   return *this;
}

Network::Network(Network &&net) noexcept : m_net(std::exchange(net.m_net, nullptr)) {}

Network &Network::operator=(Network &&net) noexcept {
   m_net = std::exchange(net.m_net, nullptr);
   return *this;
}

void Network::save(std::string_view filename) const {
   network_save(m_net, filename.data());
}

Network NetworkBuilder::build() {
   if (m_neuron_count.empty()) {
      throw std::runtime_error("network has not layers");
   }
   if (!m_weights.empty()) {
      return Network(network_create_raw(m_input_count, m_neuron_count.size(), m_neuron_count.data(), static_cast<unsigned char>(m_activation_func), m_weights.data()));
   }
   return Network(network_create(m_input_count, m_neuron_count.size(), m_neuron_count.data(), static_cast<unsigned char>(m_activation_func)));
}


TrainResult Network::train_result(std::span<double> data) {
   if (data.empty())
      return TrainResult{};
   
   auto it = data.begin();
   double output[output_count()];

   double mse = 0.0;
   double mae = 0.0;

   do {
      pass(it, output);
      it += input_count();

      for (std::size_t i = 0; i < output_count(); ++i) {
         auto err = *(it++) - output[i];
         mse += err*err;
         mae += fabs(err);
      }

      it += input_count() + output_count();
   } while (it != data.end());


   return TrainResult{
      .mse = mse,
      .mae = mae,
   };
}

NoLogger g_no_logger;

}