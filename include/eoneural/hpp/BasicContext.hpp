#ifndef EONEURAL_CONTEXT_H
#define EONEURAL_CONTEXT_H
#include <cstddef>
#include <span>
#include <vector>

namespace eoneural {

class Network;
struct TrainResult;

class BasicContext {
   std::span<double> m_train_data;
   double m_learn;
   double m_momentum;

   std::span<double>::iterator m_at;

   int m_input_count;
   int m_output_count;

   bool m_epoch_done = false;

   double *m_output = nullptr;
   double *m_partial_output = nullptr;
   double *m_errors = nullptr;

 public:
   BasicContext(const Network &net, std::span<double> train_data, double learn, double momentum);
   ~BasicContext();

   [[nodiscard]] constexpr double momentum() const {
      return m_momentum;
   }

   [[nodiscard]] constexpr double learn() const {
      return m_learn;
   }

   [[nodiscard]] constexpr const double *next_input() const {
      return &(*m_at);
   }

   [[nodiscard]] constexpr const double *next_target() const {
      return &(*(m_at + m_input_count));
   }

   [[nodiscard]] constexpr bool has_finished_epoch() const {
      return m_epoch_done;
   }

   [[nodiscard]] constexpr double *output() const {
      return m_output;
   }

   [[nodiscard]] constexpr double *partial_output() const {
      return m_partial_output;
   }

   [[nodiscard]] constexpr double *errors() const {
      return m_errors;
   }

   constexpr void start_epoch() {
      m_epoch_done = false;
   }

   void on_iteration_finished();
};

}// namespace eoneural

#endif//EONEURAL_CONTEXT_H
