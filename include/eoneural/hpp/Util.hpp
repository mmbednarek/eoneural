#ifndef EONEURAL_UTIL_HPP
#define EONEURAL_UTIL_HPP
#include <ostream>
#include <random>

namespace eoneural {

class RandomWeightGenerator {
   std::mt19937 m_generator;
   std::uniform_real_distribution<double> m_dist;

 public:
   using SeedType = std::mt19937::result_type;

   inline explicit RandomWeightGenerator(SeedType seed) : m_generator(seed),
                                                          m_dist(0, 1) {}

   constexpr double operator()() {
      return m_dist(m_generator);
   }
};

class CSVWriter {
   std::ostream &m_stream;

 public:
   constexpr explicit CSVWriter(std::ostream &stream) : m_stream(stream) {}

 private:
   template<typename TF, typename... TR>
   constexpr void internal_write(TF first, TR... other) {
      m_stream << first;
      if constexpr (sizeof...(other) > 0) {
         m_stream << ',';
         internal_write(other...);
      }
   }

 public:
   template<typename TF, typename... TR>
   constexpr void write(TF first, TR... other) {
      internal_write(first, other...);
      m_stream << '\n';
   }

   template<typename TT, typename TF, typename... TR>
   constexpr void write_trailing(std::span<TT> trailing, TF first, TR... other) {
      internal_write(first, other...);
      for (const auto &rest: trailing) {
         m_stream << ',' << rest;
      }
      m_stream << '\n';
   }
};

}// namespace eoneural

#endif//EONEURAL_UTIL_HPP
