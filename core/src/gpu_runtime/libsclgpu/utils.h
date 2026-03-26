#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "thrust/device_vector.h"

#define RED "\033[0;31m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define BLUE "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN "\033[0;36m"
#define WHITE "\033[0;37m"
#define RESET "\033[0m"

#define ORACLE false
#define TRACK_ALLOC false
#define LOG_ALLOC false

// Comment out the following line to disable recording of tracing spans
#define TRACE

#ifdef __NVCC__
#ifdef __CUDA_ARCH__
// Indicates an exceptional event that can't be recovered from. Logs a message
// and exits
#define PANIC(msg, args...)                                         \
  do {                                                              \
    printf("PANIC (%s:%d): " msg "\n", __FILE__, __LINE__, ##args); \
    asm("trap;");                                                   \
  } while (0)
#else
// Indicates an exceptional event that can't be recovered from. Logs a message
// and exits
#define PANIC(msg, args...)                                            \
  do {                                                                 \
    printf("PANIC (%s:%d): " msg "\n", __FILE__, __LINE__, ##args);    \
    throw std::runtime_error("C++ panic occurred, see prior logging"); \
  } while (0)
#endif
#else
#define PANIC(msg, args...)            \
  do {                                 \
    do_panic(msg, __FILE__, __LINE__); \
  } while (0)

__host__ [[noreturn]] inline void do_panic(const char *msg, const char *file,
                                           int line) {
  printf("PANIC (%s:%d): %s\n", file, line, msg);
  throw std::runtime_error("C++ panic occurred, see prior logging");
}
__device__ [[noreturn]] inline void do_panic(const char *msg, const char *file,
                                             int line) {
  printf("PANIC (%s:%d): %s\n", file, line, msg);
  asm("trap;");
  while (1) {
  }
}
#endif

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
inline void cudaErrorCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
           cudaGetErrorName(error), cudaGetErrorString(error));
    throw std::runtime_error("CUDA error");
  }
};

#define ROUND_UP_TO_NEAREST(M, N) \
  (static_cast<unsigned int>(((M) + (N)-1) / (N)))

// generic to_string template for objects with ostream operator
template <typename T>
std::string to_string(const T &value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

// ostream operator for thrust::device_vector
template <class T>
std::ostream &operator<<(std::ostream &os, const thrust::device_vector<T> &v) {
  os << "[";
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  os << "]";
  std::cout << std::endl;
  return os;
}

// Functor to check equality against a compile time value
template <typename T, T V>
struct equals_const {
  __device__ bool operator()(const T a) const { return a == V; }
};

// device-compatible function to check an integer's sign bit
__host__ __device__ inline int sign(int x) { return (x > 0) - (x < 0); }

// Conveniance macro for using parameter pack expressions in a statement context
template <typename... T>
__host__ __device__ inline void sink(T...) {}
__host__ __device__ inline void sink(void) {}
#define SINK(body) sink(body)

// Template to bundle together type-level values. Identical compile-time usage
// to std::tuple, except compatible with device code
template <typename... T>
struct Product {};

#define EPSILON 0.00001

// Defines a memoized function to check for existance of an environment variable
#define MAKE_ENV_GETTER(name)                                \
  inline bool name() {                                       \
    static std::optional<bool> trigger = std::nullopt;       \
    if (!trigger.has_value()) {                              \
      std::getenv(#name) ? trigger = true : trigger = false; \
    }                                                        \
    return trigger.value();                                  \
  }

MAKE_ENV_GETTER(NO_CUB);
MAKE_ENV_GETTER(SIZES);
MAKE_ENV_GETTER(HASH_APPEND);
MAKE_ENV_GETTER(HASH_INFO);
MAKE_ENV_GETTER(THRUST_TAG_JOIN);
MAKE_ENV_GETTER(THRUST_COPY_IF)
MAKE_ENV_GETTER(DOUBLE_SORT)
MAKE_ENV_GETTER(TRACE_FINE)
MAKE_ENV_GETTER(COMPACT)
MAKE_ENV_GETTER(LOG)

// Allows setting hash table overhead size from environment variable
inline size_t OVERHEAD() {
  static std::optional<size_t> trigger = std::nullopt;
  if (!trigger.has_value()) {
    auto *s = std::getenv("OVERHEAD");
    if (s) {
      trigger = std::stoul(s);
    } else {
      trigger = 8;
    }
  }
  return trigger.value();
}

// Log informational data (usable from host code)
#define hINFO(msg)                                                       \
  if (LOG()) {                                                           \
    std::cout << "INFO (" << __FILE__ << ":" << __LINE__ << "): " << msg \
              << std::endl;                                              \
  }
// Log data that indicates potentially problematic behavior (usable from host
// code)
#define hWARN(msg)                                                       \
  if (LOG()) {                                                           \
    std::cout << "WARN (" << __FILE__ << ":" << __LINE__ << "): " << msg \
              << std::endl;                                              \
  }

// Log informational data (usable from device code)
#define dINFO(msg, args...)                                        \
  if (LOG()) {                                                     \
    printf("INFO (%s:%d): " msg "\n", __FILE__, __LINE__, ##args); \
  }
// Log data that indicates potentially problematic behavior (usable from device
// code)
#define dWARN(msg, args...)                                        \
  if (LOG()) {                                                     \
    printf("WARN (%s:%d): " msg "\n", __FILE__, __LINE__, ##args); \
  }

int get_batch_size();
void set_batch_size(int size);

// Generic timer class for benchmarking
class Timer {
  std::string name;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
  std::chrono::microseconds duration;

 public:
  Timer(std::string name) : name(name), duration(0) {}
  void start() { start_ = std::chrono::high_resolution_clock::now(); }
  void stop() {
    end_ = std::chrono::high_resolution_clock::now();
    duration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
  }
  void print() {
    std::cout << "Timer [" << name << "]: " << duration.count() << "us"
              << std::endl;
  }
};

// Template trickery to allow writing constexpr for loops
template <std::size_t N>
struct num {
  static const constexpr auto value = N;
};
template <class F, std::size_t... Is>
__host__ __device__ void const_expr_for(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}
template <std::size_t N, typename F>
__host__ __device__ void const_expr_for(F func) {
  const_expr_for(func, std::make_index_sequence<N>());
}

// Raises run-time value to compile-time C++ type via switching on known
// possible run-time values
#define DISPATCH_ON_TYPE(kind, type_param, body)    \
  do {                                              \
    auto k = kind.tag();                            \
    if (k == ValueType::Tag::U32) {                 \
      using T = uint32_t;                           \
      body                                          \
    } else if (k == ValueType::Tag::U64) {          \
      using T = uint64_t;                           \
      body                                          \
    } else if (k == ValueType::Tag::USize) {        \
      using T = uintptr_t;                          \
      body                                          \
    } else if (k == ValueType::Tag::F32) {          \
      using T = float;                              \
      body                                          \
    } else if (k == ValueType::Tag::Symbol) {       \
      using T = uintptr_t;                          \
      body                                          \
    } else {                                        \
      throw std::runtime_error("Unsupported kind"); \
    }                                               \
  } while (0)

// Raises run-time value to compile-time equivalent type via switching on known
// possible run-time values
#define DISPATCH_ON_KIND(kind, type_param, body)    \
  do {                                              \
    auto k = kind.tag();                            \
    if (k == ValueType::Tag::U32) {                 \
      using T = ValueU32;                           \
      body                                          \
    } else if (k == ValueType::Tag::USize) {        \
      using T = ValueUSize;                         \
      body                                          \
    } else if (k == ValueType::Tag::F32) {          \
      using T = ValueF32;                           \
      body                                          \
    } else if (k == ValueType::Tag::Symbol) {       \
      using T = ValueSymbol;                        \
      body                                          \
    } else {                                        \
      throw std::runtime_error("Unsupported kind"); \
    }                                               \
  } while (0)

#define DISPATCH_ON_PROV(prov, prov_param, ctx_param, body)      \
  do {                                                           \
    auto _prov = prov;                                           \
    if (_prov->tag == Provenance::Tag::Unit) {                   \
      using prov_param = UnitProvenance;                         \
      Prov ctx_param(&_prov->unit);                              \
      body                                                       \
    } else if (_prov->tag == Provenance::Tag::MinMaxProb) {      \
      using prov_param = MinMaxProbProvenance;                   \
      Prov ctx_param(&_prov->min_max_prob);                      \
      body                                                       \
    } else if (_prov->tag == Provenance::Tag::DiffMinMaxProb) {  \
      using prov_param = DiffMinMaxProbProvenance;               \
      Prov ctx_param(&_prov->diff_min_max_prob);                 \
      body                                                       \
    } else if (_prov->tag == Provenance::Tag::DiffAddMultProb) { \
      using prov_param = DiffAddMultProbProvenance<AMN>;         \
      Prov ctx_param(&_prov->diff_add_mult_prob);                \
      body                                                       \
    } else if (_prov->tag == Provenance::Tag::DiffTopKProofs) {  \
      using prov_param = DiffTopKProofsProvenance<TKN>;          \
      Prov ctx_param(&_prov->diff_top_k_proofs);                 \
      body                                                       \
    } else {                                                     \
      throw std::runtime_error("Unsupported provenance type");   \
    }                                                            \
  } while (0)
