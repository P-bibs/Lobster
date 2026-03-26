#pragma once

#include <string>

#include "bindings.h"
#include "device_vec.h"
#include "utils.h"

// Add-Mult N (max dual number size for add-mult provenance)
#define AMN 40
// Top-K N (max proof size for top-k provenance)
#define TKN 140

struct Unit {
  friend std::ostream &operator<<(std::ostream &os, const Unit &) {
    return os << "unit";
  }
};
class UnitProvenance {
 public:
  static constexpr bool is_unit = true;

  using BatchDeviceContext = void *;
  using DeviceContext = void *;

  UnitProvenance(Provenance::Unit *) {}
  std::string name() { return "unit"; }
  static constexpr bool large_tags = false;
  BatchDeviceContext device_context() const { return nullptr; }
  __device__ static DeviceContext sample_context(BatchDeviceContext, uint32_t) {
    return nullptr;
  }

  using Tag = Unit;
  using FFITag = Unit;
  __host__ __device__ static Tag from_foreign(const FFITag &t) { return t; }
  __host__ __device__ static FFITag to_foreign(Tag &t) { return t; }
  __host__ __device__ static Tag zero() { return Unit(); }
  __host__ __device__ static Tag one() { return Unit(); }
  __host__ __device__ static Tag add(DeviceContext, Tag , Tag ) {
    PANIC("unit add");
    return Unit();
  }
  __host__ __device__ static Tag mult(DeviceContext, Tag , Tag ) {
    PANIC("unit mult");
    return Unit();
  }
  __host__ __device__ static void validate(Tag) {}
};

class MinMaxProbProvenance {
 private:
  Provenance::MinMaxProb *storage;

 public:
  static constexpr bool is_unit = false;
  MinMaxProbProvenance(Provenance::MinMaxProb *storage) : storage(storage) {}

  using Tag = double;

  using InputTag = double;

  using OutputTag = double;

  using FFITag = double;

  using BatchDeviceContext = void *;
  using DeviceContext = void *;

  std::string name() { return "minmaxprob"; }

  static constexpr bool large_tags = false;

  BatchDeviceContext device_context() const { return nullptr; }
  __device__ static DeviceContext sample_context(BatchDeviceContext, uint32_t) {
    return nullptr;
  }

  __host__ __device__ static Tag tagging_fn(InputTag p) { return p; }

  __host__ __device__ static OutputTag recover_fn(Tag t) { return t; }

  __host__ __device__ static Tag from_foreign(const FFITag &t) { return t; }
  __host__ __device__ static FFITag to_foreign(Tag &t) { return t; }

  __host__ __device__ bool discard(Tag p) {
    return p <= storage->valid_threshold;
  }

  __host__ __device__ static Tag zero() { return 0.0; }

  __host__ __device__ static Tag one() { return 1.0; }

  __host__ __device__ static Tag add(DeviceContext, Tag t1, Tag t2) {
    return max(t1, t2);
  }

  __host__ __device__ static bool saturated(Tag t_old, Tag t_new) {
    return t_old == t_new;
  }

  __host__ __device__ static Tag mult(DeviceContext, Tag t1, Tag t2) {
    return min(t1, t2);
  }

  __host__ __device__ static void validate(Tag) {}

  __host__ __device__ static Tag negate(Tag p) { return 1.0 - p; }

  __host__ __device__ static double weight(Tag t) { return t; }
};

class DiffMinMaxProbProvenance {
 public:
  static constexpr bool is_unit = false;
  using Deriv = intptr_t;

  struct Tag {
    double value;
    Deriv deriv;
    __host__ __device__ Tag() : value(0.0), deriv(0) {}
    __host__ __device__ Tag(double value, Deriv deriv)
        : value(value), deriv(deriv) {}

    __host__ __device__ bool operator<(const Tag &other) const {
      return value < other.value;
    }
    __host__ __device__ bool operator>(const Tag &other) const {
      return value > other.value;
    }
    __host__ __device__ bool operator<=(const Tag &other) const {
      return value <= other.value;
    }
    __host__ __device__ bool operator>=(const Tag &other) const {
      return value >= other.value;
    }

    friend std::ostream &operator<<(std::ostream &os, const Tag &t) {
      os << "[" << t.value << "|" << t.deriv << "]";
      return os;
    }
  };

  using InputTag = double;

  using OutputTag = double;

  using FFITag = Tag;

  using BatchDeviceContext = void *;
  using DeviceContext = void *;

 private:
  Provenance::DiffMinMaxProb *storage;

 public:
  DiffMinMaxProbProvenance(Provenance::DiffMinMaxProb *storage)
      : storage(storage) {}

  std::string name() { return "diffminmaxprob"; }

  static constexpr bool large_tags = false;

  BatchDeviceContext device_context() const { return nullptr; }
  __device__ static DeviceContext sample_context(BatchDeviceContext, uint32_t) {
    return nullptr;
  }

  //__host__ __device__ static Tag tagging_fn(InputTag p) { return p; }

  //__host__ __device__ static OutputTag recover_fn(Tag t) { return t; }

  __host__ __device__ static Tag from_foreign(const FFITag &t) { return t; }
  __host__ __device__ static FFITag to_foreign(Tag &t) { return t; }

  __host__ __device__ bool discard(Tag p) {
    return p.value <= storage->valid_threshold;
  }

  __host__ __device__ static Tag zero() { return Tag(0.0, 0); }

  __host__ __device__ static Tag one() { return Tag(1.0, 0); }

  __host__ __device__ static Tag add(DeviceContext, Tag t1, Tag t2) {
    if (t1.value > t2.value) {
      return t1;
    } else {
      return t2;
    }
  }

  __host__ __device__ static bool saturated(Tag t_old, Tag t_new) {
    return t_old.value == t_new.value;
  }

  __host__ __device__ static Tag mult(DeviceContext, Tag t1, Tag t2) {
    if (t1.value < t2.value) {
      return t1;
    } else {
      return t2;
    }
  }

  __host__ __device__ static void validate(Tag) {}

  __host__ __device__ static Tag negate(Tag p) {
    return Tag(1.0 - p.value, -1. * p.deriv);
  }

  __host__ __device__ static double weight(Tag t) { return t.value; }
};

template <int N>
struct DualNumber {
  struct Deriv {
    int index;
    float gradient;

    __host__ __device__ Deriv operator*(float value) const {
      Deriv result;
      result.index = index;
      result.gradient = gradient * value;
      return result;
    }
    __host__ __device__ bool operator!=(const Deriv &other) const {
      return this->index != other.index ||
             abs(this->gradient - other.gradient) > EPSILON;
    }
  };

  float value;
  int deriv_count;
  Deriv derivs[N];

  __host__ __device__ DualNumber(float value) : value(value), deriv_count(0) {
    if (SAFETY) {
      for (int i = 0; i < N; i++) {
        derivs[i].index = -1;
        derivs[i].gradient = 0.0;
      }
    }
  }

  __host__ __device__ DualNumber() : value(0.0), deriv_count(0) {
    if (SAFETY) {
      for (int i = 0; i < N; i++) {
        derivs[i].index = -1;
        derivs[i].gradient = 0.0;
      }
    }
  }

  __host__ __device__ static DualNumber one() { return DualNumber(1.0); }
  __host__ __device__ static DualNumber zero() { return DualNumber(0.0); }

  __host__ __device__ DualNumber operator+(const DualNumber &other) const {
    DualNumber result(value + other.value);

    int left_ptr = 0;
    int right_ptr = 0;

    while (true) {
      auto left_in_range = left_ptr < this->deriv_count;
      auto right_in_range = right_ptr < other.deriv_count;
      if (left_in_range && right_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        if (this->derivs[left_ptr].index < other.derivs[right_ptr].index) {
          result.derivs[result.deriv_count] = this->derivs[left_ptr];
          result.deriv_count++;
          left_ptr++;
        } else if (this->derivs[left_ptr].index >
                   other.derivs[right_ptr].index) {
          result.derivs[right_ptr] = other.derivs[right_ptr];
          result.deriv_count++;
          right_ptr++;
        } else {
          result.derivs[result.deriv_count] = this->derivs[left_ptr];
          result.derivs[result.deriv_count].gradient +=
              other.derivs[right_ptr].gradient;
          result.deriv_count++;
          left_ptr++;
          right_ptr++;
        }
      } else if (left_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        result.derivs[result.deriv_count] = this->derivs[left_ptr];
        result.deriv_count++;
        left_ptr++;
      } else if (right_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        result.derivs[result.deriv_count] = other.derivs[right_ptr];
        result.deriv_count++;
        right_ptr++;
      } else {
        break;
      }
    }

    return result;
  }

  __host__ __device__ DualNumber operator*(const DualNumber &other) const {
    DualNumber result(value * other.value);

    int left_ptr = 0;
    int right_ptr = 0;

    while (true) {
      auto left_in_range = left_ptr < this->deriv_count;
      auto right_in_range = right_ptr < other.deriv_count;
      if (left_in_range && right_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        if (this->derivs[left_ptr].index < other.derivs[right_ptr].index) {
          result.derivs[result.deriv_count] =
              this->derivs[left_ptr] * other.value;
          result.deriv_count++;
          left_ptr++;
        } else if (this->derivs[left_ptr].index >
                   other.derivs[right_ptr].index) {
          result.derivs[result.deriv_count] =
              other.derivs[right_ptr] * this->value;
          result.deriv_count++;
          right_ptr++;
        } else {
          result.derivs[result.deriv_count] = this->derivs[left_ptr];
          result.derivs[result.deriv_count].gradient =
              this->derivs[left_ptr].gradient * other.value +
              other.derivs[right_ptr].gradient * this->value;
          result.deriv_count++;
          left_ptr++;
          right_ptr++;
        }
      } else if (left_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        result.derivs[result.deriv_count] =
            this->derivs[left_ptr] * other.value;
        result.deriv_count++;
        left_ptr++;
      } else if (right_in_range) {
        if (SAFETY) {
          if (result.deriv_count >= N) {
            PANIC("too many derivatives: %d", result.deriv_count);
          }
        }
        result.derivs[result.deriv_count] =
            other.derivs[right_ptr] * this->value;
        result.deriv_count++;
        right_ptr++;
      } else {
        break;
      }
    }
    return result;
  }

  __host__ __device__ DualNumber operator-() const {
    DualNumber result(1.0 - value);

    for (int i = 0; i < this->deriv_count; i++) {
      result.derivs[i] = derivs[i];
      result.derivs[i].gradient = -result.derivs[i].gradient;
    }
    result.deriv_count = this->deriv_count;

    return result;
  }

  __host__ __device__ bool operator==(const DualNumber &other) const {
    if (abs(this->value - other.value) > EPSILON) {
      return false;
    }
    if (this->deriv_count != other.deriv_count) {
      return false;
    }
    for (int i = 0; i < this->deriv_count; i++) {
      if (this->derivs[i] != other.derivs[i]) {
        return false;
      }
    }
    return true;
  }

  __host__ __device__ bool operator!=(const DualNumber &other) const {
    return !(*this == other);
  }

  friend std::ostream &operator<<(std::ostream &os, const DualNumber &t) {
    os << "[" << t.value << "|";
    for (int i = 0; i < t.deriv_count; i++) {
      os << t.derivs[i].index << ":" << t.derivs[i].gradient << ", ";
    }
    os << "]";
    return os;
  }
};

class DummyProvenance {
 public:
  static constexpr bool is_unit = false;
  using Tag = double;
  using InputTag = double;
  using OutputTag = double;
  using FFITag = double;
  using BatchDeviceContext = void *;
  using DeviceContext = void *;

  DummyProvenance(Provenance::DiffAddMultProb *) {}
  std::string name() { return "diffaddmultprob"; }
  static constexpr bool large_tags = false;
  BatchDeviceContext device_context() const { return nullptr; }
  __device__ static DeviceContext sample_context(BatchDeviceContext, uint32_t) {
    return nullptr;
  }
  __host__ __device__ static Tag from_foreign(const FFITag &t) { return t; }
  static FFITag to_foreign(Tag &t) { return t; }
  __host__ __device__ bool discard(Tag) { return false; }
  __host__ __device__ static Tag zero() { return 0.0; }
  __host__ __device__ static Tag one() { return 1.0; }
  __host__ __device__ static Tag add(DeviceContext, Tag t1, Tag t2) {
    return t1 + t2;
  }
  __host__ __device__ static bool saturated(Tag t_old, Tag t_new) {
    return t_old == t_new;
  }
  __host__ __device__ static Tag mult(DeviceContext, Tag t1, Tag t2) {
    return t1 * t2;
  }
  __host__ __device__ static void validate(Tag) {}
  __host__ __device__ static Tag negate(Tag p) { return -p; }
  __host__ __device__ static double weight(Tag t) { return t; }
};

template <int N = 10>
using DiffAddMultProbProvenance = DummyProvenance;

// template <int N = AMN> class DiffAddMultProbProvenance {
// public:
//   using Tag = DualNumber<N>;
//   using InputTag = double;
//   using OutputTag = double;
//   using FFITag = C_DualNumber;
//   using DeviceContext = void;
//
// private:
//   Provenance::DiffAddMultProb *storage;
//
// public:
//   DiffAddMultProbProvenance(Provenance::DiffAddMultProb *storage)
//       : storage(storage) {}
//
//   std::string name() { return "diffaddmultprob"; }
//   static constexpr bool large_tags = true;
//
//   DeviceContext *device_context() const { return nullptr; }
//
//   //__host__ __device__ static Tag tagging_fn(InputTag p) { return p; }
//
//   //__host__ __device__ static OutputTag recover_fn(Tag t) { return t; }
//
//   __host__ __device__ static Tag from_foreign(const FFITag &t) {
//     if (t.gradient.indices.size() > N) {
//       PANIC("too many derivatives");
//     }
//     Tag result;
//     result.value = t.real;
//     for (size_t i = 0; i < t.gradient.indices.size(); i++) {
//       result.derivs[i].index = t.gradient.indices[i];
//       result.derivs[i].gradient = t.gradient.values[i];
//     }
//     result.deriv_count = t.gradient.indices.size();
//     return result;
//   }
//
//   static FFITag to_foreign(Tag &t) {
//     FFITag result;
//     result.real = t.value;
//     new (&result.gradient.indices) Array<uint32_t>((size_t)t.deriv_count);
//     new (&result.gradient.values) Array<float>((size_t)t.deriv_count);
//
//     for (int i = 0; i < t.deriv_count; i++) {
//       result.gradient.indices[i] = t.derivs[i].index;
//       result.gradient.values[i] = t.derivs[i].gradient;
//     }
//     return result;
//   }
//
//   __host__ __device__ bool discard(Tag p) {
//     return p.value <= storage->valid_threshold;
//   }
//
//   __host__ __device__ static Tag zero() { return DualNumber<N>::zero(); }
//
//   __host__ __device__ static Tag one() { return DualNumber<N>::one(); }
//
//   __host__ __device__ static Tag add(DeviceContext , Tag t1, Tag t2) {
//     return t1 + t2;
//   }
//
//   __host__ __device__ static bool saturated(Tag t_old, Tag t_new) {
//     return t_old == t_new;
//   }
//
//   __host__ __device__ static Tag mult(DeviceContext , Tag t1, Tag t2) {
//     return t1 * t2;
//   }
//
//   __host__ __device__ static Tag negate(Tag p) { return -p; }
//
//   __host__ __device__ static double weight(Tag t) { return t.value; }
// };

#define PROOFVAL(x)                                    \
  if (SAFETY) {                                        \
    if (x.padding_[0] != Proof<TKN>::MAGIC ||          \
        x.padding_[1] != Proof<TKN>::MAGIC) {          \
      PANIC("device difftopkproof corrupted padding"); \
    }                                                  \
    if (x.lit_count_ > TKN) {                          \
      PANIC("too many literals: %d", x.lit_count_);    \
    }                                                  \
  }
// if (x.lit_count_ < 0) {
//   PANIC("negative literal count: %d", x.lit_count_);
// }

template <int N>
struct Proof {
  using Lit = int16_t;

  const static uint32_t MAGIC = 0xdeadbeef;
  // This padding prevents another Proof that is directly before this one in
  // memory from overwriting the lit_count_ field.
  uint32_t padding_[2];
  uint32_t lit_count_;
  // empty is impossible to prove, empty with no literals is trivially true
  bool empty_;
  Lit literals_[N];

  __host__ __device__ Proof() : lit_count_(0), empty_(false) {
    if (SAFETY) {
      padding_[0] = MAGIC;
      padding_[1] = MAGIC;
      for (int i = 0; i < N; i++) {
        literals_[i] = 0;
      }
    }
  }
  __host__ __device__ Proof(Lit l) : lit_count_(1), empty_(false) {
    literals_[0] = l;
    if (SAFETY) {
      padding_[0] = MAGIC;
      padding_[1] = MAGIC;
      for (int i = 1; i < N; i++) {
        literals_[i] = 0;
      }
    }
  }

  __host__ __device__ static Proof zero() {
    Proof p;
    p.empty_ = true;
    return p;
  }
  __host__ __device__ static Proof one() { return Proof(); }

  __host__ __device__ Proof operator-() const {
    PANIC("device difftopkproof negation not implemented");
  }

  friend std::ostream &operator<<(std::ostream &os, const Proof &t) {
    PROOFVAL(t);
    if (t.empty_) {
      os << "empty";
      return os;
    } else {
      os << "[";
      for (int i = 0; i < t.lit_count_; i++) {
        os << t.literals_[i] << " ";
      }
      os << "]";
    }
    return os;
  }
};

template <size_t N = TKN>
class DiffTopKProofsProvenance {
 public:
  static constexpr bool is_unit = false;
  using Tag = Proof<N>;
  using InputTag = Proof<N>;
  using OutputTag = Proof<N>;
  using FFITag = C_Proof;
  using BatchDeviceContext = device_vec<float> *;
  using DeviceContext = device_vec<float> *;

 private:
  ManagedArray<device_vec<float>> literal_probabilities_;

 public:
  const ManagedArray<device_vec<float>> &literal_probabilities() const {
    return literal_probabilities_;
  }
  __device__ static float clause_probability(DeviceContext ctx, Tag t);

  DiffTopKProofsProvenance(Provenance::DiffTopKProofs *storage) {
    auto batch_size = storage->literal_probabilities.size();
    ManagedArray<device_vec<float>> batch_probs(batch_size);
    for (size_t sample = 0; sample < batch_size; sample++) {
      auto &sample_probs = storage->literal_probabilities[sample];
      new (&batch_probs[sample]) device_vec<float>(sample_probs.size());
      cudaCheck(cudaMemcpy(batch_probs[sample].data(), sample_probs.data(),
                           sizeof(float) * sample_probs.size(),
                           cudaMemcpyHostToDevice));
    }
    literal_probabilities_ = std::move(batch_probs);
  }

  std::string name() { return "difftopkproofs"; }

  static constexpr bool large_tags = true;

  BatchDeviceContext device_context() const {
    return literal_probabilities_.data();
  }

  __device__ static DeviceContext sample_context(BatchDeviceContext ctxs,
                                                 uint32_t sample) {
    return ctxs + sample;
  }

  //__host__ __device__ static Tag tagging_fn(InputTag p) { return p; }

  //__host__ __device__ static OutputTag recover_fn(Tag t) { return t; }

  __host__ __device__ static Tag from_foreign(const FFITag &t);

  static FFITag to_foreign(Tag &t) {
    FFITag result;
    if (t.empty_) {
      result.empty = true;
    } else {
      result.empty = false;
      new (&result.literals) Array<float>((size_t)t.lit_count_);
      for (uint32_t i = 0; i < t.lit_count_; i++) {
        result.literals[i] = t.literals_[i];
      }
    }
    return result;
  }

  __host__ __device__ bool discard(Tag) {
    PANIC("device difftopkproof discard not implemented");
  }

  __host__ __device__ static Tag zero() { return Tag::zero(); }

  __host__ __device__ static Tag one() { return Tag::one(); }

  __device__ static Tag add(DeviceContext ctx, Tag t1, Tag t2);

  __host__ __device__ static bool saturated(Tag, Tag) {
    PANIC("device difftopkproof saturated not implemented");
  }

  __device__ static Tag mult(DeviceContext, Tag t1, Tag t2);

  __host__ __device__ static void validate(Tag t) {
    if (t.padding_[0] != Proof<TKN>::MAGIC ||
        t.padding_[1] != Proof<TKN>::MAGIC) {
      PANIC("device difftopkproof corrupted padding");
    }
    if (t.lit_count_ > TKN) {
      PANIC("too many literals: %d", t.lit_count_);
    }
  }

  __host__ __device__ static Tag negate(Tag) {
    PANIC("device difftopkproof negate not implemented");
  }

  //__host__ __device__ static double weight(Tag t) { return t.value; }
};
