#include <string>

#include "bindings.h"
#include "device_vec.h"
#include "provenance.h"
#include "utils.h"
template <>
__device__ float DiffTopKProofsProvenance<>::clause_probability(
    typename DiffTopKProofsProvenance<>::DeviceContext ctx,
    typename DiffTopKProofsProvenance<>::Tag t) {
  PROOFVAL(t)
  if (t.empty_) {
    return 0.0;
  }
  float result = 1.0;
  // auto probs = ctx->data();
  for (int i = 0; i < t.lit_count_; i++) {
    auto lit = t.literals_[i];
    auto prob = ctx->at_device(abs(lit));
    result *= (lit >= 0) ? prob : (1.0 - prob);
  }
  return result;
}

template <>
__device__ typename DiffTopKProofsProvenance<>::Tag
DiffTopKProofsProvenance<>::from_foreign(
    const typename DiffTopKProofsProvenance<>::FFITag &t) {
  if (t.literals.size() > TKN) {
    PANIC("too many literals: %lu", t.literals.size());
  }
  Tag result;
  if (t.empty) {
    result.empty_ = true;
  } else {
    if (t.literals.size() > TKN) {
      PANIC("too many literals: %lu", t.literals.size());
    }
    for (size_t i = 0; i < t.literals.size(); i++) {
      result.literals_[i] = t.literals[i];
    }
    result.lit_count_ = t.literals.size();
  }
  return result;
}
template <>
__device__ typename DiffTopKProofsProvenance<>::Tag
DiffTopKProofsProvenance<>::add(
    typename DiffTopKProofsProvenance<>::DeviceContext ctx,
    typename DiffTopKProofsProvenance<>::Tag t1,
    typename DiffTopKProofsProvenance<>::Tag t2) {
  PROOFVAL(t1);
  PROOFVAL(t2);
  if (clause_probability(ctx, t1) > clause_probability(ctx, t2)) {
    return t1;
  } else {
    return t2;
  }
}

template <>
__host__ __device__ typename DiffTopKProofsProvenance<>::Tag
DiffTopKProofsProvenance<>::mult(
    typename DiffTopKProofsProvenance<>::DeviceContext,
    typename DiffTopKProofsProvenance<>::Tag t1,
    typename DiffTopKProofsProvenance<>::Tag t2) {
  Tag result;
  auto left_ptr = 0;
  auto right_ptr = 0;

  while (true) {
    PROOFVAL(result);
    if (SAFETY) {
      if (left_ptr < t1.lit_count_ || right_ptr < t2.lit_count_) {
        if (result.lit_count_ >= TKN) {
          PANIC("too many literals: %d", result.lit_count_);
        }
      }
    }

    if (left_ptr < t1.lit_count_ && right_ptr < t2.lit_count_) {
      auto left_fact = t1.literals_[left_ptr];
      auto right_fact = t2.literals_[right_ptr];

      auto left_id = abs(left_fact);
      auto right_id = abs(right_fact);

      if (left_id == right_id) {
        if (sign(left_fact) != sign(right_fact)) {
          // if they have opposite signs, the proof is unsatisfiable
          return Tag::zero();
        }
        result.literals_[result.lit_count_] = left_fact;
        result.lit_count_ += 1;
        left_ptr += 1;
        right_ptr += 1;
      } else if (left_id < right_id) {
        result.literals_[result.lit_count_] = left_fact;
        result.lit_count_ += 1;
        left_ptr += 1;
      } else {
        result.literals_[result.lit_count_] = right_fact;
        result.lit_count_ += 1;
        right_ptr += 1;
      }
    } else if (left_ptr < t1.lit_count_) {
      result.literals_[result.lit_count_] = t1.literals_[left_ptr];
      result.lit_count_ += 1;
      left_ptr += 1;
    } else if (right_ptr < t2.lit_count_) {
      result.literals_[result.lit_count_] = t2.literals_[right_ptr];
      result.lit_count_ += 1;
      right_ptr += 1;
    } else {
      break;
    }
  }

  PROOFVAL(result);
  if (SAFETY) {
    if (result.lit_count_ > TKN) {
      PANIC("too many literals: %d", result.lit_count_);
    }
  }
  return result;
}
