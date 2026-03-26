#include <thrust/count.h>

#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "provenance.h"
#include "table.h"
#include "table_index.h"
#include "utils.h"

__device__ bool HashIndexView::try_insert(size_t key, int64_t value) {
  unsigned long long int *lock_location =
      reinterpret_cast<unsigned long long int *>(&this->value_at(key));
  auto last_value =
      atomicCAS(lock_location, static_cast<unsigned long long int>(-1),
                static_cast<unsigned long long int>(value));
  if (last_value == static_cast<unsigned long long int>(-1)) {
    return true;
  }
  return false;
}

template <typename Prov, typename... T>
__global__ void construct_multi_column_hash_table(TableView<Prov> t,
                                                  HashIndexView h) {
  uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= t.size()) {
    return;
  }

  // TODO: add for loop here with stride
  auto position = h.hash_row<Prov, T...>(t, index);

  for (size_t i = 0; i < h.size(); i++) {
    bool success = h.try_insert(position, index);
    if (success) {
      return;
    }
    position = h.incr_position(position);
  }

  PANIC("construct_hash_table: failed to insert row %d\n", index);
}

__global__ void fill(int64_t *values, size_t size, int64_t value) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    values[index] = value;
  }
}

template <typename Prov, typename... T>
HashIndex::HashIndex(const Table<Prov> &t, Product<T...>)
    : width_(sizeof...(T)) {
  TRACE_START(construct_hash_index_kernel);

  constexpr size_t N = sizeof...(T);

  if (SAFETY) {
    size_t type_sizes[N] = {sizeof(typename T::type)...};
    for (size_t i = 0; i < N; i++) {
      if (t.schema().flatten().at(i).singleton().size() != type_sizes[i]) {
        throw std::runtime_error("Column type mismatch");
      }
    }
  }

  // The size of the hash table is the next power of 2 after the size of the
  // input table multiplied by the overhead factor.
  auto length = std::pow(2, std::ceil(std::log2(t.size() * OVERHEAD())));

  if (length == 0) {
    mask_ = 0;
  } else {
    mask_ = length - 1;
  }

  values_ = device_vec<int64_t>(length);
  fill<<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(values_.data(), length, -1);
  cudaCheck(cudaDeviceSynchronize());

  if (t.size() == 0) {
    return;
  }

  TRACE_START(hash_table_kernel);
  dim3 blockDim = {32, 1, 1};
  dim3 gridDim = {ROUND_UP_TO_NEAREST(t.size(), 32), 1, 1};
  construct_multi_column_hash_table<Prov, T...>
      <<<gridDim, blockDim>>>(t.view(), HashIndexView(*this));
  cudaCheck(cudaDeviceSynchronize());
  TRACE_END(hash_table_kernel);
}

HashIndex::HashIndex(size_t size, uint32_t width) : width_(width) {
  TRACE_START(construct_hash_index);
  auto length = std::pow(2, std::ceil(std::log2(size * OVERHEAD())));

  if (length == 0) {
    mask_ = 0;
  } else {
    mask_ = length - 1;
  }

  values_ = device_vec<int64_t>(length);
  thrust::fill(thrust::device, values_.begin(), values_.end(), -1);
}

template <typename Prov>
std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<Prov> &t,
                                               size_t width) {
  auto schema_flattened = t.schema().to_flat_vector();
  schema_flattened.resize(width);
  if (schema_flattened == std::vector{ValueType::USize(), ValueType::USize()}) {
    return std::make_unique<HashIndex>(t, Product<ValueUSize, ValueUSize>());
  } else if (schema_flattened ==
             std::vector{ValueType::U32(), ValueType::U32()}) {
    return std::make_unique<HashIndex>(t, Product<ValueU32, ValueU32>());
  } else if (schema_flattened ==
             std::vector{ValueType::F32(), ValueType::F32()}) {
    return std::make_unique<HashIndex>(t, Product<ValueF32, ValueF32>());
  } else if (schema_flattened ==
             std::vector{ValueType::Symbol(), ValueType::Symbol()}) {
    return std::make_unique<HashIndex>(t, Product<ValueSymbol, ValueSymbol>());
  } else if (schema_flattened == std::vector{ValueType::U32(), ValueType::U32(),
                                             ValueType::U32(),
                                             ValueType::U32()}) {
    return std::make_unique<HashIndex>(t, Product<ValueU32, ValueU32, ValueU32, ValueU32>());
  }

  if (width == 1) {
    DISPATCH_ON_KIND(schema_flattened.at(0), T,
                     return std::make_unique<HashIndex>(t, Product<T>()););
  }

  throw std::runtime_error("Unsupported join schema: " + to_string(t.schema()) +
                           ", width: " + to_string(width));
}

__host__ __device__ size_t HashIndex::width() const { return width_; }

__host__ __device__ device_vec<int64_t> &HashIndex::values() { return values_; }

__host__ __device__ size_t HashIndex::mask() const { return mask_; }

float HashIndex::load_factor() const {
  return 1. - (static_cast<float>(thrust::count_if(
                   thrust::device, values_.cbegin(), values_.cend(),
                   equals_const<int64_t, -1>())) /
               static_cast<float>(values_.size()));
}

template <typename Prov, typename... T>
void HashIndex::grow(size_t size, Table<Prov> &t, Product<T...>) {
  TRACE_START(grow_hash_index);
  if (size < values_.size()) {
    return;
  }
  if (size == 0) {
    return;
  }

  constexpr size_t N = sizeof...(T);

  if (SAFETY) {
    if (N != t.schema().width()) {
      throw std::runtime_error("Column count mismatch");
    }
    size_t type_sizes[N] = {sizeof(typename T::type)...};
    for (size_t i = 0; i < sizeof...(T); i++) {
      if (t.schema().flatten().at(i).singleton().size() != type_sizes[i]) {
        throw std::runtime_error("Column type mismatch");
      }
    }
  }

  auto length = std::pow(2, std::ceil(std::log2(size)));

  if (length == 0) {
    mask_ = 0;
  } else {
    mask_ = length - 1;
  }

  values_.resize(length);
  thrust::fill(thrust::device, values_.begin(), values_.end(), -1);

  if (t.size() == 0) {
    return;
  }

  construct_multi_column_hash_table<Prov, T...>
      <<<ROUND_UP_TO_NEAREST(t.size(), 128), 128>>>(t.view(), HashIndexView(*this));
  cudaCheck(cudaDeviceSynchronize());
}
template <typename Prov>
void HashIndex::grow(size_t size, Table<Prov> &t) {
  auto schema_flattened = t.schema().to_flat_vector();
  schema_flattened.resize(width());
  if (schema_flattened == std::vector{ValueType::USize(), ValueType::USize()}) {
    return this->grow(size, t, Product<ValueUSize, ValueUSize>());
  } else if (schema_flattened ==
             std::vector{ValueType::U32(), ValueType::U32()}) {
    return this->grow(size, t, Product<ValueU32, ValueU32>());
  } else if (schema_flattened ==
             std::vector{ValueType::F32(), ValueType::F32()}) {
    return this->grow(size, t, Product<ValueF32, ValueF32>());
  } else if (schema_flattened == std::vector{ValueType::U32(), ValueType::U32(),
                                             ValueType::U32(),
                                             ValueType::U32()}) {
    return this->grow(size, t,
                      Product<ValueU32, ValueU32, ValueU32, ValueU32>());
  }

  DISPATCH_ON_KIND(schema_flattened.at(0), T,
                   return this->grow(size, t, Product<T>()););

  throw std::runtime_error(
      "Unsupported growth schema: " + to_string(t.schema()) +
      ", width: " + to_string(width()));
}

std::unique_ptr<HashIndex> HashIndex::to(size_t size, const Allocator &alloc) const {
  // TODO: this performs an unecessary allocation
  auto output = std::make_unique<HashIndex>(size, this->width());
  assert(output->values_.size() == this->values_.size());
  output->values_ = this->values_.to(alloc);

  return output;
}

//__device__ bool HashIndex::try_insert(size_t key, int64_t value) {
//  unsigned long long int *lock_location =
//      reinterpret_cast<unsigned long long int *>(&this->value_at(key));
//  auto last_value =
//      atomicCAS(lock_location, static_cast<unsigned long long int>(-1),
//                static_cast<unsigned long long int>(value));
//  if (last_value == static_cast<unsigned long long int>(-1)) {
//    return true;
//  }
//  return false;
//}

std::ostream &operator<<(std::ostream &os, const HashIndex &index) {
  os << "MultiColumnHashIndex(load_: " << index.load_factor() << "\n";
  os << index.values_ << ")";
  return os;
}

#define PROV UnitProvenance
template std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<PROV> &t,
                                                        size_t);
template void HashIndex::grow(size_t size, Table<PROV> &t);
#undef PROV

#define PROV MinMaxProbProvenance
template std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<PROV> &t,
                                                        size_t);
template void HashIndex::grow(size_t size, Table<PROV> &t);
#undef PROV

#define PROV DiffMinMaxProbProvenance
template std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<PROV> &t,
                                                        size_t);
template void HashIndex::grow(size_t size, Table<PROV> &t);

#undef PROV
#define PROV DiffAddMultProbProvenance<>
template std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<PROV> &t,
                                                        size_t);
template void HashIndex::grow(size_t size, Table<PROV> &t);

#undef PROV
#define PROV DiffTopKProofsProvenance<>
template std::unique_ptr<HashIndex> HashIndex::make_managed(const Table<PROV> &t,
                                                        size_t);
template void HashIndex::grow(size_t size, Table<PROV> &t);
#undef PROV
