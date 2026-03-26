#pragma once

#include "alloc.h"
#include "device_vec.h"
#include "utils.h"

template <typename Prov>
class Table;
template <typename Prov>
class TableView;

class HashIndex {
 private:
  // the mask is used as an efficient modulo for hashes
  size_t mask_;
  // the number of columns included in the index
  uint32_t width_;
  // TODO: is this faster if we use int32_t? If so, how to deal with tables
  // larger than 2^16?
  device_vec<int64_t> values_;

 public:
  template <typename Prov, uint32_t... Indices, typename... T>
  __device__ size_t hash_row_unsafe(const TableView<Prov> &t, uint32_t row,
                                    std::integer_sequence<uint32_t, Indices...>,
                                    Product<T...>) const {
    // SAFETY: this function needs to be called with the correct set of indices
    return multi_hash(t.template at<typename T::type>(row, Indices)...);
  }

  template <typename Prov, typename... T>
  HashIndex(const Table<Prov> &t, Product<T...>);
  HashIndex(size_t size, uint32_t width);

  template <typename Prov>
  static std::unique_ptr<HashIndex> make_managed(const Table<Prov> &t,
                                             size_t width);

  __host__ __device__ size_t size() const { return this->values_.size(); };
  __host__ __device__ size_t width() const;

  __host__ __device__ device_vec<int64_t> &values();
  __host__ __device__ size_t mask() const;

  __device__ bool occupied(size_t i) const {
    return values_.at_device(i) != -1;
  };

  __device__ const int64_t &value_at(size_t i) const {
    return values_.at_device(i);
  };
  __device__ int64_t &value_at(size_t i) { return values_.at_device(i); };

  float load_factor() const;

  template <typename Prov, typename... T>
  void grow(size_t size, Table<Prov> &t, Product<T...>);
  template <typename Prov>
  void grow(size_t size, Table<Prov> &t);

  std::unique_ptr<HashIndex> to(size_t size, const Allocator &alloc) const;

  __host__ __device__ size_t incr_position(size_t position) const {
    return (position + 1) & mask_;
  };

  template <typename Prov, typename... T>
  __host__ __device__ size_t hash_row(const TableView<Prov> &t,
                                      uint32_t row) const {
    return hash_row_unsafe(t, row,
                           std::make_integer_sequence<uint32_t, sizeof...(T)>(),
                           Product<T...>()) &
           mask_;
  }
  friend std::ostream &operator<<(std::ostream &os, const HashIndex &index);
};

// TODO: fix const-correctness
class HashIndexView {
 private:
  size_t mask_;
  uint32_t width_;
  int64_t *values_;
  size_t size_;

 public:
  HashIndexView(const HashIndex &index)
      : mask_(index.mask()),
        width_(index.width()),
        values_(const_cast<HashIndex &>(index).values().data()),
        size_(index.size()) {}
  HashIndexView(HashIndex &index)
      : mask_(index.mask()),
        width_(index.width()),
        values_(index.values().data()),
        size_(index.size()) {}
  __host__ __device__ int64_t *values() { return this->values_; };
  __host__ __device__ const int64_t *values() const { return this->values_; };

  __host__ __device__ size_t size() const { return this->size_; };
  __host__ __device__ size_t width() const { return this->width_; };
  __host__ __device__ size_t mask() const { return this->mask_; };


  __device__ bool try_insert(size_t key, int64_t value);

  __device__ bool occupied(size_t i) const {
    if (SAFETY) {
      if (i >= size_) {
        PANIC("index out of bounds");
      }
    }
    return values_[i] != -1;
  };

  __device__ const int64_t &value_at(size_t i) const {
    if (SAFETY) {
      if (i >= size_) {
        PANIC("index out of bounds");
      }
    }
    return values_[i];
  };
  __device__ int64_t &value_at(size_t i) {
    if (SAFETY) {
      if (i >= size_) {
        PANIC("index out of bounds");
      }
    }
    return values_[i];
  };
  __host__ __device__ size_t incr_position(size_t position) const {
    return (position + 1) & mask_;
  };

  template <typename Prov, uint32_t... Indices, typename... T>
  __device__ size_t hash_row_unsafe(const TableView<Prov> &t, uint32_t row,
                                    std::integer_sequence<uint32_t, Indices...>,
                                    Product<T...>) const {
    // SAFETY: this function needs to be called with the correct set of indices
    return multi_hash(t.template at<typename T::type>(row, Indices)...);
  }
  template <typename Prov, typename... T>
  __host__ __device__ size_t hash_row(const TableView<Prov> &t,
                                      uint32_t row) const {
    return this->hash_row_unsafe(
               t, row, std::make_integer_sequence<uint32_t, sizeof...(T)>(),
               Product<T...>()) &
           mask_;
  }
};
