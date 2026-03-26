#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <memory>

#include "alloc.h"
#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "table_index.h"
#include "utils.h"

void print_all_tables();

/**
 * Given a tuple of iterators `t` and an index `index`, returns a tuple of the
 * values at index `index` in each iterator in `t`.
 */
template <typename... T, int... Index>
__host__ __device__ std::tuple<T...> index_tuple(
    std::tuple<device_buffer::device_buffer_iter<T>...> &t, int index,
    std::integer_sequence<int, Index...>) {
  return std::make_tuple(std::get<Index>(t)[index]...);
}

/**
 * A non-owning view of the fact columns in a table
 */
template <typename... T>
class FactsView {
 private:
  std::tuple<device_buffer::device_buffer_iter<T>...> columns_;

  template <int... Index>
  static std::tuple<device_buffer::device_buffer_iter<T>...> make_columns(
      Array<device_buffer> &facts, std::integer_sequence<int, Index...>) {
    return std::make_tuple(facts[Index].begin<T>()...);
  }

 public:
  FactsView(Array<device_buffer> &facts) {
    columns_ =
        make_columns(facts, std::make_integer_sequence<int, sizeof...(T)>());
  }

  FactsView(device_buffer::device_buffer_iter<T>... columns)
      : columns_(columns...) {}

  template <size_t I>
  __host__ __device__ auto column() {
    return std::get<I>(columns_);
  }

  __host__ __device__ std::tuple<device_buffer::device_buffer_iter<T>...> &
  columns() {
    return columns_;
  }
};

template <typename T>
class ColumnView {
 private:
  size_t length_;
  const T *data_;

 public:
  ColumnView(const device_buffer &data)
      : length_(data.size()), data_(reinterpret_cast<T *>(data.data())) {
    assert(sizeof(T) == data.stride());
  }
  __host__ __device__ size_t size() const { return length_; }
  const __host__ __device__ T &at(size_t i) const { return data_[i]; }

  const T *begin() const { return data_; }
  const T *end() const { return data_ + length_; }
};

template <typename T>
class ColumnViewMut {
 private:
  size_t length_;
  T *data_;

 public:
  ColumnViewMut(device_buffer &data)
      : length_(data.size()), data_(reinterpret_cast<T *>(data.data())) {
    assert(sizeof(T) == data.stride());
  }
  __host__ __device__ size_t size() const { return length_; }
  const __host__ __device__ T &at(size_t i) const { return data_[i]; }
  __host__ __device__ T &at(size_t i) { return data_[i]; }

  T *begin() { return data_; }
  T *end() { return data_ + length_; }
};

template <typename Prov>
class TableView {
 private:
  static const int MAX_COLUMNS = 6;
  device_buffer::device_buffer_iter<char> columns_[MAX_COLUMNS];
  typename device_vec<typename Prov::Tag>::device_vec_iter tags_;
  TupleType schema;
  size_t size_;
  char strides_[MAX_COLUMNS];

 public:
  TableView(const Array<device_buffer> &facts,
            const device_vec<typename Prov::Tag> &tags, TupleType schema)
      : tags_(tags.cbegin()), schema(schema), size_(facts[0].size()) {
    if (facts.size() > MAX_COLUMNS) {
      PANIC("too many columns");
    }
    for (size_t i = 0; i < facts.size(); i++) {
      columns_[i] = facts[i].cbegin<char>();
      strides_[i] = facts[i].stride();
    }
  }

  template <typename T>
  __device__ const T &at(size_t row, size_t column) const {
    const device_buffer::device_buffer_iter<T> col =
        *reinterpret_cast<const device_buffer::device_buffer_iter<T> *>(
            &columns_[column]);
    return col[row];
  }
  __host__ __device__ const void *at_raw(size_t row, size_t column) const {
    return (columns_[column] + row * strides_[column]).data();
  }
  __host__ __device__ size_t size() const { return size_; }
};

template <typename Prov>
class TableStorage {
 public:
  using Tag = typename Prov::Tag;

 private:
  Array<device_buffer> facts_;
  device_vec<Tag> tags_;
  uint32_t refcount_;
  std::unique_ptr<HashIndex> index_;
  device_vec<char> sample_mask_;
  Allocator alloc_;

 public:
  // Construct a Table by moving data from device memory
  TableStorage(device_vec<Tag> &&tags, Array<device_buffer> &&values,
               device_vec<char> &&sample_mask);

  // Construct a Table by copying data from host memory
  TableStorage(const Array<Tag> &tags, const Array<Array<Value>> &values,
               const std::vector<char> &sample_mask, TupleType schema);

  // Construct an empty table
  TableStorage(TupleType schema);

  uint32_t *refcount();
  void increment_refcount();
  void decrement_refcount();

  device_vec<Tag> &tags();
  const device_vec<Tag> &tags() const;

  template <typename T>
  ColumnView<T> column(size_t i) const {
    return ColumnView<T>(facts_[i]);
  }
  template <typename T>
  ColumnViewMut<T> column_mut(size_t i) {
    return ColumnViewMut<T>(facts_[i]);
  }

  Array<device_buffer> &values();
  const Array<device_buffer> &values() const;
  const device_buffer &column(size_t i) const;
  device_buffer &column(size_t i);

  device_vec<char> &sample_mask();
  const device_vec<char> &sample_mask() const;

  const void *at_raw(size_t row, size_t column) const;

  size_t size() const;
  size_t width() const;
  std::unique_ptr<HashIndex> &index();
  const std::unique_ptr<HashIndex> &index() const;
  HashIndex *index(const Table<Prov> &schema, size_t width);

  void validate(const TupleType &schema) const;
  ~TableStorage();
};

template <typename Prov>
class Table {
 private:
  TupleType schema_;
  TableStorage<Prov> *storage_;

 public:
  using Tag = typename Prov::Tag;

  static device_vec<char> sizes_to_mask(std::vector<uint32_t> sizes);

  ~Table();

  Table(const TupleType &schema);
  Table(const TupleType &schema, TableStorage<Prov> *storage);
  Table(TupleType schema, device_vec<Tag> &&tags, Array<device_buffer> &&values,
        device_vec<char> &&sample_mask);
  Table(const TupleType &schema, const Array<Tag> &tags,
        const Array<Array<Value>> &values,
        const std::vector<char> &sample_mask);
  Table(const Table &other);
  Table &operator=(const Table &other);

  void append(const Table &other);

  device_vec<char> &sample_mask();
  const device_vec<char> &sample_mask() const;

  std::vector<size_t> sample_sizes() const;

  TableView<Prov> view() const;

  Table<Prov> as_flattened() const;

  managed_ptr<Table> to_managed();

  Table to(const Allocator &alloc) const;

  Table<Prov> clone() const;

  size_t sorted_until() const;
  void validate() const;

  void clear();

  const TupleType &schema() const;
  size_t size() const;
  size_t width() const;
  const TableStorage<Prov> *storage() const;
  TableStorage<Prov> *storage();

  device_vec<Tag> &tags();
  const device_vec<Tag> &tags() const;

  template <typename T>
  ColumnView<T> column(size_t i) const {
    return storage_->template column<T>(i);
  }
  template <typename T>
  ColumnViewMut<T> column_mut(size_t i) {
    return storage_->template column_mut<T>(i);
  }

  const device_buffer &column_buffer(size_t i) const;

  const Array<device_buffer> &values() const;

  template <typename T>
  const T host_at(size_t row, size_t column) const {
    auto *ptr = this->storage_->at_raw(row, column);
    T val;
    cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheck(cudaDeviceSynchronize());
    return val;
  }

  const void *at_raw(size_t row, size_t column) const;

  // TODO: consolidate this API to make more sense and expose
  // fewer internal details
  const device_buffer &column(size_t i) const;
  device_buffer &column_mut(size_t i);

  template <typename T>
  device_buffer::iterator<T> column_begin(size_t column) {
    return storage_->column(column).template begin<T>();
  }
  template <typename T>
  device_buffer::iterator<T> column_end(size_t column) {
    return storage_->column(column).template end<T>();
  }

  template <typename T>
  const device_buffer::iterator<T> column_cbegin(size_t column) const {
    return storage_->column(column).template begin<T>();
  }
  template <typename T>
  const device_buffer::iterator<T> column_cend(size_t column) const {
    return storage_->column(column).template end<T>();
  }

  HashIndex *index_mut(size_t width) {
    if (!storage_) {
      return nullptr;
    }
    return storage_->index(*this, width);
  }
  const HashIndex *index(size_t width) const {
    if (!storage_) {
      return nullptr;
    }
    return storage_->index(*this, width);
  }

 private:
  template <typename... T, int... Indexes>
  __device__ static bool row_matches_unsafe(
      const TableView<Prov> &left_table, const TableView<Prov> &right_table,
      uint32_t left_index, uint32_t right_index,
      std::integer_sequence<int, Indexes...>) {
    return ((left_table.template at<typename T::type>(left_index, Indexes) ==
             right_table.template at<typename T::type>(right_index, Indexes)) &&
            ...);
  }

 public:
  template <typename... T>
  __device__ static bool row_matches(const TableView<Prov> &left_table,
                                     const TableView<Prov> &right_table,
                                     uint32_t left_index,
                                     uint32_t right_index) {
    return Table::row_matches_unsafe<T...>(
        left_table, right_table, left_index, right_index,
        std::make_integer_sequence<int, sizeof...(T)>());
  }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &os, const Table<T> &table);
};
