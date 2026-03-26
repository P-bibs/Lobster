#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>
#include <set>

#include "alloc.h"
#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "provenance.h"
#include "table.h"
#include "utils.h"

std::set<void *> ALL_TABLES;
int TOTAL_TABLES = 0;
void print_all_tables() {
  std::cout << "Total tables: " << TOTAL_TABLES << std::endl;
  std::cout << "Tables remaining: " << ALL_TABLES.size() << std::endl;

  int i = 0;
  for (auto table : ALL_TABLES) {
    TableStorage<DiffMinMaxProbProvenance> *t =
        (TableStorage<DiffMinMaxProbProvenance> *)table;

    std::cout << "Table " << i << ": width=" << t->width()
              << " size=" << t->size() << " refcount=" << *t->refcount()
              << std::endl;
    i++;
  }
}

void add_table(void *table) {
  TOTAL_TABLES += 1;
  ALL_TABLES.insert(table);
}
void remove_table(void *table) { ALL_TABLES.erase(table); }

template <typename Prov>
TableStorage<Prov>::TableStorage(device_vec<Tag> &&tags,
                                 Array<device_buffer> &&values,
                                 device_vec<char> &&sample_mask)
    : refcount_(1),
      tags_(std::move(tags)),
      facts_(std::move(values)),
      sample_mask_(std::move(sample_mask)),
      index_(nullptr) {
  add_table(this);
}

template <typename Prov>
TableStorage<Prov>::TableStorage(const Array<typename Prov::Tag> &tags,
                                 const Array<Array<Value>> &values,
                                 const std::vector<char> &sample_mask,
                                 TupleType schema)
    : refcount_(1),
      tags_(tags),
      facts_(values.size()),
      sample_mask_(sample_mask),
      index_(nullptr) {
  for (size_t i = 0; i < values.size(); i++) {
    facts_[i] = device_buffer(values[i], schema.flatten().at(i).singleton());
  }
  add_table(this);
}
template <typename Prov>
TableStorage<Prov>::TableStorage(TupleType schema)
    : refcount_(1), tags_(), facts_(schema.width()), index_(nullptr) {
  for (size_t i = 0; i < facts_.size(); i++) {
    new (&facts_[i]) device_buffer(schema.flatten().at(i).singleton());
  }
  add_table(this);
}

template <typename Prov>
uint32_t *TableStorage<Prov>::refcount() {
  return &refcount_;
}
template <typename Prov>
void TableStorage<Prov>::increment_refcount() {
  refcount_ += 1;
}
template <typename Prov>
void TableStorage<Prov>::decrement_refcount() {
  refcount_ -= 1;
}

template <typename Prov>
device_vec<typename Prov::Tag> &TableStorage<Prov>::tags() {
  return tags_;
}
template <typename Prov>
const device_vec<typename Prov::Tag> &TableStorage<Prov>::tags() const {
  return tags_;
}
template <typename Prov>
Array<device_buffer> &TableStorage<Prov>::values() {
  return facts_;
}
template <typename Prov>
const Array<device_buffer> &TableStorage<Prov>::values() const {
  return facts_;
}

template <typename Prov>
const device_buffer &TableStorage<Prov>::column(size_t i) const {
  return facts_[i];
}
template <typename Prov>
device_buffer &TableStorage<Prov>::column(size_t i) {
  return facts_[i];
}
template <typename Prov>
device_vec<char> &TableStorage<Prov>::sample_mask() {
  return sample_mask_;
}
template <typename Prov>
const device_vec<char> &TableStorage<Prov>::sample_mask() const {
  return sample_mask_;
}

template <typename Prov>
const void *TableStorage<Prov>::at_raw(size_t row, size_t column) const {
  return facts_[column].at_raw(row);
}

template <typename Prov>
size_t TableStorage<Prov>::size() const {
  return facts_[0].size();
}
template <typename Prov>
size_t TableStorage<Prov>::width() const {
  return facts_.size();
}

template <typename Prov>
std::unique_ptr<HashIndex> &TableStorage<Prov>::index() {
  return index_;
}
template <typename Prov>
const std::unique_ptr<HashIndex> &TableStorage<Prov>::index() const {
  return index_;
}
template <typename Prov>
HashIndex *TableStorage<Prov>::index(const Table<Prov> &table, size_t width) {
  auto index = index_.get();
  if (index && index->width() == width) {
    return index;
  } else {
    hINFO("Creating new index for table with schema "
          << table.schema() << " and size " << table.size());
    index_ = HashIndex::make_managed(table, width);
    return index_.get();
  }
}

template <typename Prov>
__global__ void validate_kernel(typename Prov::Tag *tags, size_t size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  Prov::validate(tags[index]);
}

template <typename Prov>
void TableStorage<Prov>::validate(const TupleType &schema) const {
  if (!SAFETY) {
    return;
  }
  TRACE_START(table_validate);

  if (refcount_ == 0) {
    PANIC("Table: refcount is 0");
  }

  size_t row_count = this->size();
  for (size_t i = 0; i < this->width(); i++) {
    if (row_count != facts_[i].size()) {
      PANIC("Table: column lengths do not match: %zu vs %zu", row_count,
            facts_[i].size());
    }
  }

  if (Prov::is_unit) {
    if (tags_.size() != 0) {
      PANIC("Table: tags should be empty for unit provenance");
    }
  } else {
    if (row_count != tags_.size()) {
      PANIC("Table: column lengths do not match tag length: %zu vs %zu",
            row_count, tags_.size());
    }
  }
  if (row_count != sample_mask_.size()) {
    PANIC("Table: column lengths do not match sample mask length: %zu vs %zu",
          row_count, sample_mask_.size());
  }

  if (facts_.size() != schema.width()) {
    PANIC(
        "Table: column count does not match schema: %zu in table vs "
        "%zu in schema",
        facts_.size(), schema.width());
  }

  for (size_t i = 0; i < facts_.size(); i++) {
    if (facts_[i].type() != schema.flatten().at(i).singleton()) {
      PANIC("Table: column %zu type does not match schema: %zu vs %zu", i,
            (size_t)facts_[i].type().tag(),
            (size_t)schema.flatten().at(i).singleton().tag());
    }
  }

  if (!Prov::is_unit) {
    validate_kernel<Prov>
        <<<ROUND_UP_TO_NEAREST(row_count, 128), 128>>>(tags_.data(), row_count);
    cudaCheck(cudaDeviceSynchronize());
  }
}
template <typename Prov>
TableStorage<Prov>::~TableStorage() {
  assert(refcount_ == 0);
  remove_table(this);
}

template <typename Prov>
Table<Prov>::~Table() {
  if (storage_) {
    storage_->decrement_refcount();
    if (*(storage_->refcount()) == 0) {
      storage_->~TableStorage();
      HostAlloc::destroy(storage_);
    }
  }
}

template <typename Prov>
Table<Prov>::Table(const TupleType &schema)
    : schema_(schema), storage_(nullptr) {
  validate();
}
template <typename Prov>
Table<Prov>::Table(const TupleType &schema, TableStorage<Prov> *storage)
    : schema_(schema), storage_(storage) {
  if (storage_) {
    storage_->increment_refcount();
  }
  validate();
}
template <typename Prov>
Table<Prov>::Table(TupleType schema, device_vec<Tag> &&tags,
                   Array<device_buffer> &&values,
                   device_vec<char> &&sample_mask)
    : schema_(schema),
      storage_(HostAlloc::make<TableStorage<Prov>>(
          std::move(tags), std::move(values), std::move(sample_mask))) {
  validate();
}
template <typename Prov>
Table<Prov>::Table(const TupleType &schema, const Array<Tag> &tags,
                   const Array<Array<Value>> &values,
                   const std::vector<char> &sample_mask)
    : schema_(schema),
      storage_(HostAlloc::make<TableStorage<Prov>>(tags, values, sample_mask,
                                                   schema)) {
  validate();
}
template <typename Prov>
Table<Prov>::Table(const Table &other)
    : schema_(other.schema_), storage_(other.storage_) {
  other.validate();
  validate();
  if (storage_) {
    storage_->increment_refcount();
  }
}
template <typename Prov>
Table<Prov> &Table<Prov>::operator=(const Table<Prov> &other) {
  this->validate();
  other.validate();

  if (this->storage_ == other.storage_) {
    if (SAFETY && this->storage_ != nullptr) {
      assert(this->schema_ == other.schema_);
    }
    return *this;
  }

  if (storage_) {
    storage_->decrement_refcount();
    if (*(storage_->refcount()) == 0) {
      storage_->~TableStorage();
      HostAlloc::destroy(storage_);
    }
  }
  schema_ = other.schema_;
  storage_ = other.storage_;
  if (storage_) {
    storage_->increment_refcount();
  }
  this->validate();
  return *this;
}

template <typename Prov>
void Table<Prov>::append(const Table &other) {
  if (this->schema() != other.schema()) {
    throw std::runtime_error("Schema mismatch: " + to_string(this->schema()) +
                             " vs " + to_string(other.schema()));
  }
  if (storage_ == nullptr && other.storage() == nullptr) {
    return;
  } else if (storage_ == nullptr) {
    *this = other;
  } else if (other.storage() == nullptr) {
    return;
  } else {
    for (size_t col = 0; col < other.width(); col++) {
      this->column_mut(col).append(other.column(col));
    }
    this->tags().append(other.tags());
    this->sample_mask().append(other.sample_mask());
    this->validate();
  }
}

template <typename Prov>
TableView<Prov> Table<Prov>::view() const {
  return TableView<Prov>(this->storage()->values(), this->storage()->tags(),
                         schema_);
}

template <typename Prov>
Table<Prov> Table<Prov>::as_flattened() const {
  // TODO: if we use the non-copy constructor, the refcounts get messed up.
  Table<Prov> t(*this);
  t.schema_ = schema_.flatten();
  return t;
}

template <typename Prov>
managed_ptr<Table<Prov>> Table<Prov>::to_managed() {
  return managed_ptr<Table>::make(*this);
}

template <typename Prov>
Table<Prov> Table<Prov>::to(const Allocator &alloc) const {
  // TODO: make this faster by not moving if storage_ is already `alloc`
  TRACE_START(Table_to);
  if (!storage_) {
    return Table<Prov>(schema_);
  }
  Array<device_buffer> values(this->width());
  for (size_t i = 0; i < this->width(); i++) {
    new (&values[i]) device_buffer(schema_.flatten().at(i).singleton());
    values[i] = this->values()[i].to(alloc);
  }
  Table<Prov> output(schema_, this->tags().to(alloc), std::move(values),
                     this->sample_mask().to(alloc));

  const auto &index = this->storage()->index();
  if (index.get()) {
    output.storage()->index() = index->to(this->size(), alloc);
  }

  return output;
}

template <typename Prov>
Table<Prov> Table<Prov>::clone() const {
  TRACE_START(Table_clone);
  if (!storage_) {
    return Table<Prov>(schema_);
  }
  auto output = Table<Prov>(schema_, std::move(storage_->tags().clone()),
                            std::move(storage_->values().clone()),
                            std::move(this->storage_->sample_mask().clone()));
  validate();
  return output;
}
template <typename Prov>
device_vec<char> &Table<Prov>::sample_mask() {
  return storage_->sample_mask();
}
template <typename Prov>
const device_vec<char> &Table<Prov>::sample_mask() const {
  return storage_->sample_mask();
}

template <typename Prov>
std::vector<size_t> Table<Prov>::sample_sizes() const {
  TRACE_START(Table_sample_sizes);

  auto batch_size = get_batch_size();
  if (!storage_) {
    std::vector<size_t> empty;
    for (size_t i = 0; i < batch_size; i++) {
      empty.push_back(0);
    }
    return empty;
  }

  auto sample_mask = this->sample_mask().to_host();
  std::vector<size_t> sample_sizes(batch_size, 0);

  for (size_t i = 0; i < this->size(); i++) {
    sample_sizes[sample_mask[i]] += 1;
  }

  return sample_sizes;
}

template <typename Prov>
void Table<Prov>::validate() const {
  if (!SAFETY) {
    return;
  }
  if (storage_ == nullptr) {
    return;
  }
  storage_->validate(schema_);
}

template <typename Prov>
void Table<Prov>::clear() {
  if (storage_) {
    storage_->decrement_refcount();
    if (*(storage_->refcount()) == 0) {
      storage_->~TableStorage();
      HostAlloc::destroy(storage_);
    }
  }
  storage_ = nullptr;
}

template <typename Prov>
const TupleType &Table<Prov>::schema() const {
  return schema_;
}
template <typename Prov>
const TableStorage<Prov> *Table<Prov>::storage() const {
  return storage_;
}
template <typename Prov>
device_vec<typename Prov::Tag> &Table<Prov>::tags() {
  if (!storage_) {
    throw std::runtime_error("Cannot access tags of empty table");
  }
  return storage_->tags();
}
template <typename Prov>
const device_vec<typename Prov::Tag> &Table<Prov>::tags() const {
  static device_vec<typename Prov::Tag> *empty;
  if (!storage_) {
    if (!empty) {
      empty = new device_vec<typename Prov::Tag>();
    }
    return *empty;
  }
  return storage_->tags();
}

template <typename Prov>
const device_buffer &Table<Prov>::column_buffer(size_t i) const {
  static device_buffer *empty;
  if (!storage_) {
    if (!empty) {
      empty = new device_buffer();
    }
    return *empty;
  }
  return values()[i];
}

template <typename Prov>
const Array<device_buffer> &Table<Prov>::values() const {
  return storage_->values();
}

template <typename Prov>
const void *Table<Prov>::at_raw(size_t row, size_t column) const {
  return this->storage_->at_raw(row, column);
}

template <typename Prov>
const device_buffer &Table<Prov>::column(size_t i) const {
  return storage_->column(i);
}
template <typename Prov>
device_buffer &Table<Prov>::column_mut(size_t i) {
  return storage_->column(i);
}

template <typename Prov>
size_t Table<Prov>::size() const {
  if (!storage_) {
    return 0;
  }
  return storage_->size();
}
template <typename Prov>
size_t Table<Prov>::width() const {
  return schema_.width();
}
template <typename Prov>
TableStorage<Prov> *Table<Prov>::storage() {
  return storage_;
}

template <typename Prov>
std::ostream &operator<<(std::ostream &os, const Table<Prov> &table) {
  if (table.size() == 0) {
    os << "<empty>";
    return os;
  }
  Array<Array<Value>> facts_host(table.width());
  for (size_t i = 0; i < table.width(); i++) {
    new (&facts_host[i]) Array<Value>();
    facts_host[i] = table.column_buffer(i).to_host_tagged(
        Value::tag_from_type(table.schema().flatten().at(i).singleton().tag()));
  }
  auto sample_mask_host = table.sample_mask().to_host();
  auto tags_host = table.tags().to_host();
  os << "\t{schema: {";

  os << table.schema();
  std::cout << "}" << std::endl;
  os << "\t";
  for (size_t row = 0; row < table.size(); row++) {
    os << "<" << (int)sample_mask_host.at(row) << ">";
    if (!Prov::is_unit) {
      os << tags_host.at(row) << "::";
    }
    os << "(";
    for (size_t col = 0; col < table.width(); col++) {
      auto value = facts_host[col][row];
      os << value;
      if (col < table.width() - 1) {
        os << ", ";
      }
    }
    if (row < table.size() - 1) {
      os << "), ";
    } else {
      os << ")";
    }
  }
  os << "\n\t}";
  return os;
}

#define PROV UnitProvenance
template class Table<PROV>;
template class TableStorage<PROV>;
template std::ostream &operator<<(std::ostream &, const Table<PROV> &);
#undef PROV
#define PROV MinMaxProbProvenance
template class Table<PROV>;
template class TableStorage<PROV>;
template std::ostream &operator<<(std::ostream &, const Table<PROV> &);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template class Table<PROV>;
template class TableStorage<PROV>;
template std::ostream &operator<<(std::ostream &, const Table<PROV> &);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template class Table<PROV>;
template class TableStorage<PROV>;
template std::ostream &operator<<(std::ostream &, const Table<PROV> &);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template class Table<PROV>;
template class TableStorage<PROV>;
template std::ostream &operator<<(std::ostream &, const Table<PROV> &);
#undef PROV
