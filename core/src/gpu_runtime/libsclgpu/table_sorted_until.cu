#include <thrust/sort.h>

#include <utility>

#include "bindings.h"
#include "provenance.h"
#include "table.h"
#include "utils.h"

template <typename Prov, typename... T, int... Index>
size_t impl(const Table<Prov> &table, Product<T...>,
            std::integer_sequence<int, Index...>) {
  auto iter = thrust::make_zip_iterator(thrust::make_tuple(
      table.sample_mask().cbegin(),
      table.template column_cbegin<typename T::sort_type>(Index)...));

  auto sorted_til =
      thrust::is_sorted_until(thrust::device, iter, iter + table.size());
  auto sorted_length = thrust::distance(iter, sorted_til);
  return sorted_length;
}

template <typename Prov>
size_t Table<Prov>::sorted_until() const {
  hINFO("Sorted until: " << this->schema());
  hINFO("Input table:\n" << *this);

  if (this->size() == 0) {
    return 0;
  }
  auto schema_flattened = this->schema().flatten();
  if (schema_flattened == TupleType(std::vector{TupleType(ValueType::U32())})) {
    return impl(*this, Product<ValueU32>(),
                std::make_integer_sequence<int, 1>());
  } else if (schema_flattened == TupleType(std::vector{TupleType(ValueType::F32())})) {
    return impl(*this, Product<ValueF32>(),
                std::make_integer_sequence<int, 1>());
  } else if (schema_flattened ==
             TupleType({ValueType::U32(), ValueType::U32()})) {
    return impl(*this, Product<ValueU32, ValueU32>(),
                std::make_integer_sequence<int, 2>());
  } else if (schema_flattened == TupleType({ValueType::U32(), ValueType::U32(),
                                            ValueType::U32()})) {
    return impl(*this, Product<ValueU32, ValueU32, ValueU32>(),
                std::make_integer_sequence<int, 3>());
  } else if (schema_flattened ==
             TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),
                        ValueType::U32()})) {
    return impl(*this, Product<ValueU32, ValueU32, ValueU32, ValueU32>(),
                std::make_integer_sequence<int, 4>());
  } else if (schema_flattened ==
             TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),
                        ValueType::U32(),ValueType::U32()})) {
    return impl(*this, Product<ValueU32, ValueU32, ValueU32, ValueU32,ValueU32>(),
                std::make_integer_sequence<int, 5>());
  } else if (schema_flattened == TupleType({ValueType::U32(), ValueType::U32(),
                                            ValueType::F32()})) {
    return impl(*this, Product<ValueU32, ValueU32, ValueF32>(),
                std::make_integer_sequence<int, 3>());
  } else if (schema_flattened == TupleType({ValueType::U32(), ValueType::F32(),
                                            ValueType::U32()})) {
    return impl(*this, Product<ValueU32, ValueF32, ValueU32>(),
                std::make_integer_sequence<int, 3>());
  } else if (schema_flattened == TupleType({ValueType::F32(), ValueType::U32(),
                                            ValueType::U32()})) {
    return impl(*this, Product<ValueF32, ValueU32, ValueU32>(),
                std::make_integer_sequence<int, 3>());
  }
  std::cout << "schema: " << this->schema() << std::endl;
  PANIC(
      "Unsupported schema for sorted_until. See prior logging for more "
      "info.");
  return 0;
}

#define PROV UnitProvenance
template size_t Table<PROV>::sorted_until() const;
#undef PROV
#define PROV MinMaxProbProvenance
template size_t Table<PROV>::sorted_until() const;
#undef PROV
#define PROV DiffMinMaxProbProvenance
template size_t Table<PROV>::sorted_until() const;
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template size_t Table<PROV>::sorted_until() const;
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template size_t Table<PROV>::sorted_until() const;
#undef PROV
