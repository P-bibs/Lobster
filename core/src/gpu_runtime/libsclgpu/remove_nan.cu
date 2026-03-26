#include <cmath>
#include <type_traits>

#include "provenance.h"
#include "remove_nan.h"
#include "utils.h"

template <typename... T>
struct IsntNanFunctor {
  using input_tuple = thrust::tuple<T...>;
  __host__ __device__ bool operator()(input_tuple t) const {
    bool is_nan = false;
    const_expr_for<sizeof...(T)>([&](auto i) {
      if constexpr (std::is_floating_point<
                        thrust::tuple_element<i.value, input_tuple>>::value) {
        is_nan |= std::isnan(thrust::get<i.value>(t));
      }
    });
    return !is_nan;
  }
};

template <typename Prov, typename... T, int... Index>
Table<Prov> remove_nan_specialized(const Table<Prov> &table, Product<T...>,
                                   std::integer_sequence<int, Index...>) {
  auto iter = thrust::make_zip_iterator(thrust::make_tuple(
      table.sample_mask().data(),
      table.template column_cbegin<typename T::type>(Index).data()...,
      table.tags().data()));

  device_vec<char> output_sample_mask(table.size());
  Array<device_buffer> output_facts(sizeof...(T));
  SINK((new (&output_facts[Index]) device_buffer(table.size(), T::tag()))...);
  device_vec<typename Prov::Tag> output_tags(table.size());

  auto output = thrust::make_zip_iterator(thrust::make_tuple(
      output_sample_mask.data(),
      output_facts[Index].template begin<typename T::type>().data()...,
      output_tags.data()));

  auto new_end = thrust::copy_if(
      thrust::device, iter, iter + table.size(), output,
      IsntNanFunctor<char, typename T::type..., typename Prov::Tag>{});
  auto output_size = thrust::distance(output, new_end);

  output_sample_mask.resize(output_size);
  SINK((output_facts[Index].resize(output_size), 0)...);
  output_tags.resize(output_size);

  Table<Prov> output_table(table.schema(), std::move(output_tags),
                           std::move(output_facts),
                           std::move(output_sample_mask));
  return output_table;
}

template <typename Prov>
Table<Prov> remove_nan(Table<Prov> table) {
  TRACE_START(remove_nan);
  hINFO("Remove NaN: ");
  hINFO("table:\n" << table);
  table.validate();

  if (table.size() == 0) {
    return Table<Prov>(table.schema());
  }

  auto schema_flattened = table.schema().flatten();
  Table<Prov> output(table.schema());
  if (schema_flattened ==
      TupleType({ValueType::U32(), ValueType::U32(), ValueType::F32()})) {
    output =
        remove_nan_specialized(table, Product<ValueU32, ValueU32, ValueF32>(),
                               std::make_integer_sequence<int, 3>());
  } else if (schema_flattened == TupleType({ValueType::U32(), ValueType::F32(),
                                            ValueType::U32()})) {
    output =
        remove_nan_specialized(table, Product<ValueU32, ValueF32, ValueU32>(),
                               std::make_integer_sequence<int, 3>());
  } else if (schema_flattened == TupleType({ValueType::F32(), ValueType::U32(),
                                            ValueType::U32()})) {
    output =
        remove_nan_specialized(table, Product<ValueF32, ValueU32, ValueU32>(),
                               std::make_integer_sequence<int, 3>());
  } else {
    std::cout << "schema: " << table.schema() << std::endl;
    PANIC(
        "Unsupported schema for remove_nan. See prior logging for more "
        "info.");
  }

  hINFO("result table:" << output);
  return output;
}

#define PROV UnitProvenance
template Table<PROV> remove_nan(Table<PROV> table);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> remove_nan(Table<PROV> table);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> remove_nan(Table<PROV> table);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> remove_nan(Table<PROV> table);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> remove_nan(Table<PROV> table);
#undef PROV
