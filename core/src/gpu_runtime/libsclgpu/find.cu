#include "find.h"
#include "flame.h"
#include "provenance.h"
#include "utils.h"

template <typename Prov, typename T>
struct FindKernel {
  ColumnView<T> col;
  T key;

  FindKernel(ColumnView<T> col, T key) : col(col), key(key) {}

  __device__ bool operator()(uint32_t i) { return col.at(i) == key; }
};

template <typename Prov>
Table<Prov> find(Table<Prov> source, Value key, TupleType result_schema) {
  TRACE_START(find);
  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }

  device_vec<uint32_t> result_indices(source.size());

  auto indices = thrust::make_counting_iterator<uint32_t>(0);
  int output_size;

  TRACE_START(find_kernel);
  DISPATCH_ON_TYPE(
      source.schema().flatten().at(0).singleton(), T,
      auto end = thrust::copy_if(
          thrust::device, indices, indices + source.size(),
          result_indices.data(),
          FindKernel<Prov, T>(source.template column<T>(0), key.downcast<T>()));
      output_size = end - result_indices.data(););
  TRACE_END(find_kernel);

  if (output_size == 0) {
    return Table<Prov>(result_schema);
  }

  TRACE_START(find_facts_gather);
  // gather
  Array<device_buffer> output_facts(source.schema().width());
  for (size_t col = 0; col < source.schema().width(); col++) {
    output_facts[col] = device_buffer(
        output_size, source.schema().flatten().at(col).singleton());
    DISPATCH_ON_TYPE(
        source.schema().flatten().at(col).singleton(), T,
        thrust::gather(thrust::device, result_indices.begin(),
                       result_indices.begin() + output_size,
                       reinterpret_cast<const T *>(source.at_raw(0, col)),
                       output_facts[col].begin<T>()););
  }
  TRACE_END(find_facts_gather);

  TRACE_START(find_tags_gather);
  device_vec<typename Prov::Tag> output_tags(output_size);
  thrust::gather(thrust::device, result_indices.begin(),
                 result_indices.begin() + output_size, source.tags().data(),
                 output_tags.begin());
  TRACE_END(find_tags_gather);

  device_vec<char> output_sample_mask(output_size);
  thrust::gather(thrust::device, result_indices.begin(),
                 result_indices.begin() + output_size,
                 source.sample_mask().data(), output_sample_mask.begin());

  Table<Prov> result(result_schema, std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
  hINFO("Find(" << key << ")");
  hINFO("input table:\n" << source);
  hINFO("result table:\n" << result);
  return result;
}
#define PROV UnitProvenance
template Table<PROV> find(Table<PROV> left, Value key, TupleType result_schema);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> find(Table<PROV> left, Value key, TupleType result_schema);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> find(Table<PROV> left, Value key, TupleType result_schema);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> find(Table<PROV> left, Value key, TupleType result_schema);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> find(Table<PROV> left, Value key, TupleType result_schema);
#undef PROV
