#include "flame.h"
#include "overwrite_one.h"
#include "provenance.h"
#include "utils.h"


template <typename Prov>
Table<Prov> overwrite_one(Table<Prov> source, TupleType result_schema) {
  TRACE_START(overwrite_one);
  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }

  auto output = source.clone();

  // fill tags with one
  auto new_tags = device_vec<typename Prov::Tag>(source.size());
  thrust::fill(thrust::device, new_tags.begin(), new_tags.end(), Prov::one());
  cudaCheck(cudaDeviceSynchronize());

  output.tags() = std::move(new_tags);

  return output;
}

#define PROV UnitProvenance
template Table<PROV> overwrite_one(
    Table<PROV> left, TupleType result_schema);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> overwrite_one(
    Table<PROV> left, TupleType result_schema);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> overwrite_one(
    Table<PROV> left, TupleType result_schema);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> overwrite_one(
    Table<PROV> left, TupleType result_schema);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> overwrite_one(
    Table<PROV> left, TupleType result_schema);
#undef PROV
