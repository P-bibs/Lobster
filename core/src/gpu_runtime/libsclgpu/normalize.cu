#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cub/device/device_merge_sort.cuh>

#include "device_vec.h"
#include "normalize.h"
#include "provenance.h"
#include "utils.h"

template <typename Prov>
Table<Prov> normalized(Table<Prov> source, const Prov &ctx, const Allocator &alloc) {
  TRACE_START(TableHandle_normalized);
  hINFO("SCHEMA: NORMALIZE: " << source.schema());
  hINFO("Normalize input:");
  hINFO(source);

  if (source.size() == 0) {
    return Table<Prov>(source.schema());
  }

  auto schema = source.schema();
  if (schema.flatten() == TupleType(std::vector{TupleType(ValueType::U32())})) {
    return normalized_specialized(source, ctx, alloc, Product<ValueU32>(), std::make_integer_sequence<int, 1>());
  } else if (schema.flatten() ==TupleType({ValueType::U32(), ValueType::U32()})) {
    return normalized_specialized(source, ctx, alloc, Product<ValueU32, ValueU32>(), std::make_integer_sequence<int,2>());
  } else if (schema.flatten() == TupleType({ValueType::U32(), ValueType::U32(),ValueType::U32()})) {
    return normalized_specialized(source,ctx, alloc, Product<ValueU32, ValueU32,ValueU32>(), std::make_integer_sequence<int, 3>());
  } else if (schema.flatten() ==TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),ValueType::U32()})) {
    return normalized_specialized(source, ctx, alloc, Product<ValueU32, ValueU32, ValueU32, ValueU32>(), std::make_integer_sequence<int, 4>());
  } else if (schema.flatten() == TupleType({ValueType::U32(), ValueType::F32(),ValueType::U32()})) {
    return normalized_specialized(source,ctx, alloc, Product<ValueU32, ValueF32,ValueU32>(), std::make_integer_sequence<int, 3>());
  } else if (schema.flatten() == TupleType({ValueType::F32(), ValueType::U32(),ValueType::U32()})) {
    return normalized_specialized(source,ctx, alloc, Product<ValueF32, ValueU32,ValueU32>(), std::make_integer_sequence<int, 3>());
  } else if (schema.flatten() == TupleType({ValueType::U32(), ValueType::U32(),ValueType::F32()})) {
    return normalized_specialized(source,ctx, alloc, Product<ValueU32, ValueU32,ValueF32>(), std::make_integer_sequence<int, 3>());
  } else if (schema.flatten() ==TupleType(std::vector{TupleType(ValueType::F32())})) {
    return normalized_specialized(source, ctx, alloc, Product<ValueF32>(), std::make_integer_sequence<int, 1>());
  } else if (schema.flatten() ==TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),ValueType::U32(), ValueType::U32()})) {
    return normalized_specialized(source, ctx, alloc, Product<ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>(), std::make_integer_sequence<int, 5>());
    //} else if (schema.flatten() ==
    //           TupleType({ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32()})) {
    //  return Specialized<Prov, 0, 1, 2, 3, 4>::template normalize<
    //      ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>(table, tags,
    //      schema,
    //                                                        ctx);
    //} else if (schema.flatten() ==
    //           TupleType({ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32(),
    //                      ValueType::U32()})) {
    //  return Specialized<Prov, 0, 1, 2, 3, 4, 5>::template normalize<
    //      ValueU32, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>(
    //      table, tags, schema, ctx);
  }

  std::cout << "schema: " << schema << std::endl;
  PANIC(
      "Unsupported schema for normalize. See prior logging for more "
      "info.");
  return Table<Prov>(schema);
}

#ifdef SPECIALIZE_PARAMS
#define PROV UnitProvenance
#define PROV MinMaxProbProvenance
#define PROV DiffMinMaxProbProvenance
#define PROV DiffAddMultProbProvenance<>
#define PROV DiffTopKProofsProvenance<>

#define TYPE u32
#define TYPE u32 u32
#define TYPE u32 u32 u32
#define TYPE u32 u32 u32 u32
#define TYPE u32 u32 u32 u32 u32
#define TYPE f32 u32 u32
#define TYPE u32 f32 u32
#define TYPE u32 u32 f32
#define TYPE f32
#endif

#ifdef SPECIALIZE_BODY
#include "../specialize_normalize.cu"
template 
Table<PROV> normalized_specialized(Table<PROV> table, const PROV &ctx, const Allocator &, Product<TYPE>,
                       std::integer_sequence<int, TYPE_INDEX>);
#endif


#define PROV UnitProvenance
template 
Table<PROV> normalized(Table<PROV> source, const PROV &ctx, const Allocator&);
#undef PROV
#define PROV MinMaxProbProvenance
template 
Table<PROV> normalized(Table<PROV> source, const PROV &ctx, const Allocator&);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template 
Table<PROV> normalized(Table<PROV> source, const PROV &ctx, const Allocator&);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template 
Table<PROV> normalized(Table<PROV> source, const PROV &ctx, const Allocator&);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template 
Table<PROV> normalized(Table<PROV> source, const PROV &ctx, const Allocator&);
#undef PROV
