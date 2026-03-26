#include <thrust/adjacent_difference.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <type_traits>
#include <utility>

#include "bindings.h"
#include "flame.h"
#include "merge.h"
#include "normalize.h"
#include "provenance.h"
#include "utils.h"

template <typename Prov>
Table<Prov> merge_tables(const Table<Prov> &left, const Table<Prov> &right,
                         const Prov &ctx, const Allocator &alloc) {
  TRACE_START(merge_tables);
  hINFO("SCHEMA: MERGE: " << left.schema());
  hINFO("Merge: ");
  hINFO("left table:\n" << left);
  hINFO("right table:\n" << right);
  left.validate();
  right.validate();
  assert(left.schema() == right.schema());

  if (left.size() == 0) {
    return right.to(alloc);
  }
  if (right.size() == 0) {
    return left.to(alloc);
  }

  Table<Prov> output(left.schema());
  if (left.schema().flatten() ==
      TupleType(std::vector{TupleType(ValueType::U32())})) {
    output =
        merge_tables_specialized(left, right, ctx, alloc, Product<ValueU32>(),
                                 std::make_integer_sequence<int, 1>());
  } else if (left.schema().flatten() ==
             TupleType({ValueType::U32(), ValueType::U32()})) {
    output = merge_tables_specialized(left, right, ctx, alloc,
                                      Product<ValueU32, ValueU32>(),
                                      std::make_integer_sequence<int, 2>());
  } else if (left.schema().flatten() ==
             TupleType(
                 {ValueType::U32(), ValueType::U32(), ValueType::U32()})) {
    output = merge_tables_specialized(left, right, ctx, alloc,
                                      Product<ValueU32, ValueU32, ValueU32>(),
                                      std::make_integer_sequence<int, 3>());
  } else if (left.schema().flatten() ==
             TupleType(
                 {ValueType::F32(), ValueType::U32(), ValueType::U32()})) {
    output = merge_tables_specialized(left, right, ctx, alloc,
                                      Product<ValueF32, ValueU32, ValueU32>(),
                                      std::make_integer_sequence<int, 3>());
  } else if (left.schema().flatten() ==
             TupleType(
                 {ValueType::U32(), ValueType::U32(), ValueType::F32()})) {
    output = merge_tables_specialized(left, right, ctx, alloc,
                                      Product<ValueU32, ValueU32, ValueF32>(),
                                      std::make_integer_sequence<int, 3>());
  } else if (left.schema().flatten() ==
             TupleType(
                 {ValueType::U32(), ValueType::F32(), ValueType::U32()})) {
    output = merge_tables_specialized(left, right, ctx, alloc,
                                      Product<ValueU32, ValueF32, ValueU32>(),
                                      std::make_integer_sequence<int, 3>());
  } else if (left.schema().flatten() ==
             TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),
                        ValueType::U32()})) {
    output = merge_tables_specialized(
        left, right, ctx, alloc,
        Product<ValueU32, ValueU32, ValueU32, ValueU32>(),
        std::make_integer_sequence<int, 4>());
  } else if (left.schema().flatten() ==
             TupleType({ValueType::U32(), ValueType::U32(), ValueType::U32(),
                        ValueType::U32(), ValueType::U32()})) {
    output = merge_tables_specialized(
        left, right, ctx, alloc,
        Product<ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>(),
        std::make_integer_sequence<int, 5>());
  } else {
    std::cout << "left schema: " << left.schema() << std::endl;
    std::cout << "right_schema: " << right.schema() << std::endl;
    PANIC(
        "Unsupported schema for merge_tables. See prior logging for more "
        "info.");
  }

  hINFO("result table:" << output);
  return output;
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
#define TYPE f32 u32 u32
#define TYPE u32 f32 u32
#define TYPE u32 u32 f32
#define TYPE u32 u32 u32 u32
#define TYPE u32 u32 u32 u32 u32

#endif

#ifdef SPECIALIZE_BODY
#include "../merge_specialized.cu"
template Table<PROV> merge_tables_specialized(
    const Table<PROV> &left, const Table<PROV> &right, const PROV &ctx,
    const Allocator &, Product<TYPE>, std::integer_sequence<int, TYPE_INDEX>);
#endif

template Table<UnitProvenance> merge_tables(const Table<UnitProvenance> &left,
                                            const Table<UnitProvenance> &right,
                                            const UnitProvenance &ctx,
                                            const Allocator &alloc);
template Table<MinMaxProbProvenance> merge_tables(
    const Table<MinMaxProbProvenance> &left,
    const Table<MinMaxProbProvenance> &right, const MinMaxProbProvenance &ctx,
    const Allocator &alloc);

template Table<DiffMinMaxProbProvenance> merge_tables(
    const Table<DiffMinMaxProbProvenance> &left,
    const Table<DiffMinMaxProbProvenance> &right,
    const DiffMinMaxProbProvenance &ctx, const Allocator &alloc);

template Table<DiffAddMultProbProvenance<>> merge_tables(
    const Table<DiffAddMultProbProvenance<>> &left,
    const Table<DiffAddMultProbProvenance<>> &right,
    const DiffAddMultProbProvenance<> &ctx, const Allocator &alloc);

template Table<DiffTopKProofsProvenance<>> merge_tables(
    const Table<DiffTopKProofsProvenance<>> &left,
    const Table<DiffTopKProofsProvenance<>> &right,
    const DiffTopKProofsProvenance<> &ctx, const Allocator &alloc);
