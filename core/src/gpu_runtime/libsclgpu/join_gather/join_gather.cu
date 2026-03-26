#include <algorithm>
#include <cstdint>

#include "../bindings.h"
#include "../device_vec.h"
#include "../flame.h"
#include "../provenance.h"
#include "../table.h"
#include "../table_index.h"
#include "../utils.h"
#include "join_gather.h"

template <int JoinColumnWidth, int LeftWidth, typename T, int Column>
__device__ void copy_val(
    device_buffer::device_buffer_iter<typename T::type> input_column,
    uint32_t *left_indices, uint32_t *right_indices,
    device_buffer::device_buffer_iter<typename T::type> output_column,
    int index) {
  if constexpr (Column < LeftWidth) {
    auto &left_column = input_column[Column];
    output_column[index] = input_column[left_indices[index]];
  } else {
    output_column[index] = input_column[right_indices[index]];
  }
}

template <int JoinColumnWidth, int LeftWidth, typename... T, int... Index>
__device__ void copy_row(FactsView<typename T::type...> input_facts,
                         uint32_t *left_indices, uint32_t *right_indices,
                         FactsView<typename T::type...> output_columns,
                         int output_length, int index, Product<T...>,
                         std::integer_sequence<int, Index...>) {
  SINK((copy_val<JoinColumnWidth, LeftWidth, T, Index>(
            input_facts.template column<Index>(), left_indices, right_indices,
            output_columns.template column<Index>(), index),
        0)...);
}

template <int JoinColumnWidth, int LeftWidth, typename... T>
__global__ void kernel(FactsView<typename T::type...> input_facts,
                       uint32_t *left_indices, uint32_t *right_indices,
                       FactsView<typename T::type...> output_columns,
                       int output_length) {
  size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= output_length) {
    return;
  }
  copy_row<JoinColumnWidth, LeftWidth>(
      input_facts, left_indices, right_indices, output_columns, output_length,
      index, Product<T...>{}, std::make_integer_sequence<int, sizeof...(T)>{});
}

/** given two tables `left` and `right`, and two vectors of length `l` called
 * `left_indices` and `right_indices`, produces a new table of length `l`.
 * Row `i` of the output table is the concatenation of row
 * `left[left_indices[i]]` and row `right[right_indices[i]]`.
 */
template <typename Prov>
Array<device_buffer> gather_join_indices(const Table<Prov> &left,
                                         const Table<Prov> &right,
                                         device_vec<uint32_t> &left_indices,
                                         device_vec<uint32_t> &right_indices) {
  TRACE_START(gather_join_indices);
  assert(left_indices.size() == right_indices.size());

  auto length = left_indices.size();

  // join on the first logical column (which may be nested and therefore contain
  // many physical columns)
  auto join_column_width = left.schema().at(0).width();
  Array<device_buffer> output_table(left.width() + right.width() -
                                    join_column_width);

  auto left_schema_flattened = left.schema().flatten();
  auto right_schema_flattened = right.schema().flatten();

  for (size_t col = 0; col < left.width(); col++) {
    new (&output_table[col])
        device_buffer(length, left_schema_flattened.at(col).singleton());
  }

  for (size_t col = join_column_width; col < right.width(); col++) {
    new (&output_table[col + left.width() - join_column_width])
        device_buffer(length, right_schema_flattened.at(col).singleton());
  }

  if (left_schema_flattened ==
          TupleType({ValueType::U32(), ValueType::U32()}) &&
      right_schema_flattened ==
          TupleType({ValueType::U32(), ValueType::U32()}) &&
      join_column_width == 1) {
    FactsView<uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        right.column(1).template cbegin<uint32_t>());
    kernel<1, 2, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type>(
                output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());

    //thrust::gather(thrust::device, left_indices.begin(),
    //               left_indices.begin() + length, left.column(0).cbegin<uint32_t>(),
    //               output_table[0].begin<uint32_t>());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32(),
                            ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>(),
        left.column(3).template cbegin<uint32_t>());
    kernel<2, 4, ValueU32, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueU32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        right.column(1).template cbegin<uint32_t>());
    kernel<1, 1, ValueU32, ValueU32><<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
        input_facts, left_indices.data(), right_indices.data(),
        FactsView<ValueU32::type, ValueU32::type>(output_table), length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>());
    kernel<1, 2, ValueU32, ValueU32><<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
        input_facts, left_indices.data(), right_indices.data(),
        FactsView<ValueU32::type, ValueU32::type>(output_table), length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::F32()}) &&
             right_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             join_column_width == 1) {
    FactsView<uint32_t, float> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<float>());
    kernel<1, 2, ValueU32, ValueF32><<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
        input_facts, left_indices.data(), right_indices.data(),
        FactsView<ValueU32::type, ValueF32::type>(output_table), length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>());
    kernel<2, 2, ValueU32, ValueU32><<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
        input_facts, left_indices.data(), right_indices.data(),
        FactsView<ValueU32::type, ValueU32::type>(output_table), length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             right_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             join_column_width == 1) {
    FactsView<uint32_t> input_facts(left.column(0).template cbegin<uint32_t>());
    kernel<1, 1, ValueU32><<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
        input_facts, left_indices.data(), right_indices.data(),
        FactsView<ValueU32::type>(output_table), length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(std::vector{TupleType(ValueType::U32())}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>());
    kernel<1, 3, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type>(
                output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>());
    kernel<2, 3, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type>(
                output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::F32()}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t, uint32_t, float> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        right.column(1).template cbegin<uint32_t>(),
        right.column(2).template cbegin<float>());
    kernel<1, 2, ValueU32, ValueU32, ValueU32, ValueF32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueF32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::F32()}) &&
             right_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::F32()}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t, float, uint32_t, float> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<float>(),
        right.column(1).template cbegin<uint32_t>(),
        right.column(2).template cbegin<float>());
    kernel<1, 3, ValueU32, ValueU32, ValueF32, ValueU32, ValueF32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueF32::type,
                      ValueU32::type, ValueF32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 1) {
    FactsView<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>(),
        right.column(1).template cbegin<uint32_t>(),
        right.column(2).template cbegin<uint32_t>());
    kernel<1, 3, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueU32::type, ValueU32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        right.column(2).template cbegin<uint32_t>());
    kernel<2, 2, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type>(
                output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32(),
                            ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>(),
        left.column(3).template cbegin<uint32_t>(),
        right.column(2).template cbegin<uint32_t>());
    kernel<2, 4, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueU32::type, ValueU32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32(),
                            ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32(),
                            ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
        input_facts(left.column(0).template cbegin<uint32_t>(),
                    left.column(1).template cbegin<uint32_t>(),
                    left.column(2).template cbegin<uint32_t>(),
                    left.column(3).template cbegin<uint32_t>(),
                    right.column(2).template cbegin<uint32_t>(),
                    right.column(3).template cbegin<uint32_t>());
    kernel<2, 4, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueU32::type, ValueU32::type, ValueU32::type>(
                output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  } else if (left_schema_flattened ==
                 TupleType(
                     {ValueType::U32(), ValueType::U32(), ValueType::U32()}) &&
             right_schema_flattened ==
                 TupleType({ValueType::U32(), ValueType::U32(),
                            ValueType::U32(), ValueType::U32()}) &&
             join_column_width == 2) {
    FactsView<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> input_facts(
        left.column(0).template cbegin<uint32_t>(),
        left.column(1).template cbegin<uint32_t>(),
        left.column(2).template cbegin<uint32_t>(),
        right.column(2).template cbegin<uint32_t>(),
        right.column(3).template cbegin<uint32_t>());
    kernel<2, 3, ValueU32, ValueU32, ValueU32, ValueU32, ValueU32>
        <<<ROUND_UP_TO_NEAREST(length, 128), 128>>>(
            input_facts, left_indices.data(), right_indices.data(),
            FactsView<ValueU32::type, ValueU32::type, ValueU32::type,
                      ValueU32::type, ValueU32::type>(output_table),
            length);
    cudaCheck(cudaDeviceSynchronize());
    return output_table;
  }

  std::cout << "left_schema_flattened: " << left_schema_flattened << std::endl;
  std::cout << "right_schema_flattened: " << right_schema_flattened
            << std::endl;
  std::cout << "join_column_width: " << join_column_width << std::endl;
  PANIC(
      "Unsupported schema for gather_join_indices. See prior logging for more "
      "info.");
}

#define PROV UnitProvenance
template Array<device_buffer> gather_join_indices(
    const Table<PROV> &left, const Table<PROV> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
#undef PROV
#define PROV MinMaxProbProvenance
template Array<device_buffer> gather_join_indices(
    const Table<PROV> &left, const Table<PROV> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Array<device_buffer> gather_join_indices(
    const Table<PROV> &left, const Table<PROV> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Array<device_buffer> gather_join_indices(
    const Table<PROV> &left, const Table<PROV> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Array<device_buffer> gather_join_indices(
    const Table<PROV> &left, const Table<PROV> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
#undef PROV
