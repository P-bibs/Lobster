#include "bindings.h"
#include "device_vec.h"
#include "intersect.h"
#include "provenance.h"
#include "utils.h"

template <int N, typename... T>
__global__ void intersect_kernel(
    FactsView<typename T::type...> indexed_table,
    FactsView<typename T::type...> nonindexed_table,
    char *indexed_table_sample_mask, char *nonindexed_table_sample_mask,
    int nonindexed_table_size, const HashIndexView table_index,
    uint32_t *output_mask, uint32_t *output_indices) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nonindexed_table_size) {
    return;
  }
  int sample = nonindexed_table_sample_mask[index];

  auto columns = nonindexed_table.columns();
  std::tuple<typename T::type...> nonindexed_table_row = index_tuple(
      nonindexed_table.columns(), index, std::make_integer_sequence<int, N>{});

  // Manually calculate the hash
  size_t position = 0;
  const_expr_for<N>([&](auto i) {
    position += simple_hash(std::get<i.value>(nonindexed_table_row));
  });
  position = position & table_index.mask();

  int32_t index_size = table_index.size();
  for (int32_t i = 0; i < index_size; i++) {
    auto occupied = table_index.occupied(position);
    if (occupied) {
      auto indexed_table_row_index = table_index.value_at(position);
      auto sample_matches =
          indexed_table_sample_mask[indexed_table_row_index] == sample;
      if (sample_matches) {
        auto indexed_table_row_index = table_index.value_at(position);

        auto indexed_table_row =
            index_tuple(indexed_table.columns(), indexed_table_row_index,
                        std::make_integer_sequence<int, N>{});

        bool matches = nonindexed_table_row == indexed_table_row;
        if (matches) {
          output_mask[index] = 1;
          output_indices[index] = indexed_table_row_index;
          return;
        }
      }
    } else {
      output_mask[index] = 0;
      return;
    }
    position = table_index.incr_position(position);
  }
  printf("Failed to intersect kernel for row %d\n", index);
}

template <typename T>
__global__ void intersect_gather(const uint32_t *mask, const uint32_t *offset,
                                 const T *input, T *output, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  if (mask[index]) {
    auto i = offset[index];
    auto v = input[index];
    output[i] = v;
  }
}

template <typename Prov>
__global__ void intersect_tag_gather(
    char *sample_mask, const uint32_t *mask, const uint32_t *destination_index,
    const typename Prov::Tag *indexed_table_tags,
    const typename Prov::Tag *nonindexed_table_tags,
    const uint32_t *indexed_tags_offset, typename Prov::Tag *output, int size,
    typename Prov::BatchDeviceContext ctxs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  auto sample = sample_mask[index];
  auto ctx = Prov::sample_context(ctxs, sample);

  if (mask[index]) {
    auto left_tag = indexed_table_tags[indexed_tags_offset[index]];
    auto right_tag = nonindexed_table_tags[index];
    auto new_tag = Prov::mult(ctx, left_tag, right_tag);
    auto out_index = destination_index[index];
    // printf("Writing from indexed_table %d and nonindexed_table %d to output
    // %d\n", indexed_tags_offset[index], index, out_index);
    output[out_index] = new_tag;
  }
}

template <typename Prov, typename... T, int... Index>
Table<Prov> specialized_intersect(Table<Prov> left, Table<Prov> right,
                                  const TupleType &result_schema,
                                  const Prov &ctx, bool index_on_right,
                                  Product<T...>,
                                  std::integer_sequence<int, Index...>) {
  if (left.size() == 0 || right.size() == 0) {
    return Table<Prov>(result_schema);
  }

  auto result_width = left.width();

  const auto *index =
      index_on_right ? right.index(right.width()) : left.index(left.width());
  const auto indexed_table = index_on_right ? right : left;
  const auto nonindexed_table = index_on_right ? left : right;

  // Make light-weight views of the facts in each table
  FactsView<typename T::type...> indexed_table_facts(
      indexed_table.template column_cbegin<typename T::type>(Index)...);
  FactsView<typename T::type...> nonindexed_table_facts(
      nonindexed_table.template column_cbegin<typename T::type>(Index)...);

  // compute mask
  device_vec<uint32_t> mask(nonindexed_table.size());
  device_vec<uint32_t> indexed_table_indices(nonindexed_table.size());
  TRACE_START(intersect_kernel_execution);
  intersect_kernel<sizeof...(T), T...>
      <<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), 512), 512>>>(
          indexed_table_facts, nonindexed_table_facts,
          indexed_table.sample_mask().data(),
          nonindexed_table.sample_mask().data(), nonindexed_table.size(), HashIndexView(*index),
          mask.data(), indexed_table_indices.data());
  cudaCheck(cudaDeviceSynchronize());
  TRACE_END(intersect_kernel_execution);
  hINFO("Mask: \n" << mask);
  hINFO("Indices: \n" << indexed_table_indices);

  // Perform a prefix sum to find offsets
  TRACE_START(intersect_prefix_sum);
  device_vec<uint32_t> offset(nonindexed_table.size());
  thrust::exclusive_scan(thrust::device, mask.begin(), mask.end(),
                         offset.begin());
  TRACE_END(intersect_prefix_sum);
  hINFO("Offsets: \n" << offset);

  uint32_t last_offset = offset.at_host(offset.size() - 1);
  uint32_t last_occurence = mask.at_host(mask.size() - 1);
  uint32_t output_size = last_offset + last_occurence;
  if (output_size == 0) {
    return Table<Prov>(result_schema);
  }

  hINFO("Output size: " << output_size);

  // gather the facts
  Array<device_buffer> output_facts(result_width);
  for (size_t column = 0; column < result_width; column++) {
    new (&output_facts[column]) device_buffer(
        output_size, nonindexed_table.schema().at(column).singleton());
  }
  SINK((intersect_gather<<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), 128),
                           128>>>(
            mask.data(), offset.data(),
            nonindexed_table_facts.template column<Index>().data(),
            output_facts[Index].begin<typename T::type>().data(),
            nonindexed_table.size()),
        0)...);
  cudaCheck(cudaDeviceSynchronize());

  device_vec<char> output_sample_mask(output_size);
  intersect_gather<<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), 128), 128>>>(
      mask.data(), offset.data(), nonindexed_table.sample_mask().data(),
      output_sample_mask.data(), nonindexed_table.size());
  cudaCheck(cudaDeviceSynchronize());

  // gather the tags
  TRACE_START(intersect_combine_tags);
  // FIXME: this does not produce proper tags unless the indexed table does
  // not have any duplicates.
  device_vec<typename Prov::Tag> output_tags(output_size);
  intersect_tag_gather<Prov>
      <<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), 128), 128>>>(
          nonindexed_table.sample_mask().data(), mask.data(), offset.data(),
          indexed_table.tags().data(), nonindexed_table.tags().data(),
          indexed_table_indices.data(), output_tags.data(),
          nonindexed_table.size(), ctx.device_context());
  cudaCheck(cudaDeviceSynchronize());
  TRACE_END(intersect_combine_tags);

  Table<Prov> output(result_schema, std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
  return output;
}

template <typename Prov>
Table<Prov> intersect(Table<Prov> left, Table<Prov> right,
                      const TupleType &result_schema, const Prov &ctx,
                      bool index_on_right) {
  TRACE_START(intersect);
  hINFO("Intersect: ");
  hINFO("left table:\n" << left);
  hINFO("right table:\n" << right);

  Table<Prov> output(result_schema);
  auto schema_flattened = left.schema().flatten();
  if (schema_flattened == TupleType({ValueType::U32(), ValueType::U32()})) {
    output = specialized_intersect(
        left, right, result_schema, ctx, index_on_right,
        Product<ValueU32, ValueU32>{}, std::make_integer_sequence<int, 2>{});
  } else if (schema_flattened == TupleType({ValueType::U32(), ValueType::U32(),
                                            ValueType::U32()})) {
    output =
        specialized_intersect(left, right, result_schema, ctx, index_on_right,
                              Product<ValueU32, ValueU32, ValueU32>{},
                              std::make_integer_sequence<int, 3>{});
  } else {
    std::cout << "schema: " << schema_flattened << std::endl;
    PANIC("Unsupported schema for intersect. See prior logs for schema.");
  }
  hINFO("result table:" << output);
  return output;
}

#define PROV UnitProvenance
template Table<PROV> intersect(const Table<PROV> left, const Table<PROV> right,
                               const TupleType &result_schema, const PROV &ctx,
                               bool);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> intersect(const Table<PROV> left, const Table<PROV> right,
                               const TupleType &result_schema, const PROV &ctx,
                               bool);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> intersect(const Table<PROV> left, const Table<PROV> right,
                               const TupleType &result_schema, const PROV &ctx,
                               bool);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> intersect(const Table<PROV> left, const Table<PROV> right,
                               const TupleType &result_schema, const PROV &ctx,
                               bool);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> intersect(const Table<PROV> left, const Table<PROV> right,
                               const TupleType &result_schema, const PROV &ctx,
                               bool);
#undef PROV
