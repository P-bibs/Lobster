#include <thrust/adjacent_difference.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdint>
#include <numeric>

#include "normalize.h"
#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "join.h"
#include "join_gather/join_gather.h"
#include "provenance.h"
#include "table.h"
#include "table_index.h"
#include "utils.h"

/**
 * Counts the occurences of each row in `nonindexed_table` in `indexed_table`
 * using the provided hash index. Note that only the first `sizeof...(T)`
 * columns are used for the join.
 *
 * Specifically, for each row `r` in `nonindexed_table`, this function counts
 * the the number of rows `l` in `indexed_table` such that `l` and `r` match on
 * the first `sizeof...(T)` columns. The resulting value is written to the
 * `occurences` buffer
 */
template <int Count, typename... T>
__global__ void count_occurrences(
    FactsView<typename T::type...> indexed_table,
    FactsView<typename T::type...> nonindexed_table,
    char *indexed_table_sample_mask, char *nonindexed_table_sample_mask,
    const HashIndexView table_index, uint32_t *occurrences, int size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= size) {
    return;
  }
  int sample = nonindexed_table_sample_mask[row];

  size_t num_occurences = 0;
  // Get the row once here to avoid repeated dereference of managed memory
  auto columns = nonindexed_table.columns();
  std::tuple<typename T::type...> nonindexed_table_row =
      index_tuple(nonindexed_table.columns(), row,
                  std::make_integer_sequence<int, Count>{});

  // Manually calculate the hash
  size_t position = 0;
  const_expr_for<Count>([&](auto i) {
    position += simple_hash(std::get<i.value>(nonindexed_table_row));
  });
  position = position & table_index.mask();

  // TODO: is this logic still correct in the presense of batching?
  int32_t index_size = table_index.size();
  for (int32_t i = 0; i < index_size; i++) {
    auto occupied = table_index.occupied(position);
    if (occupied) {
      auto indexed_table_row_index = table_index.value_at(position);
      auto sample_matches =
          indexed_table_sample_mask[indexed_table_row_index] == sample;
      if (sample_matches) {
        auto indexed_table_row =
            index_tuple(indexed_table.columns(), indexed_table_row_index,
                        std::make_integer_sequence<int, Count>{});

        bool fact_matches = nonindexed_table_row == indexed_table_row;

        if (fact_matches) {
          num_occurences += 1;
        }
      }
    } else {
      occurrences[row] = num_occurences;
      return;
    }
    position = table_index.incr_position(position);
  }
  printf("Failed to count occurrences for row %d\n", row);
}

/**
 * Computes an argjoin `indexed_table` and `nonindexed_table` using the provided
 * hash index and offset count. Resulting indices are written to
 * `indexed_table_result_indices` and `nonindexed_table_result_indices`.
 */
template <typename Prov, typename... T>
__global__ void join_kernel(FactsView<typename T::type...> indexed_table,
                            FactsView<typename T::type...> nonindexed_table,
                            const HashIndexView table_index,
                            const uint32_t *offset,
                            uint32_t *indexed_table_result_indices,
                            uint32_t *nonindexed_table_result_indices,
                            const char *indexed_table_sample_mask,
                            const char *nonindexed_table_sample_mask,
                            int size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= size) {
    return;
  }
  int sample = nonindexed_table_sample_mask[row];

  // Get the row values once here to avoid repeated dereference of managed
  // memory
  auto columns = nonindexed_table.columns();
  std::tuple<typename T::type...> nonindexed_table_row =
      index_tuple(nonindexed_table.columns(), row,
                  std::make_integer_sequence<int, sizeof...(T)>{});

  // Calculate the position of this row in the hash table
  size_t position = 0;
  const_expr_for<sizeof...(T)>([&](auto i) {
    position += simple_hash(std::get<i.value>(nonindexed_table_row));
  });
  position = position & table_index.mask();

  uint32_t write_to_index = offset[row];

  auto index_size = table_index.size();
  for (size_t i = 0; i < index_size; i++) {
    auto occupied = table_index.occupied(position);
    if (occupied) {
      auto indexed_table_row_index = table_index.value_at(position);
      auto sample_matches =
          indexed_table_sample_mask[indexed_table_row_index] == sample;
      if (sample_matches) {
        auto indexed_table_row =
            index_tuple(indexed_table.columns(), indexed_table_row_index,
                        std::make_integer_sequence<int, sizeof...(T)>{});
        bool fact_matches = nonindexed_table_row == indexed_table_row;

        if (fact_matches) {
          indexed_table_result_indices[write_to_index] =
              indexed_table_row_index;
          nonindexed_table_result_indices[write_to_index] = row;
          write_to_index++;
        }
      }
    } else {
      // TODO: would this be faster if we also passed the "occurrences"
      // vector and terminated after we had seen the correct number of
      // rows instead of continuing until we see -1?
      return;
    }
    position = table_index.incr_position(position);
  }
  printf("Failed to join row %d\n", row);
}

template <typename Prov>
__global__ void join_combine_tags(
    char *sample_mask, const typename Prov::Tag *indexed_table_tags,
    const typename Prov::Tag *nonindexed_table_tags,
    uint32_t *indexed_table_indices, uint32_t *nonindexed_table_indices,
    typename Prov::Tag *result, int size,
    typename Prov::BatchDeviceContext ctxs) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= size) {
    return;
  }
  auto ctx = Prov::sample_context(ctxs, sample_mask[row]);
  result[row] =
      Prov::mult(ctx, indexed_table_tags[indexed_table_indices[row]],
                 nonindexed_table_tags[nonindexed_table_indices[row]]);
}

// Produces the result of joining two tables `left` and `right` on the first
// `sizeof...(T)` columns. It is undefined behavior to provide `T...` that do
// not match the types of the first `sizeof...(T)` columns of `left` and
// `right`.
template <typename Prov, typename... T, int... Index>
Table<Prov> specialized_join(Table<Prov> left, Table<Prov> right,
                             const TupleType &result_schema, const Prov &ctx,
                             bool index_on_right, Product<T...>,
                             std::integer_sequence<int, Index...>) {
  TRACE_START(multi_column_join);
  if (left.size() == 0 || right.size() == 0) {
    return Table<Prov>(result_schema);
  }

  hINFO("Join input left: \n" << left);
  hINFO("Join input right: \n" << right);

  auto join_width = left.schema().at(0).width();
  const auto *index =
      index_on_right ? right.index(join_width) : left.index(join_width);

  if (index_on_right) {
      hINFO("Right index:");
      hINFO(*index);
  } else {
      hINFO("Left index:");
      hINFO(*index);
  }

  const auto indexed_table = index_on_right ? right : left;
  const auto nonindexed_table = index_on_right ? left : right;
  // Make light-weight views of the facts in each table
  FactsView<typename T::type...> indexed_table_facts(
      indexed_table.template column_cbegin<typename T::type>(Index)...);
  FactsView<typename T::type...> nonindexed_table_facts(
      nonindexed_table.template column_cbegin<typename T::type>(Index)...);

  // count occurrences in the hash table
  device_vec<uint32_t> occurrences(nonindexed_table.size());

  TRACE_START(join_count_occurrences);
  count_occurrences<sizeof...(T), T...>
      <<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), 512), 512>>>(
          indexed_table_facts, nonindexed_table_facts,
          indexed_table.sample_mask().data(),
          nonindexed_table.sample_mask().data(), HashIndexView(*index), occurrences.data(),
          nonindexed_table.size());
  cudaCheck(cudaDeviceSynchronize());
  TRACE_END(join_count_occurrences);

  hINFO("Occurrences: \n" << occurrences);

  // Perform a prefix sum to find offsets
  TRACE_START(join_prefix_sum);
  device_vec<uint32_t> offset(nonindexed_table.size());
  thrust::exclusive_scan(thrust::device, occurrences.begin(), occurrences.end(),
                         offset.begin());
  TRACE_END(join_prefix_sum);

  hINFO("Offsets: \n" << offset);

  uint32_t last_offset = offset.at_host(offset.size() - 1);
  uint32_t last_occurence = occurrences.at_host(occurrences.size() - 1);
  uint32_t new_facts_size = last_offset + last_occurence;

  if (new_facts_size == 0) {
    return Table<Prov>(result_schema);
  }

  device_vec<uint32_t> indexed_table_result_indices(new_facts_size);
  device_vec<uint32_t> nonindexed_table_result_indices(new_facts_size);

  // perform the join
  // read block size from environment variable
  auto join_block_size = std::getenv("JOIN_BLOCK_SIZE")
                             ? std::stoi(std::getenv("JOIN_BLOCK_SIZE"))
                             : 128;
  TRACE_START(join_join);
  join_kernel<Prov, T...>
      <<<ROUND_UP_TO_NEAREST(nonindexed_table.size(), join_block_size),
         join_block_size>>>(indexed_table_facts, nonindexed_table_facts, HashIndexView(*index),
                            offset.data(), indexed_table_result_indices.data(),
                            nonindexed_table_result_indices.data(),
                            indexed_table.sample_mask().data(),
                            nonindexed_table.sample_mask().data(),
                            nonindexed_table.size());
  cudaCheck(cudaDeviceSynchronize());
  TRACE_END(join_join);

  hINFO("Indexed table result indices: \n" << indexed_table_result_indices);
  hINFO("Nonindexed table result indices: \n"
        << nonindexed_table_result_indices);

  auto output_facts =
      index_on_right
          ? gather_join_indices(left, right, nonindexed_table_result_indices,
                                indexed_table_result_indices)
          : gather_join_indices(left, right, indexed_table_result_indices,
                                nonindexed_table_result_indices);

  device_vec<char> output_sample_mask(new_facts_size);
  thrust::gather(thrust::device, nonindexed_table_result_indices.begin(),
                 nonindexed_table_result_indices.end(),
                 nonindexed_table.sample_mask().cbegin(),
                 output_sample_mask.begin());

  device_vec<typename Prov::Tag> output_tags;
  if (!Prov::is_unit) {
    output_tags = device_vec<typename Prov::Tag>(new_facts_size);
    TRACE_START(join_combine_tags_kernel);
    join_combine_tags<Prov><<<ROUND_UP_TO_NEAREST(new_facts_size, 128), 128>>>(
        output_sample_mask.data(), indexed_table.tags().data(),
        nonindexed_table.tags().data(), indexed_table_result_indices.data(),
        nonindexed_table_result_indices.data(), output_tags.data(),
        new_facts_size, ctx.device_context());
    cudaCheck(cudaDeviceSynchronize());
    TRACE_END(join_combine_tags_kernel);
  }
  Table<Prov> output(result_schema, std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
  hINFO("Join()");
  hINFO("result: \n" << output);

  //std::cout << "Join result size " << output.size() << std::endl;

  return output;
}

/**
 * Compute the relational join of two tables `left` and `right` on their first
 * logical column
 */
template <typename Prov>
Table<Prov> join(Table<Prov> left, Table<Prov> right,
                 const TupleType &result_schema, const Prov &ctx,
                 bool index_on_right) {
  TRACE_START(join);

  auto schema_flattened = left.schema().at(0).flatten();
  if (schema_flattened == TupleType({ValueType::USize(), ValueType::USize()})) {
    return specialized_join(left, right, result_schema, ctx, index_on_right,
                            Product<ValueUSize, ValueUSize>{},
                            std::make_integer_sequence<int, 2>{});
  } else if (schema_flattened ==
             TupleType({ValueType::U32(), ValueType::U32()})) {
    return specialized_join(left, right, result_schema, ctx, index_on_right,
                            Product<ValueU32, ValueU32>{},
                            std::make_integer_sequence<int, 2>{});
  } else if (schema_flattened ==
             TupleType({ValueType::Symbol(), ValueType::Symbol()})) {
    return specialized_join(left, right, result_schema, ctx, index_on_right,
                            Product<ValueSymbol, ValueSymbol>{},
                            std::make_integer_sequence<int, 2>{});
  } else if (schema_flattened ==
             TupleType({ValueType::F32(), ValueType::F32()})) {
    return specialized_join(left, right, result_schema, ctx, index_on_right,
                            Product<ValueF32, ValueF32>{},
                            std::make_integer_sequence<int, 2>{});
  }

  DISPATCH_ON_KIND(schema_flattened.singleton(), T,
                   return (specialized_join(
                       left, right, result_schema, ctx, index_on_right,
                       Product<T>{}, std::make_integer_sequence<int, 1>{})););
}

#define PROV UnitProvenance
template Table<PROV> join(const Table<PROV> left, const Table<PROV> right,
                          const TupleType &result_schema, const PROV &ctx,
                          bool index_on_right);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> join(const Table<PROV> left, const Table<PROV> right,
                          const TupleType &result_schema, const PROV &ctx,
                          bool index_on_right);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> join(const Table<PROV> left, const Table<PROV> right,
                          const TupleType &result_schema, const PROV &ctx,
                          bool index_on_right);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> join(const Table<PROV> left, const Table<PROV> right,
                          const TupleType &result_schema, const PROV &ctx,
                          bool index_on_right);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> join(const Table<PROV> left, const Table<PROV> right,
                          const TupleType &result_schema, const PROV &ctx,
                          bool index_on_right);
#undef PROV
