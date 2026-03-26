#include "bindings.h"
#include "device_vec.h"
#include "provenance.h"
#include "set_difference.h"
#include "table.h"
#include "utils.h"

template <int N, typename... T>
__global__ void difference_kernel(
    FactsView<typename T::type...> left,
    FactsView<typename T::type...> right,
    char *left_sample_mask,
    char *right_sample_mask,
    int left_size,
    const HashIndex *table_index,
    uint32_t *output_mask,
    uint32_t *output_indices) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left_size) {
    return;
  }
  int sample = left_sample_mask[index];

  // Get the row once here to avoid repeated dereference of managed memory
  auto columns = left.columns();
  std::tuple<typename T::type...> left_table_row =
      index_tuple(left.columns(), index, std::make_integer_sequence<int, N>{});

  // Manually calculate the hash
  size_t position = 0;
  const_expr_for<N>([&](auto i) {
    position += simple_hash(std::get<i.value>(left_table_row));
  });
  position = position & table_index->mask();

  int32_t index_size = table_index->size();
  for (int32_t i = 0; i < index_size; i++) {
    auto occupied = table_index->occupied(position);
    if (occupied) {
      auto right_table_row_index = table_index->value_at(position);
      auto sample_matches =
          right_table_sample_mask[right_table_row_index] == sample;
      if (sample_matches) {
        auto right_table_row_index = table_index->value_at(position);

        auto right_table_row =
            index_tuple(right.columns(), right_table_row_index,
                        std::make_integer_sequence<int, N>{});

        bool matches = left_table_row == right_table_row;
        if (matches) {
          output_mask[index] = 1;
          output_indices[index] = right_table_row_index;
          return;
        }
      }
    } else {
      output_mask[index] = 0;
      return;
    }
    position = table_index->incr_position(position);
  }
  printf("Failed to difference kernel for row %d\n", index);
}

template <typename T>
__global__ void difference_gather(const uint32_t *mask, const uint32_t *offset,
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

template <typename Prov, typename... T, int... Index>
Table<Prov> specialized_difference(Table<Prov> left, Table<Prov> right,
                                   const TupleType &result_schema,
                                   Allocator &alloc,
                                   const Prov &ctx, Product<T...>,
                                   std::integer_sequence<int, Index...>) {
  hINFO("Difference:");
  hINFO("left table:\n" << left);
  hINFO("right table:\n" << right);

  if (left.size() == 0 || right.size() == 0) {
    return Table<Prov>(result_schema);
  }

  if (left.schema() != right.schema()) {
    std::cout << "Left schema: " << left.schema() << std::endl;
    std::cout << "Right schema: " << right.schema() << std::endl;
    PANIC("Mismatched schemas. See prior logs for schema.");
  }

  auto result_width = left.width();

  const auto *index = right.index(right.width());

  if (Prov::is_unit) {
    // Make light-weight views of the facts in each table
    FactsView<typename T::type...> right_facts(
        right.template column_cbegin<typename T::type>(Index)...);
    FactsView<typename T::type...> left_facts(
        left.template column_cbegin<typename T::type>(Index)...);

    // compute mask
    device_vec<uint32_t> mask(left.size());
    TRACE_START(difference_kernel_execution);
    difference_kernel<sizeof...(T), T...>
        <<<ROUND_UP_TO_NEAREST(left.size(), 512), 512>>>(
            left_facts, right_facts, left.sample_mask().data(),
            right.sample_mask().data(), left.size(), index, mask.data(),
            output_facts);
    cudaCheck(cudaDeviceSynchronize());
    TRACE_END(difference_kernel_execution);
    hINFO("Mask: \n" << mask);

    // Perform a prefix sum to find offsets
    TRACE_START(difference_prefix_sum);
    device_vec<uint32_t> offset(left.size());
    thrust::exclusive_scan(thrust::device, mask.begin(), mask.end(),
                           offset.begin());
    TRACE_END(difference_prefix_sum);
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
    SINK((new (&output_facts[Index]) device_buffer(output_size, T::tag(), alloc))...);

    SINK(
        (difference_gather<<<ROUND_UP_TO_NEAREST(left.size(), 128), 128>>>(
             mask.data(), offset.data(),
             left_facts.template column<Index>().data(),
             output_facts[Index].begin<typename T::type>().data(), left.size()),
         0)...);
    cudaCheck(cudaDeviceSynchronize());

    device_vec<char> output_sample_mask(output_size);
    difference_gather<<<ROUND_UP_TO_NEAREST(left.size(), 128), 128>>>(
        mask.data(), offset.data(), left.sample_mask().data(),
        output_sample_mask.data(), left.size());
    cudaCheck(cudaDeviceSynchronize());

    device_vec<typename Prov::Tag> output_tags;
    Table<Prov> output(result_schema, output_tags,
                       std::move(output_facts), std::move(output_sample_mask));
    return output;
  } else {
    PANIC("Not implemented");
  }
}

template <typename Prov>
Table<Prov> difference(Table<Prov> left, Table<Prov> right,
                       const TupleType &result_schema, const Prov &ctx) {
  TRACE_START(difference);
  hINFO("Difference:");
  hINFO("left table:\n" << left);
  hINFO("right table:\n" << right);

  Table<Prov> output(result_schema);
  auto schema_flattened = left.schema().flatten();
  if (schema_flattened == TupleType({ValueType::U32(), ValueType::U32()})) {
    output = specialized_difference(left, right, result_schema, ctx,
                                    Product<ValueU32, ValueU32>{},
                                    std::make_integer_sequence<int, 2>{});
  } else {
    std::cout << "schema: " << schema_flattened << std::endl;
    PANIC("Unsupported schema for difference. See prior logs for schema.");
  }
  hINFO("result table:" << output);
  return output;
}

#define PROV UnitProvenance
template Table<PROV> difference(const Table<PROV> left, const Table<PROV> right,
                                const TupleType &result_schema, const PROV &ctx,
                                bool);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> difference(const Table<PROV> left, const Table<PROV> right,
                                const TupleType &result_schema, const PROV &ctx,
                                bool);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> difference(const Table<PROV> left, const Table<PROV> right,
                                const TupleType &result_schema, const PROV &ctx,
                                bool);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> difference(const Table<PROV> left, const Table<PROV> right,
                                const TupleType &result_schema, const PROV &ctx,
                                bool);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> difference(const Table<PROV> left, const Table<PROV> right,
                                const TupleType &result_schema, const PROV &ctx,
                                bool);
#undef PROV
