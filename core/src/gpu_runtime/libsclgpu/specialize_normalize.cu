#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cub/device/device_segmented_sort.cuh>

#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "provenance.h"
#include "table.h"
#include "table_index.h"
#include "utils.h"

template <typename Prov>
__global__ void combine_tags_specialized(
    uint32_t *run_start_indices, char *sample_mask,
    typename Prov::Tag *input_tags, typename Prov::Tag *output_tags,
    size_t input_size, size_t output_length,
    typename Prov::BatchDeviceContext ctxs) {
  size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= output_length) {
    return;
  }

  auto ctx = Prov::sample_context(ctxs, sample_mask[index]);

  auto run_start_index = run_start_indices[index];
  typename Prov::Tag accum = input_tags[run_start_index];

  auto run_length =
      index == output_length - 1
          ? input_size - run_start_indices[index]
          : run_start_indices[index + 1] - run_start_indices[index];

  for (size_t i = 1; i < run_length; i++) {
    auto tag = input_tags[run_start_index + i];
    accum = Prov::add(ctx, accum, tag);
  }
  output_tags[index] = accum;
}

// template to prevent ODR violation
template <typename>
struct MyCompare {
  template <typename T>
  __device__ bool operator()(const T &lhs, const T &rhs) {
    return lhs < rhs;
  }
};

template <typename Prov, typename... T, int... Indices>
Table<Prov> normalized_specialized(Table<Prov> table, const Prov &ctx, const Allocator &alloc,
                                   Product<T...>,
                                   std::integer_sequence<int, Indices...>) {
  TRACE_START(specialized_normalize);
  hINFO("Normalize() input:");
  hINFO(table);

  int N = sizeof...(Indices);

  auto input_size = table.size();
  if (input_size == 0) {
    return Table<Prov>(table.schema());
  }

  auto input = thrust::make_zip_iterator(
      thrust::make_tuple(table.sample_mask().data(),
                         reinterpret_cast<const typename T::sort_type *>(
                             table.values()[Indices].at_raw(0))...));

  TRACE_START(specialized_normalize_output_alloc);
  device_vec<char> output_sample_mask(input_size, alloc);
  Array<device_buffer> output_facts(N);
  SINK((new (&output_facts[Indices]) device_buffer(input_size, T::tag(), alloc))...);
  auto output = thrust::make_zip_iterator(thrust::make_tuple(
      output_sample_mask.data(),
      output_facts[Indices].template begin<typename T::sort_type>().data()...));
  thrust::copy(thrust::device, input, input + input_size, output);
  TRACE_END(specialized_normalize_output_alloc);

  device_vec<typename Prov::Tag> sorted_tags(std::move(table.tags().clone()));
  device_vec<uint32_t> sorted_indices(input_size);
  thrust::sequence(thrust::device, sorted_indices.begin(),
                   sorted_indices.end());

  if (SIZES()) {
    std::cout << "Normalizing table of size " << input_size << std::endl;
  }

  TRACE_START(specialized_normalize_sort);
  if (Prov::is_unit) {
    if (COMPACT()) {
      TRACE_START(specialized_normalize_sort_compact);
      device_vec<thrust::tuple<char, typename T::sort_type...>> materialized(
          input_size);
      thrust::copy(thrust::device, input, input + input_size,
                   materialized.data());
      TRACE_END(specialized_normalize_sort_compact);
      thrust::sort(thrust::device, materialized.data(),
                   materialized.data() + input_size);
      thrust::copy(thrust::device, materialized.data(),
                   materialized.data() + input_size, output);
    } else if (!NO_CUB()) {
      void *d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                      output, input_size, MyCompare<void>());

      device_vec<char> temp_storage(temp_storage_bytes);
      d_temp_storage = temp_storage.data();

      cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                      output, input_size, MyCompare<void>());
    } else {
      thrust::sort(thrust::device, output, output + input_size);
    }
  } else {
    if (!NO_CUB()) {
      // auto input_no_mask = thrust::make_zip_iterator(
      //     thrust::make_tuple(reinterpret_cast<const typename T::type *>(
      //         table.values()[Indices].at_raw(0))...));
      // device_vec<thrust::tuple<typename T::type...>>
      // materialized(input_size); thrust::copy(thrust::device, input_no_mask,
      // input_no_mask + input_size,
      //              materialized.begin());
      // device_vec<uint32_t> sorted_indices_out(input_size);

      // device_vec<thrust::tuple<typename T::type...>>
      // materialized_out(input_size);

      // using FactTuple = thrust::tuple<typename T::type...>;
      // auto sample_sizes = table.sample_sizes();
      // int num_items = table.size();
      // int num_segments = sample_sizes.size();
      // size_t *d_offsets = sample_sizes.data();
      // FactTuple *d_keys_in = materialized.data();
      // FactTuple *d_keys_out = materialized_out.data();
      // uint32_t *d_values_in = sorted_indices.data();
      // uint32_t *d_values_out = sorted_indices_out.data();

      // void *d_temp_storage = nullptr;
      // size_t temp_storage_bytes = 0;
      // cub::DeviceSegmentedSort::SortPairs(
      //     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
      //     d_values_in, d_values_out, num_items, num_segments, d_offsets,
      //     d_offsets + 1);
      // cudaMalloc(&d_temp_storage, temp_storage_bytes);
      // cub::DeviceSegmentedSort::SortPairs(
      //     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
      //     d_values_in, d_values_out, num_items, num_segments, d_offsets,
      //     d_offsets + 1);
      // cudaFree(d_temp_storage);

      // thrust::copy(thrust::device, materialized_out.begin(),
      //              materialized_out.end(), output);

      // device_vec<typename Prov::Tag> sorted_tags_temp(sorted_tags.size());
      // thrust::gather(thrust::device, sorted_indices_out.begin(),
      // sorted_indices_out.end(),
      //                sorted_tags.begin(), sorted_tags_temp.begin());
      // sorted_tags = std::move(sorted_tags_temp);
      void *d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                      output, sorted_indices.data(),
                                      table.size(), MyCompare<void>());
      device_vec<char> temp_storage(temp_storage_bytes);
      d_temp_storage = temp_storage.data();
      cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                      output, sorted_indices.data(),
                                      table.size(), MyCompare<void>());

      device_vec<typename Prov::Tag> sorted_tags_temp(sorted_tags.size());
      thrust::gather(thrust::device, sorted_indices.begin(),
                     sorted_indices.end(), sorted_tags.begin(),
                     sorted_tags_temp.begin());
      sorted_tags = std::move(sorted_tags_temp);
    } else {
      if constexpr (Prov::large_tags) {
        thrust::sort_by_key(thrust::device, output, output + input_size,
                            sorted_indices.begin());

        device_vec<typename Prov::Tag> sorted_tags_temp(sorted_tags.size());
        thrust::gather(thrust::device, sorted_indices.begin(),
                       sorted_indices.end(), sorted_tags.begin(),
                       sorted_tags_temp.begin());
        sorted_tags = std::move(sorted_tags_temp);
      } else {
        thrust::sort_by_key(thrust::device, output, output + input_size,
                            sorted_tags.begin());
      }
    }
  }

  TRACE_END(specialized_normalize_sort);

  hINFO("Table after sorting:");
  hINFO(Table<Prov>(table.schema(), sorted_tags.clone(), output_facts.clone(),
                    output_sample_mask.clone()));

  if (SAFETY) {
    auto sorted_til =
        thrust::is_sorted_until(thrust::device, output, output + input_size);
    auto sorted_length = thrust::distance(output, sorted_til);
    if (sorted_length != input_size) {
      std::cout << "Error: only sorted until " << sorted_length
                << " but length is " << input_size << std::endl;
      PANIC("Sorting in normalize didnt actually sort");
    }
  }

  TRACE_START(specialized_normalize_unique);
  int output_size;
  device_vec<uint32_t> run_start_indices(input_size);
  if (!NO_CUB() && Prov::is_unit) {
    device_vec<char> tmp_mask(input_size, alloc);

    Array<device_buffer> tmp_facts(N);
    SINK((new (&tmp_facts[Indices])
              device_buffer(input_size, T::tag(), alloc))...);
    auto tmp_facts_iter = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_mask.data(),
        tmp_facts[Indices].template begin<typename T::sort_type>().data()...));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    device_vec<int> d_num_selected_out(1);
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              output, tmp_facts_iter,
                              d_num_selected_out.data(), input_size);

    device_vec<char> temp_storage(temp_storage_bytes);
    d_temp_storage = temp_storage.data();
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              output, tmp_facts_iter,
                              d_num_selected_out.data(), input_size);

    output_sample_mask = std::move(tmp_mask);
    output_facts = std::move(tmp_facts);
    output_size = d_num_selected_out.at_host(0);
  } else {
    thrust::sequence(thrust::device, run_start_indices.begin(),
                     run_start_indices.end());
    auto new_end = thrust::unique_by_key(
        thrust::device, output, output + input_size, run_start_indices.data());
    output_size = thrust::distance(output, new_end.first);
  }
  TRACE_END(specialized_normalize_unique);

  hINFO("Unique sample mask:");
  hINFO(output_sample_mask);
  hINFO("Uniqued facts:");
  hINFO(output_facts);

  if (output_size == 0) {
    PANIC("Normalizing table resulted in an empty table");
  }

  SINK((output_facts[Indices].resize(output_size), 0)...);
  output_sample_mask.resize(output_size);

  hINFO("Run start indices:");
  hINFO(run_start_indices);
  hINFO("Sorted tags:");
  hINFO(sorted_tags);
  hINFO("Input size: " << input_size);
  hINFO("Output size: " << output_size);

  device_vec<typename Prov::Tag> output_tags;
  if (!Prov::is_unit) {
    TRACE_START(normalize_combine_tags);
    output_tags = device_vec<typename Prov::Tag>(output_size, alloc);
    dim3 blockDim = {32, 1, 1};
    dim3 gridDim = {ROUND_UP_TO_NEAREST(output_size, 32), 1, 1};
    combine_tags_specialized<Prov><<<gridDim, blockDim>>>(
        run_start_indices.data(), output_sample_mask.data(), sorted_tags.data(),
        output_tags.data(), input_size, output_size, ctx.device_context());
    cudaCheck(cudaDeviceSynchronize());

    TRACE_END(normalize_combine_tags);
  }

  Table<Prov> output_table(table.schema(), std::move(output_tags),
                           std::move(output_facts),
                           std::move(output_sample_mask));

  hINFO("Normalize() output:");
  hINFO(output_table);
  hINFO("");

  return output_table;
}
