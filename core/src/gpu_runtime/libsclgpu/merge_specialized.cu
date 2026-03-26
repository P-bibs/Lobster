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
__global__ void gather_merge_tags(typename Prov::Tag *left_tags,
                                  typename Prov::Tag *right_tags,
                                  uint32_t *indices,
                                  typename Prov::Tag *output_tags,
                                  int output_size) {
  size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= output_size) {
    return;
  }

  uint32_t row = indices[index];
  uint32_t mask = 1u << 31;
  if (row & mask) {
    row &= ~mask;
    output_tags[index] = right_tags[row];
  } else {
    output_tags[index] = left_tags[row];
  }
}

/**
 * Combines the tags of deduplicated facts after a merge operation
 */
template <typename Prov>
__global__ void specialized_merge_combine_tags(
    char *sample_mask, typename Prov::Tag *source_tags, int32_t *tag_indices,
    typename Prov::Tag *output_tags, int output_size,
    typename Prov::BatchDeviceContext ctxs) {
  size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= output_size) {
    return;
  }
  auto ctx = Prov::sample_context(ctxs, sample_mask[index]);

  // special case last element
  if (index == output_size - 1) {
    output_tags[index] = source_tags[tag_indices[index]];
    return;
  }

  // After a merge, a fact can be duplicated at most once; therefore,
  // we know whether or not this fact is duplicated (and whether we need to
  // merge tags) based on whether or not the next index is contiguous
  auto this_tag_index = tag_indices[index];
  auto next_tag_index = tag_indices[index + 1];
  if (this_tag_index + 1 == next_tag_index) {
    output_tags[index] = source_tags[this_tag_index];
  } else {
    output_tags[index] = Prov::add(ctx, source_tags[this_tag_index],
                                   source_tags[this_tag_index + 1]);
  }
}

/**
 * Merges two sorted tables with identical schema, removing duplicates and
 * combining tags
 */
template <typename Prov, typename... T, int... Index>
Table<Prov> merge_tables_specialized(const Table<Prov> &left,
                                     const Table<Prov> &right, const Prov &ctx,
                                     const Allocator &alloc, Product<T...>,
                                     std::integer_sequence<int, Index...>) {
  int N = sizeof...(T);
  TRACE_START(merge_tables_specialized);

  // Merge
  TRACE_START(specalized_merge_merge);
  auto left_zipped = thrust::make_zip_iterator(thrust::make_tuple(
      left.sample_mask().data(),
      left.template column<typename T::sort_type>(Index).begin()...));
  auto right_zipped = thrust::make_zip_iterator(thrust::make_tuple(
      right.sample_mask().data(),
      right.template column<typename T::sort_type>(Index).begin()...));

  if (SAFETY) {
    assert(thrust::is_sorted(thrust::device, left_zipped,
                             left_zipped + left.size()));
    assert(thrust::is_sorted(thrust::device, right_zipped,
                             right_zipped + right.size()));
  }

  auto merged_facts_size = left.size() + right.size();
  device_vec<char> output_sample_mask(merged_facts_size, alloc);
  Array<device_buffer> output_facts(N);
  SINK((new (&output_facts[Index])
            device_buffer(merged_facts_size, T::tag(), alloc))...);
  auto output_facts_iterator = thrust::make_zip_iterator(thrust::make_tuple(
      output_sample_mask.data(),
      output_facts[Index].template begin<typename T::sort_type>().data()...));

  device_vec<typename Prov::Tag> merged_tags;

  if (Prov::is_unit) {
    thrust::merge(thrust::device, left_zipped, left_zipped + left.size(),
                  right_zipped, right_zipped + right.size(),
                  output_facts_iterator);
  } else if (Prov::large_tags) {
    merged_tags = device_vec<typename Prov::Tag>(merged_facts_size);
    device_vec<uint32_t> left_indices(merged_facts_size);
    thrust::sequence(thrust::device, left_indices.begin(), left_indices.end());
    device_vec<uint32_t> right_indices(merged_facts_size);
    thrust::sequence(thrust::device, right_indices.begin(), right_indices.end(),
                     1u << 31);
    device_vec<uint32_t> output_indices(merged_facts_size);

    thrust::merge_by_key(thrust::device, left_zipped, left_zipped + left.size(),
                         right_zipped, right_zipped + right.size(),
                         left_indices.data(), right_indices.data(),
                         output_facts_iterator, output_indices.data());

    gather_merge_tags<Prov>
        <<<ROUND_UP_TO_NEAREST(merged_facts_size, 128), 128>>>(
            left.tags().data(), right.tags().data(), output_indices.data(),
            merged_tags.data(), merged_facts_size);
    cudaCheck(cudaDeviceSynchronize());
  } else {
    merged_tags = device_vec<typename Prov::Tag>(merged_facts_size);
    thrust::merge_by_key(thrust::device, left_zipped, left_zipped + left.size(),
                         right_zipped, right_zipped + right.size(),
                         left.tags().cbegin(), right.tags().cbegin(),
                         output_facts_iterator, merged_tags.begin());

    // auto &keys1 = left_zipped;
    // auto &keys2 = right_zipped;
    // auto &values1 = left.tags().cbegin();
    // auto &values2 = right.tags().cbegin();
    // auto &result_keys = merged_facts_iterator;
    // auto &result_values = merged_tags;

    //// 1) Get temp storage size
    // std::size_t temp_storage_bytes = 0;
    // cub::DeviceMerge::MergePairs(
    //     nullptr, temp_storage_bytes, keys1.begin(), values1.begin(),
    //     static_cast<int>(keys1.size()), keys2.begin(), values2.begin(),
    //     static_cast<int>(keys2.size()), result_keys,
    //     result_values.begin());

    //// 2) Allocate temp storage
    // thrust::device_vector<char> temp_storage(temp_storage_bytes);

    //// 3) Perform merge operation
    // cub::DeviceMerge::MergePairs(
    //     thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes,
    //     keys1.begin(), values1.begin(), static_cast<int>(keys1.size()),
    //     keys2.begin(), values2.begin(), static_cast<int>(keys2.size()),
    //     result_keys, result_values.begin());
  }
  TRACE_END(specalized_merge_merge);

  // Unique
  // Indices into the merged_facts array (before it's unique)
  TRACE_START(specalized_merge_unique);
  device_vec<int32_t> output_facts_indices(merged_facts_size);
  thrust::sequence(thrust::device, output_facts_indices.begin(),
                   output_facts_indices.end());

  hINFO("Sorted table:");
  hINFO(Table<Prov>(left.schema(), merged_tags.clone(), output_facts.clone(),
                    output_sample_mask.clone()));

  if (SAFETY) {
    auto sorted_til =
        thrust::is_sorted_until(thrust::device, output_facts_iterator,
                                output_facts_iterator + merged_facts_size);
    auto sorted_length = thrust::distance(output_facts_iterator, sorted_til);
    if (sorted_length != merged_facts_size) {
      std::cout << "Error: only sorted until " << sorted_length
                << " but length is " << merged_facts_size << std::endl;
      PANIC("Merging in merge didnt retain sort");
    }
  }

  int output_length;
  if (!NO_CUB() && Prov::is_unit) {
    device_vec<char> tmp_mask(merged_facts_size, alloc);
    Array<device_buffer> tmp_facts(N);
    SINK((new (&tmp_facts[Index])
              device_buffer(merged_facts_size, T::tag(), alloc))...);
    auto tmp_facts_iter = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_mask.data(),
        tmp_facts[Index].template begin<typename T::sort_type>().data()...));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    device_vec<int> d_num_selected_out(1);
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              output_facts_iterator, tmp_facts_iter,
                              d_num_selected_out.data(), merged_facts_size);

    device_vec<char> temp_storage(temp_storage_bytes);
    d_temp_storage = temp_storage.data();
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              output_facts_iterator, tmp_facts_iter,
                              d_num_selected_out.data(), merged_facts_size);

    output_sample_mask = std::move(tmp_mask);
    output_facts = std::move(tmp_facts);

    output_length = d_num_selected_out.at_host(0);
  } else {
    auto unique_result = thrust::unique_by_key(
        thrust::device, output_facts_iterator,
        output_facts_iterator + merged_facts_size, output_facts_indices.data());

    output_length =
        thrust::distance(output_facts_iterator, thrust::get<0>(unique_result));
  }

  SINK((output_facts[Index].resize(output_length), 0)...);
  output_sample_mask.resize(output_length);
  TRACE_END(specalized_merge_unique);

  hINFO("Merged tags:\n" << merged_tags);
  hINFO("Output facts indices:\n" << output_facts_indices);
  hINFO("Output length: " << output_length);

  // Combine tags
  device_vec<typename Prov::Tag> output_tags;
  if (!Prov::is_unit) {
    TRACE_START(specalized_merge_combine_tags);
    output_tags = device_vec<typename Prov::Tag>(output_length, alloc);
    specialized_merge_combine_tags<Prov>
        <<<ROUND_UP_TO_NEAREST(output_length, 128), 128>>>(
            output_sample_mask.data(), merged_tags.data(),
            output_facts_indices.data(), output_tags.data(), output_length,
            ctx.device_context());
    cudaCheck(cudaDeviceSynchronize());
    TRACE_END(specalized_merge_combine_tags);
  }

  return Table<Prov>(left.schema(), std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
}
