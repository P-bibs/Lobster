#include "../specialize_normalize.cu"

#define PROV DiffTopKProofsProvenance<>
template Table<DiffTopKProofsProvenance<>>
Specialized<PROV, 0>::normalized<ValueU32>(Table<PROV> source, const PROV &ctx);
#undef PROV

// We don't need to provide this specialization for correctness, but if we
// just use the normal template for the single-value case, compile times
// increase 10x for unknown reasons.
// template <>
// template <>
// Table<DiffTopKProofsProvenance<>>
// Specialized<DiffTopKProofsProvenance<>, 0>::normalized<ValueU32>(
//    Table<DiffTopKProofsProvenance<>> source,
//    const DiffTopKProofsProvenance<> &ctx) {
//  using Prov = DiffTopKProofsProvenance<>;
//  TRACE_START(specialized_normalize);
//
//  if (facts.size() == 0) {
//    throw std::runtime_error("Table: normalize called on empty table");
//  }
//  auto input_size = facts[0].size();
//  if (input_size == 0) {
//    return Table<Prov>(schema);
//  }
//
//  auto input = facts[0].cbegin<ValueU32::type>();
//
//  TRACE_START(specialized_normalize_output_alloc);
//  Array<device_buffer> output_facts(1);
//  new (&output_facts[0]) device_buffer(input_size, ValueU32::tag());
//  auto output = output_facts[0].template begin<ValueU32::type>().data();
//  thrust::copy(thrust::device, input, input + input_size, output);
//  TRACE_END(specialized_normalize_output_alloc);
//
//  device_vec<typename Prov::Tag> sorted_tags(std::move(tags.clone()));
//  device_vec<uint32_t> sorted_indices(input_size);
//  thrust::sequence(thrust::device, sorted_indices.begin(),
//                   sorted_indices.end());
//
//  if (SIZES()) {
//    std::cout << "Normalizing table of size " << input_size << std::endl;
//  }
//
//  TRACE_START(specialized_normalize_sort);
//  if constexpr (Prov::large_tags) {
//    thrust::sort_by_key(thrust::device, output, output + input_size,
//                        sorted_indices.begin());
//    device_vec<typename Prov::Tag> sorted_tags_temp(sorted_tags.size());
//    thrust::gather(thrust::device, sorted_indices.begin(),
//    sorted_indices.end(),
//                   sorted_tags.begin(), sorted_tags_temp.begin());
//    sorted_tags = std::move(sorted_tags_temp);
//  } else {
//    thrust::sort_by_key(thrust::device, output, output + input_size,
//                        sorted_tags.begin());
//  }
//  TRACE_END(specialized_normalize_sort);
//
//  TRACE_START(specialized_normalize_unique);
//  device_vec<uint32_t> run_start_indices(input_size);
//  thrust::sequence(thrust::device, run_start_indices.begin(),
//                   run_start_indices.end());
//  auto new_end = thrust::unique_by_key(
//      thrust::device, output, output + input_size, run_start_indices.data());
//  auto output_size = thrust::distance(output, new_end.first);
//  TRACE_END(specialized_normalize_unique);
//
//  output_facts[0].resize(output_size);
//
//  TRACE_START(normalize_combine_tags);
//  device_vec<typename Prov::Tag> output_tags(output_size);
//
//  dim3 blockDim = {32, 1, 1};
//  dim3 gridDim = {ROUND_UP_TO_NEAREST(output_size, 32), 1, 1};
//  combine_tags_specialized<Prov><<<gridDim, blockDim>>>(
//      run_start_indices.data(), sorted_tags.data(), output_tags.data(),
//      input_size, output_size, ctx.device_context());
//  cudaCheck(cudaPeekAtLastError());
//  cudaCheck(cudaDeviceSynchronize());
//
//  TRACE_END(normalize_combine_tags);
//
//  Table<Prov> output_table(schema, std::move(output_tags),
//                           std::move(output_facts));
//
//  hINFO("Normalize() output:");
//  hINFO(output_table);
//  hINFO("");
//
//  return output_table;
//}
