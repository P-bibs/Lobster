#include "flame.h"
#include "product.h"
#include "provenance.h"
#include "utils.h"

template <typename Prov>
__global__ void combine_product_tags(char *left_sample_mask,
                                     const typename Prov::Tag *left,
                                     const typename Prov::Tag *right,
                                     typename Prov::Tag *result, int left_size,
                                     int right_size,
                                     typename Prov::BatchDeviceContext ctxs) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  auto result_size = left_size * right_size;
  if (row >= result_size) {
    return;
  }

  auto left_row = row / right_size;
  auto right_row = row % right_size;

  auto ctx = Prov::sample_context(ctxs, left_sample_mask[left_row]);

  result[row] = Prov::mult(ctx, left[left_row], right[right_row]);
}

template <typename T>
__global__ void product_left(const T *input, T *output, int right_size,
                             int output_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= output_size) {
    return;
  }

  auto input_row = row / right_size;
  output[row] = input[input_row];
}

template <typename T>
__global__ void product_right(const T *input, T *output, int right_size,
                              int output_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= output_size) {
    return;
  }

  auto input_row = row % right_size;
  output[row] = input[input_row];
}

template <typename Prov>
Table<Prov> product(Table<Prov> left, Table<Prov> right,
                    TupleType result_schema, const Prov &ctx) {
  TRACE_START(product);

  auto result_size = left.size() * right.size();
  if (result_size == 0) {
    return Table<Prov>(result_schema);
  }

  auto result_width = left.width() + right.width();

  Array<device_buffer> result_facts(result_width);

  // TODO: what if we align warps to the size of the right table
  for (size_t column = 0; column < left.width(); column++) {
    auto data_kind = left.schema().flatten().at(column).singleton();
    new (&result_facts[column]) device_buffer(result_size, data_kind);

    DISPATCH_ON_TYPE(data_kind, T,
                     (product_left<T><<<(result_size + 255) / 256, 256>>>(
                         static_cast<const T *>(left.at_raw(0, column)),
                         static_cast<T *>(result_facts[column].data()),
                         right.size(), result_size)););
    cudaCheck(cudaDeviceSynchronize());
  }

  for (size_t column = 0; column < right.width(); column++) {
    auto data_kind = right.schema().flatten().at(column).singleton();
    auto output_column = column + left.width();
    new (&result_facts[output_column]) device_buffer(result_size, data_kind);

    DISPATCH_ON_TYPE(data_kind, T,
                     (product_right<T><<<(result_size + 255) / 256, 256>>>(
                         static_cast<const T *>(right.at_raw(0, column)),
                         static_cast<T *>(result_facts[output_column].data()),
                         right.size(), result_size)););
    cudaCheck(cudaDeviceSynchronize());
  }

  device_vec<typename Prov::Tag> result_tags(result_size);
  combine_product_tags<Prov><<<(result_size + 255) / 256, 256>>>(
      left.sample_mask().data(), left.tags().data(), right.tags().data(), result_tags.data(), left.size(),
      right.size(), ctx.device_context());
  cudaCheck(cudaDeviceSynchronize());

  auto sample_sizes = left.sample_sizes();
  device_vec<char> output_sample_mask(result_size);
  product_left<char><<<(result_size + 255) / 256, 256>>>(
      left.sample_mask().data(), output_sample_mask.data(), right.size(),
      result_size);
  cudaCheck(cudaDeviceSynchronize());

  // TODO: calculate proper sample mask
  auto result =
      Table<Prov>(result_schema, std::move(result_tags),
                  std::move(result_facts), std::move(output_sample_mask));

  hINFO("Product: ");
  hINFO("left table:\n" << left);
  hINFO("right table:\n" << right);
  hINFO("result table:" << result);
  return result;
}

#define PROV UnitProvenance
template Table<PROV> product(
    const Table<PROV> left,
    const Table<PROV> right, TupleType result_schema,
    const PROV &ctx);
#undef PROV 
#define PROV MinMaxProbProvenance
template Table<PROV> product(
    const Table<PROV> left,
    const Table<PROV> right, TupleType result_schema,
    const PROV &ctx);
#undef PROV 
#define PROV DiffMinMaxProbProvenance
template Table<PROV> product(
    const Table<PROV> left,
    const Table<PROV> right, TupleType result_schema,
    const PROV &ctx);
#undef PROV 
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> product(
    const Table<PROV> left,
    const Table<PROV> right, TupleType result_schema,
    const PROV &ctx);
#undef PROV 
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> product(
    const Table<PROV> left,
    const Table<PROV> right, TupleType result_schema,
    const PROV &ctx);
#undef PROV 
