#include <cub/device/device_segmented_reduce.cuh>
#include <numeric>

#include "../expr.h"
#include "../flame.h"
#include "../provenance.h"
#include "../utils.h"
#include "project_constant.h"

template <typename Prov>
struct TagSegmentedReduceFunctor {
  typename Prov::BatchDeviceContext ctxs;
  TagSegmentedReduceFunctor(typename Prov::BatchDeviceContext ctxs)
      : ctxs(ctxs) {}
  using T = thrust::tuple<char, typename Prov::Tag>;
  __device__ T operator()(T accum, T tag) {
    return thrust::make_tuple(
        0, Prov::add(Prov::sample_context(ctxs, thrust::get<0>(tag)),
                     thrust::get<1>(accum), thrust::get<1>(tag)));
  }
};
template <typename Prov>
struct TagReduceFunctor {
  typename Prov::BatchDeviceContext ctxs;
  TagReduceFunctor(typename Prov::BatchDeviceContext ctxs) : ctxs(ctxs) {}
  using T = typename Prov::Tag;
  __device__ T operator()(T accum, T tag) {
    return Prov::add(Prov::sample_context(ctxs, 0), accum, tag);
  }
};

static float clause_probability_host(std::vector<float> lit_probs,
                              typename DiffTopKProofsProvenance<>::Tag t) {
  if (t.empty_) {
    return 0.0;
  }
  float result = 1.0;
  for (int i = 0; i < t.lit_count_; i++) {
    auto lit = t.literals_[i];
    auto prob = lit_probs.at(abs(lit));
    result *= (lit >= 0) ? prob : (1.0 - prob);
  }
  return result;
}

static typename DiffTopKProofsProvenance<>::Tag dtkp_add_host(
    std::vector<float> lit_probs, typename DiffTopKProofsProvenance<>::Tag t1,
    typename DiffTopKProofsProvenance<>::Tag t2) {
  if (clause_probability_host(lit_probs, t1) >
      clause_probability_host(lit_probs, t2)) {
    return t1;
  } else {
    return t2;
  }
}

template <typename Prov>
Table<Prov> project_constant(Table<Prov> source, const Expr &expr,
                             TupleType result_schema, const Prov &ctx) {
  TRACE_START(project_constant);
  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }

  hINFO("Project_constant(" << expr << ")");
  hINFO("input table:\n" << source);

  Array<device_buffer> output_facts(result_schema.width());
  assert(expr.tag == Expr::Tag::Tuple);
  for (size_t i = 0; i < expr.tuple._0.size(); i++) {
    assert(expr.tuple._0[i].tag == Expr::Tag::Constant);
    std::vector<Value> host_vec;
    auto constant = expr.tuple._0[i].constant._0;
    host_vec.push_back(constant);
    new (&output_facts[i]) device_buffer(host_vec, constant.type());
  }


  TRACE_START(project_constant_tag_reduce);
  auto batch_size = get_batch_size();

  if (Prov::is_unit) {
    if (batch_size > 1) {
      PANIC("Batch size > 1 not supported by project_constant");
    }
    Table<Prov> result(result_schema, device_vec<typename Prov::Tag>(),
                       std::move(output_facts),
                       device_vec<char>(std::vector<char>{0}));
    return result;
  }

  device_vec<typename Prov::Tag> output_tags;
  if (batch_size > 1) {
    PANIC("Batch size > 1 not supported by project_constant");
    // auto sample_sizes = source.sample_sizes();
    // std::vector<size_t> offsets_host;
    // offsets_host.push_back(0);
    // std::inclusive_scan(sample_sizes.begin(), sample_sizes.end(),
    //                     std::back_inserter(offsets_host));
    // device_vec<size_t> offsets_device(offsets_host);
    //
    // auto batch_size = sample_sizes.size();
    // device_vec<typename Prov::Tag> output_tags(batch_size);
    // auto op = TagSegmentedReduceFunctor<Prov>(ctx.device_context());
    // auto init = thrust::make_tuple(0, Prov::zero());
    // void *d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    // auto it = thrust::make_zip_iterator(
    //     thrust::make_tuple(source.sample_mask().data(),
    //     source.tags().data()));
    // device_vec<char> sink(batch_size);
    // auto out = thrust::make_zip_iterator(
    //     thrust::make_tuple(sink.data(), output_tags.data()));
    // cudaCheck(cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
    // temp_storage_bytes, it,
    //                                    out, batch_size,
    //                                    offsets_device.data(),
    //                                    offsets_device.data() + 1, op, init));
    // device_vec<uint8_t> temp_storage(temp_storage_bytes);
    // d_temp_storage = temp_storage.data();
    // cudaCheck(cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
    // temp_storage_bytes, it,
    //                                    out, batch_size,
    //                                    offsets_device.data(),
    //                                    offsets_device.data() + 1, op, init));
  } else {
    // auto reduced_tag = thrust::reduce(
    //     thrust::device, source.tags().begin(), source.tags().end(),
    //     Prov::zero(), TagReduceFunctor<Prov>(ctx.device_context()));
    // output_tags = device_vec(std::vector<typename Prov::Tag>{reduced_tag});
    if constexpr (std::is_same<Prov, DiffTopKProofsProvenance<>>::value) {
      std::vector<typename Prov::Tag> tags_host = source.tags().to_host();
      std::vector<float> lit_probs = ctx.literal_probabilities()[0].to_host();
      auto result_tag = std::reduce(tags_host.begin(), tags_host.end(),
                                Prov::zero(), [&](auto accum, auto tag) {
                                  return dtkp_add_host(lit_probs, accum, tag);
                                });
      output_tags = device_vec(std::vector<typename Prov::Tag>{result_tag});
    } else {
      PANIC("Unsupported provenance type for project_constant");
    }
    // output_tags = device_vec(batch_size);
    // auto op = TagReduceFunctor<Prov>(ctx.device_context());
    // auto init = Prov::zero();
    // void *d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    // auto it = source.tags.data();
    // aut0 out = output_tags.data();
    // auto num_items = source.tags().size();
    // cudaCheck(cub::DeviceReduce::Reduce(
    //   d_temp_storage, temp_storage_bytes,
    //   it, out, num_items, op, init));
    // device_vec<uint8_t> temp_storage(temp_storage_bytes);
    // d_temp_storage = temp_storage.data();
    // cudaCheck(cub::DeviceReduce::Reduce(
    //   d_temp_storage, temp_storage_bytes,
    //   it, out, num_items, op, init));
  }
  TRACE_END(project_constant_tag_reduce);

  std::vector<char> output_sample_mask_host(batch_size);
  std::iota(output_sample_mask_host.begin(), output_sample_mask_host.end(), 0);
  device_vec<char> output_sample_mask(output_sample_mask_host);

  Table<Prov> result(result_schema, std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
  hINFO("result table:\n" << result);
  return result;
}

#define PROV UnitProvenance
template Table<PROV> project_constant(Table<PROV> source, const Expr &expr,
                                      TupleType result_schema, const PROV &ctx);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> project_constant(Table<PROV> source, const Expr &expr,
                                      TupleType result_schema, const PROV &ctx);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> project_constant(Table<PROV> source, const Expr &expr,
                                      TupleType result_schema, const PROV &ctx);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> project_constant(Table<PROV> source, const Expr &expr,
                                      TupleType result_schema, const PROV &ctx);
#undef PROV
