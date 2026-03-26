#include "expr.h"
#include "filter.h"
#include "flame.h"
#include "provenance.h"
#include "utils.h"

template <typename Prov>
struct FilterKernel {
  BakedExpr program;
  TableView<Prov> t;

  FilterKernel(BakedExpr &program, TableView<Prov> t)
      : program(program), t(t) {}

  __device__ bool operator()(uint32_t i) {
    Value val = BakedExpr::eval(program, t, i);
    if (SAFETY) {
      if (val.tag != Value::Tag::Bool) {
        PANIC("Filter expression did not evaluate to a boolean at row %u", i);
      }
    }
    return val.bool_._0;
  }
};

template <typename Prov, typename T>
struct FilterNeqKernel {
  uint8_t lhs;
  uint8_t rhs;
  TableView<Prov> t;

  FilterNeqKernel(uint8_t lhs, uint8_t rhs, TableView<Prov> t)
      : lhs(lhs), rhs(rhs), t(t) {}

  __device__ bool operator()(uint32_t i) {
    return t.template at<T>(i, lhs) != t.template at<T>(i, rhs);
  }
};



template <typename Prov>
Table<Prov> filter(Table<Prov> source, const Expr &expr,
                   TupleType result_schema) {
  TRACE_START(filter);
  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }

  auto baked_expr = BakedExpr(expr, source.schema());

  device_vec<uint32_t> result_indices(source.size());

  auto indices = thrust::make_counting_iterator<uint32_t>(0);

  size_t output_size;

  bool simple = expr.tag == Expr::Tag::Binary && expr.binary._0.op == BinaryOp::Neq &&
                expr.binary._0.op1->tag == Expr::Tag::Access &&
                expr.binary._0.op2->tag == Expr::Tag::Access;
  if (simple) {
    TRACE_START(filter_special_case);
    uint8_t lhs = expr.binary._0.op1->access._0.to_index(source.schema());
    uint8_t rhs = expr.binary._0.op2->access._0.to_index(source.schema());
    ValueType type = source.schema().flatten().at(lhs).singleton();
    //DISPATCH_ON_TYPE(type, T, 
    //  auto end = thrust::copy_if(thrust::device, indices, indices + source.size(),
    //                             result_indices.begin(),
    //                             FilterNeqKernel<Prov, T>(lhs, rhs, source.view()));
      auto end = thrust::copy_if(thrust::device, indices, indices + source.size(),
                                 result_indices.begin(),
                                 FilterNeqKernel<Prov, uint32_t>(lhs, rhs, source.view()));
      output_size = end - result_indices.begin();
  } else {
    auto end = thrust::copy_if(thrust::device, indices, indices + source.size(),
                               result_indices.begin(),
                               FilterKernel<Prov>(baked_expr, source.view()));
    output_size = end - result_indices.begin();
  }

  if (output_size == 0) {
    return Table<Prov>(result_schema);
  }

  // gather
  Array<device_buffer> output_facts(source.schema().width());
  for (size_t col = 0; col < source.schema().width(); col++) {
    output_facts[col] = device_buffer(
        output_size, source.schema().flatten().at(col).singleton());
    DISPATCH_ON_TYPE(
        source.schema().flatten().at(col).singleton(), T,
        thrust::gather(thrust::device, result_indices.begin(),
                       result_indices.begin() + output_size,
                       reinterpret_cast<const T *>(source.at_raw(0, col)),
                       output_facts[col].begin<T>()););
  }

  device_vec<typename Prov::Tag> output_tags;
  if (!Prov::is_unit) {
    output_tags = device_vec<typename Prov::Tag>(output_size);
    thrust::gather(thrust::device, result_indices.begin(),
                   result_indices.begin() + output_size, source.tags().data(),
                   output_tags.begin());
  }


  device_vec<char> output_sample_mask(output_size);
  thrust::gather(thrust::device, result_indices.begin(),
                 result_indices.begin() + output_size,
                 source.sample_mask().data(), output_sample_mask.begin());

  Table<Prov> result(result_schema, std::move(output_tags),
                     std::move(output_facts), std::move(output_sample_mask));
  hINFO("Find(" << expr << ")");
  hINFO("input table:\n" << source);
  hINFO("result table:\n" << result);
  return result;
}
#define PROV UnitProvenance
template Table<PROV> filter(Table<PROV> left, const Expr &expr,
                            TupleType result_schema);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> filter(Table<PROV> left, const Expr &expr,
                            TupleType result_schema);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> filter(Table<PROV> left, const Expr &expr,
                            TupleType result_schema);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> filter(Table<PROV> left, const Expr &expr,
                            TupleType result_schema);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> filter(Table<PROV> left, const Expr &expr,
                            TupleType result_schema);
#undef PROV
