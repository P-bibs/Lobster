#include "../expr.h"
#include "../flame.h"
#include "../provenance.h"
#include "../utils.h"
#include "project_interpret.h"
#include "../remove_nan.h"

template <typename Prov, typename T>
struct ProjectFunctor {
  BakedExpr program;
  TableView<Prov> t;

  ProjectFunctor(const BakedExpr &program, TableView<Prov> t)
      : program(program), t(t) {}

  __device__ T operator()(uint32_t i) {
    return BakedExpr::eval(program, t, i).template downcast<T>();
  }
};
template <typename Prov>
Table<Prov> project_interpret(Table<Prov> source, const Expr &expr,
                              TupleType result_schema) {
  TRACE_START(project_interpret);

  hINFO("Project: " << expr);

  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }

  auto baked_expr = BakedTupleExpr(expr, source.schema());

  hINFO("Baked expression: " << baked_expr);

  if (baked_expr.size() != result_schema.width()) {
    throw std::runtime_error(
        "Projection result schema does not match expression");
  }

  Array<device_buffer> output_facts(baked_expr.size());
  for (size_t i = 0; i < baked_expr.size(); i++) {
    output_facts[i] =
        device_buffer(source.size(), result_schema.flatten().at(i).singleton());
    auto indices = thrust::make_counting_iterator<uint32_t>(0);
    DISPATCH_ON_TYPE(
        result_schema.flatten().at(i).singleton(), T,
        thrust::transform(
            thrust::device, indices, indices + source.size(),
            output_facts[i].begin<T>(),
            ProjectFunctor<Prov, T>(baked_expr.at(i), source.view())););
  }
  Table<Prov> result(result_schema, std::move(source.tags().clone()),
                     std::move(output_facts),
                     std::move(source.sample_mask().clone()));

  if (baked_expr.has_division()) {
    result = remove_nan(result);
  }

  hINFO("Project_interpret(" << expr << ")");
  hINFO("input table:\n" << source);
  hINFO("result table:\n" << result);

  return result;
}

#define PROV UnitProvenance
template Table<PROV> project_interpret(Table<PROV> source, const Expr &expr,
                                       TupleType result_schema);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> project_interpret(Table<PROV> source, const Expr &expr,
                                       TupleType result_schema);
#undef PROV

#define PROV DiffMinMaxProbProvenance
template Table<PROV> project_interpret(Table<PROV> source, const Expr &expr,
                                       TupleType result_schema);
#undef PROV

#define PROV DiffAddMultProbProvenance<>
template Table<PROV> project_interpret(Table<PROV> source, const Expr &expr,
                                       TupleType result_schema);
#undef PROV

#define PROV DiffTopKProofsProvenance<>
template Table<PROV> project_interpret(Table<PROV> source, const Expr &expr,
                                       TupleType result_schema);
#undef PROV
