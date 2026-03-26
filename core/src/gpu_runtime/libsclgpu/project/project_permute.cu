#include "../expr.h"
#include "../flame.h"
#include "../provenance.h"
#include "../utils.h"
#include "project_permute.h"

template <typename Prov>
Table<Prov> project_permute(Table<Prov> source, const Expr &expr,
                            TupleType result_schema) {
  TRACE_START(project_permute_batched);
  if (source.size() == 0) {
    return Table<Prov>(result_schema);
  }
  auto baked_expr = BakedPermuteExpr::bake(source.schema(), expr);
  auto result = baked_expr.evaluate(source, result_schema);
  hINFO("Project_permute(" << expr << ")");
  hINFO("input table:\n" << source);
  hINFO("result table:\n" << result);
  return result;
}

#define PROV UnitProvenance
template Table<PROV> project_permute(Table<PROV> source, const Expr &expr,
                                     TupleType result_schema);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> project_permute(Table<PROV> source, const Expr &expr,
                                     TupleType result_schema);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> project_permute(Table<PROV> source, const Expr &expr,
                                     TupleType result_schema);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> project_permute(Table<PROV> source, const Expr &expr,
                                     TupleType result_schema);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> project_permute(Table<PROV> source, const Expr &expr,
                                     TupleType result_schema);
#undef PROV
