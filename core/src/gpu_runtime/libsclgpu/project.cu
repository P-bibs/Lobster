#include "expr.h"
#include "flame.h"
#include "normalize.h"
#include "project.h"
#include "project/project_constant.h"
#include "project/project_interpret.h"
#include "project/project_permute.h"
#include "provenance.h"
#include "utils.h"

MAKE_ENV_GETTER(PAR_PROJ);
template <typename Prov>
Table<Prov> project(Table<Prov> table, const Expr &expr,
                    TupleType result_schema, const Prov &ctx) {
  TRACE_START(project_batched);
  table.validate();

  if (expr.is_constant()) {
    return project_constant(table, expr, result_schema, ctx);
  } else if (expr.is_permutation()) {
    return project_permute(table, expr, result_schema);
  } else {
    return project_interpret(table, expr, result_schema);
  }
}

#define PROV UnitProvenance
template Table<PROV> project(Table<PROV> tables, const Expr &expr,
                             TupleType result_schema, const PROV &);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> project(Table<PROV> tables, const Expr &expr,
                             TupleType result_schema, const PROV &);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> project(Table<PROV> tables, const Expr &expr,
                             TupleType result_schema, const PROV &);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> project(Table<PROV> tables, const Expr &expr,
                             TupleType result_schema, const PROV &);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> project(Table<PROV> tables, const Expr &expr,
                             TupleType result_schema, const PROV &);
#undef PROV
