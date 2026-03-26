#include "project_constant.cu"

#define PROV DiffTopKProofsProvenance<>
template Table<PROV> project_constant(Table<PROV> source, const Expr &expr,
                                      TupleType result_schema, const PROV &ctx);
#undef PROV
