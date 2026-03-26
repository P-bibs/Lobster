#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<DiffAddMultProbProvenance<>> normalized_specialized(Table<DiffAddMultProbProvenance<>> table, const DiffAddMultProbProvenance<> &ctx, const Allocator &, Product<ValueU32,ValueU32>,
std::integer_sequence<int, 0,1>);