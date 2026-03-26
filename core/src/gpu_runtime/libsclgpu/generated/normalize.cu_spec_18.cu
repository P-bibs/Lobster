#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<DiffMinMaxProbProvenance> normalized_specialized(Table<DiffMinMaxProbProvenance> table, const DiffMinMaxProbProvenance &ctx, const Allocator &, Product<ValueU32>,
std::integer_sequence<int, 0>);