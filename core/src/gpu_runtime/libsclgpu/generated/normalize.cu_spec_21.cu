#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<DiffMinMaxProbProvenance> normalized_specialized(Table<DiffMinMaxProbProvenance> table, const DiffMinMaxProbProvenance &ctx, const Allocator &, Product<ValueU32,ValueU32,ValueU32,ValueU32>,
std::integer_sequence<int, 0,1,2,3>);