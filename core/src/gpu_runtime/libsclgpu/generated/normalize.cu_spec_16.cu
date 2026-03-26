#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<MinMaxProbProvenance> normalized_specialized(Table<MinMaxProbProvenance> table, const MinMaxProbProvenance &ctx, const Allocator &, Product<ValueU32,ValueU32,ValueF32>,
std::integer_sequence<int, 0,1,2>);