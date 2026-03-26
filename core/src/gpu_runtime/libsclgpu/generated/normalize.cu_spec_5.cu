#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<UnitProvenance> normalized_specialized(Table<UnitProvenance> table, const UnitProvenance &ctx, const Allocator &, Product<ValueF32,ValueU32,ValueU32>,
std::integer_sequence<int, 0,1,2>);