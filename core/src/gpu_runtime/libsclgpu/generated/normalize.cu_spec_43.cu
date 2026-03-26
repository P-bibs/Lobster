#include "../provenance.h"
#include "../specialize_normalize.cu"
template
Table<DiffTopKProofsProvenance<>> normalized_specialized(Table<DiffTopKProofsProvenance<>> table, const DiffTopKProofsProvenance<> &ctx, const Allocator &, Product<ValueU32,ValueU32,ValueF32>,
std::integer_sequence<int, 0,1,2>);