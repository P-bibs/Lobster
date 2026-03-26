#include "../provenance.h"
#include "../merge_specialized.cu"
template Table<DiffTopKProofsProvenance<>> merge_tables_specialized(
const Table<DiffTopKProofsProvenance<>> &left, const Table<DiffTopKProofsProvenance<>> &right, const DiffTopKProofsProvenance<> &ctx,
const Allocator &, Product<ValueU32,ValueU32,ValueF32>, std::integer_sequence<int, 0,1,2>);