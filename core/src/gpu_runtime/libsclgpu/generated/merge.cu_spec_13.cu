#include "../provenance.h"
#include "../merge_specialized.cu"
template Table<MinMaxProbProvenance> merge_tables_specialized(
const Table<MinMaxProbProvenance> &left, const Table<MinMaxProbProvenance> &right, const MinMaxProbProvenance &ctx,
const Allocator &, Product<ValueU32,ValueU32,ValueF32>, std::integer_sequence<int, 0,1,2>);