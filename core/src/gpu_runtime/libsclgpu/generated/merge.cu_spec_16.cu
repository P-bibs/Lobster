#include "../provenance.h"
#include "../merge_specialized.cu"
template Table<DiffMinMaxProbProvenance> merge_tables_specialized(
const Table<DiffMinMaxProbProvenance> &left, const Table<DiffMinMaxProbProvenance> &right, const DiffMinMaxProbProvenance &ctx,
const Allocator &, Product<ValueU32>, std::integer_sequence<int, 0>);