#include "../provenance.h"
#include "../merge_specialized.cu"
template Table<UnitProvenance> merge_tables_specialized(
const Table<UnitProvenance> &left, const Table<UnitProvenance> &right, const UnitProvenance &ctx,
const Allocator &, Product<ValueU32,ValueU32,ValueU32>, std::integer_sequence<int, 0,1,2>);