#include "../provenance.h"
#include "../merge_specialized.cu"
template Table<DiffAddMultProbProvenance<>> merge_tables_specialized(
const Table<DiffAddMultProbProvenance<>> &left, const Table<DiffAddMultProbProvenance<>> &right, const DiffAddMultProbProvenance<> &ctx,
const Allocator &, Product<ValueU32,ValueU32,ValueF32>, std::integer_sequence<int, 0,1,2>);