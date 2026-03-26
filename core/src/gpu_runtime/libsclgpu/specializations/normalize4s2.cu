#include "../specialize_normalize.cu"

#define PROV DiffTopKProofsProvenance<>
template Table<DiffTopKProofsProvenance<>>
Specialized<PROV, 0, 1, 2>::normalized<ValueU32, ValueU32, ValueU32>(
    Table<PROV> source, const PROV &ctx);
#undef PROV
