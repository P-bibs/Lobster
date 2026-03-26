#include "../specialize_normalize.cu"

#define PROV DiffTopKProofsProvenance<>
template Table<DiffTopKProofsProvenance<>>
Specialized<PROV, 0>::normalized<ValueF32>(Table<PROV> source, const PROV &ctx);
#undef PROV
