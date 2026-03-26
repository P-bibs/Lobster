#include "../normalize.cu"

#define PROV DiffTopKProofsProvenance<>
template Table<PROV> normalized(Table<PROV> schema, const PROV &ctx);
#undef PROV
