#include "../normalize.cu"
#include "../specialize_normalize.cu"

#define PROV DiffMinMaxProbProvenance
template Table<PROV> normalized(Table<PROV> schema, const PROV &ctx);
#undef PROV
