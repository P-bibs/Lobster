#include "../normalize.cu"
#include "../specialize_normalize.cu"

#define PROV DiffAddMultProbProvenance<>
template Table<PROV> normalized(Table<PROV> schema, const PROV &ctx);
#undef PROV
