#include "table.h"

template <typename Prov>
Table<Prov> product(Table<Prov> left, Table<Prov> right,
                    TupleType result_schema, const Prov &ctx);
