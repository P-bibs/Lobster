#pragma once

#include "bindings.h"
#include "table.h"
#include "table_index.h"

template <typename Prov>
Table<Prov> intersect(Table<Prov> left, Table<Prov> right,
                 const TupleType &result_schema, const Prov &ctx, bool index_on_right);
