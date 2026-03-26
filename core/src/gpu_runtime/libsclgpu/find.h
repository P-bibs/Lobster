#pragma once
#include "table.h"

template <typename Prov>
Table<Prov> find(Table<Prov> source, const Value key, TupleType result_schema);
