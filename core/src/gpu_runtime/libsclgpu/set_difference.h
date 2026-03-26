#pragma once

#include "alloc.h"
#include "device_vec.h"

template <typename... T>
Table<Prov> difference(Table<Prov> left, Table<Prov> right,
                      const TupleType &result_schema, const Prov &ctx);
