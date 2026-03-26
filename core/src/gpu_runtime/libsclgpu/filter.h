#pragma once
#include "table.h"

template <typename Prov>
Table<Prov> filter(Table<Prov> left, const Expr &expr, TupleType result_schema);
