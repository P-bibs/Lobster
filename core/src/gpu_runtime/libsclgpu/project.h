#pragma once
#include "table.h"

template <typename Prov>
Table<Prov> project(Table<Prov> tables, const Expr &expr,
                    TupleType result_schema, const Prov &ctx);
