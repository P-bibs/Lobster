#pragma once
#include "../table.h"

template <typename Prov>
Table<Prov> project_interpret(Table<Prov> source, const Expr &expr,
                              TupleType result_schema);
