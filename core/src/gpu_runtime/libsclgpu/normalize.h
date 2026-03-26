#pragma once

#include "bindings.h"
#include "device_vec.h"
#include "flame.h"
#include "table.h"

template <typename Prov>
Table<Prov> normalized(Table<Prov> table, const Prov &ctx, const Allocator &alloc);

template <typename Prov, typename... T, int... Indices>
Table<Prov> normalized_specialized(Table<Prov> table, const Prov &ctx, const Allocator &alloc, Product<T...>,
                       std::integer_sequence<int, Indices...>);
