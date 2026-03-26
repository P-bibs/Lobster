#pragma once

#include "flame.h"
#include "table.h"

template <typename Prov>
Table<Prov> merge_tables(const Table<Prov> &left, const Table<Prov> &right,
                         const Prov &ctx, const Allocator &alloc);

template <typename Prov, typename... T, int... Index>
Table<Prov> merge_tables_specialized(const Table<Prov> &left,
                                     const Table<Prov> &right, const Prov &ctx,
                                     const Allocator &alloc, Product<T...>,
                                     std::integer_sequence<int, Index...>);
