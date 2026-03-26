#pragma once
#include "../device_vec.h"
#include "../table.h"

template <typename Prov>
Array<device_buffer> gather_join_indices(
    const Table<Prov> &left, const Table<Prov> &right,
    device_vec<uint32_t> &left_indices, device_vec<uint32_t> &right_indices);
