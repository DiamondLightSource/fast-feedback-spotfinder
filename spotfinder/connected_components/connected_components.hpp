#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <builtin_types.h>

#include <cstdint>
#include <vector>

#include "h5read.h"

struct Reflection {
    int l, t, r, b;  // Bounding box: left, top, right, bottom
    int num_pixels = 0;
};

void connected_components_2d(const std::vector<int2> px_coords,
                             const std::vector<size_t> px_kvals,
                             const std::vector<pixel_t> px_values,
                             const ushort width,
                             const ushort height,
                             const uint32_t min_spot_size,
                             std::vector<Reflection> *boxes_ptr);

#endif  // CONNECTED_COMPONENTS_HPP