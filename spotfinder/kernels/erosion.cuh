#pragma once

__global__ void erosion(uint8_t __restrict__ *dispersion_mask_ptr,
                        // uint8_t __restrict__ *mask,
                        size_t dispersion_mask_pitch,
                        uint8_t radius);