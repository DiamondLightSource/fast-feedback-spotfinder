#pragma once

#include "h5read.h"

__global__ void dispersion(pixel_t __restrict__ *image_ptr,
                           uint8_t __restrict__ *mask_ptr,
                           uint8_t __restrict__ *result_mask_ptr);

__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image_ptr,
                                               uint8_t __restrict__ *mask_ptr,
                                               uint8_t __restrict__ *result_mask_ptr);

__global__ void dispersion_extended_second_pass(
  pixel_t __restrict__ *image_ptr,
  uint8_t __restrict__ *mask_ptr,
  uint8_t __restrict__ *dispersion_mask_ptr,
  uint8_t __restrict__ *result_mask_ptr,
  size_t dispersion_mask_pitch);
