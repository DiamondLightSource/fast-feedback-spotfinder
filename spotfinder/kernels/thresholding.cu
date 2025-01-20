/**
 * @file thresholding.cu
 * @brief Contains CUDA kernel implementations for thresholding image 
 *        data.
 *
 * This file contains CUDA kernels for thresholding image data based on
 * the variance and mean of the local neighbourhood of each pixel. These
 * thresholds are used to identify potential signal spots against a
 * background in the input image based on local variance, mean, and
 * user-defined significance levels.
 * 
 * @note The __restrict__ keyword is used to indicate to the compiler
 *       that the two pointers are not aliased, allowing the compiler to
 *       perform more aggressive optimizations.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda/std/tuple>

#include "cuda_common.hpp"
#include "device_common.cuh"
#include "thresholding.cuh"

namespace cg = cooperative_groups;

#pragma region Global constants
/*
 * Constants for kernels
 * extern keyword is used to declare a variable that is defined in
 * another file. This links the constant global variable to the
 * kernel_constants variable defined in spotfinder.cu
 */
extern __constant__ KernelConstants kernel_constants;
#pragma endregion Global constants

#pragma region Device Functions
/**
 * @brief Calculate the dispersion flags for a given pixel.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param this_pixel The pixel value at the current position.
 * @param x The x-coordinate of the pixel.
 * @param y The y-coordinate of the pixel.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 * @return A tuple <bool not_background, bool is_signal, uint8_t n> where:
 *        - not_background: True if the pixel is not part of the background.
 *        - is_signal: True if the pixel is a strong signal.
 *        - n: The number of valid pixels in the local neighbourhood.
 */
__device__ cuda::std::tuple<bool, bool, uint8_t> calculate_dispersion_flags(
  PitchedArray2D<pixel_t> image,
  PitchedArray2D<uint8_t> mask,
  pixel_t this_pixel,
  int local_x,
  int local_y,
  ushort width,
  ushort height,
  uint8_t kernel_radius,
  uint8_t min_count,
  float n_sig_b,
  float n_sig_s) {
    // Initialize variables for computing the local sum and count
    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

#pragma unroll
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
#pragma unroll
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            // Calculate the local coordinates
            int lx = local_x + j;
            int ly = local_y + i;
            uint8_t mask_pixel = mask(lx, ly);
            if (mask_pixel) {
                pixel_t pixel = image(lx, ly);
                sum += pixel;
                sumsq += pixel * pixel;
                ++n;
            }
        }
    }

    // Check if the pixel has enough valid neighbours
    if (n < min_count) {
        return {false, false, n};
    }

    // Compute local mean and variance
    float sum_f = static_cast<float>(sum);
    float sumsq_f = static_cast<float>(sumsq);

    float mean = sum_f / n;
    float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
    float dispersion = variance / mean;

    // Compute the background threshold and signal threshold
    float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
    bool not_background = dispersion > background_threshold;
    float signal_threshold = mean + n_sig_s * sqrt(mean);

    // Check if the pixel is a strong pixel
    bool is_signal = this_pixel > signal_threshold;

    return {not_background, is_signal, n};
}
#pragma endregion

#pragma region Thresholding kernels
/**
 * @brief Kernel for computing the basic threshold based on variance and mean.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param threshold The global threshold for intensity.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void dispersion(pixel_t __restrict__ *image_ptr,
                           uint8_t __restrict__ *mask_ptr,
                           uint8_t __restrict__ *result_mask_ptr) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.mask_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int global_x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int global_y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (global_x >= kernel_constants.width || global_y >= kernel_constants.height)
        return;  // Out of bounds guard

    // Allocate shared memory for the image data
    extern __shared__ uint8_t shared_mem[];
    // Partition shared memory for image and mask data
    uint8_t *shared_mask_ptr = shared_mem;
    size_t shared_partition_size =
      (blockDim.x + KERNEL_RADIUS * 2) * (blockDim.y + KERNEL_RADIUS * 2);
    pixel_t *shared_image_ptr =
      reinterpret_cast<pixel_t *>(&shared_mask_ptr[shared_partition_size]);

    // Create pitched arrays for shared memory access
    size_t shared_pitch = blockDim.x + KERNEL_RADIUS * 2;
    PitchedArray2D<pixel_t> shared_image(shared_image_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_mask(shared_mask_ptr, &shared_pitch);

    int local_x = threadIdx.x + KERNEL_RADIUS;
    int local_y = threadIdx.y + KERNEL_RADIUS;

    // Get the pixel value at the current position
    pixel_t this_pixel = image(global_x, global_y);

    // Load this pixel into shared memory
    shared_image(local_x, local_y) = this_pixel;
    shared_mask(local_x, local_y) = mask(global_x, global_y);

    // Load the surrounding pixels into shared memory
    load_halo(block,
              global_x,
              global_y,
              kernel_constants.width,
              kernel_constants.height,
              KERNEL_RADIUS,
              KERNEL_RADIUS,
              cuda::std::make_tuple(image, shared_image),
              cuda::std::make_tuple(mask, shared_mask));

    // Sync threads to ensure all shared memory is loaded
    block.sync();

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid = shared_mask(local_x, local_y) != 0
                       && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Validity guard
    if (!px_is_valid) {
        result_mask(global_x, global_y) = 0;
        return;
    }

    // Calculate the dispersion flags
    auto [not_background, is_signal, n] =
      calculate_dispersion_flags(shared_image,
                                 shared_mask,
                                 this_pixel,
                                 local_x,
                                 local_y,
                                 kernel_constants.width,
                                 kernel_constants.height,
                                 KERNEL_RADIUS,
                                 kernel_constants.min_count,
                                 kernel_constants.n_sig_b,
                                 kernel_constants.n_sig_s);

    result_mask(global_x, global_y) = not_background && is_signal && n > 1;
}

/**
 * @brief Kernel for computing the dispersion threshold.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image_ptr,
                                               uint8_t __restrict__ *mask_ptr,
                                               uint8_t __restrict__ *result_mask_ptr) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.mask_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int global_x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int global_y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (global_x >= kernel_constants.width || global_y >= kernel_constants.height)
        return;  // Out of bounds guard

    // Allocate shared memory for the image data
    extern __shared__ uint8_t shared_mem[];
    // Partition shared memory for image and mask data
    uint8_t *shared_mask_ptr = shared_mem;
    size_t shared_partition_size =
      (blockDim.x + KERNEL_RADIUS * 2) * (blockDim.y + KERNEL_RADIUS * 2);
    pixel_t *shared_image_ptr =
      reinterpret_cast<pixel_t *>(&shared_mask_ptr[shared_partition_size]);

    // Create pitched arrays for shared memory access
    size_t shared_pitch = blockDim.x + KERNEL_RADIUS * 2;
    PitchedArray2D<pixel_t> shared_image(shared_image_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_mask(shared_mask_ptr, &shared_pitch);

    int local_x = threadIdx.x + KERNEL_RADIUS;
    int local_y = threadIdx.y + KERNEL_RADIUS;

    // Get the pixel value at the current position
    pixel_t this_pixel = image(global_x, global_y);

    // Load this pixel into shared memory
    shared_image(local_x, local_y) = this_pixel;
    shared_mask(local_x, local_y) = mask(global_x, global_y);

    // Load the surrounding pixels into shared memory
    load_halo(block,
              global_x,
              global_y,
              kernel_constants.width,
              kernel_constants.height,
              KERNEL_RADIUS,
              KERNEL_RADIUS,
              cuda::std::make_tuple(image, shared_image),
              cuda::std::make_tuple(mask, shared_mask));

    // Sync threads to ensure all shared memory is loaded
    block.sync();

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid = shared_mask(local_x, local_y) != 0
                       && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Validity guard
    if (!px_is_valid) {
        result_mask(global_x, global_y) = 0;
        return;
    }

    // Calculate the dispersion flags
    auto [not_background, is_signal, n] =
      calculate_dispersion_flags(shared_image,
                                 shared_mask,
                                 this_pixel,
                                 local_x,
                                 local_y,
                                 kernel_constants.width,
                                 kernel_constants.height,
                                 KERNEL_RADIUS,
                                 kernel_constants.min_count,
                                 kernel_constants.n_sig_b,
                                 kernel_constants.n_sig_s);

    result_mask(global_x, global_y) = not_background && n > 1;
}

/**
 * @brief Kernel for computing the final threshold after dispersion mask.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param dispersion_mask Pointer to the dispersion mask used for extended algorithm.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param dispersion_mask_pitch The pitch of the dispersion mask data.
 * @param result_mask_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param n_sig_s Signal significance level.
 * @param threshold Global threshold for the intensity.
 */
__global__ void dispersion_extended_second_pass(
  pixel_t __restrict__ *image_ptr,
  uint8_t __restrict__ *mask_ptr,
  uint8_t __restrict__ *dispersion_mask_ptr,
  uint8_t __restrict__ *result_mask_ptr,
  size_t dispersion_mask_pitch) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.result_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> dispersion_mask(dispersion_mask_ptr,
                                            &dispersion_mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int global_x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int global_y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (global_x >= kernel_constants.width || global_y >= kernel_constants.height)
        return;  // Out of bounds guard

    // Allocate shared memory for the image data
    extern __shared__ uint8_t shared_mem[];
    // Partition shared memory for image, mask and dispersion mask data
    uint8_t *shared_mask_ptr = shared_mem;
    size_t shared_partition_size = (blockDim.x + KERNEL_RADIUS_EXTENDED * 2)
                                   * (blockDim.y + KERNEL_RADIUS_EXTENDED * 2);
    uint8_t *shared_dispersion_mask_ptr = &shared_mask_ptr[shared_partition_size];
    pixel_t *shared_image_ptr =
      reinterpret_cast<pixel_t *>(&shared_dispersion_mask_ptr[shared_partition_size]);

    // Create pitched arrays for shared memory access
    size_t shared_pitch = blockDim.x + KERNEL_RADIUS_EXTENDED * 2;
    PitchedArray2D<pixel_t> shared_image(shared_image_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_mask(shared_mask_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_dispersion_mask(shared_dispersion_mask_ptr,
                                                   &shared_pitch);

    int local_x = threadIdx.x + KERNEL_RADIUS_EXTENDED;
    int local_y = threadIdx.y + KERNEL_RADIUS_EXTENDED;

    pixel_t this_pixel = image(global_x, global_y);

    // Load this pixel into shared memory
    shared_image(local_x, local_y) = this_pixel;
    shared_mask(local_x, local_y) = mask(global_x, global_y);
    shared_dispersion_mask(local_x, local_y) = dispersion_mask(global_x, global_y);

    // Load the surrounding pixels into shared memory
    load_halo(block,
              global_x,
              global_y,
              kernel_constants.width,
              kernel_constants.height,
              KERNEL_RADIUS_EXTENDED,
              KERNEL_RADIUS_EXTENDED,
              cuda::std::make_tuple(image, shared_image),
              cuda::std::make_tuple(mask, shared_mask),
              cuda::std::make_tuple(dispersion_mask, shared_dispersion_mask));

    // Sync threads to ensure all shared memory is loaded
    block.sync();

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid = shared_mask(local_x, local_y) != 0
                       && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint sum = 0;
    uint8_t n = 0;

#pragma unroll
    for (int i = -KERNEL_RADIUS_EXTENDED; i <= KERNEL_RADIUS_EXTENDED; ++i) {
#pragma unroll
        for (int j = -KERNEL_RADIUS_EXTENDED; j <= KERNEL_RADIUS_EXTENDED; ++j) {
            // Calculate the local coordinates
            int lx = local_x + j;
            int ly = local_y + i;

            pixel_t pixel = shared_image(lx, ly);
            uint8_t mask_pixel = shared_mask(lx, ly);
            uint8_t disp_mask_pixel = shared_dispersion_mask(lx, ly);
            /*
               * Check if the pixel is valid. That means that it is not
               * masked and was not marked as potentially signal in the
               * dispersion mask.
              */
            bool include_pixel =
              mask_pixel != MASKED_PIXEL && disp_mask_pixel != MASKED_PIXEL;
            if (include_pixel) {
                sum += pixel;
                n += 1;
            }
        }
    }

    constexpr float threshold = 0.0f;  // DIALS default value

    // Calculate the thresholding
    if (px_is_valid && n > 0) {
        float sum_f = static_cast<float>(sum);

        // The pixel must have been marked as potentially signal in the dispersion mask
        bool disp_mask = shared_dispersion_mask(local_x, local_y) == MASKED_PIXEL;
        // The pixel must be above the global threshold
        bool global_mask = shared_image(local_x, local_y) > threshold;
        // Calculate the local mean
        float mean = (n > 1 ? sum_f / n : 0);  // If n is less than 1, set mean to 0
        // The pixel must be above the local threshold
        bool local_mask = shared_image(local_x, local_y)
                          >= (mean + kernel_constants.n_sig_s * sqrtf(mean));

        result_mask(global_x, global_y) = disp_mask && global_mask && local_mask;
    } else {
        result_mask(global_x, global_y) = 0;
    }
}
#pragma endregion