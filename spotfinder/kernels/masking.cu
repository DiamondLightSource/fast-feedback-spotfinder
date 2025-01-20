/**
 * @file masking.cu
 * @brief Contains CUDA device functions and kernel implementations for
 *        applying a resolution mask to an image based on pixel 
 *        distances and beam properties.
 *
 * This file modifies a mask based on the resolution of each pixel in an
 * image. The resolution is calculated based on the distance of the
 * pixel from the beam center and the detector's properties. Pixels
 * with resolutions outside a specified range are set as masked.
 * 
 * The `apply_resolution_mask` kernel is complemented by a host function 
 * `call_apply_resolution_mask`, which prepares the CUDA execution
 * configuration and launches the kernel.
 * 
 * @note The distance from center calculation is done with the
 *       assumption that the detector is perpendicular to the beam.
 */

#include "device_common.cuh"
#include "masking.cuh"

#define VALID_PIXEL 1
#define MASKED_PIXEL 0

#pragma region Device functions
/**
 * @brief Function to calculate the distance of a pixel from the beam center.
 * @param x The x-coordinate of the pixel in the image
 * @param y The y-coordinate of the pixel in the image
 * @param center_x The x-coordinate of the pixel beam center in the image
 * @param center_y The y-coordinate of the pixel beam center in the image
 * @param pixel_size_x The pixel size of the detector in the x-direction in m
 * @param pixel_size_y The pixel size of the detector in the y-direction in m
 * @return The calculated distance from the beam center in m
*/
__device__ float get_distance_from_center(float x,
                                          float y,
                                          float center_x,
                                          float center_y,
                                          float pixel_size_x,
                                          float pixel_size_y) {
    /*
     * Since this calculation is for a broad, general exclusion, we can
     * use basic Pythagoras to calculate the distance from the center.
     * 
     * We add 0.5 in order to move to the center of the pixel. Since
     * starting at 0, 0, for instance, would infact be the corner.
    */
    float dx = ((x + 0.5f) - center_x) * pixel_size_x;
    float dy = ((y + 0.5f) - center_y) * pixel_size_y;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Function to calculate the interplanar distance of a reflection.
 * The interplanar distance is calculated using the formula:
 *         d = λ / (2 * sin(ϴ))
 * @param wavelength The wavelength of the X-ray beam in Å
 * @param distance_to_detector The distance from the sample to the detector in m
 * @param distance_from_center The distance of the reflection from the beam center in m
 * @return The calculated d value
*/
__device__ float get_resolution(float wavelength,
                                float distance_to_detector,
                                float distance_from_centre) {
    /*
     * Since the angle calculated is, in fact, 2ϴ, we halve to get the
     * proper value of ϴ
    */
    float theta = 0.5 * atanf(distance_from_centre / distance_to_detector);
    return wavelength / (2 * sinf(theta));
}
#pragma endregion Res Mask Functions

#pragma region Mask kernel
/**
 * @brief CUDA kernel to apply a resolution mask for an image.
 *
 * This kernel calculates the resolution for each pixel in an image based on the
 * distance from the beam center and the detector properties. It then masks out
 * pixels whose resolution falls outside the specified range [dmin, dmax],
 * provided that the pixel is not already masked, by setting the mask value of
 * the pixel to 0 in the mask data.
 *
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param wavelength The wavelength of the X-ray beam in Ångströms.
 * @param distance_to_detector The distance from the sample to the detector in m.
 * @param beam_center_x The x-coordinate of the beam center in the image.
 * @param beam_center_y The y-coordinate of the beam center in the image.
 * @param pixel_size_x The pixel size of the detector in the x-direction in m.
 * @param pixel_size_y The pixel size of the detector in the y-direction in m.
 * @param dmin The minimum resolution (d-spacing) threshold.
 * @param dmax The maximum resolution (d-spacing) threshold.
 */
__global__ void apply_resolution_mask(uint8_t *mask_ptr,
                                      size_t mask_pitch,
                                      int width,
                                      int height,
                                      float wavelength,
                                      float distance_to_detector,
                                      float beam_center_x,
                                      float beam_center_y,
                                      float pixel_size_x,
                                      float pixel_size_y,
                                      float dmin,
                                      float dmax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height) return;  // Out of bounds

    // Create pitched array for data access
    PitchedArray2D<uint8_t> mask(mask_ptr, &mask_pitch);

    if (mask(x, y) == MASKED_PIXEL) {  // Check if the pixel is masked
        /*
        * If the pixel is already masked, we don't need to calculate the
        * resolution for it, so we can just leave it masked
        */
        return;
    }

    float distance_from_centre = get_distance_from_center(
      x, y, beam_center_x, beam_center_y, pixel_size_x, pixel_size_y);
    float resolution =
      get_resolution(wavelength, distance_to_detector, distance_from_centre);

    // Check if dmin is set and if the resolution is below it
    if (dmin > 0 && resolution < dmin) {
        mask(x, y) = MASKED_PIXEL;
        return;
    }

    // Check if dmax is set and if the resolution is above it
    if (dmax > 0 && resolution > dmax) {
        mask(x, y) = MASKED_PIXEL;
        return;
    }

    // If the pixel is not masked and the resolution is within the limits, set the resolution mask to 1
    mask(x, y) = VALID_PIXEL;
    // ⛔🧊
}
#pragma endregion Mask Kernel

#pragma region Launch wrapper
/**
 * @brief Host function to launch the apply_resolution_mask kernel.
 *
 * This function sets up the kernel execution parameters and launches the
 * apply_resolution_mask kernel to generate and apply a resolution mask
 * onto the base mask for the detector.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param params The parameters required to calculate the resolution mask.  
 */
void call_apply_resolution_mask(dim3 blocks,
                                dim3 threads,
                                size_t shared_memory,
                                cudaStream_t stream,
                                uint8_t *mask,
                                ResolutionMaskParams params) {
    // Launch the kernel
    apply_resolution_mask<<<blocks, threads, shared_memory, stream>>>(
      mask,
      params.mask_pitch,
      params.width,
      params.height,
      params.wavelength,
      params.detector.distance,
      params.detector.beam_center_x,
      params.detector.beam_center_y,
      params.detector.pixel_size_x,
      params.detector.pixel_size_y,
      params.dmin,
      params.dmax);
}
#pragma endregion Launch wrapper