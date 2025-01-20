/**
 * @file device_common.hu
 * @brief Common device functions
 */

#ifndef DEVICE_COMMON_H
#define DEVICE_COMMON_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <cuda/std/tuple>
#include <type_traits>

/* 
 * Kernel radii 🔳
 *
 * Kernel, here, referring to a sliding window of pixels. Such as in
 * convolution or erosion. This is not to be confused with the CUDA
 * kernel, which is a function that runs on the GPU.
 * 
 * One-direction width of kernel. Total kernel span is (R * 2 + 1)
 * The kernel is a square, so the height is the same as the width.
*/
constexpr uint8_t KERNEL_RADIUS = 3;           // 7x7 kernel
constexpr uint8_t KERNEL_RADIUS_EXTENDED = 5;  // 11x11 kernel

/**
 * @brief Struct to act as a global container for constant values
 * necessary for spotfinding
 * 
 * @note This struct is intended to be copied to the device's
 * constant memory before any kernel calls
 */
struct KernelConstants {
    size_t image_pitch;           // Pitch of the image
    size_t mask_pitch;            // Pitch of the mask
    size_t result_pitch;          // Pitch of the result
    ushort width;                 // Width of the image
    ushort height;                // Height of the image
    float max_valid_pixel_value;  // Maximum valid pixel value
    uint8_t min_count;            // Minimum number of pixels in a spot
    float n_sig_b;                // Number of standard deviations for background
    float n_sig_s;                // Number of standard deviations for signal
};

/*
 * Constants for kernels
 * extern keyword is used to declare a variable that is defined in
 * another file. This links the constant global variable to the
 * kernel_constants variable defined in `spotfinder.cu`
 */
extern __constant__ KernelConstants kernel_constants;

/**
 * @brief Struct to represent a 2D pitched array on the device and provide
 * a convenient way to access pitched elements
 * 
 * @tparam T Type of the elements in the array
 * @param array Pointer to the array
 * @param pitch Pointer to the pitch of the array
 */
template <typename T>
struct PitchedArray2D {
    T *array;
    const size_t *pitch;

    /**
     * @brief Construct a new PitchedArray2D object
     */
    __device__ PitchedArray2D(T *array, const size_t *pitch)
        : array(array), pitch(pitch) {}

    /**
     * @brief Access the element at the given coordinates
     */
    __device__ T operator()(uint x, uint y) const {
        return array[(y * (*pitch)) + x];
    }

    /**
     * @brief Access the element at the given coordinates
     */
    __device__ T &operator()(uint x, uint y) {
        return array[(y * (*pitch)) + x];
    }

    /**
     * @brief Get the pitch value
     */
    __device__ size_t get_pitch() const {
        // Dereference the pointer to get the value
        return *pitch;
    }
};

/*
 * Type validation helpers for load_halo
 *
 * These helpers ensure that only valid types are passed to the variadic
 * parameter pack in `load_halo`. A valid type is defined as 
 * `cuda::std::tuple<PitchedArray2D<T>, PitchedArray2D<T>>`.
 *
 * `is_valid_pitched_array_tuple` checks a single type for validity.
 * `are_valid_pitched_array_tuples` recursively validates all types 
 * in a parameter pack.
 *
 * These checks enforce compile-time correctness and prevent runtime 
 * errors caused by invalid types.
 * 
 * The helpers use a technique called specialization to provide a custom 
 * implementation for specific cases. The default implementation of 
 * `is_valid_pitched_array_tuple` returns `false` for all types, but a 
 * specialized version explicitly recognizes and validates 
 * `cuda::std::tuple<PitchedArray2D<T>, PitchedArray2D<T>>`, returning `true`.
 * This ensures only the intended types are considered valid.
 */

// Default case: a type is not a valid tuple
template <typename T>
struct is_valid_pitched_array_tuple : std::false_type {};

// Specialization for a valid tuple
// Checks if a type is a `cuda::std::tuple` of two `PitchedArray2D` objects.
template <typename T1, typename T2>
struct is_valid_pitched_array_tuple<
  cuda::std::tuple<PitchedArray2D<T1>, PitchedArray2D<T2>>> : std::true_type {};

// Validate all types in a variadic parameter pack
template <typename... Args>
struct are_valid_pitched_array_tuples;

// Base case for parameter pack validation: an empty pack is always valid.
template <>
struct are_valid_pitched_array_tuples<> : std::true_type {};

// Recursive case: checks the first type and continues with the rest.
// If any type is invalid, the entire pack is considered invalid.
template <typename First, typename... Rest>
struct are_valid_pitched_array_tuples<First, Rest...>
    : std::conditional_t<is_valid_pitched_array_tuple<First>::value,
                         are_valid_pitched_array_tuples<Rest...>,
                         std::false_type> {};
/**
 * @brief Load the halo region of an image and mask into shared memory.
 * 
 * This function loads the surrounding pixels (halo) of the current
 * block into shared memory. It ensures that the halo region is correctly
 * loaded for boundary conditions.
 * 
 * @tparam MappedPairs Variadic template parameter pack representing
 * tuples of an image source and its corresponding shared memory
 * destination (e.g., std::tuple<src, dst>). This allows arbitrary
 * pairs to be passed and processed.
 * @param block The cooperative groups thread block.
 * @param x The x-coordinate of the current thread in the global memory.
 * @param y The y-coordinate of the current thread in the global memory.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param kernel_width The width of the kernel (halo region).
 * @param kernel_height The height of the kernel (halo region).
 * @param mapped_pairs Variadic parameter pack of tuples, each
 * containing a source image and a corresponding shared memory array
 * as PitchedArray2D.
 */
template <typename... MappedPairs>
__device__ void load_halo(const cooperative_groups::thread_block block,
                          const int x,
                          const int y,
                          const ushort width,
                          const ushort height,
                          const uint8_t kernel_width,
                          const uint8_t kernel_height,
                          MappedPairs... mapped_pairs) {
    // Validate the types in the parameter pack
    static_assert(are_valid_pitched_array_tuples<MappedPairs...>::value,
                  "All mapped_pairs must be cuda::std::tuple<PitchedArray2D<T>, "
                  "PitchedArray2D<T>>");

    // Compute local shared memory coordinates
    int local_x = threadIdx.x + kernel_width;
    int local_y = threadIdx.y + kernel_height;

    /*
     * A lambda function is used here to load a single pixel from the
     * global memory (src) into the shared memory (dst). The lambda
     * accepts a mapped pair (src, dst) and offsets (offset_x, offset_y),
     * and uses structured binding to extract the source and
     * destination objects from the tuple.
     */
    auto load_pixel = [&](auto &mapped_pair, int offset_x, int offset_y) {
        auto &[src, dst] = mapped_pair;  // Destucture the mapped pair tuple
        dst(local_x + offset_x, local_y + offset_y) = src(x + offset_x, y + offset_y);
    };

    // Precompute boundary checks to avoid repeated evaluation

    bool load_left = threadIdx.x < kernel_width;
    bool left_valid = x >= kernel_width;

    bool load_right = threadIdx.x >= blockDim.x - kernel_width;
    bool right_valid = x + kernel_width < width;

    bool load_top = threadIdx.y < kernel_height;
    bool top_valid = y >= kernel_height;

    bool load_bottom = threadIdx.y >= blockDim.y - kernel_height;
    bool bottom_valid = y + kernel_height < height;

    /*
     * These are C++ fold expressions, which applies the lambda function
     * (load_pixel) to each tuple in the variadic parameter pack
     * (mapped_pairs...). By using this approach, we can load pixels
     * for multiple source/destination pairs in a single operation,
     * without needing explicit loops for each pair.
     */

    // Load vertically and horizontally adjacent pixels
    if (load_left && left_valid) {
        // Load left halo
        (load_pixel(mapped_pairs, -kernel_width, 0), ...);
    }
    if (load_right && right_valid) {
        // Load right halo
        (load_pixel(mapped_pairs, kernel_width, 0), ...);
    }
    if (load_top && top_valid) {
        // Load top halo
        (load_pixel(mapped_pairs, 0, -kernel_height), ...);
    }
    if (load_bottom && bottom_valid) {
        // Load bottom halo
        (load_pixel(mapped_pairs, 0, kernel_height), ...);
    }

    // Load diagonally adjacent pixels
    if (load_left && load_top && left_valid && top_valid) {
        // Load top-left corner
        (load_pixel(mapped_pairs, -kernel_width, -kernel_height), ...);
    }
    if (load_right && load_top && right_valid && top_valid) {
        // Load top-right corner
        (load_pixel(mapped_pairs, kernel_width, -kernel_height), ...);
    }
    if (load_left && load_bottom && left_valid && bottom_valid) {
        // Load bottom-left corner
        (load_pixel(mapped_pairs, -kernel_width, kernel_height), ...);
    }
    if (load_right && load_bottom && right_valid && bottom_valid) {
        // Load bottom-right corner
        (load_pixel(mapped_pairs, kernel_width, kernel_height), ...);
    }
}

#endif  // DEVICE_COMMON_H