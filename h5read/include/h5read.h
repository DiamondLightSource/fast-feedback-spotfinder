#ifndef _H5READ_H
#define _H5READ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <cassert>
#include <mutex>
extern "C" {
#endif

typedef struct _h5read_handle h5read_handle;

// Define a data type alias so that other users don't have to hardcode
typedef uint16_t image_t_type;

typedef struct image_t {
    uint16_t *data;
    uint8_t *mask;
    size_t slow;
    size_t fast;
} image_t;

/* data as modules i.e. 3D array */
typedef struct image_modules_t {
    uint16_t *data;  ///< Module image data in a 3D array block of [module][slow][fast]
    uint8_t *mask;   ///< Image mask, in the same shape as the module data
    size_t modules;  ///< Total number of modules
    size_t slow;     ///< Number of pixels in slow direction per module
    size_t fast;     ///< Number of pixels in fast direction per module
} image_modules_t;

/// Read an h5 file. Returns NULL if failed.
h5read_handle *h5read_open(const char *master_filename);

/// Generate sample data
h5read_handle *h5read_generate_samples();

/// Cleanup and release an h5 file object
void h5read_free(h5read_handle *);

/// Get the number of images in a dataset
size_t h5read_get_number_of_images(h5read_handle *obj);
/// Get the number of image pixels in the slow dimension
size_t h5read_get_image_slow(h5read_handle *obj);
/// Get the number of image pixels in the fast dimension
size_t h5read_get_image_fast(h5read_handle *obj);
/// Get the trusted range for this dataset
void h5read_get_trusted_range(h5read_handle *obj, image_t_type *min, image_t_type *max);
/// Get the wavelength for this dataset
float h5read_get_wavelength(h5read_handle *obj);
/// Get the pixel size for this dataset
float h5read_get_pixel_size_slow(h5read_handle *obj);
float h5read_get_pixel_size_fast(h5read_handle *obj);
/// Get the oscillation for this dataset
float h5read_get_oscillation_start(h5read_handle *obj);
float h5read_get_oscillation_width(h5read_handle *obj);
float h5read_get_detector_distance(h5read_handle *obj);
float h5read_get_beam_center_x(h5read_handle *obj);
float h5read_get_beam_center_y(h5read_handle *obj);

/** Borrow a pointer to the image mask.
 *
 * This must not be released by the caller, and must not be used beyond
 * the point that h5read_free is called. */
uint8_t *h5read_get_mask(h5read_handle *obj);

/// Read an image from a dataset
image_t *h5read_get_image(h5read_handle *obj, size_t number);
/// Free a previously read image
void h5read_free_image(image_t *image);

/** Read an image from a dataset into a preallocated buffer.
 *
 * The caller is responsible for both allocating and releasing the image
 * data buffer. This buffer *must* be at least large enough to hold an
 * image of slow*fast, or else undefined memory could be overwritten.
 */
void h5read_get_image_into(h5read_handle *obj, size_t index, image_t_type *data);

void h5read_get_raw_chunk(h5read_handle *obj,
                          size_t index,
                          size_t *size,
                          uint8_t *data,
                          size_t max_size);

size_t h5read_get_chunk_size(h5read_handle *obj, size_t index);

/// Read an image from a dataset, split up into modules
image_modules_t *h5read_get_image_modules(h5read_handle *obj, size_t frame_number);
/// Free an image read as modules
void h5read_free_image_modules(image_modules_t *modules);

/// Parse basic command arguments with verbose, filename in form:
///     Usage: <prog> [-h|--help] [-v] [FILE.nxs]
h5read_handle *h5read_parse_standard_args(int argc, char **argv);

#ifdef __cplusplus
}

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

class Image {
  private:
    std::shared_ptr<h5read_handle> _handle;
    std::shared_ptr<image_t> _image;

  public:
    Image(std::shared_ptr<h5read_handle> reader, size_t i) noexcept;

    const std::span<image_t_type> data;
    const std::span<uint8_t> mask;
    const size_t slow;
    const size_t fast;
};

class ImageModules {
  private:
    std::shared_ptr<h5read_handle> _handle;
    std::shared_ptr<image_modules_t> _modules;
    // Mutable vectors to point to the appropriate parts of the module data
    std::vector<std::span<image_t_type>> _modules_data;
    std::vector<std::span<uint8_t>> _modules_masks;

  public:
    ImageModules(std::shared_ptr<h5read_handle> handle, size_t i) noexcept;

    const std::span<image_t_type> data;
    const std::span<uint8_t> mask;

    const size_t n_modules;  ///< Number of modules
    const size_t slow;       ///< Height of a module, in pixels
    const size_t fast;       ///< Width of a module, in pixels
    /// Convenience lookup to get data for a particular module
    const std::span<std::span<image_t_type>> modules;
    /// Convenience lookup to get mask for a particular module
    const std::span<std::span<uint8_t>> masks;
};

/// Base class object to provide a unified reader interface
class Reader {
  public:
    enum ChunkCompression {
        BITSHUFFLE_LZ4,
        BYTE_OFFSET_32,
    };

    virtual ~Reader() {};

    virtual bool is_image_available(size_t index) = 0;

    virtual std::span<uint8_t> get_raw_chunk(size_t index,
                                             std::span<uint8_t> destination) = 0;
    virtual ChunkCompression get_raw_chunk_compression() = 0;
    virtual size_t get_number_of_images() const = 0;
    virtual std::array<image_t_type, 2> get_trusted_range() const = 0;
    virtual std::array<size_t, 2> image_shape() const = 0;
    virtual std::optional<std::span<const uint8_t>> get_mask() const = 0;
    virtual std::optional<float> get_wavelength() const = 0;
    virtual std::optional<std::array<float, 2>> get_pixel_size()
      const = 0;  ///< Pixel size (y, x), in meters
    virtual std::optional<std::array<float, 2>> get_beam_center()
      const = 0;  ///< Beam center (y, x), in pixels
    virtual std::optional<float> get_detector_distance()
      const = 0;  ///< Distance to detector, in meters.
    virtual std::array<float, 2> get_oscillation()
      const = 0;  ///< Oscillation (start, width), in degrees
};

// Declare a C++ "object" version so we don't have to keep track of allocations
class H5Read : public Reader {
  public:
    using image_type = image_t_type;

    /// Create a reader using generated sample data
    H5Read();
    /// Create a reader from a Nexus file
    H5Read(const std::string &filename);
    /// Create a reader by parsing command arguments
    H5Read(int argc, char **argv) noexcept;

    /// Read image data into an existing buffer
    void get_image_into(size_t index, uint16_t *data) {
        h5read_get_image_into(_handle.get(), index, data);
    }
    /// Read image data into an existing buffer
    void get_image_into(size_t index, std::span<uint16_t> data) {
#ifndef NDEBUG
        assert(data.size() >= get_image_slow() * get_image_fast());
#endif
        h5read_get_image_into(_handle.get(), index, data.data());
    }

    /// See if an image is available for raw chunk read
    bool is_image_available(size_t index) {
        return h5read_get_chunk_size(_handle.get(), index) > 0;
    }

    std::span<uint8_t> get_raw_chunk(size_t index, std::span<uint8_t> destination) {
        size_t chunk_bytes;
        h5read_get_raw_chunk(_handle.get(),
                             index,
                             &chunk_bytes,
                             destination.data(),
                             destination.size_bytes());
        return {destination.data(), chunk_bytes};
    }

    virtual auto get_raw_chunk_compression() -> ChunkCompression {
        return Reader::ChunkCompression::BITSHUFFLE_LZ4;
    }

    Image get_image(size_t index) {
        return Image(_handle, index);
    }

    virtual std::optional<std::span<const uint8_t>> get_mask() const {
        auto mask = h5read_get_mask(_handle.get());
        if (mask == nullptr) {
            return std::nullopt;
        }
        return {{h5read_get_mask(_handle.get()), get_image_slow() * get_image_fast()}};
    }

    ImageModules get_image_modules(size_t index) {
        return ImageModules(_handle, index);
    }

    /// Get the total number of image frames
    virtual size_t get_number_of_images() const {
        return h5read_get_number_of_images(_handle.get());
    }
    /// Get the number of pixels in the slow dimension
    size_t get_image_slow() const {
        return h5read_get_image_slow(_handle.get());
    }
    /// Get the number of pixels in the fast dimension
    size_t get_image_fast() const {
        return h5read_get_image_fast(_handle.get());
    }
    /// Get the image shape, in (slow, fast) pixels
    virtual std::array<size_t, 2> image_shape() const {
        return {get_image_slow(), get_image_fast()};
    }
    /// Get the (min, max) inclusive trusted range of pixel values
    virtual std::array<image_t_type, 2> get_trusted_range() const {
        image_t_type min, max;
        h5read_get_trusted_range(_handle.get(), &min, &max);
        return {min, max};
    }
    /// Get the wavelength of the X-ray beam
    virtual std::optional<float> get_wavelength() const {
        float wavelength = h5read_get_wavelength(_handle.get());
        if (wavelength == -1) {  // No wavelength provided
            return std::nullopt;
        } else {
            std::optional<float> result = wavelength;
            return result;
        }
    }
    /// Get the oscillation range, in (start, width) degrees
    virtual std::array<float, 2> get_oscillation() const {
        float start = h5read_get_oscillation_start(_handle.get());
        float width = h5read_get_oscillation_width(_handle.get());

        return {{start, width}};
    }

    virtual std::optional<std::array<float, 2>> get_pixel_size() const {
        return {{h5read_get_pixel_size_slow(_handle.get()),
                 h5read_get_pixel_size_fast(_handle.get())}};
    }

    virtual std::optional<std::array<float, 2>> get_beam_center() const {
        return {{h5read_get_beam_center_y(_handle.get()),
                 h5read_get_beam_center_x(_handle.get())}};
    }
    virtual std::optional<float> get_detector_distance() const {
        return {h5read_get_detector_distance(_handle.get())};
    }
    std::mutex mutex;

  protected:
    std::shared_ptr<h5read_handle> _handle;
};

using pixel_t = H5Read::image_type;

template <typename T>
bool is_ready_for_read(const std::string &path);

template <>
inline bool is_ready_for_read<H5Read>(const std::string &path) {
    return true;
}
#endif

#endif
