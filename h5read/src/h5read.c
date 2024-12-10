#include "h5read.h"

#include <assert.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef HAVE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#else
typedef uint8_t hid_t;
#endif

#include "eiger2xe.h"

// VDS stuff

#define MAXFILENAME 256
#define MAXDATAFILES 100
#define MAXDIM 3

typedef struct h5_data_file {
    char filename[MAXFILENAME];
    char dsetname[MAXFILENAME];
    hid_t file;
    hid_t dataset;
    size_t frames;
    size_t offset;
} h5_data_file;

struct _h5read_handle {
    hid_t master_file;
    int data_file_count;
    h5_data_file *data_files;
    size_t frames;  ///< Number of frames in this dataset
    size_t slow;    ///< Pixel dimension of images in the slow direction
    size_t fast;    ///< Pixel dimensions of images in the fast direction

    uint8_t *mask;         ///< Shared image mask
    uint8_t *module_mask;  ///< Shared module mask
    size_t mask_size;      ///< Total size(in pixels) of mask
    image_t_type trusted_range_min,
      trusted_range_max;  ///< Trusted range of this dataset
    float wavelength;     ///< Wavelength of the X-ray beam

    float pixel_size_x, pixel_size_y;
    float detector_distance;
    float beam_center_x, beam_center_y;
    float oscillation_start;
    float oscillation_width;
};

void h5read_free(h5read_handle *obj) {
#ifdef HAVE_HDF5
    for (int i = 0; i < obj->data_file_count; i++) {
        H5Dclose(obj->data_files[i].dataset);
        H5Fclose(obj->data_files[i].file);
    }
    if (obj->master_file) H5Fclose(obj->master_file);
#endif
    if (obj->data_files) free(obj->data_files);
    free(obj->mask);
    free(obj->module_mask);

    free(obj);
}

/// Get the number of frames available
size_t h5read_get_number_of_images(h5read_handle *obj) {
    return obj->frames;
}

size_t h5read_get_image_slow(h5read_handle *obj) {
    return obj->slow;
}

size_t h5read_get_image_fast(h5read_handle *obj) {
    return obj->fast;
}

void h5read_free_image(image_t *i) {
    free(i->data);
    // Mask is a pointer to the file-global file mask so isn't freed
    free(i);
}

uint8_t *h5read_get_mask(h5read_handle *obj) {
    return obj->mask;
}

/// blit the relevent pixel data across from a single image into a collection
/// of image modules - will allocate the latter
///
/// @param image    The image to blit from
/// @param modules  The modules object to fill
void _blit(image_t *image, image_modules_t *modules) {
    // Number of modules in fast, slow directions
    size_t fast, slow;
    if (image->slow == E2XE_16M_SLOW) {
        fast = 4;
        slow = 8;
    } else {
        fast = 2;
        slow = 4;
    }

    modules->slow = E2XE_MOD_SLOW;
    modules->fast = E2XE_MOD_FAST;
    modules->modules = slow * fast;

    size_t module_pixels = E2XE_MOD_SLOW * E2XE_MOD_FAST;

    modules->data = (uint16_t *)malloc(sizeof(uint16_t) * slow * fast * module_pixels);

    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image->fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                size_t offset =
                  (row0 + row * image->fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                size_t target =
                  (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&modules->data[target],
                       (void *)&image->data[offset],
                       sizeof(uint16_t) * E2XE_MOD_FAST);
            }
        }
    }
}

image_modules_t *h5read_get_image_modules(h5read_handle *obj, size_t n) {
    image_t *image = h5read_get_image(obj, n);
    image_modules_t *modules = malloc(sizeof(image_modules_t));
    modules->data = NULL;
    modules->mask = obj->module_mask;
    modules->modules = -1;
    modules->fast = -1;
    modules->slow = -1;
    _blit(image, modules);
    h5read_free_image(image);
    return modules;
}

void h5read_free_image_modules(image_modules_t *i) {
    free(i->data);
    // Like image, mask is held on the central h5read_handle object
    free(i);
}

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;
uint32_t pcg32_random_r(pcg32_random_t *rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

#define NUM_SAMPLE_IMAGES 6

/// Generate a sample image from number
void _generate_sample_image(h5read_handle *obj, size_t n, image_t_type *data) {
    assert(n >= 0 && n <= NUM_SAMPLE_IMAGES);

    if (n == 0) {
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
    } else if (n == 1) {
        // Image 1: I=1 for every unmasked pixel
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = 1;
                    }
                }
            }
        }
    } else if (n == 2) {
        // Image 2: High pixel (100) every 42 pixels across the detector
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int y = 0; y < E2XE_16M_SLOW; y += 42) {
            for (int x = 0; x < E2XE_16M_FAST; x += 42) {
                int k = y * E2XE_16M_FAST + x;
                data[k] = 100;
            }
        }
    } else if (n == 3) {
        // Image 3 - I=x
        for (int y = 0; y < E2XE_16M_SLOW; y++) {
            for (int x = 0; x < E2XE_16M_FAST; x++) {
                int k = y * E2XE_16M_FAST + x;
                data[k] = x;
            }
        }
    } else if (n == 4) {
        // Image 3 - I=y
        for (int y = 0; y < E2XE_16M_SLOW; y++) {
            for (int x = 0; x < E2XE_16M_FAST; x++) {
                int k = y * E2XE_16M_FAST + x;
                data[k] = y;
            }
        }

    } else if (n == 5) {
        // Image 3: "Random" background, zero on masks

        pcg32_random_t state = {0};
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        uint32_t num = pcg32_random_r(&state);
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = (num % 10);
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "Error: Unhandled sample image %d\n", (int)n);
        exit(2);
    }
}

/// Find the data file index for a particular image number.
/// If the image isn't found on any data files, returns obj->data_file_count
/// FIXME returns updated in index in place
int _find_data_file_for_image(h5read_handle *obj, size_t *index) {
    int data_file = 0;
    size_t subtracted = *index;
    for (; data_file < obj->data_file_count; data_file++) {
        if ((subtracted + obj->data_files[data_file].offset)
            < (obj->data_files[data_file].frames + obj->data_files[data_file].offset)) {
            break;
        } else {
            subtracted -= obj->data_files[data_file].frames;
        }
    }
    *index = subtracted;
    return data_file;
}

size_t h5read_get_chunk_size(h5read_handle *obj, size_t index) {
    if (obj->data_files == 0) {
        fprintf(stderr, "Error: Cannot do direct chunk read with sample data\n", index);
        exit(1);
    }
#ifdef HAVE_HDF5
    int data_file = _find_data_file_for_image(obj, &index);
    if (data_file == obj->data_file_count) {
        fprintf(stderr, "Error: Could not find data file for frame %ld\n", index);
        exit(1);
    }
    h5_data_file *current = &(obj->data_files[data_file]);

    hsize_t offset[3] = {index + current->offset, 0, 0};
    hsize_t chunk_size = 0;
    // H5Drefresh(current->dataset);
    H5Dget_chunk_storage_size(current->dataset, offset, &chunk_size);
#endif
    if (chunk_size < 0) {
        return 0;
    }
    return (size_t)chunk_size;
}

void h5read_get_raw_chunk(h5read_handle *obj,
                          size_t index,
                          size_t *size,
                          uint8_t *data,
                          size_t max_size) {
    if (obj->data_files == 0) {
        fprintf(stderr, "Error: Cannot do direct chunk read with sample data\n", index);
        exit(1);
    }
#ifdef HAVE_HDF5
    int data_file = _find_data_file_for_image(obj, &index);
    if (data_file == obj->data_file_count) {
        fprintf(stderr, "Error: Could not find data file for frame %ld\n", index);
        exit(1);
    }
    h5_data_file *current = &(obj->data_files[data_file]);

    hsize_t offset[3] = {index + current->offset, 0, 0};
    hsize_t chunk_size = 0;
    H5Dget_chunk_storage_size(current->dataset, offset, &chunk_size);
    *size = chunk_size;

    if (chunk_size > max_size) {
        fprintf(stderr, "Error: Not enough room to store compressed chunk\n");
        exit(1);
    }

    hid_t datatype = H5Dget_type(current->dataset);
    hsize_t datasize = H5Tget_size(datatype);
    if (datasize != 2) {
        fprintf(stderr, "Error: Unexpected datasize\n");
        exit(1);
    }

    uint32_t filter = 0;
    herr_t err = H5Dread_chunk(current->dataset, H5P_DEFAULT, offset, &filter, data);
    if (err < 0) {
        fprintf(stderr, "Error: Failed to read chunk\n");
        exit(1);
    }
#endif
}

void h5read_get_image_into(h5read_handle *obj, size_t index, image_t_type *data) {
    if (index >= obj->frames) {
        fprintf(stderr,
                "Error: image %ld greater than number of frames (%ld)\n",
                index,
                obj->frames);
        exit(1);
    }
    // Check if we are using sample data
    if (obj->data_files == 0) {
        // We are using autogenerated image data. Return that.
        _generate_sample_image(obj, index, data);
        return;
    }

#ifdef HAVE_HDF5
    /* first find the right data file - having to do this lookup is annoying
       but probably cheap */
    int data_file = _find_data_file_for_image(obj, &index);

    if (data_file == obj->data_file_count) {
        fprintf(stderr, "Error: Could not find data file for frame %ld\n", index);
        exit(1);
    }

    h5_data_file *current = &(obj->data_files[data_file]);

    hid_t space = H5Dget_space(current->dataset);
    hid_t datatype = H5Dget_type(current->dataset);

    hsize_t block[3] = {1, obj->slow, obj->fast};
    hsize_t offset[3] = {index + current->offset, 0, 0};

    // select data to read #todo add status checks
    H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, block, NULL);
    hid_t mem_space = H5Screate_simple(3, block, NULL);

    if (H5Dread(current->dataset, datatype, mem_space, space, H5P_DEFAULT, data) < 0) {
        H5Eprint(H5E_DEFAULT, NULL);
        exit(1);
    }

    H5Sclose(space);
    H5Sclose(mem_space);
#endif
}

image_t *h5read_get_image(h5read_handle *obj, size_t n) {
    // Make an image_t to write into
    image_t *result = malloc(sizeof(image_t));
    result->mask = obj->mask;
    result->fast = obj->fast;
    result->slow = obj->slow;
    // Create the buffer here. This will be freed by h5read_free_image
    result->data = malloc(sizeof(image_t_type) * obj->slow * obj->fast);
    // Use our read-into-buffer function to fill this
    h5read_get_image_into(obj, n, result->data);

    return result;
}

void h5read_get_trusted_range(h5read_handle *obj,
                              image_t_type *min,
                              image_t_type *max) {
    if (min != NULL) {
        *min = obj->trusted_range_min;
    }
    if (max != NULL) {
        *max = obj->trusted_range_max;
    }
}

float h5read_get_wavelength(h5read_handle *obj) {
    return obj->wavelength;
}

float h5read_get_pixel_size_slow(h5read_handle *obj) {
    return obj->pixel_size_x;
}
float h5read_get_pixel_size_fast(h5read_handle *obj) {
    return obj->pixel_size_y;
}
float h5read_get_detector_distance(h5read_handle *obj) {
    return obj->detector_distance;
}
float h5read_get_beam_center_x(h5read_handle *obj) {
    return obj->beam_center_x;
}
float h5read_get_beam_center_y(h5read_handle *obj) {
    return obj->beam_center_y;
}
float h5read_get_oscillation_start(h5read_handle *obj) {
    return obj->oscillation_start;
}
float h5read_get_oscillation_width(h5read_handle *obj) {
    return obj->oscillation_width;
}

#ifdef HAVE_HDF5
void read_mask(h5read_handle *obj) {
    char mask_path[] = "/entry/instrument/detector/pixel_mask";

    hid_t mask_dataset = H5Dopen(obj->master_file, mask_path, H5P_DEFAULT);

    if (mask_dataset < 0) {
        // We are allowed to not have a mask in NXmx
        fprintf(stderr, "Warning: no mask data found at %s\n", mask_path);
        // Make an empty mask
        obj->mask_size = 0;
        obj->mask = NULL;
        return;
    }

    hid_t datatype = H5Dget_type(mask_dataset);
    hid_t mask_info = H5Dget_space(mask_dataset);

    size_t mask_dsize = H5Tget_size(datatype);
    if (mask_dsize == 4) {
        printf("mask dtype uint32\n");
    } else if (mask_dsize == 8) {
        printf("mask dtype uint64\n");
    } else {
        fprintf(stderr, "Error: mask data size (%ld) != 4,8\n", H5Tget_size(datatype));
        exit(1);
    }

    obj->mask_size = H5Sget_simple_extent_npoints(mask_info);

    printf("Mask has %ld elements\n", obj->mask_size);

    void *buffer = NULL;

    uint32_t *raw_mask = NULL;
    uint64_t *raw_mask_64 = NULL;  // why?
    if (mask_dsize == 4) {
        raw_mask = (uint32_t *)malloc(sizeof(uint32_t) * obj->mask_size);
        buffer = (void *)raw_mask;
    } else {
        raw_mask_64 = (uint64_t *)malloc(sizeof(uint64_t) * obj->mask_size);
        buffer = (void *)raw_mask_64;
    }

    if (H5Dread(mask_dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
        fprintf(stderr, "Error: While reading mask\n");
        exit(1);
    }

    // count 0's

    size_t zero = 0;

    obj->mask = (uint8_t *)malloc(sizeof(uint8_t) * obj->mask_size);

    if (mask_dsize == 4) {
        for (size_t j = 0; j < obj->mask_size; j++) {
            if (raw_mask[j] == 0) {
                zero++;
                obj->mask[j] = 1;
            } else {
                obj->mask[j] = 0;
            }
        }
    } else {
        for (size_t j = 0; j < obj->mask_size; j++) {
            if (raw_mask_64[j] == 0) {
                zero++;
                obj->mask[j] = 1;
            } else {
                obj->mask[j] = 0;
            }
        }
    }

    // blit mask over to module mask

    size_t fast, slow, offset, target, image_slow, image_fast, module_pixels;
    module_pixels = E2XE_MOD_FAST * E2XE_MOD_SLOW;

    if (obj->mask_size == E2XE_16M_SLOW * E2XE_16M_FAST) {
        slow = 8;
        fast = 4;
        image_slow = E2XE_16M_SLOW;
        image_fast = E2XE_16M_FAST;
    } else {
        slow = 4;
        fast = 2;
        image_slow = E2XE_4M_SLOW;
        image_fast = E2XE_4M_FAST;
    }
    obj->module_mask = (uint8_t *)malloc(sizeof(uint8_t) * fast * slow * module_pixels);
    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image_fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                offset =
                  (row0 + row * image_fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                target = (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&obj->module_mask[target],
                       (void *)&obj->mask[offset],
                       sizeof(uint8_t) * E2XE_MOD_FAST);
            }
        }
    }

    printf("%ld of the pixels are valid\n", zero);

    // cleanup

    if (raw_mask) free(raw_mask);
    if (raw_mask_64) free(raw_mask_64);
    H5Dclose(mask_dataset);
}

/// Read a single float value out of an HDF5 dataset
///
/// If the dataset is present, but cannot be read, terminates the program.
///
/// @param      origin          The root file or group to read the path from
/// @param      path            The path from the origin to open
/// @param[out] destination     Where to write the output value to, if successful.
///
/// @return The HDF error code (negative) if opening the dataset failed, or 0
herr_t _read_single_value_image_t_type(hid_t origin,
                                       const char *dataset_path,
                                       image_t_type *destination) {
    hid_t dataset = H5Dopen(origin, dataset_path, H5P_DEFAULT);
    if (dataset < 0) {
        return dataset;
    }
    hid_t datatype = H5Dget_type(dataset);
    size_t datatype_size = H5Tget_size(datatype);
    hid_t dataspace = H5Dget_space(dataset);
    size_t num_elements = H5Sget_simple_extent_npoints(dataspace);

    if (num_elements > 1) {
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);

        fprintf(stderr,
                "Error: While reading %s%s: More than one element.\n",
                name,
                dataset_path);
        exit(1);
    }
    if (datatype_size < sizeof(image_t_type)) {
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);

        fprintf(
          stderr,
          "Error: While reading %s%s: Value of size %zu is smaller than data type\n",
          name,
          dataset_path,
          datatype_size);
        exit(1);
    }
    assert(sizeof(image_t_type) == 2);
    image_t_type value;
    if (H5Dread(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, destination)
        < 0) {
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);
        fprintf(stderr,
                "Error: While reading %s%s: Unspecified data reading error.\n",
                name,
                dataset_path);
        exit(1);
    }
    H5Dclose(dataset);
    return 0;
}

/// Read a single float value out of an HDF5 dataset
///
/// If the dataset is present, but contains multiple elements, sets destination
/// to -1. If an unknown error occurs, terminates the program.
///
/// @param      origin          The root file or group to read the path from
/// @param      path            The path from the origin to open
/// @param[out] destination     Where to write the output value to, if successful.
///
/// @return HDF error (negative) if opening the dataset failed. Othewise, 0.
herr_t _read_single_value_float(hid_t origin, const char *path, float *destination) {
    hid_t dataset = H5Dopen(origin, path, H5P_DEFAULT);
    if (dataset < 0) {
        return dataset;
    }
    hid_t datatype = H5Dget_type(dataset);
    size_t datatype_size = H5Tget_size(datatype);
    hid_t dataspace = H5Dget_space(dataset);
    size_t num_elements = H5Sget_simple_extent_npoints(dataspace);

    if (num_elements > 1) {
        // Get the name of the origin for output
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);
        fprintf(
          stderr, "Error: While reading %s/%s: More than one element.\n", name, path);
        *destination = -1;
    } else if (datatype_size < sizeof(float)) {
        // Get the name of the origin for output
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);

        fprintf(
          stderr,
          "Error: While reading %s/%s: Value of size %zu is smaller than data type\n",
          name,
          path,
          datatype_size);
        *destination = -1;
    } else if (H5Dread(
                 dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, destination)
               < 0) {
        // Get the name of the origin for output
        char name[256] = "\0";
        H5Iget_name(origin, name, 256);

        fprintf(stderr,
                "Error: While reading %s/%s: Unspecified data reading error.\n",
                name,
                path);
        exit(1);
    }
    H5Dclose(dataset);
    return 0;
}

void read_trusted_range(h5read_handle *obj) {
    obj->trusted_range_min = 0;
    obj->trusted_range_max = (image_t_type)(-1);
    _read_single_value_image_t_type(obj->master_file,
                                    "/entry/instrument/detector/saturation_value",
                                    &obj->trusted_range_max);
    _read_single_value_image_t_type(obj->master_file,
                                    "/entry/instrument/detector/underload_value",
                                    &obj->trusted_range_min);
}

void read_wavelength(h5read_handle *obj) {
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/beam/incident_wavelength",
                                 &obj->wavelength)
        < 0) {
        fprintf(stderr, "No wavelength data found...\n");
        obj->wavelength = -1;
    }
}

void read_oscillation_start_and_width(h5read_handle *obj) {
    char omega_path[] = "/entry/sample/sample_omega/omega";

    hid_t omega_dataset = H5Dopen(obj->master_file, omega_path, H5P_DEFAULT);
    if (omega_dataset < 0) {
        //We're allowed no omega scan e.g. grid
        obj->oscillation_start = 0;
        obj->oscillation_width = 0;
        return;
    }
    hid_t datatype = H5Dget_type(omega_dataset);
    hid_t omega_info = H5Dget_space(omega_dataset);
    int size = H5Sget_simple_extent_npoints(omega_info);
    if (size < 2) {
        fprintf(stderr, "Error: While reading oscillation, size<2\n");
        exit(1);
    }
    double *raw_omega = (double *)malloc(sizeof(double) * size);
    void *buffer = (void *)raw_omega;
    if (H5Dread(omega_dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
        fprintf(stderr, "Error: While reading oscillation\n");
        exit(1);
    }
    obj->oscillation_start = (float)raw_omega[0];
    obj->oscillation_width = (float)raw_omega[1] - raw_omega[0];
    free(raw_omega);
    H5Dclose(omega_dataset);
}

void read_detector_metadata(h5read_handle *obj) {
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/detector/x_pixel_size",
                                 &obj->pixel_size_x)
        < 0) {
        obj->pixel_size_x = -1;
    }
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/detector/y_pixel_size",
                                 &obj->pixel_size_y)
        < 0) {
        obj->pixel_size_y = -1;
    }
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/detector/beam_center_x",
                                 &obj->beam_center_x)
        < 0) {
        obj->beam_center_x = -1;
    }
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/detector/beam_center_y",
                                 &obj->beam_center_y)
        < 0) {
        obj->beam_center_y = -1;
    }
    if (_read_single_value_float(obj->master_file,
                                 "/entry/instrument/detector/distance",
                                 &obj->detector_distance)
        < 0) {
        obj->detector_distance = 0;
    }

    printf("Read pixel size: %f\n", obj->pixel_size_x);
}

/// Get number of VDS and read info about all the sub-files.
///
/// @param master           HDF5 File object pointing to the master file
/// @param dataset          The root dataset to search for VDS from
/// @param data_files_array Pointer to an array variable, that will be
///                         allocated and filled with basic information
///                         about the VDS sub-files.
/// @returns The number of VDS found and allocated into data_files_array
int vds_info(char *root, hid_t master, hid_t dataset, h5_data_file **data_files_array) {
    hid_t plist, vds_source;
    size_t vds_count;
    herr_t status;

    plist = H5Dget_create_plist(dataset);

    status = H5Pget_virtual_count(plist, &vds_count);

    *data_files_array = calloc(vds_count, sizeof(h5_data_file));
    // Used to use vds parameter directly - put here so no mass-changes
    h5_data_file *vds = *data_files_array;

    for (int j = 0; j < vds_count; j++) {
        hsize_t start[MAXDIM], stride[MAXDIM], count[MAXDIM], block[MAXDIM];
        size_t dims;

        vds_source = H5Pget_virtual_srcspace(plist, j);
        dims = H5Sget_simple_extent_ndims(vds_source);

        if (dims != 3) {
            H5Sclose(vds_source);
            fprintf(stderr, "incorrect data dimensionality: %d\n", (int)dims);
            return -1;
        }

        H5Sget_regular_hyperslab(vds_source, start, stride, count, block);
        H5Sclose(vds_source);

        H5Pget_virtual_filename(plist, j, vds[j].filename, MAXFILENAME);
        H5Pget_virtual_dsetname(plist, j, vds[j].dsetname, MAXFILENAME);

        for (int k = 1; k < dims; k++) {
            if (start[k] != 0) {
                fprintf(stderr, "incorrect chunk start: %d\n", (int)start[k]);
                return -1;
            }
        }

        vds[j].frames = block[0];
        vds[j].offset = start[0];

        if ((strlen(vds[j].filename) == 1) && (vds[j].filename[0] == '.')) {
            H5L_info_t info;
            status = H5Lget_info(master, vds[j].dsetname, &info, H5P_DEFAULT);

            if (status) {
                fprintf(stderr, "error from H5Lget_info on %s\n", vds[j].dsetname);
                return -1;
            }

            /* if the data file points to an external source, dereference */

            if (info.type == H5L_TYPE_EXTERNAL) {
                char buffer[MAXFILENAME], scr[MAXFILENAME];
                unsigned flags;
                const char *nameptr, *dsetptr;

                H5Lget_val(master, vds[j].dsetname, buffer, MAXFILENAME, H5P_DEFAULT);
                H5Lunpack_elink_val(
                  buffer, info.u.val_size, &flags, &nameptr, &dsetptr);

                /* assumptions herein:
                    - external link references are local paths
                    - only need to worry about UNIX paths e.g. pathsep is /
                    - ASCII so chars are ... chars
                   so manually assemble...
                 */

                strcpy(scr, root);
                scr[strlen(root)] = '/';
                strcpy(scr + strlen(root) + 1, nameptr);

                strcpy(vds[j].filename, scr);
                strcpy(vds[j].dsetname, dsetptr);
            }
        } else {
            char scr[MAXFILENAME];
            assert(snprintf(scr, MAXFILENAME, "%s/%s", root, vds[j].filename)
                   < MAXFILENAME);
            strcpy(vds[j].filename, scr);
        }

        // do I want to open these here? Or when they are needed...
        vds[j].file = 0;
        vds[j].dataset = 0;
    }

    status = H5Pclose(plist);

    return vds_count;
}

/// Extracts the h5_data_file dictionary for information on all VDS
///
/// @param filename         The name of the master file
/// @param h5_data_file     The data_files array to be allocated and filled
///
/// @returns The number of VDS files
int unpack_vds(const char *filename, h5_data_file **data_files) {
    // TODO if we want this to become SWMR aware in the future will need to
    // allow for that here
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);

    if (file < 0) {
        fprintf(stderr, "Error: Opening for VDS read %s\n", filename);
        return -1;
    }

    hid_t dataset = H5Dopen(file, "/entry/data/data", H5P_DEFAULT);
    if (dataset < 0) {
        H5Fclose(file);
        fprintf(stderr, "Error: Reading H5 entry %s\n", "/entry/data/data");
        return -1;
    }

    /* always set the absolute path to file information */
    char rootpath[MAXFILENAME];
    strncpy(rootpath, filename, MAXFILENAME);
    char *root = dirname(rootpath);
    char cwd[MAXFILENAME];
    if ((strlen(root) == 1) && (root[0] == '.')) {
        root = getcwd(cwd, MAXFILENAME);
    }

    int vds_count = vds_info(root, file, dataset, data_files);

    H5Dclose(dataset);
    H5Fclose(file);

    return vds_count;
}

void setup_data(h5read_handle *obj) {
    hid_t dataset = obj->data_files[0].dataset;
    hid_t datatype = H5Dget_type(dataset);

    if (H5Tget_size(datatype) != 2) {
        fprintf(stderr, "native data size != 2 (%ld)\n", H5Tget_size(datatype));
        exit(1);
    }

    hid_t space = H5Dget_space(dataset);

    if (H5Sget_simple_extent_ndims(space) != 3) {
        fprintf(stderr, "raw data not three dimensional\n");
        exit(1);
    }

    hsize_t dims[3];
    H5Sget_simple_extent_dims(space, dims, NULL);

    obj->slow = dims[1];
    obj->fast = dims[2];

    printf("Total data size: %ldx%ldx%ld\n", obj->frames, obj->slow, obj->fast);
    H5Sclose(space);
}

h5read_handle *h5read_open(const char *master_filename) {
    hid_t master_file =
      H5Fopen(master_filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);

    if (master_file < 0) {
        fprintf(stderr, "Error: Reading %s\n", master_filename);
        return NULL;
    }

    // Create the H5 handle object
    h5read_handle *file = calloc(1, sizeof(h5read_handle));
    file->master_file = master_file;

    file->data_file_count = unpack_vds(master_filename, &file->data_files);

    if (file->data_file_count < 0) {
        fprintf(stderr, "Error: While reading VDS of %s\n", master_filename);
        H5Fclose(master_file);
        free(file);
        return NULL;
    }

    // open up the actual data files, count all the frames
    file->frames = 0;
    h5_data_file *data_files = file->data_files;
    for (int j = 0; j < file->data_file_count; j++) {
        data_files[j].file = H5Fopen(
          data_files[j].filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);
        if (data_files[j].file < 0) {
            fprintf(stderr, "Error: Opening child file %s\n", data_files[j].filename);
            H5Fclose(master_file);
            free(file);
            return NULL;
        }
        data_files[j].dataset =
          H5Dopen(data_files[j].file, data_files[j].dsetname, H5P_DEFAULT);
        if (data_files[j].dataset < 0) {
            fprintf(stderr,
                    "Error: Reading datasets of child file %s\n",
                    data_files[j].filename);
            H5Fclose(data_files[j].file);
            H5Fclose(master_file);
            free(file);
            return NULL;
        }
        file->frames += data_files[j].frames;
    }

    read_trusted_range(file);

    read_wavelength(file);

    read_detector_metadata(file);

    read_oscillation_start_and_width(file);

    read_mask(file);

    setup_data(file);

    return file;
}
#endif

// Generate a mask with just module bounds masked off
uint8_t *_generate_e2xe_16m_mask() {
    assert(E2XE_MOD_SLOW * E2XE_16M_NSLOW + E2XE_GAP_SLOW * (E2XE_16M_NSLOW - 1)
           == E2XE_16M_SLOW);
    assert(E2XE_MOD_FAST * E2XE_16M_NFAST + E2XE_GAP_FAST * (E2XE_16M_NFAST - 1)
           == E2XE_16M_FAST);
    uint8_t *mask = calloc(E2XE_16M_SLOW * E2XE_16M_SLOW, sizeof(uint8_t));
    for (size_t i = 0; i < E2XE_16M_SLOW * E2XE_16M_SLOW; ++i) {
        mask[i] = 1;
    }
    // Horizontal gaps
    for (int gap = 1; gap < E2XE_16M_NSLOW; ++gap) {
        // First gap has 1 module 0 gap, second gap has 2 modules 1 gap etc
        size_t y = gap * E2XE_MOD_SLOW + (gap - 1) * E2XE_GAP_SLOW;
        // Horizontal gaps can just be bulk memset for each gap
        memset(mask + y * E2XE_16M_FAST, 0, E2XE_GAP_SLOW * E2XE_16M_FAST);
    }
    // Vertical gaps
    for (int gap = 1; gap < E2XE_16M_NFAST; ++gap) {
        // First gap has 1 module 0 gap, second gap has 2 modules 1 gap etc
        size_t x = gap * E2XE_MOD_FAST + (gap - 1) * E2XE_GAP_FAST;
        for (int y = 0; y < E2XE_16M_SLOW; ++y) {
            memset(mask + y * E2XE_16M_FAST + x, 0, E2XE_GAP_FAST);
        }
    }
    return mask;
}

h5read_handle *h5read_generate_samples() {
    h5read_handle *file = calloc(1, sizeof(h5read_handle));

    // Generate the mask - with module gaps masked off
    file->slow = E2XE_16M_SLOW;
    file->fast = E2XE_16M_FAST;
    file->mask = _generate_e2xe_16m_mask();
    file->trusted_range_max = (image_t_type)(-1);
    file->trusted_range_min = 0;
    file->beam_center_x = E2XE_16M_FAST / 2.0;
    file->beam_center_y = E2XE_16M_SLOW / 2.0;
    file->pixel_size_x = 0.75e-6;
    file->pixel_size_y = 0.75e-6;
    file->detector_distance = 0.5;

    // Module mask is just empty for now
    file->module_mask =
      malloc(E2XE_16M_NSLOW * E2XE_16M_NFAST * E2XE_MOD_FAST * E2XE_MOD_SLOW);
    for (size_t i = 0;
         i < E2XE_16M_NSLOW * E2XE_16M_NFAST * E2XE_MOD_FAST * E2XE_MOD_SLOW;
         ++i) {
        file->module_mask[i] = 1;
    }

    // Debug - writing mask to a test file
    // FILE *fo = fopen("mask.dat", "w");
    // fwrite(file->mask, sizeof(uint8_t), E2XE_16M_SLOW * E2XE_16M_FAST, fo);
    // fclose(fo);

    file->frames = NUM_SAMPLE_IMAGES;
    return file;
}

h5read_handle *h5read_parse_standard_args(int argc, char **argv) {
    bool implicit_sample = getenv("H5READ_IMPLICIT_SAMPLE") != NULL;
#ifndef HAVE_HDF5
    // Always assume implicit sample if we can't read HDF5 files
    implicit_sample = true;
#endif
    const char *USAGE = implicit_sample
                          ? "Usage: %s [-h|--help] [-v] [FILE.nxs | --sample]\n"
                          : "Usage: %s [-h|--help] [-v] (FILE.nxs | --sample)\n";
    const char *HELP =
      "Options:\n\
  FILE.nxs      Path to the Nexus file to parse\n\
  -h, --help    Show this message\n\
  -v            Verbose HDF5 message output\n\
  --sample      Don't load a data file, instead use generated test data.\n\
                If H5READ_IMPLICIT_SAMPLE is set, then this is assumed,\n\
                if a file is not provided.";

    bool verbose = false;
    bool sample_data = false;

    // Handle simple case of -h or --help
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, USAGE, argv[0]);
            fprintf(stderr, "\n%s\n", HELP);
            exit(0);
        }
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
            // Shift the rest over this one so that we only have positionals
            for (int j = i; j < argc; j++) {
                argv[i] = argv[j];
            }
            argc -= 1;
        }
        if (!strcmp(argv[i], "--sample")) {
            sample_data = true;
            // Shift the rest over this one so that we only have positionals
            for (int j = i; j < argc; j++) {
                argv[i] = argv[j];
            }
            argc -= 1;
        }
    }
#ifdef HAVE_HDF5
    if (!verbose) {
        // Turn off verbose hdf5 errors
        H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    }
#endif
    if (implicit_sample && argc == 1 && !sample_data) {
#ifdef HAVE_HDF5
        fprintf(
          stderr,
          "No input file but H5READ_IMPLICIT_SAMPLE is set - defaulting to sample "
          "data\n");
#else
        fprintf(
          stderr,
          "Not built with HDF5 support and no files specified; using sample data\n");
#endif
        sample_data = true;
    }

    if (argc == 1 && !sample_data) {
        fprintf(stderr, USAGE, argv[0]);
        exit(1);
    }
    // If we specifically requested --sample, then we can have no more arguments
    if (argc > 1 && sample_data) {
        fprintf(stderr, "Unrecognised extra arguments with --sample\n");
        exit(1);
    }

    h5read_handle *handle = 0;
    if (sample_data) {
        fprintf(stderr, "Using SAMPLE dataset\n");
        handle = h5read_generate_samples();
    } else {
#ifdef HAVE_HDF5
        handle = h5read_open(argv[1]);
#else
        fprintf(stderr, "Error: Cannot read nexus file as h5read built without HDF5\n");
#endif
    }
    if (handle == NULL) {
        fprintf(stderr, "Error: Could not open nexus file %s\n", argv[1]);
        exit(1);
    }
    return handle;
}
