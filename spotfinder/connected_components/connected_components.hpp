#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <builtin_types.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <vector>

#include "cuda_common.hpp"
#include "h5read.h"

struct Reflection {
    int l, t, r, b;  // Bounding box: left, top, right, bottom
    int num_pixels = 0;
};

struct Signal {
    int2 coord;           // Coordinates of the pixel (x, y)
    pixel_t intensity;    // Pixel intensity
    size_t linear_index;  // Linear index of the pixel
};

/**
 * @brief Class to find connected components in a 2D image
 * 
 * The `ConnectedComponents` class identifies connected components (clusters) in a
 * binary 2D image. It stores signal pixels and their corresponding graph structure
 * to determine connected regions.
 * 
 * @param result_image the result image
 * @param original_image the original image
 * @param width the width of the image
 * @param height the height of the image
 * @param z_index the z-index of the image
 * @param min_spot_size the minimum spot size
 * 
 * @note `result_image` and `original_image` are only used to construct the
 * signals and are not stored in the object, nor are they used again once
 * initialization is complete.
 */
class ConnectedComponents {
  public:
    ConnectedComponents(const uint8_t *result_image,
                        const pixel_t *original_image,
                        const ushort width,
                        const ushort height,
                        const uint32_t min_spot_size);

    uint get_num_strong_pixels() const {
        return num_strong_pixels;
    }

    uint get_num_strong_pixels_filtered() const {
        return num_strong_pixels_filtered;
    }

    std::vector<Reflection> get_boxes() const {
        return boxes;
    }

  private:
    uint num_strong_pixels;           // Number of strong pixels
    uint num_strong_pixels_filtered;  // Number of strong pixels after filtering
    std::vector<Reflection> boxes;    // Bounding boxes
    std::unordered_map<size_t, Signal>
      signals;  // Maps pixel linear index -> Signal (used to store signal pixels)
    std::unordered_map<size_t, size_t>
      vertex_map;  // Maps linear_index -> graph vertex ID

    /**
     * @brief Builds a graph where vertices represent signal pixels and edges represent 
     * connectivity between neighboring pixels.
     * 
     * This function uses the `signals` map to find neighboring pixels efficiently.
     * The `vertex_map` is used to map linear indices (from `signals`) to graph vertex IDs.
     */
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> build_graph(
      const ushort width,
      const ushort height);
    /**
     * @brief Generates bounding boxes for connected components using the graph labels.
     * 
     * The `labels` vector maps each graph vertex to its connected component ID.
     * The `vertex_map` is used to relate graph vertex IDs back to the original pixels.
     */
    void generate_boxes(std::vector<int> labels,
                        int num_labels,
                        const ushort width,
                        const ushort height,
                        const uint32_t min_spot_size);
};

#endif  // CONNECTED_COMPONENTS_HPP