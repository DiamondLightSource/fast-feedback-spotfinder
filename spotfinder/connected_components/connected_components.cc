#include "connected_components.hpp"

#include <builtin_types.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "h5read.h"

/**
 * Build a graph from the pixel coordinates
 * 
 * The graph is built by iterating over the pixel coordinates and connecting
 * pixels that are adjacent to each other.
 * 
 * @param px_coords a vector of pixel coordinates
 * @param px_kvals a vector of pixel indices
 * @param px_values a vector of pixel intensity values
 * @param width the width of the image
 * @param height the height of the image
 * @param graph (output) empty graph to be filled with edges
 */
void build_graph(
  const std::vector<int2> &px_coords,
  const std::vector<size_t> &px_kvals,
  const std::vector<pixel_t> &px_values,
  const ushort width,
  const ushort height,
  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> &graph) {
    // Index for next pixel to search when looking for pixels
    // below the current one. This will only ever increase, because
    // we are guaranteed to always look for one after the last found
    // pixel.
    int idx_pixel_below = 1;

    for (int i = 0; i < static_cast<int>(px_coords.size()) - 1; ++i) {
        auto coord = px_coords[i];
        auto coord_right = int2{coord.x + 1, coord.y};
        auto k = px_kvals[i];

        if (px_coords[i + 1] == coord_right) {
            // Since we generate strong pixels coordinates horizontally,
            // if there is a pixel to the right then it is guaranteed
            // to be the next one in the list. Connect these.
            boost::add_edge(i, i + 1, graph);
        }
        // Now, check the pixel directly below this one. We need to scan
        // to find it, because _if_ there is a matching strong pixel,
        // then we don't know how far ahead it is in the coordinates array
        if (coord.y < height - 1) {
            auto coord_below = int2{coord.x, coord.y + 1};
            auto k_below = k + width;
            // int idx = i + 1;
            while (idx_pixel_below < px_coords.size() - 1
                   && px_kvals[idx_pixel_below] < k_below) {
                ++idx_pixel_below;
            }
            // Either we've got the pixel below, past that - or the
            // last pixel in the coordinate set.
            if (px_coords[idx_pixel_below] == coord_below) {
                boost::add_edge(i, idx_pixel_below, graph);
            }
        }
    }
}

/**
 * Build boxes from the connected components
 * 
 * The boxes are built by iterating over the labels and pixel coordinates
 * and updating the bounding box for each label.
 * 
 * @param labels a vector of labels for each pixel
 * @param px_coords a vector of pixel coordinates
 * @param width the width of the image
 * @param height the height of the image
 * @param boxes (output) empty vector of Reflections to be filled with boxes
 */
void build_boxes(const std::vector<int> &labels,
                 const std::vector<int2> &px_coords,
                 const ushort width,
                 const ushort height,
                 std::vector<Reflection> &boxes) {
    for (int i = 0; i < labels.size(); ++i) {
        auto label = labels[i];
        auto coord = px_coords[i];
        Reflection &box = boxes[label];
        box.l = std::min(box.l, coord.x);
        box.r = std::max(box.r, coord.x);
        box.t = std::min(box.t, coord.y);
        box.b = std::max(box.b, coord.y);
        box.num_pixels += 1;
    }
}

/**
 * Find connected components in a 2D image
 * 
 * This function finds connected components in a 2D image using a disjoint-set
 * data structure. The image is represented as a graph where each pixel is a
 * vertex and adjacent pixels are connected by edges. The connected components
 * are found by iterating over the pixel coordinates and connecting pixels that
 * are adjacent to each other. The connected components are then used to build
 * bounding boxes around the connected pixels.
 * 
 * @param px_coords a vector of pixel coordinates
 * @param px_kvals a vector of pixel indices
 * @param px_values a vector of pixel intensity values
 * @param width the width of the image
 * @param height the height of the image
 * @param min_spot_size the minimum size of a spot, set to 0 to disable filtering
 * @param boxes (output) a vector of Reflections to be filled with boxes
 * @return a tuple containing the number of strong pixels and the number of
 *        strong pixels after filtering
 */
void connected_components_2d(const std::vector<int2> px_coords,
                             const std::vector<size_t> px_kvals,
                             const std::vector<pixel_t> px_values,
                             const ushort width,
                             const ushort height,
                             const uint32_t min_spot_size,
                             std::vector<Reflection> *boxes_ptr) {
    size_t num_strong_pixels = 0;
    size_t num_strong_pixels_filtered = 0;

    // Build graph
    auto graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>{};
    build_graph(px_coords, px_kvals, px_values, width, height, graph);

    // Find connected components
    auto labels = std::vector<int>(boost::num_vertices(graph));
    auto num_labels = boost::connected_components(graph, labels.data());

    // Build boxes
    auto boxes = std::vector<Reflection>(num_labels, {width, height, 0, 0});
    build_boxes(labels, px_coords, width, height, boxes);

    // Copy boxes to output
    *boxes_ptr = boxes;
}