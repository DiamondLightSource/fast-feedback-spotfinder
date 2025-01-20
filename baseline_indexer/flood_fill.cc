#include <math.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <map>
#include <stack>
#include <tuple>
#define _USE_MATH_DEFINES
#include <spdlog/spdlog.h>

#include <cmath>
#include <numeric>

using Eigen::Vector3d;
using Eigen::Vector3i;

// Define a modulo function that returns python style modulo for negative numbers.
int modulo(int i, int n) {
    return (i % n + n) % n;
}

/**
 * @brief Perform a flood fill algorithm on a grid of data to determine connected areas of signal.
 * @param grid The input array (grid) of data
 * @param rmsd_cutoff Filter out grid points below this cutoff value
 * @param n_points The size of each dimension of the FFT grid.
 * @returns A tuple of grid points per peak and centres of mass of the peaks in fractional coordinates.
 */
std::tuple<std::vector<int>, std::vector<Vector3d>> flood_fill(
  std::vector<double> const& grid,
  double rmsd_cutoff = 15.0,
  int n_points = 256) {
    auto start = std::chrono::system_clock::now();
    assert(grid.size() == n_points * n_points * n_points);
    //  First calculate the rmsd and use this to create a binary grid
    double sumg = std::accumulate(grid.begin(), grid.end(), 0.0);
    double meang = sumg / grid.size();
    double sum_delta_sq = std::accumulate(
      grid.begin(), grid.end(), 0.0, [meang](double total, const double& val) {
          return total + std::pow(val - meang, 2);
      });
    double rmsd = std::pow(sum_delta_sq / grid.size(), 0.5);
    std::vector<int> grid_binary(n_points * n_points * n_points, 0);
    double cutoff = rmsd_cutoff * rmsd;
    for (int i = 0; i < grid.size(); i++) {
        if (grid[i] >= cutoff) {
            grid_binary[i] = 1;
        }
    }
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - start;
    spdlog::debug("Time for first part of flood fill: {0:.5f}s", elapsed_time.count());

    // Now do the flood fill.
    // Wrap around the edge in all three dimensions to replicate the DIALS
    // results exactly.
    int n_voids = 0;
    std::stack<Vector3i> stack;
    std::vector<std::vector<Vector3i>> accumulators;
    int target = 1;
    int replacement = 2;
    std::vector<int> grid_points_per_void;
    int accumulator_index = 0;

    // precalculate a few constants
    int total = n_points * n_points * n_points;
    int n_sq = n_points * n_points;
    int n_sq_minus_n = n_points * (n_points - 1);
    int nn_sq_minus_n = n_points * n_points * (n_points - 1);

    for (int i = 0; i < grid_binary.size(); i++) {
        if (grid_binary[i] == target) {
            // Convert the array index into xyz coordinates.
            // Store xyz coordinates on the stack, but index the array with 1D index.
            int x = i % n_points;
            int y = (i % n_sq) / n_points;
            int z = i / n_sq;
            Vector3i xyz = {x, y, z};
            stack.push(xyz);
            grid_binary[i] = replacement;
            std::vector<Vector3i> this_accumulator;
            accumulators.push_back(this_accumulator);
            n_voids++;
            grid_points_per_void.push_back(0);

            while (!stack.empty()) {
                Vector3i this_xyz = stack.top();
                stack.pop();
                accumulators[accumulator_index].push_back(this_xyz);
                grid_points_per_void[accumulator_index]++;

                // Predefined neighbor offsets for 6-connected neighbors
                static const std::array<Vector3i, 6> neighbors = {Vector3i{1, 0, 0},
                                                                  Vector3i{-1, 0, 0},
                                                                  Vector3i{0, 1, 0},
                                                                  Vector3i{0, -1, 0},
                                                                  Vector3i{0, 0, 1},
                                                                  Vector3i{0, 0, -1}};

                int modx = modulo(this_xyz[0], n_points);
                int mody = modulo(this_xyz[1], n_points) * n_points;
                int modz = modulo(this_xyz[2], n_points) * n_sq;

                for (const Vector3i& offset : neighbors) {
                    // Compute the neighbor position
                    Vector3i neighbor = this_xyz + offset;
                    // Compute the flattened 1D array index for the neighbor
                    int array_index =
                      (offset[0] ? modulo(neighbor[0], n_points) : modx) +  // x
                      (offset[1] ? (modulo(neighbor[1], n_points) * n_points) : mody)
                      +                                                             // y
                      (offset[2] ? (modulo(neighbor[2], n_points) * n_sq) : modz);  // z

                    // Check if the neighbor matches the target and push to the stack
                    if (grid_binary[array_index] == target) {
                        grid_binary[array_index] = replacement;
                        stack.push(neighbor);
                    }
                }
            }
            replacement++;
            accumulator_index++;
        }
    }
    auto t3 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time2 = t3 - t2;
    spdlog::debug("Time for second part of flood fill: {0:.5f}s",
                  elapsed_time2.count());

    // Now calculate the unweighted centres of mass of each group, in fractional coordinates.
    std::vector<Vector3d> centres_of_mass_frac(n_voids);
    for (int i = 0; i < accumulators.size(); i++) {
        std::vector<Vector3i> values = accumulators[i];
        int n = values.size();
        double divisor = static_cast<double>(n * n_points);
        Vector3i sum = std::accumulate(values.begin(), values.end(), Vector3i{0, 0, 0});
        centres_of_mass_frac[i] = {
          sum[2] / divisor, sum[1] / divisor, sum[0] / divisor};  //z,y,x
    }
    return std::make_tuple(grid_points_per_void, centres_of_mass_frac);
}

/**
 * @brief Perform a filter on the flood fill results.
 * @param grid_points_per_void The number of grid points in each peak
 * @param centres_of_mass_frac The centres of mass of each peak, in fractional coordinates
 * @param peak_volume_cutoff The minimum fractional threshold for peaks to be included.
 * @returns A tuple of grid points per peak and centres of mass of the peaks in fractional coordinates.
 */
std::tuple<std::vector<int>, std::vector<Vector3d>> flood_fill_filter(
  std::vector<int> grid_points_per_void,
  std::vector<Vector3d> centres_of_mass_frac,
  double peak_volume_cutoff = 0.15) {
    // Filter out based on iqr range and peak_volume_cutoff
    std::vector<int> grid_points_per_void_unsorted(grid_points_per_void);
    std::sort(grid_points_per_void.begin(), grid_points_per_void.end());
    // The peak around the origin of the FFT can be very large in volume,
    // so use the IQR range to filter high-volume peaks out that are not
    // from the typical distribution of points, before applying the peak
    // volume cutoff based on a fraction of the max volume in the remaining
    // array.
    int Q3_index = grid_points_per_void.size() * 3 / 4;
    int Q1_index = grid_points_per_void.size() / 4;
    int iqr = grid_points_per_void[Q3_index] - grid_points_per_void[Q1_index];
    constexpr int iqr_multiplier = 5;
    int cut = (iqr * iqr_multiplier) + grid_points_per_void[Q3_index];

    // Remove abnormally high volumes
    while (grid_points_per_void[grid_points_per_void.size() - 1] > cut) {
        grid_points_per_void.pop_back();
    }
    int max_val = grid_points_per_void[grid_points_per_void.size() - 1];
    // Cut based on a fraction of the max volume.
    int peak_cutoff = static_cast<int>(peak_volume_cutoff * max_val);
    for (int i = grid_points_per_void_unsorted.size() - 1; i >= 0; i--) {
        if (grid_points_per_void_unsorted[i] <= peak_cutoff) {
            grid_points_per_void_unsorted.erase(grid_points_per_void_unsorted.begin()
                                                + i);
            centres_of_mass_frac.erase(centres_of_mass_frac.begin() + i);
        }
    }
    return std::make_tuple(grid_points_per_void_unsorted, centres_of_mass_frac);
}
