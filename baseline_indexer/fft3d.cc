#include <assert.h>
#include <math.h>
#include <pocketfft_hdronly.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <map>
#include <stack>
#include <tuple>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

#define _USE_MATH_DEFINES
#include <cmath>

using namespace pocketfft;

/**
 * @brief map reciprocal space vectors onto a grid of size n_points^3.
 * @param reciprocal_space_vectors Reciprocal space vectors to be mapped.
 * @param data_in The vector (grid) which the data will be mapped to.
 * @param selection The vector of the selection of points mapped to the grid.
 * @param d_min A resolution limit for mapping to the grid.
 * @param b_iso The isotropic B-factor used to weight the points as a function of resolution.
 * @param n_points The size of each dimension of the FFT grid.
 */
void map_centroids_to_reciprocal_space_grid(
  std::vector<Vector3d> const &reciprocal_space_vectors,
  std::vector<std::complex<double>> &data_in,
  std::vector<bool> &selection,
  double d_min,
  double b_iso = 0,
  uint32_t n_points = 256) {
    assert(data_in.size() == n_points * n_points * n_points);
    // Determine the resolution span of the grid so we know how to map
    // each coordinate to the grid.
    const double rlgrid = 2 / (d_min * n_points);
    const double one_over_rlgrid = 1 / rlgrid;
    const int half_n_points = n_points / 2;
    int count = 0;

    for (int i = 0; i < reciprocal_space_vectors.size(); i++) {
        const Vector3d v = reciprocal_space_vectors[i];
        const double v_length = v.norm();
        const double d_spacing = 1 / v_length;
        if (d_spacing < d_min) {
            selection[i] = false;
            continue;
        }
        Vector3i coord;
        // map to the nearest point in each dimension.
        for (int j = 0; j < 3; j++) {
            coord[j] = static_cast<int>(round(v[j] * one_over_rlgrid)) + half_n_points;
        }
        if ((coord.maxCoeff() >= n_points) || coord.minCoeff() < 0) {
            selection[i] = false;
            continue;
        }
        // Use the b_iso to determine the weight for each coordinate.
        double T;
        if (b_iso != 0) {
            T = std::exp(-b_iso * v_length * v_length / 4.0);
        } else {
            T = 1;
        }
        // unravel to the 1d index and write the complex<double> value.
        size_t index =
          coord[2] + (n_points * coord[1]) + (n_points * n_points * coord[0]);
        if (!data_in[index].real()) {
            count++;
        }
        data_in[index] = {T, 0.0};
    }
    spdlog::info("Number of centroids used: {0}", count);
}

/**
 * @brief Perform a 3D FFT of the reciprocal space coordinates (spots).
 * @param reciprocal_space_vectors The input vector of reciprocal space coordinates.
 * @param real_out The (real) array that the FFT result will be written to.
 * @param d_min Cut the data at this resolution limit for the FFT
 * @param b_iso The isotropic B-factor used to weight the points as a function of resolution.
 * @param n_points The size of each dimension of the FFT grid.
 * @returns A boolean array indicating which coordinates were used for the FFT.
 */
std::vector<bool> fft3d(std::vector<Vector3d> const &reciprocal_space_vectors,
                        std::vector<double> &real_out,
                        double d_min,
                        double b_iso = 0,
                        uint32_t n_points = 256,
                        size_t nthreads = 1) {
    auto start = std::chrono::system_clock::now();
    assert(real_out.size() == n_points * n_points * n_points);

    // We want to write out the real part of the FFT, but the pocketfft functions require
    // complex vectors (we are using c2c i.e. complex to complex), so initialise these vectors.
    // Note we should be able to use c2r rather than c2c, but I couldn't get this to work with
    // the output ordering in c2r (JBE).
    std::vector<std::complex<double>> complex_data_in(n_points * n_points * n_points);
    std::vector<std::complex<double>> data_out(n_points * n_points * n_points);

    // A boolean array of whether the vectors were used for the FFT.
    std::vector<bool> used_in_indexing(reciprocal_space_vectors.size(), true);
    auto t1 = std::chrono::system_clock::now();

    // Map the vectors onto a discrete grid. The values of the grid points are weights
    // determined by b_iso.
    map_centroids_to_reciprocal_space_grid(reciprocal_space_vectors,
                                           complex_data_in,
                                           used_in_indexing,
                                           d_min,
                                           b_iso,
                                           n_points);
    auto t2 = std::chrono::system_clock::now();

    // Prepare the required objects for the FFT.
    shape_t shape_in{n_points, n_points, n_points};
    int stride_x = sizeof(std::complex<double>);
    int stride_y = static_cast<int>(sizeof(std::complex<double>) * n_points);
    int stride_z = static_cast<int>(sizeof(std::complex<double>) * n_points * n_points);
    stride_t stride_in{
      stride_x, stride_y, stride_z};  // must have the size of each element. Must have
                                      // size() equal to shape_in.size()
    stride_t stride_out{
      stride_x, stride_y, stride_z};  // must have the size of each element. Must
                                      // have size() equal to shape_in.size()
    shape_t axes{0, 1, 2};            // 0 to shape.size()-1 inclusive
    bool forward{FORWARD};

    double fct{1.0f};
    // note, threads can be higher than the number of hardware threads.
    // It is not clear what the best value is for this.
    spdlog::info("Performing FFT with nthreads={0}", nthreads);
    // Do the FFT.
    c2c(shape_in,
        stride_in,
        stride_out,
        axes,
        forward,
        complex_data_in.data(),
        data_out.data(),
        fct,
        nthreads);
    auto t3 = std::chrono::system_clock::now();

    // Take the square of the real part as the output.
    for (int i = 0; i < real_out.size(); ++i) {
        real_out[i] = std::pow(data_out[i].real(), 2);
    }
    auto t4 = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = t4 - start;
    std::chrono::duration<double> elapsed_map = t2 - t1;
    std::chrono::duration<double> elapsed_make_arrays = t1 - start;
    std::chrono::duration<double> elapsed_c2c = t3 - t2;
    std::chrono::duration<double> elapsed_square = t4 - t3;
    spdlog::debug("Total time for fft3d: {0:.5f}s", elapsed_seconds.count());
    spdlog::debug("elapsed time for making data arrays: {0:.5f}s",
                  elapsed_make_arrays.count());
    spdlog::debug("elapsed time for map_to_recip: {0:.5f}s", elapsed_map.count());
    spdlog::debug("elapsed time for c2c: {0:.5f}s", elapsed_c2c.count());
    spdlog::debug("elapsed time for squaring: {0:.5f}s", elapsed_square.count());

    return used_in_indexing;
}
