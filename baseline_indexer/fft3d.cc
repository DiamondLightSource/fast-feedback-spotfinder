#include <pocketfft_hdronly.h>
#include <map>
#include <stack>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <math.h>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

#define _USE_MATH_DEFINES
#include <cmath>

using namespace pocketfft;

void map_centroids_to_reciprocal_space_grid_cpp(
  std::vector<Vector3d> const& reciprocal_space_vectors,
  std::vector<std::complex<double>> &data_in,
  std::vector<bool> &selection,
  double d_min,
  double b_iso = 0,
  uint32_t n_points = 256) {
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
    for (int j = 0; j < 3; j++) {
      coord[j] = ((int)round(v[j] * one_over_rlgrid)) + half_n_points;
    }
    if ((coord.maxCoeff() >= n_points) || coord.minCoeff() < 0) {
      selection[i] = false;
      continue;
    }
    double T;
    if (b_iso != 0) {
      T = std::exp(-b_iso * v_length * v_length / 4.0);
    } else {
      T = 1;
    }
    size_t index = coord[2] + (n_points * coord[1]) + (n_points * n_points * coord[0]);
    if (!data_in[index].real()){  
      count++;
    }
    data_in[index] = {T, 0.0};
  }
  std::cout << "Number of centroids used: " << count << std::endl;
}

std::vector<bool> fft3d(
  std::vector<Vector3d> const& reciprocal_space_vectors,
  std::vector<double> &real_out,
  double d_min,
  double b_iso = 0,
  uint32_t n_points = 256) {
  auto start = std::chrono::system_clock::now();

  std::vector<std::complex<double>> complex_data_in(n_points * n_points * n_points);
  std::vector<std::complex<double>> data_out(n_points * n_points * n_points);

  std::vector<bool> used_in_indexing(reciprocal_space_vectors.size(), true);
  auto t1 = std::chrono::system_clock::now();

  map_centroids_to_reciprocal_space_grid_cpp(reciprocal_space_vectors, complex_data_in, used_in_indexing, d_min, b_iso, n_points);
  auto t2 = std::chrono::system_clock::now();
  shape_t shape_in{n_points, n_points, n_points};
  int stride_x = sizeof(std::complex<double>);
  int stride_y = static_cast<int>(sizeof(std::complex<double>) * n_points);
  int stride_z = static_cast<int>(sizeof(std::complex<double>) * n_points * n_points);
  stride_t stride_in{stride_x, stride_y, stride_z}; // must have the size of each element. Must have
                                // size() equal to shape_in.size()
  stride_t stride_out{stride_x, stride_y, stride_z};  // must have the size of each element. Must
                                 // have size() equal to shape_in.size()
  shape_t axes{0, 1, 2};         // 0 to shape.size()-1 inclusive
  bool forward{FORWARD};
  
  double fct{1.0f};
  size_t nthreads = 20;  // use all threads available - is this working?
  
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

  for (int i = 0; i < real_out.size(); ++i) {
    real_out[i] = std::pow(data_out[i].real(), 2);
  }
  auto t4 = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = t4 - start;
  std::chrono::duration<double> elapsed_map = t2 - t1;
  std::chrono::duration<double> elapsed_make_arrays = t1 - start;
  std::chrono::duration<double> elapsed_c2c = t3 - t2;
  std::chrono::duration<double> elapsed_square = t4 - t3;
  std::cout << "Total time for fft3d: " << elapsed_seconds.count() << "s" << std::endl;

  std::cout << "elapsed time for making data arrays: " << elapsed_make_arrays.count() << "s" << std::endl;
  std::cout << "elapsed time for map_to_recip: " << elapsed_map.count() << "s" << std::endl;
  std::cout << "elapsed time for c2c: " << elapsed_c2c.count() << "s" << std::endl;
  std::cout << "elapsed time for squaring: " << elapsed_square.count() << "s" << std::endl;

  return used_in_indexing;
}

