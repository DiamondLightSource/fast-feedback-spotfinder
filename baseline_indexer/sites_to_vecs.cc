#include <dx2/utils.h>
#include <math.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <tuple>

using Eigen::Vector3d;

#define _USE_MATH_DEFINES
#include <cmath>

// Define a few simple data structures to help with the calculations below.
class VectorGroup {
  public:
    void add(Vector3d vec, int weight) {
        vectors.push_back(vec);
        weights.push_back(weight);
    }
    Vector3d mean() const {
        Vector3d sum =
          std::accumulate(vectors.begin(), vectors.end(), Vector3d{0, 0, 0});
        return sum / static_cast<double>(vectors.size());
    }
    std::vector<Vector3d> vectors{};
    std::vector<int> weights{};
};

struct SiteData {
    Vector3d site;
    double length;
    int volume;
};
bool compare_site_data(const SiteData& a, const SiteData& b) {
    return a.length < b.length;
}
bool compare_site_data_volume(const SiteData& a, const SiteData& b) {
    return a.volume > b.volume;
}

bool is_approximate_integer_multiple(Vector3d v1,
                                     Vector3d v2,
                                     double relative_length_tolerance = 0.2,
                                     double angular_tolerance = 5.0) {
    double angle = angle_between_vectors_degrees(v1, v2);
    if ((angle < angular_tolerance) || (std::abs(180 - angle) < angular_tolerance)) {
        double l1 = v1.norm();
        double l2 = v2.norm();
        if (l1 > l2) {
            std::swap(l1, l2);
        }
        double n = l2 / l1;
        if (std::abs(std::round(n) - n) < relative_length_tolerance) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Rescale the fractional coordinates on the grid to vectors (in reciprocal space).
 * @param centres_of_mass_frac The fractional centres of mass of FFT peaks.
 * @param grid_points_per_void The number of FFT grid points corresponding to each peak.
 * @param d_min The resolution limit that was applied to the FFT.
 * @param min_cell Don't consider vectors below this length
 * @param max_cell Don't consider vectors above this length
 * @param n_points The size of each dimension of the FFT grid.
 * @returns Unique vectors, sorted by volume, that give describe the FFT peaks.
 */
std::vector<Vector3d> sites_to_vecs(std::vector<Vector3d> centres_of_mass_frac,
                                    std::vector<int> grid_points_per_void,
                                    double d_min,
                                    double min_cell = 3.0,
                                    double max_cell = 92.3,
                                    uint32_t n_points = 256) {
    auto start = std::chrono::system_clock::now();
    // Calculate the scaling between the FFT grid and reciprocal space.
    double fft_cell_length = n_points * d_min / 2.0;
    // Use 'sites_mod_short' and convert to cartesian (but keep in same array)
    for (Vector3d& vec : centres_of_mass_frac) {
        // Apply the scaling across the vector
        std::transform(
          vec.begin(), vec.end(), vec.begin(), [fft_cell_length](double& val) {
              if (val > 0.5) val -= 1.0;
              return val * fft_cell_length;
          });
    }

    // now do some filtering based on the min and max cell
    std::vector<SiteData> filtered_data;
    for (int i = 0; i < centres_of_mass_frac.size(); ++i) {
        double length = centres_of_mass_frac[i].norm();
        if (length > min_cell && length < 2 * max_cell) {
            filtered_data.push_back(
              {centres_of_mass_frac[i], length, grid_points_per_void[i]});
        }
    }

    // Now sort the filtered data. Ggroup together those
    // with similar angles and lengths (e.g. inverse pairs from the FFT).
    double relative_length_tolerance = 0.1;
    double angular_tolerance = 5.0;
    std::vector<VectorGroup> vector_groups{};
    for (const SiteData& data : filtered_data) {
        bool matched_group = false;
        for (VectorGroup& group : vector_groups) {
            Vector3d mean_v = group.mean();
            double mean_v_length = mean_v.norm();
            if ((std::abs(mean_v_length - data.length)
                 / std::max(mean_v_length, data.length))
                < relative_length_tolerance) {
                double angle = angle_between_vectors_degrees(mean_v, data.site);
                if (angle < angular_tolerance) {
                    group.add(data.site, data.volume);
                    matched_group = true;
                    break;
                } else if (std::abs(180 - angle) < angular_tolerance) {
                    group.add(-1.0 * data.site, data.volume);
                    matched_group = true;
                    break;
                }
            }
        }
        // If it didn't match any existing group, create a new one.
        if (!matched_group) {
            VectorGroup group;
            group.add(data.site, data.volume);
            vector_groups.push_back(group);
        }
    }

    // Create "site"s based on the data from the groups.
    std::vector<SiteData> grouped_data;
    for (const VectorGroup& group : vector_groups) {
        Vector3d site = group.mean();
        int max = *std::max_element(group.weights.begin(), group.weights.end());
        grouped_data.push_back({site, site.norm(), max});
    }

    // Sort by volume, then by length.
    std::stable_sort(
      grouped_data.begin(), grouped_data.end(), compare_site_data_volume);
    std::stable_sort(grouped_data.begin(), grouped_data.end(), compare_site_data);

    // Now check if any sites are integer multiples of other sites.
    std::vector<SiteData> unique_sites;
    for (const SiteData& data : grouped_data) {
        bool is_unique = true;
        const Vector3d& v = data.site;
        for (const SiteData& unique_site : unique_sites) {
            // If the volume of the unique site is less than the current site, skip
            if (unique_site.volume <= data.volume) {
                continue;
            }
            // If the current site is an integer multiple of the unique site, exit
            if (is_approximate_integer_multiple(unique_site.site, v)) {
                spdlog::info("rejecting {0} : is integer multiple of {1}",
                             v.norm(),
                             unique_site.site.norm());
                is_unique = false;
                break;
            }
        }
        if (is_unique) {
            unique_sites.push_back({v, v.norm(), data.volume});
        }
    }
    // now sort by peak volume again
    std::stable_sort(
      unique_sites.begin(), unique_sites.end(), compare_site_data_volume);
    std::vector<Vector3d> unique_vectors_sorted;
    spdlog::info("Candidate basis vectors:");
    for (int i = 0; i < unique_sites.size(); i++) {
        unique_vectors_sorted.push_back(unique_sites[i].site);
        spdlog::info(
          "{0} {1:.5f} {2}", i, unique_sites[i].length, unique_sites[i].volume);
    }

    std::chrono::duration<double> elapsed_seconds =
      std::chrono::system_clock::now() - start;
    spdlog::debug("elapsed time for sites_to_vecs: {0:.5f}s", elapsed_seconds.count());
    return unique_vectors_sorted;
}
