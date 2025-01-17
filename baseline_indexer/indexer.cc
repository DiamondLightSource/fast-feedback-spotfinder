#include <dx2/beam.h>
#include <dx2/crystal.h>
#include <dx2/detector.h>
#include <dx2/goniometer.h>
#include <dx2/h5read_processed.h>
#include <dx2/scan.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/os.h>

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "fft3d.cc"
#include "flood_fill.cc"
#include "gemmi/symmetry.hpp"
#include "sites_to_vecs.cc"
#include "xyz_to_rlp.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using json = nlohmann::json;

int main(int argc, char** argv) {
    // The purpose of an indexer is to determine the lattice model that best
    // explains the positions of the strong spots found during spot-finding.
    // The lattice model is a set of three vectors that define the crystal
    // lattice translations.
    // The experiment models (beam, detector) can also be refined during the
    // indexing process. The output is a set of models - a new crystal model that
    // describes the crystal lattice and an updated set of experiment models.
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    parser.add_argument("-e", "--expt").help("Path to the DIALS expt file");
    parser.add_argument("-r", "--refl")
      .help("Path to the h5 reflection table file containing spotfinding results");
    parser.add_argument("--dmin")
      .help("The resolution limit of spots to use in the indexing process.")
      .scan<'f', float>();
    parser.add_argument("--max-cell")
      .help("The maximum possible cell length to consider during indexing")
      .scan<'f', float>();
    parser
      .add_argument(
        "--fft-npoints")  // mainly for testing, likely would always want to keep it as 256.
      .help(
        "The number of grid points to use for the fft. Powers of two are most "
        "efficient.")
      .default_value<uint32_t>(256)
      .scan<'u', uint32_t>();
    parser.parse_args(argc, argv);

    if (!parser.is_used("--expt")) {
        fmt::print("Error: must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("--refl")) {
        fmt::print(
          "Error: must specify spotfinding results file (in DIALS HDF5 format) with "
          "--refl\n");
        std::exit(1);
    }
    // In DIALS, the max cell is automatically determined through a nearest
    // neighbour analysis that requires the annlib package. For now,
    // let's make this a required argument to help with testing/comparison
    // to DIALS.
    if (!parser.is_used("--max-cell")) {
        fmt::print("Error: must specify --max-cell\n");
        std::exit(1);
    }
    // FIXME use highest resolution by default to remove this requirement.
    if (!parser.is_used("--dmin")) {
        fmt::print("Error: must specify --dmin\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("--expt");
    std::string filename = parser.get<std::string>("--refl");
    double max_cell = parser.get<float>("max-cell");
    double d_min = parser.get<float>("dmin");

    // Parse the experiment list (a json file) and load the models.
    // Will be moved to dx2.
    std::ifstream f(imported_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error& ex) {
        std::cerr << "Error: Unable to read " << imported_expt.c_str()
                  << "; json parse error at byte " << ex.byte << std::endl;
        std::exit(1);
    }

    // Load the models
    json beam_data = elist_json_obj["beam"][0];
    MonoXrayBeam beam(beam_data);
    json scan_data = elist_json_obj["scan"][0];
    Scan scan(scan_data);
    json gonio_data = elist_json_obj["goniometer"][0];
    Goniometer gonio(gonio_data);
    json panel_data = elist_json_obj["detector"][0]["panels"][0];
    Panel detector(panel_data);

    // Read data from a reflection table. Again, this should be moved to
    // dx2 and only require the data array name (xyzobs.px.value) with some
    // logic to step through the directory structure
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    // Note, xyzobs_px is the flattened, on-disk representation of the array
    // i.e. if there are 100 spots, the length of xyzobs_px is 300, and
    // contains the elements [x0, y0, z0, x1, y1, z1, ..., x99, y99, z99]
    std::vector<double> xyzobs_px =
      read_array_from_h5_file<double>(filename, array_name);

    // The diffraction spots form a lattice in reciprocal space (if the experimental
    // geometry is accurate). So use the experimental models to transform the spot
    // coordinates on the detector into reciprocal space.
    std::vector<Vector3d> rlp = xyz_to_rlp(xyzobs_px, detector, beam, scan, gonio);
    std::cout << "Number of reflections: " << rlp.size() << std::endl;

    // b_iso is an isotropic b-factor used to weight the points when doing the fft.
    // i.e. high resolution (weaker) spots are downweighted by the expected
    // intensity fall-off as as function of resolution.
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    uint32_t n_points = parser.get<uint32_t>("--fft-npoints");
    std::cout << "Setting b_iso = " << b_iso << std::endl;

    // Create an array to store the fft result. This is a 3D grid of points, typically 256^3.
    std::vector<double> real_fft_result(n_points * n_points * n_points, 0.0);

    // Do the fft of the reciprocal lattice coordinates.
    // the used in indexing array denotes whether a coordinate was used for the
    // fft (might not be if dmin filter was used for example). The used_in_indexing array
    // is sometimes used onwards in the dials indexing algorithms, so keep for now.
    std::vector<bool> used_in_indexing =
      fft3d(rlp, real_fft_result, d_min, b_iso, n_points);

    // The fft result is noisy. We want to extract the peaks, which may be spread over several
    // points on the fft grid. So we use a flood fill algorithm (https://en.wikipedia.org/wiki/Flood_fill)
    // to determine the connected regions in 3D. This is how it is done in DIALS, but I note that
    // perhaps this could be done with connected components analysis.
    // So do the flood fill, and extract the centres of mass of the peaks and the number of grid points
    // that contribute to each peak.
    std::vector<int> grid_points_per_void;
    std::vector<Vector3d> centres_of_mass_frac;
    // 15.0 is the DIALS 'rmsd_cutoff' parameter to filter out weak peaks.
    std::tie(grid_points_per_void, centres_of_mass_frac) =
      flood_fill(real_fft_result, 15.0, n_points);
    // Do some further filtering, 0.15 is the DIALS peak_volume_cutoff parameter.
    std::tie(grid_points_per_void, centres_of_mass_frac) =
      flood_fill_filter(grid_points_per_void, centres_of_mass_frac, 0.15);

    // Convert the peak centres from the fft grid into vectors in reciprocal space. These are our candidate
    // lattice vectors.
    // 3.0 is the min cell parameter.
    std::vector<Vector3d> candidate_lattice_vectors = sites_to_vecs(
      centres_of_mass_frac, grid_points_per_void, d_min, 3.0, max_cell, n_points);

    // at this point, we will test combinations of the candidate vectors, use those to index the spots, do some
    // refinement of the candidates and choose the best one. Then we will do some more refinement including extra
    // model parameters. At then end, we will have a list of refined experiment models (including a crystal)

    // For now, let's just write out the candidate vectors and write out the unrefined experiment models with the
    // first combination of candidate vectors as an example crystal, to demonstrate an example experiment list data
    // structure.

    // dump the candidate vectors to json
    std::string n_vecs = std::to_string(candidate_lattice_vectors.size() - 1);
    size_t n_zero = n_vecs.length();
    json vecs_out;
    for (int i = 0; i < candidate_lattice_vectors.size(); i++) {
        std::string s = std::to_string(i);
        auto pad_s = std::string(n_zero - std::min(n_zero, s.length()), '0') + s;
        vecs_out[pad_s] = candidate_lattice_vectors[i];
    }
    std::string outfile = "candidate_vectors.json";
    std::cout << "Saving candidate vectors to " << outfile << std::endl;
    std::ofstream vecs_file(outfile);
    vecs_file << vecs_out.dump(4);

    // Now make a crystal and save an experiment list with the models.
    if (candidate_lattice_vectors.size() < 3) {
        std::cout << "Insufficient number of candidate vectors to make a crystal model."
                  << std::endl;
    } else {
        gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
        Crystal best_xtal{candidate_lattice_vectors[0],
                          candidate_lattice_vectors[1],
                          candidate_lattice_vectors[2],
                          space_group};
        json cryst_out = best_xtal.to_json();

        // save an example experiment list
        json elist_out;  // a list of potentially multiple experiments
        elist_out["__id__"] = "ExperimentList";
        json expt_out;  // our single experiment
        // no imageset (for now?).
        expt_out["__id__"] = "Experiment";
        expt_out["identifier"] = "test";
        expt_out["beam"] =
          0;  // the indices of the models that will correspond to our experiment
        expt_out["detector"] = 0;
        expt_out["goniometer"] = 0;
        expt_out["scan"] = 0;
        expt_out["crystal"] = 0;
        elist_out["experiment"] = std::array<json, 1>{expt_out};
        elist_out["crystal"] =
          std::array<json, 1>{cryst_out};  // add the the actual models
        elist_out["scan"] = std::array<json, 1>{scan.to_json()};
        elist_out["goniometer"] = std::array<json, 1>{gonio.to_json()};
        elist_out["beam"] = std::array<json, 1>{beam.to_json()};
        elist_out["detector"] = std::array<json, 1>{detector.to_json()};

        std::ofstream efile("elist.json");
        efile << elist_out.dump(4);
    }

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for indexer: " << elapsed_time.count() << "s" << std::endl;
}