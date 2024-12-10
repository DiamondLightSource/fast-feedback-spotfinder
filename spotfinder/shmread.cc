

#include "shmread.hpp"

#include <fmt/core.h>

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

#include "cuda_common.hpp"

using json = nlohmann::json;
using namespace fmt;

SHMRead::SHMRead(const std::string &path) : _base_path(path) {
    // Read the header
    auto header_path = path + "/start_1";
    std::ifstream f(header_path);
    json data = json::parse(f);

    _num_images =
      data["nimages"].template get<size_t>() * data["ntrigger"].template get<size_t>();
    _image_shape = {
      data["y_pixels_in_detector"].template get<size_t>(),
      data["x_pixels_in_detector"].template get<size_t>(),
    };

    uint8_t bit_depth_image = data["bit_depth_image"].template get<uint8_t>();

    if (bit_depth_image != 16) {
        throw std::runtime_error(format(
          "Can not read image with bit_depth_image={}, only 16", bit_depth_image));
    }
    _trusted_range = {
      0, data["countrate_correction_count_cutoff"].template get<image_t_type>()};

    if (data.contains("wavelength")) {
        _wavelength = data["wavelength"].template get<float>();
    } else {
        _wavelength = std::nullopt;
    }

    _detector_distance = data["detector_distance"].template get<float>() / 1000;
    _pixel_size = {data["y_pixel_size"].template get<float>(),
                   data["x_pixel_size"].template get<float>()};
    _beam_center = {data["beam_center_y"].template get<float>(),
                    data["beam_center_x"].template get<float>()};

    if (data.contains("omega_start") && data.contains("omega_increment")) {
        _oscillation = {data["omega_start"].template get<float>(),
                        data["omega_increment"].template get<float>()};
    } else {
        _oscillation = std::nullopt;
    }

    // Read the mask
    std::vector<int32_t> raw_mask;
    raw_mask.resize(_image_shape[0] * _image_shape[1]);
    auto mask_filename = format("{}/start_5", _base_path);
    if (std::filesystem::file_size(mask_filename)
        != raw_mask.size() * sizeof(decltype(raw_mask)::value_type)) {
        throw std::runtime_error("Error: Mask file does not match expected size");
    }
    std::ifstream f_mask(mask_filename, std::ios::in | std::ios::binary);
    f_mask.read(reinterpret_cast<char *>(raw_mask.data()),
                raw_mask.size() * sizeof(decltype(raw_mask)::value_type));
    // draw_image_data(raw_mask.data(), 0, 0, 20, 20, _image_shape[1], _image_shape[0]);
    _mask.reserve(_image_shape[0] * _image_shape[1]);
    for (auto &v : raw_mask) {
        _mask.push_back(!v);
    }
    // return {destination.data(), static_cast<size_t>(f.gcount())};
}

bool SHMRead::is_image_available(size_t index) {
    return std::filesystem::exists(format("{}/image_{:06d}_2", _base_path, index));
}

std::span<uint8_t> SHMRead::get_raw_chunk(size_t index,
                                          std::span<uint8_t> destination) {
    std::ifstream f(format("{}/image_{:06d}_2", _base_path, index),
                    std::ios::in | std::ios::binary);
    f.read(reinterpret_cast<char *>(destination.data()), destination.size());
    return {destination.data(), static_cast<size_t>(f.gcount())};
}

template <>
bool is_ready_for_read<SHMRead>(const std::string &path) {
    // We need headers.1, and headers.5, to read the metadata
    return std::filesystem::exists(format("{}/start_1", path))
           && std::filesystem::exists(format("{}/start_4", path));
}
