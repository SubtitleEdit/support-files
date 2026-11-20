#include "renderer.h"
#include "generator.h"
#include "raft.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {
using namespace propainter;
std::vector<uint8_t> read_bytes(std::istream & stream, size_t count) {
    std::vector<uint8_t> bytes(count);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), count))
        throw std::runtime_error("Input shorter than declared");
    return bytes;
}
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--stream") {
            const std::string size = argv[2];
            const auto split = size.find('x');
            if (split == std::string::npos) throw std::runtime_error("Expected WIDTHxHEIGHT");
            const std::string width = size.substr(0, split), height = size.substr(split + 1);
            size_t width_end, height_end;
            const int w = std::stoi(width, &width_end), h = std::stoi(height, &height_end);
            if (width_end != width.size() || height_end != height.size() || w < 1 || h < 1 || w > 8192 || h > 8192)
                throw std::runtime_error("Invalid stream dimensions");
            run_stream(std::filesystem::canonical(argv[0]).parent_path() / "models", w, h);
            return 0;
        }
        if (argc < 8 || argc > 13)
            throw std::runtime_error("Usage: propainter-native MODEL_DIR WIDTH HEIGHT FRAMES RGB24 MASK8 OUTPUT_RGB24 "
                                     "[RAFT_ITERATIONS [MASK_DILATION [NEIGHBOR_LENGTH [REFERENCE_STRIDE [SUBVIDEO_LENGTH]]]]]");
        const int w = std::stoi(argv[2]), h = std::stoi(argv[3]), n = std::stoi(argv[4]);
        const int iters = argc > 8 ? std::stoi(argv[8]) : 20;
        const int dilation = argc > 9 ? std::stoi(argv[9]) : 4;
        const int neighbors = argc > 10 ? std::stoi(argv[10]) : 10;
        const int ref_stride = argc > 11 ? std::stoi(argv[11]) : 10;
        const int subvideo = argc > 12 ? std::stoi(argv[12]) : 80;
        if (w < 8 || h < 8 || w > 640 || h > 640 || w % 8 || h % 8 || n < 1 ||
            iters < 1 || iters > 32 || dilation < 0 || dilation > 64 ||
            neighbors < 2 || neighbors > 24 || ref_stride < 1 || subvideo < 1 || n > 1000 ||
            size_t(w) * h * n > 40000000)
            throw std::runtime_error("Invalid dimensions, frame count, or inference settings");
        std::ifstream rgb(argv[5], std::ios::binary), mask(argv[6], std::ios::binary);
        Sequence images, masks;
        std::vector<std::vector<uint8_t>> pixels;
        const size_t cells = size_t(w) * h;
        for (int t = 0; t < n; ++t) {
            auto bytes = read_bytes(rgb, 3 * cells);
            images.push_back(normalized_rgb(bytes, w, h));
            masks.push_back(dilated_mask(read_bytes(mask, cells), w, h, dilation));
            pixels.push_back(std::move(bytes));
        }
        if (rgb.peek() != std::istream::traits_type::eof() || mask.peek() != std::istream::traits_type::eof())
            throw std::runtime_error("Unexpected trailing input data");
        const auto result = render(argv[1], images, masks, std::move(pixels), iters, neighbors / 2, ref_stride, subvideo);
        std::ofstream output(argv[7], std::ios::binary);
        for (const auto & frame : result)
            output.write(reinterpret_cast<const char *>(frame.data()), frame.size());
        if (!output) throw std::runtime_error("Cannot write inpainted frames");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
