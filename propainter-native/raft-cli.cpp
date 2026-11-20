#include "raft.h"
#include "ggml.h"
#include "gguf.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

using namespace propainter;

int main(int argc, char ** argv) {
    try {
        if (argc != 7 && argc != 8) {
            throw std::runtime_error("Usage: propainter-raft CHECKPOINT fnet|cnet|flow WIDTH HEIGHT INPUT OUTPUT [ITERATIONS]");
        }
        const std::string_view mode = argv[2];
        if ((mode != "fnet" && mode != "cnet" && mode != "flow") || (argc == 8 && mode != "flow")) {
            throw std::runtime_error("Choose fnet, cnet, or flow");
        }
        const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
        if (width < 1 || height < 1 || width > 640 || height > 640 ||
            (mode != "flow" && (width < 128 || height < 128 || width % 8 || height % 8))) {
            throw std::runtime_error("Invalid RAFT frame size");
        }
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(argv[1], {false, &raw}), &gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> model(raw, &ggml_free);
        if (!metadata || !model) {
            throw std::runtime_error("Cannot load the RAFT checkpoint");
        }
        const auto arch = gguf_find_key(metadata.get(), "general.architecture");
        if (arch < 0 || gguf_get_kv_type(metadata.get(), arch) != GGUF_TYPE_STRING ||
            std::string_view(gguf_get_val_str(metadata.get(), arch)) != "propainter-raft") {
            throw std::runtime_error("Expected a ProPainter RAFT checkpoint");
        }

        FeatureMap image{width, height, 3, std::vector<float>(3ull * width * height)};
        std::ifstream input(argv[5], std::ios::binary);
        const auto read_image = [&](FeatureMap & frame) {
            if (!input.read(reinterpret_cast<char *>(frame.data.data()), frame.data.size() * sizeof(float))) {
                throw std::runtime_error("Expected normalized float32 RGB in channel-first order");
            }
        };
        read_image(image);
        FeatureMap result;
        if (mode == "flow") {
            const int iterations = argc == 8 ? std::stoi(argv[7]) : 12;
            if (iterations < 1 || iterations > 32) {
                throw std::runtime_error("Use between 1 and 32 flow iterations");
            }
            FeatureMap second{width, height, 3, std::vector<float>(3ull * width * height)};
            read_image(second);
            result = raft_flow(model.get(), image, second, iterations);
        } else {
            result = raft_encode(model.get(), image, mode == "cnet");
        }
        if (input.peek() != std::istream::traits_type::eof()) {
            throw std::runtime_error("Unexpected trailing input data");
        }
        std::ofstream output(argv[6], std::ios::binary);
        if (!output.write(reinterpret_cast<const char *>(result.data.data()), result.data.size() * sizeof(float))) {
            throw std::runtime_error("Cannot write RAFT encoder output");
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
