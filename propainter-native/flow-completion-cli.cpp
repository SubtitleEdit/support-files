#include "flow-completion.h"
#include "ggml.h"
#include "gguf.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

int main(int argc, char ** argv) {
    using namespace propainter;
    try {
        if (argc != 7)
            throw std::runtime_error("Usage: propainter-complete-flow CHECKPOINT WIDTH HEIGHT FRAMES INPUT OUTPUT");
        const int w = std::stoi(argv[2]), h = std::stoi(argv[3]), n = std::stoi(argv[4]);
        if (w < 8 || h < 8 || w > 2048 || h > 2048 || w % 8 || h % 8 || n < 2 || n > 128)
            throw std::runtime_error("Invalid input dimensions or frame count");
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(argv[1], {false, &raw}), &gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> model(raw, &ggml_free);
        if (!metadata || !model) throw std::runtime_error("Cannot load flow completion checkpoint");
        const auto arch = gguf_find_key(metadata.get(), "general.architecture");
        if (arch < 0 || gguf_get_kv_type(metadata.get(), arch) != GGUF_TYPE_STRING ||
            std::string_view(gguf_get_val_str(metadata.get(), arch)) != "propainter-flow-completion")
            throw std::runtime_error("Expected a ProPainter flow completion checkpoint");
        std::ifstream input(argv[5], std::ios::binary);
        const auto read = [&](int frames, int channels) {
            Sequence sequence;
            for (int t = 0; t < frames; ++t) {
                auto & frame = sequence.emplace_back(FeatureMap{w, h, channels, std::vector<float>(size_t(w) * h * channels)});
                if (!input.read(reinterpret_cast<char *>(frame.data.data()), frame.data.size() * sizeof(float)))
                    throw std::runtime_error("Expected planar float32 flow and mask data");
            }
            return sequence;
        };
        BidirectionalFlow flows{read(n - 1, 2), read(n - 1, 2)};
        const auto masks = read(n, 1);
        if (input.peek() != std::istream::traits_type::eof())
            throw std::runtime_error("Unexpected trailing input data");
        const auto result = complete_flow(model.get(), flows, masks);
        std::ofstream output(argv[6], std::ios::binary);
        for (const auto * sequence : {&result.forward, &result.backward})
            for (const auto & frame : *sequence)
                output.write(reinterpret_cast<const char *>(frame.data.data()), frame.data.size() * sizeof(float));
        if (!output) throw std::runtime_error("Cannot write completed flows");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
