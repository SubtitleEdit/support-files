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
using Model = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

Model load_model(const std::filesystem::path & path, std::string_view architecture) {
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.string().c_str(), {false, &raw}), &gguf_free);
    Model model(raw, &ggml_free);
    if (!metadata || !model) throw std::runtime_error("Cannot load " + path.string());
    const auto arch = gguf_find_key(metadata.get(), "general.architecture");
    if (arch < 0 || gguf_get_kv_type(metadata.get(), arch) != GGUF_TYPE_STRING ||
        gguf_get_val_str(metadata.get(), arch) != architecture)
        throw std::runtime_error("Unexpected architecture in " + path.string());
    return model;
}


}

namespace propainter {
FeatureMap dilated_mask(const Bytes & bytes, int w, int h, int dilation) {
    FeatureMap mask{w, h, 1, std::vector<float>(size_t(w) * h)};
    std::transform(bytes.begin(), bytes.end(), mask.data.begin(), [](uint8_t value) { return value > 0; });
    for (int i = 0; i < dilation; ++i) {
        auto old = mask.data;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int p = y * w + x;
                mask.data[p] = old[p] || (x > 0 && old[p - 1]) || (x + 1 < w && old[p + 1]) ||
                               (y > 0 && old[p - w]) || (y + 1 < h && old[p + w]);
            }
    }
    return mask;
}

FeatureMap normalized_rgb(const Bytes & bytes, int w, int h) {
    const size_t cells = size_t(w) * h;
    FeatureMap image{w, h, 3, std::vector<float>(3 * cells)};
    for (size_t p = 0; p < cells; ++p)
        for (int c = 0; c < 3; ++c)
            image.data[c * cells + p] = bytes[p * 3 + c] / 127.5f - 1;
    return image;
}

namespace {
Sequence subset(const Sequence & sequence, int start, int end) {
    return {sequence.begin() + start, sequence.begin() + end};
}

BidirectionalFlow subset(const BidirectionalFlow & flows, int start, int end) {
    return {subset(flows.forward, start, end), subset(flows.backward, start, end)};
}

void append_crop(Sequence & target, Sequence & source, int start, int end) {
    target.insert(target.end(), std::make_move_iterator(source.begin() + start),
                  std::make_move_iterator(source.begin() + end));
}

}

std::vector<std::vector<uint8_t>> render(const std::filesystem::path & models, const Sequence & images,
                                         const Sequence & masks, std::vector<std::vector<uint8_t>> pixels,
                                         int iterations, int neighbor_stride, int ref_stride, int subvideo) {
    if (std::none_of(masks.begin(), masks.end(), [](const FeatureMap & mask) {
        return std::any_of(mask.data.begin(), mask.data.end(), [](float value) { return value != 0; });
    })) return pixels;
    const int frames = images.size();
    const int links = frames - 1;
    BidirectionalFlow flows;
    if (links > 0) {
        {
            auto raft = load_model(models / "raft-things.gguf", "propainter-raft");
            for (int i = 0; i < links; ++i) {
                flows.forward.push_back(raft_flow(raft.get(), images[i], images[i + 1], iterations));
                flows.backward.push_back(raft_flow(raft.get(), images[i + 1], images[i], iterations));
                std::cerr << "RAFT " << i + 1 << '/' << links << '\n';
            }
        }
        auto completion = load_model(models / "recurrent_flow_completion.gguf", "propainter-flow-completion");
        BidirectionalFlow completed;
        constexpr int context = 5;
        for (int cursor = 0; cursor < links; cursor += subvideo) {
            const int start = std::max(0, cursor - context);
            const int end = std::min(links, cursor + subvideo + context);
            auto section = complete_flow(completion.get(), subset(flows, start, end), subset(masks, start, end + 1));
            const int keep_end = std::min(links, cursor + subvideo) - start;
            append_crop(completed.forward, section.forward, cursor - start, keep_end);
            append_crop(completed.backward, section.backward, cursor - start, keep_end);
        }
        flows = std::move(completed);
        std::cerr << "Completed optical flow\n";
    }
    auto masked = images;
    for (int t = 0; t < frames; ++t)
        for (size_t p = 0; p < masked[t].data.size(); ++p)
            masked[t].data[p] *= 1 - masks[t].data[p % masks[t].pixels()];
    const int propagation_chunk = std::min(100, subvideo);
    PropagatedImages propagated;
    for (int cursor = 0; cursor < frames; cursor += propagation_chunk) {
        const int start = std::max(0, cursor - 10);
        const int end = std::min(frames, cursor + propagation_chunk + 10);
        auto section = propagate_images(subset(masked, start, end), subset(flows, start, end - 1),
                                        subset(masks, start, end));
        const int keep_end = std::min(frames, cursor + propagation_chunk) - start;
        append_crop(propagated.images, section.images, cursor - start, keep_end);
        append_crop(propagated.masks, section.masks, cursor - start, keep_end);
    }
    auto generator = load_model(models / "ProPainter.gguf", "propainter-generator");
    std::vector<bool> processed(frames);
    for (int center = 0; center < frames; center += neighbor_stride) {
        const int start = std::max(0, center - neighbor_stride);
        const int end = std::min(frames, center + neighbor_stride + 1);
        auto local_images = subset(propagated.images, start, end);
        auto local_masks = subset(masks, start, end);
        auto local_updated = subset(propagated.masks, start, end);
        const int radius = ref_stride * (subvideo / ref_stride / 2);
        const int ref_start = frames > subvideo ? std::max(0, center - radius) : 0;
        const int ref_end = frames > subvideo ? std::min(frames, center + radius) : frames;
        for (int index = ref_start; index < ref_end; index += ref_stride) {
            if (index >= start && index < end) continue;
            local_images.push_back(propagated.images[index]);
            local_masks.push_back(masks[index]);
            local_updated.push_back(propagated.masks[index]);
        }
        const auto patch = inpaint(generator.get(), local_images, subset(flows, start, end - 1),
                                   local_masks, local_updated, end - start);
        for (int t = start; t < end; ++t) {
            const auto & frame = patch[t - start];
            for (size_t p = 0; p < frame.pixels(); ++p) {
                if (masks[t].data[p] == 0) continue;
                for (int c = 0; c < 3; ++c) {
                    const float value = (frame.data[c * frame.pixels() + p] + 1) * 127.5f;
                    const auto predicted = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
                    auto & original = pixels[t][p * 3 + c];
                    original = processed[t] ? (int(original) + predicted) / 2 : predicted;
                }
            }
            processed[t] = true;
        }
        std::cerr << "Inpainted window at frame " << center + 1 << '/' << frames << '\n';
    }
    return pixels;
}
}
