#include "raft-correlation.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace {
float bilinear(const float * values, int width, int height, float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y) || x <= -1 || y <= -1 || x >= width || y >= height) {
        return 0.0f;
    }
    const auto ix = static_cast<int>(std::floor(x));
    const auto iy = static_cast<int>(std::floor(y));
    const float fx = x - ix;
    const float fy = y - iy;
    const auto get = [&](int xx, int yy) {
        return xx >= 0 && xx < width && yy >= 0 && yy < height ? values[yy * width + xx] : 0.0f;
    };
    return (1 - fy) * ((1 - fx) * get(ix, iy) + fx * get(ix + 1, iy))
        + fy * ((1 - fx) * get(ix, iy + 1) + fx * get(ix + 1, iy + 1));
}

}

RaftCorrelation::RaftCorrelation(const std::vector<float> & input1, const std::vector<float> & input2,
                                 int channels, int height, int width) : pixels_(0) {
    if (channels < 1 || channels > 512 || height < 16 || width < 16 || height > 80 || width > 80) {
        throw std::runtime_error("Unsupported RAFT correlation dimensions");
    }
    const int pixels = pixels_ = width * height;
    if (input1.size() != static_cast<size_t>(channels) * pixels || input2.size() != input1.size()) {
        throw std::runtime_error("Invalid RAFT feature buffers");
    }

    const size_t data_bytes = (2ull * channels * pixels + 1ull * pixels * pixels) * sizeof(float);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({data_bytes + 4 * 1024 * 1024, nullptr, false}), &ggml_free);
    if (!ctx) {
        throw std::runtime_error("Cannot allocate the RAFT correlation graph");
    }
    auto * first = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, pixels);
    auto * second = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, pixels);
    auto * first_data = static_cast<float *>(first->data);
    auto * second_data = static_cast<float *>(second->data);
    for (int p = 0; p < pixels; ++p) {
        for (int ch = 0; ch < channels; ++ch) {
            first_data[p * channels + ch] = input1[ch * pixels + p];
            second_data[p * channels + ch] = input2[ch * pixels + p];
        }
    }
    auto * products = ggml_mul_mat(ctx.get(), second, first);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, products);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 4) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("RAFT all-pairs matrix multiplication failed");
    }


    widths_[0] = width;
    heights_[0] = height;
    const auto * data = static_cast<const float *>(products->data);
    const float scale = 1.0f / std::sqrt(static_cast<float>(channels));
    maps_[0].resize(static_cast<size_t>(pixels) * pixels);
    for (size_t i = 0; i < maps_[0].size(); ++i) {
        maps_[0][i] = data[i] * scale;
    }
    for (int level = 1; level < levels; ++level) {
        const int old_w = widths_[level - 1], old_h = heights_[level - 1];
        const int new_w = widths_[level] = old_w / 2;
        const int new_h = heights_[level] = old_h / 2;
        auto & current = maps_[level];
        const auto & previous = maps_[level - 1];
        current.resize(static_cast<size_t>(pixels) * new_w * new_h);
        for (int p = 0; p < pixels; ++p) {
            const float * source = previous.data() + static_cast<size_t>(p) * old_w * old_h;
            float * output = current.data() + static_cast<size_t>(p) * new_w * new_h;
            for (int y = 0; y < new_h; ++y) {
                for (int x = 0; x < new_w; ++x) {
                    const int top = 2 * y * old_w + 2 * x;
                    output[y * new_w + x] = 0.25f *
                        (source[top] + source[top + 1] + source[top + old_w] + source[top + old_w + 1]);
                }
            }
        }
    }
}

std::vector<float> RaftCorrelation::lookup(const std::vector<float> & coords) const {
    if (coords.size() != static_cast<size_t>(2 * pixels_)) {
        throw std::runtime_error("Invalid RAFT lookup coordinates");
    }
    std::vector<float> result(output_channels * pixels_);
    for (int level = 0; level < levels; ++level) {
        const int width = widths_[level], height = heights_[level];
        const float scale = 1.0f / (1 << level);
        for (int p = 0; p < pixels_; ++p) {
            const float x = coords[p] * scale;
            const float y = coords[pixels_ + p] * scale;
            const float * map = maps_[level].data() + static_cast<size_t>(p) * width * height;
            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int feature = level * window * window + (dx + radius) * window + dy + radius;
                    result[feature * pixels_ + p] = bilinear(map, width, height, x + dx, y + dy);
                }
            }
        }
    }
    return result;
}
