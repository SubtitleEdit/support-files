#include "flow-completion.h"
#include "ggml.h"

#include <algorithm>
#include <stdexcept>

namespace propainter {
namespace {
FeatureMap replicate_pad(const FeatureMap & input, int padding) {
    FeatureMap output{input.width + 2 * padding, input.height + 2 * padding, input.channels, {}};
    output.data.resize(output.pixels() * output.channels);
    for (int c = 0; c < input.channels; ++c) {
        for (int y = 0; y < output.height; ++y) {
            const int yy = std::clamp(y - padding, 0, input.height - 1);
            for (int x = 0; x < output.width; ++x) {
                const int xx = std::clamp(x - padding, 0, input.width - 1);
                output.data[c * output.pixels() + y * output.width + x] =
                    input.data[c * input.pixels() + yy * input.width + xx];
            }
        }
    }
    return output;
}

Sequence temporal_convolution(ggml_context * model, const std::string & prefix, const Sequence & input) {
    const auto & frame = input.front();
    const int cells = frame.pixels(), channels = frame.channels, inner = channels * 3;
    const auto & weight = tensor(model, prefix + ".weight");
    if (ggml_nelements(&weight) != int64_t(inner) * channels)
        throw std::runtime_error("Invalid temporal convolution kernel");
    const auto * bias = vector(model, prefix + ".bias", channels);
    Sequence output;
    std::vector<float> columns(size_t(cells) * inner);
    std::vector<float> product(size_t(cells) * channels);
    for (int t = 0; t < static_cast<int>(input.size()); ++t) {
        for (int p = 0; p < cells; ++p) {
            for (int c = 0; c < channels; ++c) {
                for (int k = 0; k < 3; ++k) {
                    const int source = t + 2 * (k - 1);
                    columns[size_t(p) * inner + c * 3 + k] =
                        source >= 0 && source < static_cast<int>(input.size()) ?
                        input[source].data[c * cells + p] : 0;
                }
            }
        }
        multiply(static_cast<const float *>(weight.data), columns.data(), product.data(), inner, cells, channels);
        auto & result = output.emplace_back(FeatureMap{frame.width, frame.height, channels, std::vector<float>(product.size())});
        for (int c = 0; c < channels; ++c)
            for (int p = 0; p < cells; ++p)
                result.data[c * cells + p] = product[p * channels + c] + bias[c];
        leaky_relu(result, 0.2f);
    }
    return output;
}

Sequence encode(ggml_context * model, const std::string & prefix, Sequence input) {
    for (int layer : {0, 2}) {
        const auto name = prefix + "." + std::to_string(layer);
        for (auto & frame : input) {
            frame = convolution(model, name + ".conv1.0", frame, layer == 0 ? 1 : 2, 1);
            leaky_relu(frame, 0.2f);
        }
        input = temporal_convolution(model, name + ".conv2.0", input);
    }
    return input;
}

void propagate(ggml_context * model, Sequence & spatial) {
    const auto & first = spatial.front();
    const FeatureMap zero{first.width, first.height, first.channels, std::vector<float>(first.data.size())};
    const int frames = spatial.size();
    Sequence backward(frames), forward(frames);
    const std::string base = "feat_prop_module";
    for (bool reverse : {true, false}) {
        auto & sequence = reverse ? backward : forward;
        const int step = reverse ? -1 : 1;
        const std::string direction = reverse ? "backward_" : "forward_";
        const auto backbone = base + ".backbone." + direction;
        FeatureMap propagated = zero;
        for (int i = 0; i < frames; ++i) {
            const int t = reverse ? frames - 1 - i : i;
            if (i > 0) {
                const auto & second = i > 1 ? sequence[t - 2 * step] : zero;
                const auto condition = concatenate(concatenate(propagated, spatial[t]), second);
                propagated = deform(model, base + ".deform_align." + direction,
                    concatenate(propagated, second), condition, 5);
            }
            auto residual = convolution(model, backbone + ".0",
                concatenate(reverse ? spatial[t] : concatenate(spatial[t], backward[t]), propagated), 1, 1);
            leaky_relu(residual, 0.1f);
            add(propagated, convolution(model, backbone + ".2", residual, 1, 1));
            sequence[t] = propagated;
        }
    }
    for (int t = 0; t < frames; ++t)
        add(spatial[t], convolution(model, base + ".fusion", concatenate(backward[t], forward[t]), 1, 0));
}

Sequence complete_direction(ggml_context * model, const Sequence & flows, const Sequence & masks, bool reverse) {
    const int frames = flows.size();
    Sequence input;
    for (int i = 0; i < frames; ++i) {
        const int t = reverse ? frames - 1 - i : i;
        const auto & mask = masks[t + int(reverse)];
        auto flow = flows[t];
        for (size_t j = 0; j < flow.data.size(); ++j)
            flow.data[j] *= 1 - mask.data[j % mask.pixels()];
        auto down = convolution(model, "downsample.0", replicate_pad(concatenate(flow, mask), 2), 2, 0);
        leaky_relu(down, 0.2f);
        input.push_back(std::move(down));
    }
    auto skip = encode(model, "encoder1", std::move(input));
    auto mid = encode(model, "encoder2", skip);
    for (auto & frame : mid) {
        for (int dilation : {3, 2, 1}) {
            frame = convolution(model, "mid_dilation." + std::to_string(2 * (3 - dilation)), frame, 1,
                                dilation, dilation, dilation);
            leaky_relu(frame, 0.2f);
        }
    }
    propagate(model, mid);
    for (int i = 0; i < frames; ++i) {
        auto & frame = mid[i];
        for (const std::string name : {"decoder2", "decoder1", "upsample"}) {
            frame = convolution(model, name + ".0", frame, 1, 1);
            leaky_relu(frame, 0.2f);
            frame = convolution(model, name + ".2.conv", resize_twice(frame), 1, 1);
            if (name != "upsample") leaky_relu(frame, 0.2f);
            if (name == "decoder2") add(frame, skip[i]);
        }
        const int t = reverse ? frames - 1 - i : i;
        const auto & mask = masks[t + int(reverse)];
        for (size_t j = 0; j < frame.data.size(); ++j) {
            const float amount = mask.data[j % mask.pixels()];
            frame.data[j] = amount * frame.data[j] + (1 - amount) * flows[t].data[j];
        }
    }
    if (reverse) std::reverse(mid.begin(), mid.end());
    return mid;
}
}

BidirectionalFlow complete_flow(ggml_context * model, const BidirectionalFlow & input, const Sequence & masks) {
    if (masks.size() < 2 || input.forward.size() + 1 != masks.size() || input.backward.size() != input.forward.size())
        throw std::runtime_error("Invalid flow completion sequence length");
    const int width = masks[0].width, height = masks[0].height;
    if (width < 8 || height < 8 || width % 8 || height % 8)
        throw std::runtime_error("Flow completion requires dimensions divisible by eight");
    for (const auto * sequence : {&masks, &input.forward, &input.backward}) {
        const int channels = sequence == &masks ? 1 : 2;
        for (const auto & frame : *sequence)
            if (frame.width != width || frame.height != height || frame.channels != channels ||
                frame.data.size() != frame.pixels() * channels)
                throw std::runtime_error("Invalid flow completion frame shape");
    }
    return {complete_direction(model, input.forward, masks, false),
            complete_direction(model, input.backward, masks, true)};
}
}
