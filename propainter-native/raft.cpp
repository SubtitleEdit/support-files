#include "raft.h"
#include "raft-correlation.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace propainter;

void normalize(ggml_context * model, const std::string & prefix, FeatureMap & x, bool batch) {
    const size_t pixels = x.pixels();
    constexpr float epsilon = 1e-5f;
    if (batch) {
        const auto * scale = vector(model, prefix + ".weight", x.channels);
        const auto * offset = vector(model, prefix + ".bias", x.channels);
        const auto * mean = vector(model, prefix + ".running_mean", x.channels);
        const auto * variance = vector(model, prefix + ".running_var", x.channels);
        for (int ch = 0; ch < x.channels; ++ch) {
            const float gain = scale[ch] / std::sqrt(variance[ch] + epsilon);
            auto * channel = x.data.data() + ch * pixels;
            for (size_t i = 0; i < pixels; ++i) {
                channel[i] = (channel[i] - mean[ch]) * gain + offset[ch];
            }
        }
        return;
    }
    for (int ch = 0; ch < x.channels; ++ch) {
        auto * channel = x.data.data() + ch * pixels;
        double sum = 0;
        for (size_t i = 0; i < pixels; ++i) {
            sum += channel[i];
        }
        const float mean = static_cast<float>(sum / pixels);
        double sum_squares = 0;
        for (size_t i = 0; i < pixels; ++i) {
            const double delta = channel[i] - mean;
            sum_squares += delta * delta;
        }
        const float inverse_stdev = 1.0f / std::sqrt(static_cast<float>(sum_squares / pixels) + epsilon);
        for (size_t i = 0; i < pixels; ++i) {
            channel[i] = (channel[i] - mean) * inverse_stdev;
        }
    }
}

void relu(FeatureMap & x) {
    for (auto & value : x.data) {
        value = std::max(value, 0.0f);
    }
}

FeatureMap residual(ggml_context * model, const std::string & prefix, FeatureMap source, int stride, bool batch) {
    auto y = convolution(model, prefix + ".conv1", source, stride, 1);
    normalize(model, prefix + ".norm1", y, batch);
    relu(y);
    y = convolution(model, prefix + ".conv2", y, 1, 1);
    normalize(model, prefix + ".norm2", y, batch);
    relu(y);
    if (stride == 2) {
        source = convolution(model, prefix + ".downsample.0", source, 2, 0);
        normalize(model, prefix + ".downsample.1", source, batch);
    }
    if (source.data.size() != y.data.size()) {
        throw std::runtime_error("RAFT residual shape mismatch");
    }
    for (size_t i = 0; i < y.data.size(); ++i) {
        y.data[i] = std::max(0.0f, source.data[i] + y.data[i]);
    }
    return y;
}

FeatureMap encode(ggml_context * model, const FeatureMap & image, bool batch) {
    const std::string name = batch ? "module.cnet" : "module.fnet";
    auto x = convolution(model, name + ".conv1", image, 2, 3);
    normalize(model, name + ".norm1", x, batch);
    relu(x);
    for (int layer = 1; layer <= 3; ++layer) {
        const auto prefix = name + ".layer" + std::to_string(layer);
        x = residual(model, prefix + ".0", std::move(x), layer == 1 ? 1 : 2, batch);
        x = residual(model, prefix + ".1", std::move(x), 1, batch);
    }
    return convolution(model, name + ".conv2", x, 1, 0);
}

FeatureMap slice(const FeatureMap & x, int first, int channels) {
    const auto begin = x.data.begin() + first * x.pixels();
    return {x.width, x.height, channels,
            std::vector<float>(begin, begin + channels * x.pixels())};
}

FeatureMap activated_convolution(ggml_context * model, const std::string & name, const FeatureMap & input, int padding) {
    auto output = convolution(model, name, input, 1, padding);
    relu(output);
    return output;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void gru(ggml_context * model, FeatureMap & hidden, const FeatureMap & input, bool vertical) {
    const std::string prefix = "module.update_block.gru.";
    const std::string suffix = vertical ? "2" : "1";
    const int padding_x = vertical ? 0 : 2, padding_y = vertical ? 2 : 0;
    const auto joined = concatenate(hidden, input);
    const auto update = convolution(model, prefix + "convz" + suffix, joined, 1, padding_x, padding_y);
    const auto reset = convolution(model, prefix + "convr" + suffix, joined, 1, padding_x, padding_y);
    auto masked = hidden;
    for (size_t i = 0; i < masked.data.size(); ++i) {
        masked.data[i] *= sigmoid(reset.data[i]);
    }
    const auto candidate = convolution(model, prefix + "convq" + suffix, concatenate(masked, input),
                                       1, padding_x, padding_y);
    for (size_t i = 0; i < hidden.data.size(); ++i) {
        const float z = sigmoid(update.data[i]);
        hidden.data[i] = (1 - z) * hidden.data[i] + z * std::tanh(candidate.data[i]);
    }
}

FeatureMap upsample(const FeatureMap & flow, const FeatureMap & mask) {
    const int width = flow.width, height = flow.height;
    const size_t pixels = flow.pixels();
    FeatureMap output{8 * width, 8 * height, 2, std::vector<float>(128 * pixels)};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t index = y * width + x;
            for (int dy = 0; dy < 8; ++dy) {
                for (int dx = 0; dx < 8; ++dx) {
                    float coefficients[9];
                    float largest = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < 9; ++k) {
                        coefficients[k] = mask.data[(k * 64 + 8 * dy + dx) * pixels + index] * 0.25f;
                        largest = std::max(largest, coefficients[k]);
                    }
                    float sum = 0;
                    for (auto & coefficient : coefficients) {
                        coefficient = std::exp(coefficient - largest);
                        sum += coefficient;
                    }
                    const size_t destination = (8 * y + dy) * output.width + (8 * x + dx);
                    for (int channel = 0; channel < 2; ++channel) {
                        float value = 0;
                        for (int k = 0; k < 9; ++k) {
                            const int sx = x + k % 3 - 1, sy = y + k / 3 - 1;
                            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                                value += (coefficients[k] / sum) * (8 * flow.data[channel * pixels + sy * width + sx]);
                            }
                        }
                        output.data[channel * output.pixels() + destination] = value;
                    }
                }
            }
        }
    }
    return output;
}

FeatureMap resize_canvas(const FeatureMap & input, int width, int height) {
    if (width == input.width && height == input.height) {
        return input;
    }
    FeatureMap output{width, height, input.channels, std::vector<float>(static_cast<size_t>(width) * height * input.channels)};
    const int copy_width = std::min(width, input.width);
    for (int c = 0; c < input.channels; ++c) {
        for (int y = 0; y < height; ++y) {
            const auto * source = input.data.data() + c * input.pixels() + std::min(y, input.height - 1) * input.width;
            auto * destination = output.data.data() + c * output.pixels() + y * width;
            std::copy_n(source, copy_width, destination);
            std::fill(destination + copy_width, destination + width, source[copy_width - 1]);
        }
    }
    return output;
}

FeatureMap estimate_flow(ggml_context * model, const RaftFrame & first, const RaftFrame & second, int iterations) {
    const auto & first_features = first.features;
    const auto & second_features = second.features;
    const auto & context = first.context;
    const int width = first_features.width, height = first_features.height;
    const RaftCorrelation correlation(first_features.data, second_features.data, first_features.channels, height, width);
    auto hidden = slice(context, 0, 128);
    auto input = slice(context, 128, 128);
    for (auto & value : hidden.data) {
        value = std::tanh(value);
    }
    relu(input);

    const int pixels = width * height;
    FeatureMap grid{width, height, 2, std::vector<float>(2 * pixels)};
    for (int p = 0; p < pixels; ++p) {
        grid.data[p] = static_cast<float>(p % width);
        grid.data[p + pixels] = static_cast<float>(p / width);
    }
    auto coordinates = grid;
    FeatureMap flow{width, height, 2, std::vector<float>(2 * pixels)};
    const std::string base = "module.update_block.";
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const FeatureMap corr{width, height, RaftCorrelation::output_channels, correlation.lookup(coordinates.data)};
        auto corr_features = activated_convolution(model, base + "encoder.convc1", corr, 0);
        corr_features = activated_convolution(model, base + "encoder.convc2", corr_features, 1);
        auto flow_features = activated_convolution(model, base + "encoder.convf1", flow, 3);
        flow_features = activated_convolution(model, base + "encoder.convf2", flow_features, 1);
        const auto motion = activated_convolution(model, base + "encoder.conv",
                                                   concatenate(corr_features, flow_features), 1);
        const auto gru_input = concatenate(input, concatenate(motion, flow));
        gru(model, hidden, gru_input, false);
        gru(model, hidden, gru_input, true);

        const auto flow_hidden = activated_convolution(model, base + "flow_head.conv1", hidden, 1);
        const auto delta = convolution(model, base + "flow_head.conv2", flow_hidden, 1, 1);
        for (size_t i = 0; i < flow.data.size(); ++i) {
            coordinates.data[i] += delta.data[i];
            flow.data[i] = coordinates.data[i] - grid.data[i];
        }
    }
    const auto mask_hidden = activated_convolution(model, base + "mask.0", hidden, 1);
    return upsample(flow, convolution(model, base + "mask.2", mask_hidden, 1, 0));
}

}

namespace propainter {
RaftFrame raft_prepare(ggml_context * model, const FeatureMap & image) {
    const int padded_width = std::max(128, ((image.width + 7) / 8) * 8);
    const int padded_height = std::max(128, ((image.height + 7) / 8) * 8);
    const auto padded = resize_canvas(image, padded_width, padded_height);
    return {encode(model, padded, false), encode(model, padded, true), image.width, image.height};
}

FeatureMap raft_flow(ggml_context * model, const RaftFrame & first, const RaftFrame & second, int iterations) {
    return resize_canvas(estimate_flow(model, first, second, iterations), first.width, first.height);
}

FeatureMap raft_flow(ggml_context * model, const FeatureMap & first, const FeatureMap & second, int iterations) {
    return raft_flow(model, raft_prepare(model, first), raft_prepare(model, second), iterations);
}

FeatureMap raft_encode(ggml_context * model, const FeatureMap & image, bool context) {
    return encode(model, image, context);
}
}
