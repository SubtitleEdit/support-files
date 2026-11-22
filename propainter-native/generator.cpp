#include "generator.h"
#include "sparse-transformer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace propainter {
namespace {
FeatureMap warp(const FeatureMap & source, const FeatureMap & flow, bool nearest = false) {
    FeatureMap result{source.width, source.height, source.channels, std::vector<float>(source.data.size())};
    const int pixels = source.pixels();
    for (int p = 0; p < pixels; ++p) {
        const float x = p % source.width + flow.data[p], y = p / source.width + flow.data[pixels + p];
        for (int c = 0; c < source.channels; ++c) {
            const auto * plane = source.data.data() + c * pixels;
            if (nearest) {
                const int xx = std::nearbyint(x), yy = std::nearbyint(y);
                if (xx >= 0 && xx < source.width && yy >= 0 && yy < source.height)
                    result.data[c * pixels + p] = plane[yy * source.width + xx];
            } else {
                result.data[c * pixels + p] = sample(plane, source.width, source.height, x, y);
            }
        }
    }
    return result;
}

FeatureMap consistent(const FeatureMap & flow, const FeatureMap & opposite) {
    const auto warped = warp(opposite, flow);
    const int pixels = flow.pixels();
    FeatureMap result{flow.width, flow.height, 1, std::vector<float>(pixels)};
    for (int p = 0; p < pixels; ++p) {
        const float x = flow.data[p], y = flow.data[pixels + p];
        const float xx = warped.data[p], yy = warped.data[pixels + p];
        const float dx = x + xx, dy = y + yy;
        result.data[p] = dx * dx + dy * dy < 0.01f * (x * x + y * y + xx * xx + yy * yy) + 0.5f;
    }
    return result;
}

FeatureMap refine(ggml_context * model, const std::string & name, const FeatureMap & input) {
    auto residual = convolution(model, name + ".0", input, 1, 1);
    leaky_relu(residual, 0.2f);
    return convolution(model, name + ".2", residual, 1, 1);
}

PropagatedImages propagate(ggml_context * model, const Sequence & input, const BidirectionalFlow & flows,
                           const Sequence & masks) {
    const int frames = input.size();
    if (frames < 1 || masks.size() != input.size() || flows.forward.size() + 1 != input.size() ||
        flows.backward.size() != flows.forward.size())
        throw std::runtime_error("Invalid propagation sequence");
    PropagatedImages sequence{input, masks};
    Sequence backward;
    const std::string base = "feat_prop_module.";
    for (bool reverse : {true, false}) {
        const auto & motion = reverse ? flows.forward : flows.backward;
        const auto & inverse = reverse ? flows.backward : flows.forward;
        const std::string direction = reverse ? "backward_1" : "forward_1";
        FeatureMap previous, previous_mask;
        for (int i = 0; i < frames; ++i) {
            const int t = reverse ? frames - 1 - i : i;
            const auto & current = sequence.images[t];
            const auto & mask = sequence.masks[t];
            auto result = current;
            auto updated = mask;
            if (i > 0) {
                const int link = reverse ? t : t - 1;
                const auto & flow = motion[link];
                const auto valid = consistent(flow, inverse[link]);
                const auto warped = warp(previous, flow, !model);
                if (model) {
                    const auto cond = concatenate(concatenate(concatenate(current, warped), flow),
                                                  concatenate(valid, mask));
                    result = deform(model, base + "deform_align." + direction, previous, cond, 3, &flow);
                } else {
                    const auto warped_mask = warp(previous_mask, flow);
                    for (size_t p = 0; p < mask.pixels(); ++p) {
                        const bool available = valid.data[p] > 0 && warped_mask.data[p] <= 0.1f;
                        if (mask.data[p] > 0.1f && available) {
                            for (int c = 0; c < current.channels; ++c)
                                result.data[c * mask.pixels() + p] = warped.data[c * mask.pixels() + p];
                            updated.data[p] = 0;
                        }
                    }
                }
            }
            if (model)
                add(result, refine(model, base + "backbone." + direction,
                                   concatenate(concatenate(current, result), mask)));
            sequence.images[t] = result;
            sequence.masks[t] = updated;
            previous = std::move(result);
            previous_mask = std::move(updated);
        }
        if (reverse && model) backward = sequence.images;
    }
    if (model)
        for (int t = 0; t < frames; ++t) {
            sequence.images[t] = refine(model, base + "fuse",
                concatenate(concatenate(backward[t], sequence.images[t]), masks[t]));
            add(sequence.images[t], input[t]);
        }
    return sequence;
}

FeatureMap concatenate_groups(const FeatureMap & first, const FeatureMap & second, int groups) {
    FeatureMap result{first.width, first.height, first.channels + second.channels, {}};
    const int a = first.data.size() / groups, b = second.data.size() / groups;
    result.data.reserve(first.data.size() + second.data.size());
    for (int g = 0; g < groups; ++g) {
        result.data.insert(result.data.end(), first.data.begin() + g * a, first.data.begin() + (g + 1) * a);
        result.data.insert(result.data.end(), second.data.begin() + g * b, second.data.begin() + (g + 1) * b);
    }
    return result;
}

FeatureMap encode(ggml_context * model, FeatureMap frame) {
    FeatureMap skip;
    constexpr int groups_at_layer[] = {2, 4, 8, 1};
    for (int layer = 0; layer <= 16; layer += 2) {
        int groups = 1;
        if (layer == 8) skip = frame;
        if (layer > 8) {
            groups = groups_at_layer[(layer - 10) / 2];
            frame = concatenate_groups(skip, frame, groups);
        }
        frame = convolution(model, "encoder.layers." + std::to_string(layer), frame,
                            layer == 0 || layer == 4 ? 2 : 1, 1, 1, 1, groups);
        leaky_relu(frame, 0.2f);
    }
    return frame;
}

FeatureMap quarter(const FeatureMap & frame, bool flow) {
    FeatureMap result{frame.width / 4, frame.height / 4, frame.channels, {}};
    result.data.resize(result.pixels() * result.channels);
    for (int c = 0; c < frame.channels; ++c)
        for (int y = 0; y < result.height; ++y)
            for (int x = 0; x < result.width; ++x) {
                const auto * plane = frame.data.data() + c * frame.pixels();
                result.data[c * result.pixels() + y * result.width + x] = flow ?
                    sample(plane, frame.width, frame.height, 4 * x + 1.5f, 4 * y + 1.5f) / 4 :
                    plane[(4 * y) * frame.width + 4 * x];
            }
    return result;
}
}

PropagatedImages propagate_images(const Sequence & images, const BidirectionalFlow & flows, const Sequence & masks) {
    return propagate(nullptr, images, flows, masks);
}

FeatureMap encode_image(ggml_context * model, const FeatureMap & image, const FeatureMap & mask,
                        const FeatureMap & updated) {
    return encode(model, concatenate(concatenate(image, mask), updated));
}

Sequence inpaint(ggml_context * model, Sequence features, const BidirectionalFlow & flows,
                 const Sequence & masks, const Sequence & updated, size_t local_frames) {
    if (!model || local_frames < 1 || local_frames > features.size() || masks.size() != features.size() ||
        updated.size() != features.size() || flows.forward.size() + 1 != local_frames ||
        flows.backward.size() != flows.forward.size())
        throw std::runtime_error("Invalid inpainting sequence");
    Sequence local_masks, feature_masks;
    for (size_t t = 0; t < local_frames; ++t) {
        local_masks.push_back(quarter(masks[t], false));
        feature_masks.push_back(concatenate(local_masks.back(), quarter(updated[t], false)));
    }
    BidirectionalFlow scaled;
    for (size_t t = 0; t + 1 < local_frames; ++t) {
        scaled.forward.push_back(quarter(flows.forward[t], true));
        scaled.backward.push_back(quarter(flows.backward[t], true));
    }
    Sequence local(features.begin(), features.begin() + local_frames);
    auto propagated = propagate(model, local, scaled, feature_masks);
    std::move(propagated.images.begin(), propagated.images.end(), features.begin());

    transform_features(model, features, local_masks);
    features.resize(local_frames);
    for (size_t t = 0; t < local_frames; ++t) {
        auto & frame = features[t];
        if (std::none_of(masks[t].data.begin(), masks[t].data.end(), [](float value) { return value != 0; })) {
            frame = {};
            continue;
        }

        for (int layer : {0, 2, 4, 6}) {
            const bool upscale = layer == 0 || layer == 4;
            frame = convolution(model, "decoder." + std::to_string(layer) + (upscale ? ".conv" : ""),
                                upscale ? resize_twice(frame) : std::move(frame), 1, 1);
            if (layer != 6) leaky_relu(frame, 0.2f);
        }
        for (float & value : frame.data) value = std::tanh(value);
    }
    return features;
}
}
