#include "sparse-transformer.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace propainter {
namespace {
constexpr int hidden = 512;
constexpr int patch = 7;
constexpr int window_h = 5;
constexpr int window_w = 9;
constexpr int window_cells = window_h * window_w;
constexpr int head_width = hidden / 4;

struct Grid {
    int width, height, frames;
    int cells() const { return width * height; }
    int index(int t, int y, int x) const { return t * cells() + y * width + x; }
};

std::vector<float> linear(ggml_context * model, const std::string & name, const std::vector<float> & input, int in) {
    const auto & w = tensor(model, name + ".weight");
    const int out = w.ne[1], rows = input.size() / in;
    if (w.ne[0] != in || rows * size_t(in) != input.size())
        throw std::runtime_error("Invalid linear shape for " + name);
    const auto * bias = vector(model, name + ".bias", out);
    std::vector<float> result(size_t(rows) * out);
    multiply(static_cast<const float *>(w.data), input.data(), result.data(), in, rows, out);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < out; ++c)
            result[r * out + c] += bias[c];
    return result;
}

std::vector<float> normalize(ggml_context * model, const std::string & name, const std::vector<float> & input) {
    const auto * scale = vector(model, name + ".weight", hidden);
    const auto * bias = vector(model, name + ".bias", hidden);
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); i += hidden) {
        const auto * row = input.data() + i;
        const float mean = std::accumulate(row, row + hidden, 0.0f) / hidden;
        float sum = 0;
        for (int c = 0; c < hidden; ++c) {
            const float d = row[c] - mean;
            sum += d * d;
        }
        const float inverse = 1 / std::sqrt(sum / hidden + 1e-5f);
        for (int c = 0; c < hidden; ++c)
            result[i + c] = (row[c] - mean) * inverse * scale[c] + bias[c];
    }
    return result;
}

class Patches {
public:
    Patches(int width, int height)
        : width(width), height(height), tokens_w((width + 2) / 3), tokens_h((height + 2) / 3),
          coordinates(tokens_w * tokens_h * patch * patch, -1), counts(width * height) {
        for (int y = 0; y < tokens_h; ++y)
            for (int x = 0; x < tokens_w; ++x)
                for (int ky = 0; ky < patch; ++ky)
                    for (int kx = 0; kx < patch; ++kx) {
                        const int yy = y * 3 + ky - 3, xx = x * 3 + kx - 3;
                        if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;
                        const int p = yy * width + xx;
                        coordinates[((y * tokens_w + x) * patch + ky) * patch + kx] = p;
                        ++counts[p];
                    }
    }

    std::vector<float> unfold(const FeatureMap & feature) const {
        const int row = feature.channels * patch * patch;
        std::vector<float> result(size_t(tokens_w) * tokens_h * row);
        for (int t = 0; t < tokens_w * tokens_h; ++t)
            for (int c = 0; c < feature.channels; ++c)
                for (int k = 0; k < patch * patch; ++k) {
                    const int p = coordinates[t * patch * patch + k];
                    if (p >= 0)
                        result[t * row + c * patch * patch + k] = feature.data[c * feature.pixels() + p];
                }
        return result;
    }

    FeatureMap fold(const float * tokens, int channels, bool average) const {
        FeatureMap result{width, height, channels, std::vector<float>(size_t(width) * height * channels)};
        const int row = channels * patch * patch;
        for (int t = 0; t < tokens_w * tokens_h; ++t)
            for (int c = 0; c < channels; ++c)
                for (int k = 0; k < patch * patch; ++k) {
                    const int p = coordinates[t * patch * patch + k];
                    if (p >= 0)
                        result.data[c * result.pixels() + p] += tokens[t * row + c * patch * patch + k];
                }
        if (average)
            for (int c = 0; c < channels; ++c)
                for (int p = 0; p < width * height; ++p)
                    result.data[c * result.pixels() + p] /= counts[p];
        return result;
    }

    int width, height, tokens_w, tokens_h;
    std::vector<int> coordinates, counts;
};

std::vector<float> pooled(ggml_context * model, const std::string & name, const Grid & grid,
                          const std::vector<float> & input) {
    const auto & weights = tensor(model, name + ".weight");
    const auto * w = static_cast<const float *>(weights.data);
    const auto * bias = vector(model, name + ".bias", hidden);
    const int pw = grid.width / 4, ph = grid.height / 4;
    std::vector<float> result(size_t(grid.frames) * pw * ph * hidden);
    for (int t = 0; t < grid.frames; ++t)
        for (int y = 0; y < ph; ++y)
            for (int x = 0; x < pw; ++x)
                for (int c = 0; c < hidden; ++c) {
                    float total = bias[c];
                    for (int dy = 0; dy < 4; ++dy)
                        for (int dx = 0; dx < 4; ++dx)
                            total += w[c * 16 + dy * 4 + dx] *
                                input[grid.index(t, y * 4 + dy, x * 4 + dx) * hidden + c];
                    result[((t * ph + y) * pw + x) * hidden + c] = total;
                }
    return result;
}

void attend_head(const std::vector<int> & queries, const std::vector<int> & keys, int offset,
                 const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
                 std::vector<float> & output) {
    const int nq = queries.size(), nk = keys.size();
    std::vector<float> qq(size_t(nq) * head_width), kk(size_t(nk) * head_width);
    std::vector<float> vv(size_t(nk) * head_width), result(size_t(nq) * head_width);
    for (int i = 0; i < nq; ++i)
        std::copy_n(q.data() + queries[i] * hidden + offset, head_width, qq.data() + i * head_width);
    for (int i = 0; i < nk; ++i) {
        std::copy_n(k.data() + keys[i] * hidden + offset, head_width, kk.data() + i * head_width);
        for (int c = 0; c < head_width; ++c)
            vv[c * nk + i] = v[keys[i] * hidden + offset + c];
    }
    if (!metal_attention(qq.data(), kk.data(), vv.data(), result.data(), head_width, nq, nk)) {
        std::vector<float> scores(size_t(nq) * nk);
        multiply(kk.data(), qq.data(), scores.data(), head_width, nq, nk);
        const float scale = 1 / std::sqrt(float(head_width));
        for (int i = 0; i < nq; ++i) {
            auto * row = scores.data() + i * nk;
            const float max = *std::max_element(row, row + nk);
            float sum = 0;
            for (int j = 0; j < nk; ++j) {
                row[j] = std::exp((row[j] - max) * scale);
                sum += row[j];
            }
            for (int j = 0; j < nk; ++j) row[j] /= sum;
        }
        multiply(vv.data(), scores.data(), result.data(), nk, nq, head_width);
    }
    for (int i = 0; i < nq; ++i)
        std::copy_n(result.data() + i * head_width, head_width, output.data() + queries[i] * hidden + offset);
}

std::vector<float> attention(ggml_context * model, const std::string & name, const Grid & grid,
                             const std::vector<float> & normalized, const std::vector<bool> & masked, int phase) {
    const Grid padded{(grid.width + window_w - 1) / window_w * window_w,
                      (grid.height + window_h - 1) / window_h * window_h, grid.frames};
    std::vector<float> input(size_t(padded.frames) * padded.cells() * hidden);
    for (int t = 0; t < grid.frames; ++t)
        for (int y = 0; y < grid.height; ++y)
            std::copy_n(normalized.data() + grid.index(t, y, 0) * hidden, grid.width * hidden,
                        input.data() + padded.index(t, y, 0) * hidden);
    auto q = linear(model, name + ".query", input, hidden);
    const int pool_cells = (padded.width / 4) * (padded.height / 4);
    auto pool = pooled(model, name + ".pool_layer", padded, input);
    input.insert(input.end(), pool.begin(), pool.end());
    auto k = linear(model, name + ".key", input, hidden);
    auto v = linear(model, name + ".value", input, hidden);
    auto output = std::vector<float>(q.size());

    const auto * indices = ggml_get_tensor(model, (name + ".valid_ind_rolled").c_str());
    if (!indices || indices->type != GGML_TYPE_I64 || !indices->data)
        throw std::runtime_error("Missing expanded attention window indices");
    const auto * rolled = static_cast<const int64_t *>(indices->data);
    int wi = 0;
    const int spatial_count = padded.frames * padded.cells();
    for (int y = 0; y < padded.height; y += window_h) {
        for (int x = 0; x < padded.width; x += window_w, ++wi) {
            std::vector<int> queries, keys;
            for (int t = 0; t < grid.frames; ++t)
                for (int p = 0; p < window_cells; ++p)
                    queries.push_back(padded.index(t, y + p / window_w, x + p % window_w));
            const bool sparse = masked[wi];
            if (sparse) {
                for (int t = phase; t < grid.frames; t += 2) {
                    const auto first = queries.begin() + t * window_cells;
                    keys.insert(keys.end(), first, first + window_cells);
                    for (int64_t j = 0; j < ggml_nelements(indices); ++j) {
                        const int quadrant = rolled[j] / window_cells, p = rolled[j] % window_cells;
                        const int dy = quadrant < 2 ? 3 : -3, dx = quadrant % 2 == 0 ? 5 : -5;
                        const int yy = (y + p / window_w + dy + padded.height) % padded.height;
                        const int xx = (x + p % window_w + dx + padded.width) % padded.width;
                        keys.push_back(padded.index(t, yy, xx));
                    }
                    for (int p = 0; p < pool_cells; ++p)
                        keys.push_back(spatial_count + t * pool_cells + p);
                }
            }
            for (int c = 0; c < hidden; c += head_width) {
                if (sparse) {
                    if (!keys.empty()) attend_head(queries, keys, c, q, k, v, output);
                } else {
                    for (int t = 0; t < grid.frames; ++t) {
                        const auto first = queries.begin() + t * window_cells;
                        const std::vector<int> frame(first, first + window_cells);
                        attend_head(frame, frame, c, q, k, v, output);
                    }
                }
            }
        }
    }
    std::vector<float> cropped(normalized.size());
    for (int t = 0; t < grid.frames; ++t)
        for (int y = 0; y < grid.height; ++y)
            std::copy_n(output.data() + padded.index(t, y, 0) * hidden, grid.width * hidden,
                        cropped.data() + grid.index(t, y, 0) * hidden);
    return linear(model, name + ".proj", cropped, hidden);
}

void residual(std::vector<float> & x, const std::vector<float> & y) {
    for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}
}

void transform_features(ggml_context * model, Sequence & features, const Sequence & local_masks) {
    const Patches patches(features.front().width, features.front().height);
    const Grid grid{patches.tokens_w, patches.tokens_h, static_cast<int>(features.size())};
    const int windows_w = (grid.width + window_w - 1) / window_w;
    std::vector<bool> masked(windows_w * ((grid.height + window_h - 1) / window_h));
    for (const auto & mask : local_masks)
        for (int t = 0; t < grid.cells(); ++t)
            for (int k = 0; k < patch * patch; ++k) {
                const int p = patches.coordinates[t * patch * patch + k];
                if (p >= 0 && mask.data[p] > 0)
                    masked[(t / grid.width / window_h) * windows_w + (t % grid.width / window_w)] = true;
            }
    std::vector<float> x;
    for (const auto & feature : features) {
        auto encoded = linear(model, "ss.embedding", patches.unfold(feature), feature.channels * patch * patch);
        x.insert(x.end(), encoded.begin(), encoded.end());
    }
    constexpr int ffn_channels = 40;
    constexpr int ffn_width = ffn_channels * patch * patch;
    for (int layer = 0; layer < 8; ++layer) {
        const auto base = "transformers.transformer." + std::to_string(layer);
        residual(x, attention(model, base + ".attention", grid, normalize(model, base + ".norm1", x),
                              masked, layer % 2));
        auto ff = linear(model, base + ".mlp.fc1.0", normalize(model, base + ".norm2", x), hidden);
        for (int t = 0; t < grid.frames; ++t) {
            auto * first = ff.data() + t * grid.cells() * ffn_width;
            auto folded = patches.fold(first, ffn_channels, true);
            const auto unfolded = patches.unfold(folded);
            std::copy(unfolded.begin(), unfolded.end(), first);
        }
        for (float & value : ff) value *= 0.5f * (1 + std::erf(value * 0.7071067811865475f));
        residual(x, linear(model, base + ".mlp.fc2.1", ff, ffn_width));
    }
    for (size_t t = 0; t < local_masks.size(); ++t) {
        const auto * first = x.data() + t * grid.cells() * hidden;
        const auto decoded = linear(model, "sc.embedding", std::vector<float>(first, first + grid.cells() * hidden),
                                    hidden);
        auto composed = patches.fold(decoded.data(), features[t].channels, false);
        add(features[t], convolution(model, "sc.bias_conv", composed, 1, 1));
    }
}
}
