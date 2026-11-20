#include "model-ops.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <memory>
#include <stdexcept>

namespace propainter {
namespace {
ggml_backend_t accelerator() {
    static auto device = [] {
        using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        const char * requested = std::getenv("PROPAINTER_BACKEND");
#if defined(PROPAINTER_AUTO_CUDA)
        const std::string_view default_backend = "cuda";
#elif defined(PROPAINTER_AUTO_METAL)
        const std::string_view default_backend = "metal";
#else
        const std::string_view default_backend = "cpu";
#endif
        const std::string_view name = requested ? std::string_view(requested) : default_backend;
        if (name == "cpu") return Backend(nullptr, &ggml_backend_free);
        if (name != "metal" && name != "cuda")
            throw std::runtime_error("PROPAINTER_BACKEND must be cpu, metal, or cuda");
        const auto dev = ggml_backend_dev_by_name(name == "metal" ? "MTL0" : "CUDA0");
        Backend backend(dev ? ggml_backend_dev_init(dev, nullptr) : nullptr, &ggml_backend_free);
        if (!backend) {
            if (requested)
                throw std::runtime_error("Requested ggml backend is unavailable: " + std::string(name));
            std::cerr << "ProPainter ggml " << name << " unavailable, using CPU\n";
            return backend;
        }
        std::cerr << "ProPainter ggml backend: " << ggml_backend_name(backend.get()) << '\n';
        return backend;
    }();
    return device.get();
}

void execute(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * y,
             const float * input, const float * weights, float * result, ggml_backend_t backend) {
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (backend) {
        bool supported = true;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            supported = supported && ggml_backend_supports_op(backend, ggml_graph_node(graph, i));
        if (!supported) {
            // The no-allocation graph already exists. Use a CPU backend buffer for its tensors.
            using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
            static Backend cpu(ggml_backend_cpu_init(), &ggml_backend_free);
            if (!cpu) throw std::runtime_error("Cannot initialize ggml CPU fallback");
            ggml_backend_cpu_set_n_threads(cpu.get(), 4);
            backend = cpu.get();
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx, backend), &ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("Cannot allocate ggml backend tensor buffer");
        ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
        ggml_backend_tensor_set(w, weights, 0, ggml_nbytes(w));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml backend graph computation failed");
        ggml_backend_tensor_get(y, result, 0, ggml_nbytes(y));
        return;
    }
    std::memcpy(x->data, input, ggml_nbytes(x));
    std::memcpy(w->data, weights, ggml_nbytes(w));
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("ggml CPU graph computation failed");
    std::memcpy(result, y->data, ggml_nbytes(y));
}
}

const ggml_tensor & tensor(ggml_context * model, const std::string & name) {
    const auto * value = ggml_get_tensor(model, name.c_str());
    if (!value || !value->data || value->type != GGML_TYPE_F32) {
        throw std::runtime_error("Missing float32 checkpoint tensor: " + name);
    }
    return *value;
}

const float * vector(ggml_context * model, const std::string & name, int channels) {
    const auto & value = tensor(model, name);
    if (ggml_nelements(&value) != channels) {
        throw std::runtime_error("Incorrect vector size for " + name);
    }
    return static_cast<const float *>(value.data);
}

FeatureMap convolution(ggml_context * model, const std::string & prefix, const FeatureMap & source,
                       int stride, int padding, int padding_y, int dilation, int groups) {
    if (padding_y < 0) {
        padding_y = padding;
    }
    const auto & weight = tensor(model, prefix + ".weight");
    const int kernel_w = static_cast<int>(weight.ne[0]);
    const int kernel_h = static_cast<int>(weight.ne[1]);
    const int input_channels = source.channels / groups;
    const int channels = static_cast<int>(ggml_nelements(&weight) / (kernel_w * kernel_h * input_channels));
    if (ggml_nelements(&weight) != int64_t(kernel_w) * kernel_h * input_channels * channels || channels % groups || source.channels % groups || kernel_w < 1 || kernel_w > 7 || kernel_h < 1 ||
        kernel_h > 7 || channels < 1 || channels > 576) {
        throw std::runtime_error("Incorrect convolution size for " + prefix);
    }
    const int width = (source.width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    const int height = (source.height + 2 * padding_y - dilation * (kernel_h - 1) - 1) / stride + 1;
    if (width < 1 || height < 1) {
        throw std::runtime_error("Invalid convolution resolution");
    }

    const size_t cells = static_cast<size_t>(width) * height;
    const int output_channels = channels / groups;
    const size_t input_size = source.pixels() * input_channels;
    const size_t kernel_size = size_t(kernel_w) * kernel_h * input_channels * output_channels;
    const size_t workspace = (input_size + kernel_size +
        cells * (input_channels * kernel_w * kernel_h + 2 * output_channels)) * sizeof(float);
    const auto * bias = vector(model, prefix + ".bias", channels);
    FeatureMap result{width, height, channels, std::vector<float>(cells * channels)};
    auto * backend = accelerator();
    std::vector<float> computed(cells * output_channels);
    for (int group = 0; group < groups; ++group) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({(backend ? 0 : workspace) + 4 * 1024 * 1024, nullptr, backend != nullptr}), &ggml_free);
        if (!ctx) throw std::runtime_error("Cannot allocate graph for " + prefix);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, source.width, source.height, input_channels);
        auto * kernel = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kernel_w, kernel_h, input_channels, output_channels);
        auto * y = ggml_conv_2d(ctx.get(), kernel, x, stride, stride, padding, padding_y, dilation, dilation);
        execute(ctx.get(), x, kernel, y, source.data.data() + group * input_size,
                static_cast<const float *>(weight.data) + group * kernel_size, computed.data(), backend);
        for (int c = 0; c < output_channels; ++c) {
            const int channel = group * output_channels + c;
            for (size_t i = 0; i < cells; ++i)
                result.data[channel * cells + i] = computed[c * cells + i] + bias[channel];
        }
    }
    return result;
}

FeatureMap concatenate(const FeatureMap & first, const FeatureMap & second) {
    if (first.width != second.width || first.height != second.height) {
        throw std::runtime_error("RAFT concatenation shape mismatch");
    }
    FeatureMap combined{first.width, first.height, first.channels + second.channels, first.data};
    combined.data.insert(combined.data.end(), second.data.begin(), second.data.end());
    return combined;
}

void leaky_relu(FeatureMap & value, float slope) {
    for (float & x : value.data) {
        if (x < 0) x *= slope;
    }
}

void add(FeatureMap & value, const FeatureMap & residual) {
    if (value.data.size() != residual.data.size()) throw std::runtime_error("Residual shape mismatch");
    for (size_t i = 0; i < value.data.size(); ++i) value.data[i] += residual.data[i];
}

float sample(const float * plane, int width, int height, float x, float y) {
    if (x <= -1 || y <= -1 || x >= width || y >= height) return 0;
    const int x0 = std::floor(x), y0 = std::floor(y);
    const float dx = x - x0, dy = y - y0;
    const auto at = [&](int xx, int yy) -> float {
        return xx >= 0 && xx < width && yy >= 0 && yy < height ? plane[yy * width + xx] : 0;
    };
    return (1 - dy) * ((1 - dx) * at(x0, y0) + dx * at(x0 + 1, y0)) +
           dy * ((1 - dx) * at(x0, y0 + 1) + dx * at(x0 + 1, y0 + 1));
}

FeatureMap resize_twice(const FeatureMap & input) {
    FeatureMap result{2 * input.width, 2 * input.height, input.channels, {}};
    result.data.resize(result.pixels() * result.channels);
    const float sx = float(input.width - 1) / (result.width - 1);
    const float sy = float(input.height - 1) / (result.height - 1);
    for (int c = 0; c < input.channels; ++c)
        for (int y = 0; y < result.height; ++y)
            for (int x = 0; x < result.width; ++x)
                result.data[c * result.pixels() + y * result.width + x] =
                    sample(input.data.data() + c * input.pixels(), input.width, input.height, x * sx, y * sy);
    return result;
}

void multiply(const float * weight, const float * columns, float * output, int inner, int cells, int channels) {
    auto * backend = accelerator();
    const size_t workspace = (size_t(inner) * (cells + channels) + size_t(cells) * channels) * sizeof(float);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({(backend ? 0 : workspace) + 1024 * 1024, nullptr, backend != nullptr}), &ggml_free);
    if (!ctx) throw std::runtime_error("Cannot allocate matrix product");
    auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, inner, channels);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, inner, cells);
    auto * y = ggml_mul_mat(ctx.get(), w, x);
    execute(ctx.get(), x, w, y, columns, weight, output, backend);
}

FeatureMap deform(ggml_context * model, const std::string & prefix, const FeatureMap & input,
                  const FeatureMap & condition, float magnitude, const FeatureMap * flow) {
    auto offsets = condition;
    for (int layer = 0; layer <= 6; layer += 2) {
        offsets = convolution(model, prefix + ".conv_offset." + std::to_string(layer), offsets, 1, 1);
        if (layer != 6) leaky_relu(offsets, 0.1f);
    }

    constexpr int kernel = 9;
    const int groups = offsets.channels / (3 * kernel);
    const int channels = tensor(model, prefix + ".bias").ne[0];
    const auto & weight = tensor(model, prefix + ".weight");
    const int inner = input.channels * kernel;
    const int cells = input.pixels();
    if (groups <= 0 || offsets.channels != 3 * groups * kernel || input.channels % groups ||
        ggml_nelements(&weight) != int64_t(inner) * channels)
        throw std::runtime_error("Invalid deformable convolution shape");
    const int group_channels = input.channels / groups;
    std::vector<float> columns(size_t(cells) * inner);
    for (int p = 0; p < cells; ++p) {
        const int x = p % input.width, y = p / input.width;
        for (int g = 0; g < groups; ++g) {
            for (int k = 0; k < kernel; ++k) {
                const int index = g * kernel + k;
                const float dy = magnitude * std::tanh(offsets.data[(2 * index) * cells + p]);
                const float dx = magnitude * std::tanh(offsets.data[(2 * index + 1) * cells + p]);
                const float yy = y + k / 3 - 1 + dy + (flow ? flow->data[cells + p] : 0);
                const float xx = x + k % 3 - 1 + dx + (flow ? flow->data[p] : 0);
                const float mask = 1 / (1 + std::exp(-offsets.data[(2 * groups * kernel + index) * cells + p]));
                for (int c = g * group_channels; c < (g + 1) * group_channels; ++c)
                    columns[size_t(p) * inner + c * kernel + k] = mask *
                        sample(input.data.data() + c * cells, input.width, input.height, xx, yy);
            }
        }
    }
    std::vector<float> product(size_t(cells) * channels);
    multiply(static_cast<const float *>(weight.data), columns.data(), product.data(), inner, cells, channels);
    FeatureMap result{input.width, input.height, channels, std::vector<float>(product.size())};
    const auto * bias = vector(model, prefix + ".bias", channels);
    for (int p = 0; p < cells; ++p)
        for (int c = 0; c < channels; ++c)
            result.data[c * cells + p] = product[p * channels + c] + bias[c];
    return result;
}
}
