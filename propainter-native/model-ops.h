#pragma once

#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace propainter {
struct FeatureMap {
    int width;
    int height;
    int channels;
    std::vector<float> data;

    size_t pixels() const { return static_cast<size_t>(width) * height; }
};

const ggml_tensor & tensor(ggml_context * model, const std::string & name);
const float * vector(ggml_context * model, const std::string & name, int channels);
FeatureMap convolution(ggml_context * model, const std::string & prefix, const FeatureMap & source,
                       int stride, int padding, int padding_y = -1, int dilation = 1, int groups = 1);
FeatureMap concatenate(const FeatureMap & first, const FeatureMap & second);
void leaky_relu(FeatureMap & value, float slope);
void add(FeatureMap & value, const FeatureMap & residual);
float sample(const float * plane, int width, int height, float x, float y);
FeatureMap resize_twice(const FeatureMap & input);
FeatureMap deform(ggml_context * model, const std::string & prefix, const FeatureMap & input,
                  const FeatureMap & condition, float magnitude, const FeatureMap * flow = nullptr);
bool metal_attention(const float * query, const float * key, const float * value, float * result,
                     int width, int queries, int keys);
void multiply(const float * weight, const float * columns, float * output, int inner, int cells, int channels);
}
