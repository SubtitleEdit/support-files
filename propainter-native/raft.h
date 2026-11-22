#pragma once
#include "model-ops.h"

namespace propainter {
struct RaftFrame {
    FeatureMap features;
    FeatureMap context;
    int width;
    int height;
};

RaftFrame raft_prepare(ggml_context * model, const FeatureMap & image);
FeatureMap raft_flow(ggml_context * model, const RaftFrame & first, const RaftFrame & second, int iterations);
FeatureMap raft_flow(ggml_context * model, const FeatureMap & first, const FeatureMap & second, int iterations);
FeatureMap raft_encode(ggml_context * model, const FeatureMap & image, bool context);
}
