#pragma once
#include "model-ops.h"

namespace propainter {
FeatureMap raft_flow(ggml_context * model, const FeatureMap & first, const FeatureMap & second, int iterations);
FeatureMap raft_encode(ggml_context * model, const FeatureMap & image, bool context);
}
