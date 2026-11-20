#pragma once
#include "model-ops.h"

namespace propainter {
using Sequence = std::vector<FeatureMap>;

struct BidirectionalFlow {
    Sequence forward;
    Sequence backward;
};

BidirectionalFlow complete_flow(ggml_context * model, const BidirectionalFlow & input, const Sequence & masks);
}
