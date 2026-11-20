#pragma once
#include "flow-completion.h"

namespace propainter {
void transform_features(ggml_context * model, Sequence & features, const Sequence & local_masks);
}
