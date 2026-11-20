#pragma once
#include "flow-completion.h"

namespace propainter {
struct PropagatedImages {
    Sequence images;
    Sequence masks;
};

PropagatedImages propagate_images(const Sequence & images, const BidirectionalFlow & flows, const Sequence & masks);
Sequence inpaint(ggml_context * model, const Sequence & images, const BidirectionalFlow & flows,
                 const Sequence & masks, const Sequence & updated, size_t local_frames);
}
