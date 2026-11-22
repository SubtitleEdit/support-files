#pragma once
#include "flow-completion.h"

namespace propainter {
struct PropagatedImages {
    Sequence images;
    Sequence masks;
};

PropagatedImages propagate_images(const Sequence & images, const BidirectionalFlow & flows, const Sequence & masks);
FeatureMap encode_image(ggml_context * model, const FeatureMap & image, const FeatureMap & mask,
                        const FeatureMap & updated);
Sequence inpaint(ggml_context * model, Sequence features, const BidirectionalFlow & flows,
                 const Sequence & masks, const Sequence & updated, size_t local_frames);
}
