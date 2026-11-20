#pragma once
#include "flow-completion.h"
#include <cstdint>
#include <filesystem>

namespace propainter {
using Bytes = std::vector<uint8_t>;
FeatureMap dilated_mask(const Bytes & bytes, int width, int height, int dilation);
FeatureMap normalized_rgb(const Bytes & bytes, int width, int height);
std::vector<Bytes> render(const std::filesystem::path & models, const Sequence & images,
                          const Sequence & masks, std::vector<Bytes> pixels,
                          int iterations, int neighbor_stride, int ref_stride, int subvideo);
void run_stream(const std::filesystem::path & models, int width, int height);
}
