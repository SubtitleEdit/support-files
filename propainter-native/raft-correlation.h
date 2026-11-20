#pragma once

#include <array>
#include <vector>

class RaftCorrelation {
public:
    static constexpr int levels = 4;
    static constexpr int radius = 4;
    static constexpr int window = 2 * radius + 1;
    static constexpr int output_channels = levels * window * window;

    RaftCorrelation(const std::vector<float> & first, const std::vector<float> & second,
                    int channels, int height, int width);

    // coordinates and result are channel-first; the first coordinate plane is x.
    std::vector<float> lookup(const std::vector<float> & coordinates) const;

private:
    int pixels_;
    std::array<int, levels> widths_;
    std::array<int, levels> heights_;
    std::array<std::vector<float>, levels> maps_;
};
