#include "raft-correlation.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
template <typename T>
std::vector<T> read(std::istream & in, size_t length) {
    std::vector<T> data(length);
    if (!in.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T))) {
        throw std::runtime_error("Incomplete correlation fixture");
    }
    return data;
}

void correlation(std::istream & input, std::ostream & output) {
    const auto dimensions = read<std::int32_t>(input, 3);
    const int channels = dimensions[0], height = dimensions[1], width = dimensions[2];
    if (channels < 1 || channels > 512 || height < 16 || width < 16 || height > 80 || width > 80) {
        throw std::runtime_error("Unsupported RAFT correlation dimensions");
    }
    const int pixels = width * height;
    const auto input1 = read<float>(input, channels * pixels);
    const auto input2 = read<float>(input, channels * pixels);
    const auto coords = read<float>(input, pixels * 2);
    if (input.peek() != std::istream::traits_type::eof()) {
        throw std::runtime_error("Unexpected trailing correlation data");
    }

    const RaftCorrelation pyramid(input1, input2, channels, height, width);
    const auto result = pyramid.lookup(coords);
    output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
    if (!output) {
        throw std::runtime_error("Could not write the correlation output");
    }
}
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: propainter-raft-correlation INPUT OUTPUT");
        }
        std::ifstream input(argv[1], std::ios::binary);
        std::ofstream output(argv[2], std::ios::binary);
        if (!input || !output) {
            throw std::runtime_error("Cannot open correlation fixture or result");
        }
        correlation(input, output);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
