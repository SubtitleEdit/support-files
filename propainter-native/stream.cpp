#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace propainter {
namespace {
struct Record {
    Bytes rgb, mask;
};

struct Box {
    int left, top, right, bottom;
};

bool masked(uint8_t value) { return value >= 128; }

bool read_record(std::istream & input, Record & record, size_t cells) {
    if (input.peek() == std::istream::traits_type::eof()) {
        if (!input.eof() || input.bad()) throw std::runtime_error("Could not read stream");
        return false;
    }
    record = {Bytes(cells * 3), Bytes(cells)};
    for (auto * bytes : {&record.rgb, &record.mask})
        if (!input.read(reinterpret_cast<char *>(bytes->data()), bytes->size()))
            throw std::runtime_error("Truncated ProPainter frame record");
    return true;
}

Box bounds(const std::vector<Record> & records, int width, int height) {
    Box box{width, height, 0, 0};
    for (const auto & record : records)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if (masked(record.mask[y * width + x])) {
                    box.left = std::min(box.left, x);
                    box.top = std::min(box.top, y);
                    box.right = std::max(box.right, x + 1);
                    box.bottom = std::max(box.bottom, y + 1);
                }
    return {std::max(0, box.left - 80), std::max(0, box.top - 80),
            std::min(width, box.right + 80), std::min(height, box.bottom + 80)};
}

uint8_t bilinear(const Bytes & pixels, int stride, int width, int height, float x, float y, int c) {
    x = std::clamp(x, 0.0f, float(width - 1));
    y = std::clamp(y, 0.0f, float(height - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float dx = x - x0, dy = y - y0;
    const auto get = [&](int a, int b) { return pixels[(size_t(b) * stride + a) * 3 + c]; };
    return static_cast<uint8_t>(std::lround((1 - dy) * ((1 - dx) * get(x0, y0) + dx * get(x1, y0)) +
                                           dy * ((1 - dx) * get(x0, y1) + dx * get(x1, y1))));
}

Bytes thumbnail(const Bytes & rgb, int width, int height) {
    Bytes small;
    for (int y = 0; y < std::max(1, height * 3 / 4); y += std::max(1, height / 24))
        for (int x = 0; x < width; x += std::max(1, width / 32)) {
            const auto * start = rgb.data() + (size_t(y) * width + x) * 3;
            small.insert(small.end(), start, start + 3);
        }
    return small;
}

bool cut(const Bytes & previous, const Bytes & current) {
    if (previous.empty()) return false;
    size_t difference = 0;
    for (size_t i = 0; i < previous.size(); ++i) difference += std::abs(int(previous[i]) - current[i]);
    return difference > 24 * previous.size();
}

void write_frame(const Bytes & frame) {
    if (!std::cout.write(reinterpret_cast<const char *>(frame.data()), frame.size()))
        throw std::runtime_error("Could not write ProPainter stream");
}

void render_chunk(const std::filesystem::path & models, const std::vector<Record> & records,
                  int start, int end, int width, int height) {
    if (start == end) return;
    if (std::none_of(records.begin() + start, records.begin() + end, [](const Record & record) {
        return std::any_of(record.mask.begin(), record.mask.end(), masked);
    })) {
        for (int t = start; t < end; ++t) write_frame(records[t].rgb);
        std::cout.flush();
        return;
    }
    const auto box = bounds(records, width, height);
    const int rw = box.right - box.left, rh = box.bottom - box.top;
    const float scale = std::min({1.0f, 512.0f / rw, 320.0f / rh});
    const int sw = std::max(1L, std::lround(rw * scale)), sh = std::max(1L, std::lround(rh * scale));
    const int pw = std::max(64, (sw + 7) / 8 * 8), ph = std::max(64, (sh + 7) / 8 * 8);
    const float sx = float(rw) / sw, sy = float(rh) / sh;
    Sequence frames, masks;
    std::vector<Bytes> pixels;
    for (const auto & record : records) {
        Bytes rgb(size_t(pw) * ph * 3), mask(size_t(pw) * ph);
        for (int y = 0; y < ph; ++y)
            for (int x = 0; x < pw; ++x) {
                const float xx = std::clamp((std::min(x, sw - 1) + 0.5f) * sx - 0.5f, 0.0f, float(rw - 1));
                const float yy = std::clamp((std::min(y, sh - 1) + 0.5f) * sy - 0.5f, 0.0f, float(rh - 1));
                const size_t p = size_t(y) * pw + x;
                for (int c = 0; c < 3; ++c)
                    rgb[p * 3 + c] = bilinear(record.rgb, width, width, height, box.left + xx, box.top + yy, c);
                if (x >= sw || y >= sh) continue;
                for (int dy = int(y * sy); dy < std::min(rh, int(std::ceil((y + 1) * sy))); ++dy)
                    for (int dx = int(x * sx); dx < std::min(rw, int(std::ceil((x + 1) * sx))); ++dx)
                        mask[p] |= masked(record.mask[size_t(box.top + dy) * width + box.left + dx]);
            }
        frames.push_back(normalized_rgb(rgb, pw, ph));
        masks.push_back(dilated_mask(mask, pw, ph, 4));
        pixels.push_back(std::move(rgb));
    }
    if (frames.size() == 1) {
        frames.push_back(frames.front());
        masks.push_back(masks.front());
        pixels.push_back(pixels.front());
    }
    const auto filled = render(models, frames, masks, std::move(pixels), 12, 5, 10, 80);
    for (int t = start; t < end; ++t) {
        auto rgb = records[t].rgb;
        for (int y = 0; y < rh; ++y)
            for (int x = 0; x < rw; ++x) {
                const size_t p = size_t(box.top + y) * width + box.left + x;
                if (!masked(records[t].mask[p])) continue;
                for (int c = 0; c < 3; ++c)
                    rgb[p * 3 + c] = bilinear(filled[t], pw, sw, sh, (x + 0.5f) / sx - 0.5f, (y + 0.5f) / sy - 0.5f, c);
            }
        write_frame(rgb);
    }
    std::cout.flush();
}
}

void run_stream(const std::filesystem::path & models, int width, int height) {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1)
        throw std::runtime_error("Could not enable binary streaming");
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    const size_t cells = size_t(width) * height;
    const int budget = std::min<size_t>(48, std::max<size_t>(4, (256 * 1024 * 1024) / (4 * cells)));
    const int context = std::min(6, (budget - 1) / 3);
    const int chunk = budget - 2 * context;
    std::vector<Record> records;
    Record record;
    Bytes previous;
    int history = 0;
    while (read_record(std::cin, record, cells)) {
        auto current = thumbnail(record.rgb, width, height);
        if (cut(previous, current)) {
            render_chunk(models, records, history, records.size(), width, height);
            records.clear();
            history = 0;
        }
        previous = std::move(current);
        records.push_back(std::move(record));
        if (int(records.size()) >= history + chunk + context) {
            render_chunk(models, records, history, history + chunk, width, height);
            const int keep = std::min(context, history + chunk);
            records.erase(records.begin(), records.begin() + history + chunk - keep);
            history = keep;
        }
    }
    render_chunk(models, records, history, records.size(), width, height);
}
}
