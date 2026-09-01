// image.h — CPU-side image container plus load/save and quality metrics.
//
// Convention: ImageF always holds *linear* RGBA float. sRGB decode happens on
// load, encode on save, so every pixel operation in the tool works in linear
// light. Metrics are computed on sRGB-encoded values because that is the space
// PSNR/SSIM numbers are normally quoted in.
#pragma once

#include "common.h"

struct ImageF {
    int w = 0;
    int h = 0;
    std::vector<float> px;  // RGBA, linear, row-major, w*h*4

    bool empty() const { return w <= 0 || h <= 0 || px.empty(); }
    size_t pixels() const { return static_cast<size_t>(w) * static_cast<size_t>(h); }
};

float SrgbToLinear(float c);
float LinearToSrgb(float c);

// Loads PNG/JPG/BMP/TGA (8- or 16-bit) and converts to linear RGBA float.
// Alpha is passed through unchanged (alpha is not gamma encoded).
ImageF LoadImageLinear(const std::string& path);

// Writes 8-bit PNG. Input is linear; sRGB encoding is applied here.
void SavePng8(const std::string& path, const ImageF& img);

// Writes 16-bit PNG, still sRGB-encoded, for when 8 bits would clip detail.
void SavePng16(const std::string& path, const ImageF& img);

struct DiffStats {
    double maxAbs = 0.0;   // largest per-channel sRGB error, 0..1
    double meanAbs = 0.0;  // mean per-channel sRGB error
    int maxX = 0;          // where the worst error is, handy for cropping
    int maxY = 0;
};

// Writes an error map next to the result: per-pixel max absolute channel
// difference in sRGB space, multiplied by `gain` and run through a
// black-blue-green-yellow-red ramp. Returns the statistics regardless.
DiffStats SaveDiffPng(const std::string& path, const ImageF& reference, const ImageF& test,
                      float gain);

struct Metrics {
    double psnrRgb = 0.0;   // dB, over sRGB-encoded RGB
    double ssimLuma = 0.0;  // [0,1], over sRGB-encoded luma
};

// Both images must have identical dimensions.
Metrics ComputeMetrics(const ImageF& reference, const ImageF& test);

// Stable 64-bit hash of the sRGB-encoded 8-bit pixels. Used to detect two
// presets producing byte-identical output (i.e. a preset silently fell back).
uint64_t HashImage(const ImageF& img);
