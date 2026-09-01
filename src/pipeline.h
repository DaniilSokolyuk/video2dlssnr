// pipeline.h — the accumulation loop itself: synthesise a jittered frame
// sequence from a still image and let DLSS reconstruct it.
//
// Kept free of Options/CLI types so the test binary can drive it directly.
#pragma once

#include "cli.h"
#include "common.h"
#include "dlss.h"
#include "gpu.h"
#include "image.h"

// Directories handed to NGX so it prefers a DLSS build sitting beside the
// binary over whatever the display driver ships. `explicitDir` comes from
// --dll-dir and, when given, takes priority; it must exist.
std::vector<std::wstring> DefaultDllSearchPaths(const std::string& explicitDir);

// The low-res inputs DLSS consumes, one set per render resolution.
struct LowResSet {
    GpuTexture color;
    GpuTexture depth;
    GpuTexture motion;
};

// Creates the three targets and leaves them in UNORDERED_ACCESS, which is the
// state DLSS requires for every resource it touches.
LowResSet CreateLowResSet(GpuContext& gpu, unsigned w, unsigned h);

// Creates the DLSS output target, likewise left in UNORDERED_ACCESS.
GpuTexture CreateOutputTexture(GpuContext& gpu, unsigned w, unsigned h);

struct AccumulationParams {
    GpuTexture* source = nullptr;    // high-res ground truth
    GpuTexture* output = nullptr;    // DLSS target, source-sized
    LowResSet* lowRes = nullptr;
    GpuTexture* exposure = nullptr;  // null when auto-exposure is enabled

    DlssFeatureDesc feature;
    JitterSign sign;
    int frames = 32;
    int phases = 8;
    DownFilter filter = DownFilter::Point;

    // True when the shader must sRGB-encode its output because the source
    // texture holds linear values but DLSS is being fed display-referred colour.
    bool encodeSrgb = true;
    // True when DLSS output is display-referred and must be decoded to linear.
    bool decodeSrgbOnReadback = true;

    float depthValue = 0.5f;
    bool captureLowRes = false;
};

struct AccumulationResult {
    ImageF output;       // linear RGBA, source-sized
    ImageF lowResInput;  // last frame handed to DLSS; empty unless requested
    double msPerFrame = 0.0;
};

AccumulationResult RunAccumulation(GpuContext& gpu, NgxSession& ngx,
                                   const AccumulationParams& params);
