// cli.h — command line surface and the pure helpers around it.
//
// Split out of main.cpp so the argument parsing, jitter sequence and phase-count
// logic can be exercised by the test binary without pulling in a GPU device.
#pragma once

#include "common.h"
#include "gpu.h"

struct Options {
    std::string input;
    std::string outDir = "out";
    std::vector<std::string> qualities{"quality"};
    std::vector<std::string> presets{"default", "J", "K"};
    int frames = 32;
    int phases = 0;  // 0 = derive from the upscale factor
    DownFilter filter = DownFilter::Point;
    std::string jitterSign = "auto";
    bool hdr = false;
    bool displaySpace = false;  // filter in the source's own sRGB space
    bool autoExposure = true;
    bool alphaUpscaling = false;
    float depthValue = 0.5f;
    bool png16 = false;
    bool saveLowRes = false;
    bool saveDiff = true;      // write an error map beside each result
    float diffGain = 8.0f;     // amplification applied to that error map
    bool metricsOnly = false;  // compute metrics, write no images — for batch sweeps
    std::string dllDir;
    int adapter = -1;
    bool debugLayer = false;
    std::string jsonPath;
    bool verbose = false;
    // DLSS Neural Rendering (NGX feature 18) capability probe.
    bool probeNr = false;
    bool probeSl = false;
    bool nrRun = false;
    bool nrVideo = false;  // stream raw RGBA frames stdin -> SR+NR -> stdout (ffmpeg on both ends)
    // Model params (latched at feature creation).
    float nrIntensity = 1.0f;        // DLSSNR.Intensity
    unsigned nrStyle = 0;            // DLSSNR.Style: 0 Default, 1 Natural, 2 Cinematic
    float nrLocalStructure = 1.0f;   // DLSSNR.LocalStructureStrength
    float nrLocalTone = 1.0f;        // DLSSNR.LocalToneStrength
    float nrSkin = -1.0f;            // DLSSNR.SkinStructureStrength (<0 = model default)
    float nrGlobalTone = -1.0f;      // DLSSNR.GlobalToneStrength    (<0 = model default)
    bool nrAutoMask = false;         // DLSSNR.UseAutoMask
    bool nrUiCorrection = true;      // DLSSNR.UICorrection
    // Composition of the model's output over the original (host-side).
    float nrDetail = 1.0f;           // overall strength: 0 = original, 1 = full NR
    float nrColour = 1.0f;           // 0 = keep original hue (NR luma only), 1 = NR colour
    bool nrHdr = false;              // feed linear (HDR) instead of sRGB-encoded colour
    // Upscaling: NR super-resolves when the output is larger than the input.
    float nrScale = 1.0f;            // output = input * scale
    unsigned nrTargetW = 0, nrTargetH = 0;  // set one side (--nr-width/--nr-height), other by
                                            // aspect; wins over scale
    bool nrDiff = false;   // also write <name>_nr_diff.png (off by default - faster)
    bool nrOrig = false;   // also write <name>_orig.png
    bool nrMotion = true;     // video: estimate optical-flow motion vectors for NR (on by default)
    bool nrMotionVis = false; // video: output the flow visualisation instead of the NR result
    int nrMotionEngine = 0;   // video flow backend: 0 auto (NVOFA else LK), 1 nvof, 2 lk
    unsigned slFeature = 1004;
    unsigned nrInW = 1920, nrInH = 1080;
    unsigned nrOutW = 3840, nrOutH = 2160;
    unsigned nrPreset = 0;
};

void PrintUsage();

std::vector<std::string> SplitList(const std::string& s);

// Throws ToolError on any invalid combination. Sets *wantHelp and returns early
// for -h/--help without validating the rest.
Options ParseArgs(int argc, char** argv, bool* wantHelp);

// Radical-inverse (Halton) sequence — the standard DLSS jitter source.
double Halton(int index, int base);

struct JitterSign {
    float x = 1.0f;
    float y = 1.0f;
    std::string name = "++";
};

bool ParseJitterSign(const std::string& s, JitterSign* out);

// Length of the jitter sequence. DLSS wants roughly 8 phases per upscale factor
// squared, so that every output pixel gets covered. `override` wins when > 0.
int JitterPhaseCount(int sourceWidth, unsigned renderWidth, int overrideValue);
