// video2dlssnr — run a still image through DLSS Super Resolution.
//
// The image is treated as ground truth. Each pass downsamples it to the render
// resolution DLSS asks for, offset by that frame's subpixel jitter, and feeds
// the result in as a freshly rendered frame with zero motion. DLSS accumulates
// those jittered samples exactly as it would in a game, so the output is a real
// temporal reconstruction rather than a one-shot spatial upscale — which is what
// makes preset differences visible.

#include "cli.h"
#include "common.h"
#include "nr.h"
#include "slprobe.h"
#include "pipeline.h"
#include "dlss.h"
#include "gpu.h"
#include "image.h"

#include <shellapi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct RunResult {
    std::string quality;
    std::string preset;
    unsigned renderW = 0, renderH = 0;
    unsigned targetW = 0, targetH = 0;
    double psnr = 0.0;
    double ssim = 0.0;
    double msPerFrame = 0.0;
    uint64_t hash = 0;
    std::string file;
    std::string diffFile;
    double maxErr = 0.0;
    double meanErr = 0.0;
    std::string sameAs;  // earlier run this is byte-identical to, if any
};

static void WriteJson(const std::string& path, const Options& o, const ImageF& source,
                      const std::string& dllPath, const std::string& dllVersion,
                      const JitterSign& sign, const std::vector<RunResult>& results) {
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) {
        LogWarn("cannot write JSON to %s", path.c_str());
        return;
    }
    auto esc = [](const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '\\' || c == '"') {
                r += '\\';
                r += c;
            } else {
                r += c;
            }
        }
        return r;
    };

    fprintf(f, "{\n");
    fprintf(f, "  \"input\": \"%s\",\n", esc(o.input).c_str());
    fprintf(f, "  \"sourceWidth\": %d,\n", source.w);
    fprintf(f, "  \"sourceHeight\": %d,\n", source.h);
    fprintf(f, "  \"dlssDll\": \"%s\",\n", esc(dllPath).c_str());
    fprintf(f, "  \"dlssVersion\": \"%s\",\n", esc(dllVersion).c_str());
    fprintf(f, "  \"frames\": %d,\n", o.frames);
    fprintf(f, "  \"filter\": \"%s\",\n", DownFilterName(o.filter));
    fprintf(f, "  \"jitterSign\": \"%s\",\n", sign.name.c_str());
    fprintf(f, "  \"colorSpace\": \"%s\",\n", o.hdr ? "linear-hdr" : "srgb-ldr");
    fprintf(f, "  \"runs\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const RunResult& r = results[i];
        const std::string identicalTo =
            r.sameAs.empty() ? std::string("null") : ("\"" + esc(r.sameAs) + "\"");
        const std::string diffFile =
            r.diffFile.empty() ? std::string("null") : ("\"" + esc(r.diffFile) + "\"");
        fprintf(f,
                "    {\"quality\": \"%s\", \"preset\": \"%s\", \"renderWidth\": %u, "
                "\"renderHeight\": %u, \"outputWidth\": %u, \"outputHeight\": %u, "
                "\"psnr\": %.4f, \"ssim\": %.6f, \"maxError\": %.5f, \"meanError\": %.5f, "
                "\"gpuMsPerFrame\": %.4f, \"hash\": \"%016llx\", \"file\": \"%s\", "
                "\"diffFile\": %s, \"identicalTo\": %s}%s\n",
                r.quality.c_str(), r.preset.c_str(), r.renderW, r.renderH, r.targetW, r.targetH,
                r.psnr, r.ssim, r.maxErr, r.meanErr, r.msPerFrame,
                static_cast<unsigned long long>(r.hash), esc(r.file).c_str(), diffFile.c_str(),
                identicalTo.c_str(), (i + 1 == results.size()) ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

static std::string ExeDir() {
    wchar_t p[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, p, MAX_PATH);
    if (!n) return {};
    std::wstring w(p, n);
    const size_t slash = w.find_last_of(L"\\/");
    return (slash == std::wstring::npos) ? std::string{} : Widen2Narrow(w.substr(0, slash));
}

static int Run(int argc, char** argv) {
    bool wantHelp = false;
    Options o = ParseArgs(argc, argv, &wantHelp);
    if (wantHelp) {
        PrintUsage();
        return 0;
    }
    SetVerbose(o.verbose);

    // Default the DLL directory to the executable's own folder, so dropping nvngx_dlssnr.dll
    // next to video2dlssnr.exe works with no --dll-dir.
    if (o.dllDir.empty()) o.dllDir = ExeDir();

    if (o.probeNr) {
        return ProbeNeuralRendering(o.dllDir, o.adapter, o.nrInW, o.nrInH, o.nrOutW,
                                    o.nrOutH, o.nrPreset, o.verbose);
    }

    if (o.probeSl) {
        return ProbeStreamlineNR(o.dllDir, o.slFeature, o.verbose);
    }

    if (o.nrVideo) {
        NrModelParams vm{};
        vm.preset = o.nrPreset;
        vm.intensity = o.nrIntensity;
        vm.style = o.nrStyle;
        vm.localStructure = o.nrLocalStructure;
        vm.localTone = o.nrLocalTone;
        vm.skinStructure = o.nrSkin;
        vm.globalTone = o.nrGlobalTone;
        vm.autoMask = o.nrAutoMask ? 1u : 0u;
        vm.uiCorrection = o.nrUiCorrection ? 1u : 0u;
        return RunNeuralRenderingVideo(o.dllDir, o.adapter, o.nrInW, o.nrInH, vm, o.nrDetail,
                                       o.nrColour, o.nrHdr, o.nrScale, o.nrTargetW, o.nrTargetH,
                                       o.nrMotion, o.nrMotionVis, o.nrMotionEngine, o.nrSrPreset,
                                       o.verbose);
    }

    if (o.nrRun) {
        const std::string outDir = o.outDir.empty() ? std::string("nr_out") : o.outDir;
        NrModelParams model{};
        model.preset = o.nrPreset;
        model.intensity = o.nrIntensity;
        model.style = o.nrStyle;
        model.localStructure = o.nrLocalStructure;
        model.localTone = o.nrLocalTone;
        model.skinStructure = o.nrSkin;
        model.globalTone = o.nrGlobalTone;
        model.autoMask = o.nrAutoMask ? 1u : 0u;
        model.uiCorrection = o.nrUiCorrection ? 1u : 0u;
        return RunNeuralRendering(o.dllDir, o.adapter, o.input, outDir, model, o.nrDetail,
                                  o.nrColour, o.nrHdr, o.nrScale, o.nrTargetW, o.nrTargetH,
                                  o.nrSrPreset, o.nrDiff, o.nrOrig, o.verbose);
    }

    // Expand the 'all' shorthands.
    if (o.qualities.size() == 1 && o.qualities[0] == "all") {
        o.qualities.clear();
        for (const QualityMode& m : AllQualityModes()) o.qualities.push_back(m.name);
    }
    if (o.presets.size() == 1 && o.presets[0] == "all") {
        o.presets = {"default", "E", "F", "J", "K", "L", "M"};
    }

    std::vector<QualityMode> qualities;
    for (const std::string& q : o.qualities) {
        QualityMode m;
        if (!ParseQualityMode(q, &m)) throw ToolError("unknown quality mode '" + q + "'");
        qualities.push_back(m);
    }
    std::vector<unsigned> presets;
    for (const std::string& p : o.presets) {
        unsigned v = 0;
        if (!ParsePreset(p, &v)) throw ToolError("unknown preset '" + p + "'");
        presets.push_back(v);
    }

    LogInfo("Loading %s", o.input.c_str());
    const ImageF source = LoadImageLinear(o.input);
    LogInfo("  source: %d x %d", source.w, source.h);

    std::error_code ec;
    fs::create_directories(o.outDir, ec);
    if (o.jsonPath.empty()) o.jsonPath = (fs::path(o.outDir) / "results.json").string();

    GpuContext gpu;
    gpu.Initialize(o.debugLayer, o.adapter);
    LogInfo("  gpu:    %s (%zu MB)", gpu.AdapterName().c_str(), gpu.AdapterVramMB());

    const std::vector<std::wstring> dllPaths = DefaultDllSearchPaths(o.dllDir);
    for (const std::wstring& p : dllPaths) LogDebug("dll search path: %s", Widen2Narrow(p).c_str());

    NgxSession ngx;
    ngx.Init(gpu.Device(), dllPaths, o.verbose);

    const DlssCaps caps = ngx.QueryCaps();
    if (!caps.available) {
        std::string why = NgxResultToString(caps.initResult);
        if (caps.needsUpdatedDriver) {
            why += "; driver too old, needs at least " + std::to_string(caps.minDriverMajor) + "." +
                   std::to_string(caps.minDriverMinor);
        }
        throw ToolError("DLSS Super Resolution is not available on this system: " + why);
    }

    std::string dllPath = "(not found)", dllVersion = "unknown";
    if (NgxSession::FindLoadedDlssModule(&dllPath, &dllVersion)) {
        LogInfo("  dlss:   %s", dllVersion.c_str());
        LogInfo("          %s", dllPath.c_str());
    } else {
        LogWarn("could not identify the loaded nvngx_dlss.dll");
    }
    LogInfo("");

    // Shared resources: the source never changes, and every run writes into the
    // same output-sized target.
    GpuTexture src = gpu.CreateTexture(source.w, source.h, DXGI_FORMAT_R16G16B16A16_FLOAT, false,
                                       L"source");
    if (o.displaySpace) {
        // Upload the sRGB-encoded values so all filtering happens in that space.
        std::vector<float> encoded = source.px;
        for (size_t i = 0; i < source.pixels(); ++i) {
            for (int c = 0; c < 3; ++c) encoded[i * 4 + c] = LinearToSrgb(encoded[i * 4 + c]);
        }
        gpu.UploadRgbaFloat(src, encoded);
    } else {
        gpu.UploadRgbaFloat(src, source.px);
    }

    GpuTexture outTex = CreateOutputTexture(gpu, source.w, source.h);

    GpuTexture exposureTex;
    GpuTexture* exposurePtr = nullptr;
    if (!o.autoExposure) {
        exposureTex = gpu.CreateTexture(1, 1, DXGI_FORMAT_R32_FLOAT, true, L"exposure");
        gpu.UploadR32Float(exposureTex, {1.0f});
        exposurePtr = &exposureTex;
    }

    std::map<std::pair<int, int>, LowResSet> lrCache;
    auto lowResFor = [&](unsigned w, unsigned h) -> LowResSet& {
        const std::pair<int, int> key{static_cast<int>(w), static_cast<int>(h)};
        auto it = lrCache.find(key);
        if (it != lrCache.end()) return it->second;

        LowResSet set = CreateLowResSet(gpu, w, h);
        return lrCache.emplace(key, std::move(set)).first->second;
    };

    auto phasesFor = [&](unsigned renderW) {
        return JitterPhaseCount(source.w, renderW, o.phases);
    };

    auto makeDesc = [&](const QualityMode& q, unsigned preset, const OptimalSettings& s) {
        DlssFeatureDesc d;
        d.renderW = s.renderW;
        d.renderH = s.renderH;
        d.targetW = static_cast<unsigned>(source.w);
        d.targetH = static_cast<unsigned>(source.h);
        d.quality = q.value;
        d.preset = preset;
        d.hdr = o.hdr;
        d.autoExposure = o.autoExposure;
        d.alphaUpscaling = o.alphaUpscaling;
        return d;
    };

    auto makeParams = [&](LowResSet& lr, const DlssFeatureDesc& desc, const JitterSign& s,
                          int frames, int phases, bool captureLowRes) {
        AccumulationParams p;
        p.source = &src;
        p.output = &outTex;
        p.lowRes = &lr;
        p.exposure = exposurePtr;
        p.feature = desc;
        p.sign = s;
        p.frames = frames;
        p.phases = phases;
        p.filter = o.filter;
        // In display-space mode the source texture already holds sRGB-encoded
        // values, so the shader must not encode a second time.
        p.encodeSrgb = !o.hdr && !o.displaySpace;
        p.decodeSrgbOnReadback = !o.hdr;
        p.depthValue = o.depthValue;
        p.captureLowRes = captureLowRes;
        return p;
    };

    // ---- Resolve the jitter sign convention -------------------------------
    JitterSign sign;
    CHECK(ParseJitterSign("--", &sign));  // theoretical default, see README
    if (o.jitterSign == "auto") {
        // Probe on the most heavily upscaled mode requested. At DLAA the render
        // grid matches the source, so a sub-texel jitter lands on the same texel
        // whichever way it points and all four candidates score identically.
        const QualityMode* probeMode = nullptr;
        OptimalSettings probeSettings{};
        for (const QualityMode& q : qualities) {
            OptimalSettings s;
            try {
                s = ngx.GetOptimal(source.w, source.h, q.value);
            } catch (const ToolError&) {
                continue;
            }
            if (!probeMode || s.renderW < probeSettings.renderW) {
                probeMode = &q;
                probeSettings = s;
            }
        }
        if (!probeMode) throw ToolError("no usable quality mode to probe the jitter sign with");

        LowResSet& lr = lowResFor(probeSettings.renderW, probeSettings.renderH);
        const DlssFeatureDesc desc = makeDesc(*probeMode, presets[0], probeSettings);
        const int probeFrames = std::min(o.frames, 16);
        const int probePhases = phasesFor(probeSettings.renderW);

        LogInfo("Probing jitter sign convention on %s, %d frames each...", probeMode->label,
                probeFrames);
        double best = -1.0, worst = 1e9;
        JitterSign bestSign = sign;
        for (const char* candidate : {"++", "+-", "-+", "--"}) {
            JitterSign s;
            CHECK(ParseJitterSign(candidate, &s));
            const AccumulationResult probe = RunAccumulation(
                gpu, ngx, makeParams(lr, desc, s, probeFrames, probePhases, false));
            const Metrics m = ComputeMetrics(source, probe.output);
            LogInfo("   %s  PSNR %6.2f dB", candidate, m.psnrRgb);
            worst = std::min(worst, m.psnrRgb);
            if (m.psnrRgb > best) {
                best = m.psnrRgb;
                bestSign = s;
            }
        }
        // Below about a tenth of a dB the candidates are indistinguishable and
        // picking the best one would just be picking noise.
        if (best - worst < 0.1) {
            LogWarn("all four signs scored within %.2f dB - the probe cannot tell them apart "
                    "at this upscale factor; using %s",
                    best - worst, sign.name.c_str());
        } else {
            sign = bestSign;
            LogInfo("  -> using %s", sign.name.c_str());
        }
        LogInfo("");
    } else if (!ParseJitterSign(o.jitterSign, &sign)) {
        throw ToolError("--jitter-sign must be auto, ++, +-, -+ or --");
    }

    // ---- Main sweep --------------------------------------------------------
    const std::string stem = fs::path(o.input).stem().string();
    std::vector<RunResult> results;
    std::map<uint64_t, std::string> seenHashes;

    for (const QualityMode& q : qualities) {
        OptimalSettings settings;
        try {
            settings = ngx.GetOptimal(source.w, source.h, q.value);
        } catch (const ToolError& e) {
            LogWarn("%s: skipped (%s)", q.label, e.what());
            continue;
        }
        LowResSet& lr = lowResFor(settings.renderW, settings.renderH);
        const int phases = phasesFor(settings.renderW);

        for (unsigned preset : presets) {
            RunResult r;
            r.quality = q.label;
            r.preset = PresetName(preset);
            r.renderW = settings.renderW;
            r.renderH = settings.renderH;
            r.targetW = static_cast<unsigned>(source.w);
            r.targetH = static_cast<unsigned>(source.h);

            const DlssFeatureDesc desc = makeDesc(q, preset, settings);
            AccumulationResult run;
            try {
                run = RunAccumulation(gpu, ngx,
                                      makeParams(lr, desc, sign, o.frames, phases, o.saveLowRes));
            } catch (const ToolError& e) {
                LogWarn("%s preset %s: %s", q.label, r.preset.c_str(), e.what());
                continue;
            }
            const ImageF& img = run.output;
            r.msPerFrame = run.msPerFrame;

            const Metrics m = ComputeMetrics(source, img);
            r.psnr = m.psnrRgb;
            r.ssim = m.ssimLuma;
            r.hash = HashImage(img);

            const std::string name = stem + "_" + q.name + "_p" + r.preset + ".png";
            const fs::path outFile = fs::path(o.outDir) / name;
            if (!o.metricsOnly) {
                r.file = outFile.string();
                if (o.png16) {
                    SavePng16(r.file, img);
                } else {
                    SavePng8(r.file, img);
                }
            }
            if (o.saveLowRes && !run.lowResInput.empty()) {
                const fs::path lrFile =
                    fs::path(o.outDir) / (stem + "_" + q.name + "_p" + r.preset + "_input.png");
                SavePng8(lrFile.string(), run.lowResInput);
            }
            if (o.saveDiff) {
                const fs::path diffFile =
                    fs::path(o.outDir) / (stem + "_" + q.name + "_p" + r.preset + "_diff.png");
                const DiffStats d = SaveDiffPng(diffFile.string(), source, img, o.diffGain);
                r.diffFile = diffFile.string();
                r.maxErr = d.maxAbs;
                r.meanErr = d.meanAbs;
            }

            auto seen = seenHashes.find(r.hash);
            if (seen != seenHashes.end()) {
                r.sameAs = seen->second;
            } else {
                seenHashes.emplace(r.hash, q.label + std::string("/") + r.preset);
            }

            const std::string dupNote = r.sameAs.empty() ? std::string() : ("   == " + r.sameAs);
            LogInfo("%-16s preset %-7s %4ux%-4u -> %ux%u   PSNR %6.2f dB   SSIM %.4f   %6.2f ms%s",
                    q.label, r.preset.c_str(), r.renderW, r.renderH, r.targetW, r.targetH, r.psnr,
                    r.ssim, r.msPerFrame, dupNote.c_str());
            results.push_back(std::move(r));
        }
    }

    if (results.empty()) throw ToolError("no runs completed");

    // ---- Summary -----------------------------------------------------------
    LogInfo("");
    LogInfo("Ranked by PSNR:");
    std::vector<const RunResult*> ranked;
    for (const RunResult& r : results) ranked.push_back(&r);
    std::sort(ranked.begin(), ranked.end(),
              [](const RunResult* a, const RunResult* b) { return a->psnr > b->psnr; });
    for (size_t i = 0; i < ranked.size(); ++i) {
        LogInfo("  %2zu. %-16s preset %-7s  %6.2f dB   SSIM %.4f   %6.2f ms", i + 1,
                ranked[i]->quality.c_str(), ranked[i]->preset.c_str(), ranked[i]->psnr,
                ranked[i]->ssim, ranked[i]->msPerFrame);
    }

    std::vector<std::string> dupes;
    for (const RunResult& r : results) {
        if (!r.sameAs.empty()) {
            dupes.push_back(r.quality + "/" + r.preset + " == " + r.sameAs);
        }
    }
    if (!dupes.empty()) {
        LogInfo("");
        LogInfo("Byte-identical outputs (preset had no effect, most likely a silent fallback):");
        for (const std::string& d : dupes) LogInfo("  %s", d.c_str());
    }

    WriteJson(o.jsonPath, o, source, dllPath, dllVersion, sign, results);
    LogInfo("");
    LogInfo("Wrote %zu image(s) to %s and results to %s", results.size(), o.outDir.c_str(),
            o.jsonPath.c_str());

    // Tear NGX down before the D3D12 device goes away.
    lrCache.clear();
    ngx.Shutdown();
    return 0;
}

int main(int argc, char** argv) {
    // Windows hands main() ANSI (code-page) argv, but the tool treats every string as UTF-8.
    // Rebuild argv from the wide command line so non-ASCII paths (e.g. Cyrillic) survive.
    std::vector<std::string> utf8Args;
    std::vector<char*> utf8Argv;
    int wArgc = 0;
    LPWSTR* wArgv = CommandLineToArgvW(GetCommandLineW(), &wArgc);
    if (wArgv) {
        utf8Args.reserve(wArgc);
        for (int i = 0; i < wArgc; ++i) utf8Args.push_back(Widen2Narrow(wArgv[i]));
        LocalFree(wArgv);
        utf8Argv.reserve(utf8Args.size() + 1);
        for (auto& s : utf8Args) utf8Argv.push_back(s.data());
        utf8Argv.push_back(nullptr);
        argc = static_cast<int>(utf8Args.size());
        argv = utf8Argv.data();
    }
    try {
        return Run(argc, argv);
    } catch (const ToolError& e) {
        LogErr("%s", e.what());
        return 1;
    } catch (const std::exception& e) {
        LogErr("unexpected: %s", e.what());
        return 2;
    }
}
