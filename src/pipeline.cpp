#include "pipeline.h"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::wstring> DefaultDllSearchPaths(const std::string& explicitDir) {
    std::vector<std::wstring> paths;
    auto add = [&](const fs::path& p) {
        std::error_code ec;
        if (p.empty() || !fs::exists(p, ec)) return;
        const std::wstring w = fs::absolute(p, ec).wstring();
        if (std::find(paths.begin(), paths.end(), w) == paths.end()) paths.push_back(w);
    };

    if (!explicitDir.empty()) {
        // Our narrow strings are UTF-8; fs::path(std::string) would read them as the ANSI code
        // page and mangle non-ASCII paths (e.g. Cyrillic), so widen explicitly.
        const fs::path p = Narrow2Widen(explicitDir);
        if (!fs::exists(p)) throw ToolError("--dll-dir does not exist: " + explicitDir);
        add(p);
    }

    wchar_t exePath[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, exePath, MAX_PATH)) {
        const fs::path exeDir = fs::path(exePath).parent_path();
        add(exeDir);
        add(exeDir.parent_path());
        add(exeDir.parent_path().parent_path());
    }
    std::error_code ec;
    add(fs::current_path(ec));
    return paths;
}

LowResSet CreateLowResSet(GpuContext& gpu, unsigned w, unsigned h) {
    LowResSet set;
    set.color = gpu.CreateTexture(static_cast<int>(w), static_cast<int>(h),
                                  DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"lrColor");
    set.depth = gpu.CreateTexture(static_cast<int>(w), static_cast<int>(h), DXGI_FORMAT_R32_FLOAT,
                                  true, L"lrDepth");
    set.motion = gpu.CreateTexture(static_cast<int>(w), static_cast<int>(h),
                                   DXGI_FORMAT_R16G16_FLOAT, true, L"lrMotion");
    gpu.Begin();
    gpu.Transition(set.color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(set.depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(set.motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();
    return set;
}

GpuTexture CreateOutputTexture(GpuContext& gpu, unsigned w, unsigned h) {
    GpuTexture out = gpu.CreateTexture(static_cast<int>(w), static_cast<int>(h),
                                       DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"dlssOutput");
    gpu.Begin();
    gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();
    return out;
}

static void DecodeSrgbInPlace(ImageF& img) {
    for (size_t i = 0; i < img.pixels(); ++i) {
        for (int c = 0; c < 3; ++c) img.px[i * 4 + c] = SrgbToLinear(img.px[i * 4 + c]);
    }
}

AccumulationResult RunAccumulation(GpuContext& gpu, NgxSession& ngx,
                                   const AccumulationParams& p) {
    CHECK(p.source && p.output && p.lowRes);
    CHECK(p.frames >= 1);

    DlssFeature feature;
    {
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        feature.Create(ngx, cl, p.feature);
        gpu.EndAndWait();
    }

    double totalMs = 0.0;
    int timedFrames = 0;

    for (int f = 0; f < p.frames; ++f) {
        const int phase = (p.phases > 0) ? (f % p.phases) : f;
        const float jx = static_cast<float>(Halton(phase + 1, 2) - 0.5);
        const float jy = static_cast<float>(Halton(phase + 1, 3) - 0.5);

        ID3D12GraphicsCommandList* cl = gpu.Begin();

        DownsampleArgs da;
        da.src = p.source;
        da.dstColor = &p.lowRes->color;
        da.dstDepth = &p.lowRes->depth;
        da.dstMotion = &p.lowRes->motion;
        da.jitterX = p.sign.x * jx;
        da.jitterY = p.sign.y * jy;
        da.filter = p.filter;
        da.encodeSrgb = p.encodeSrgb;
        da.depthValue = p.depthValue;
        gpu.RecordDownsample(da);

        gpu.UavBarrier(p.lowRes->color);
        gpu.UavBarrier(p.lowRes->depth);
        gpu.UavBarrier(p.lowRes->motion);

        DlssEvalInputs in;
        in.color = &p.lowRes->color;
        in.depth = &p.lowRes->depth;
        in.motion = &p.lowRes->motion;
        in.output = p.output;
        in.exposure = p.exposure;
        // DLSS always receives the canonical jitter value; only the direction the
        // source is sampled in flips, which is what JitterSign selects.
        in.jitterX = jx;
        in.jitterY = jy;
        in.reset = (f == 0);

        gpu.RecordTimestampBegin();
        feature.Evaluate(cl, in);
        gpu.RecordTimestampEnd();
        gpu.EndAndWait();

        // The first frame carries feature warm-up cost; leave it out of the average.
        if (f > 0) {
            totalMs += gpu.LastGpuMs();
            ++timedFrames;
        }
    }

    AccumulationResult result;
    result.msPerFrame = timedFrames ? (totalMs / timedFrames) : 0.0;

    if (p.captureLowRes) {
        result.lowResInput.w = p.lowRes->color.w;
        result.lowResInput.h = p.lowRes->color.h;
        result.lowResInput.px = gpu.ReadbackRgbaFloat(p.lowRes->color);
        // The low-res frame is in whatever space DLSS was fed, which is exactly
        // what decodeSrgbOnReadback describes — including display-space mode,
        // where the shader did not encode because the source already was.
        if (p.decodeSrgbOnReadback) DecodeSrgbInPlace(result.lowResInput);
    }

    result.output.w = p.output->w;
    result.output.h = p.output->h;
    result.output.px = gpu.ReadbackRgbaFloat(*p.output);
    if (p.decodeSrgbOnReadback) DecodeSrgbInPlace(result.output);

    feature.Release();
    return result;
}
