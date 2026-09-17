// DLSS Neural Rendering caller-gate shim.
//
// The snippet (nvngx_dlssnr.dll) resolves the module owning its caller's return address via
// RtlPcToFileHeader and rejects any whose path does not contain "nvngx.dll" (the driver core is
// _nvngx.dll), returning FAIL_PlatformError before it looks at a single argument. video2dlssnr.exe fails
// that test. This DLL exists only to be named "nvngx.dll_dlssnr.dll", so calls into the snippet
// originate from a module the snippet accepts. It contains no NVIDIA code.
//
// The snippet's own exports are driven directly (Init_Ext, CreateFeature, EvaluateFeature,
// ReleaseFeature) on a parameter block the host allocated with NVSDK_NGX_D3D12_AllocateParameters.
// The driver core is never asked to create feature 18: from driver 616.64 on its loader routes
// that request into the snippet itself, and snippet 310.8.0.0 faults inside D3D12 on that route.
//
// Every snippet entry point is wrapped in SEH so a fault inside NVIDIA's code is reported to the host
// (stage, code, address, module) instead of killing the process silently.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include "nr_params.h"
#include "nvsdk_ngx_params.h"

namespace {

using PFN_NrInitExt = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int,
                                    const void*);
using PFN_NrCreate = int(__cdecl*)(ID3D12GraphicsCommandList*, int, const void*, void**);
using PFN_NrEvaluate = int(__cdecl*)(ID3D12GraphicsCommandList*, const void*, const void*, void*);
using PFN_NrRelease = int(__cdecl*)(void*);

struct Snippet {
    HMODULE module = nullptr;
    PFN_NrInitExt init = nullptr;
    PFN_NrCreate create = nullptr;
    PFN_NrEvaluate evaluate = nullptr;
    PFN_NrRelease release = nullptr;
    bool initialised = false;
};
Snippet g;

bool load(const wchar_t* path) {
    if (g.module) return g.create != nullptr;
    g.module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g.module) return false;
    g.init = (PFN_NrInitExt) GetProcAddress(g.module, "NVSDK_NGX_D3D12_Init_Ext");
    g.create = (PFN_NrCreate) GetProcAddress(g.module, "NVSDK_NGX_D3D12_CreateFeature");
    g.evaluate = (PFN_NrEvaluate) GetProcAddress(g.module, "NVSDK_NGX_D3D12_EvaluateFeature");
    g.release = (PFN_NrRelease) GetProcAddress(g.module, "NVSDK_NGX_D3D12_ReleaseFeature");
    return g.init && g.create && g.evaluate;
}

// Returned by the guarded calls when the snippet raised a structured exception. Not
// NVSDK_NGX_Result_Fail (0xBAD00000), so a caught fault stays distinguishable from a genuine
// generic failure; still matches NVSDK_NGX_FAILED()'s 0xFFF00000 mask.
constexpr int kFwdCrashed = (int) 0xBAD000FF;

}  // namespace

extern "C" {

__declspec(dllexport) int fwd_last_init = 0;
__declspec(dllexport) int fwd_last_create = 0;

// Filled by the __except filter below. Stage: 1 init, 2 create, 3 evaluate, 4 release.
__declspec(dllexport) int fwd_last_exception_stage = 0;
__declspec(dllexport) unsigned int fwd_last_exception_code = 0;
__declspec(dllexport) unsigned long long fwd_last_exception_addr = 0;
__declspec(dllexport) unsigned long long fwd_last_exception_module_base = 0;
__declspec(dllexport) wchar_t fwd_last_exception_module[MAX_PATH] = {};

}  // extern "C"

namespace {

// The __except filter: records where the snippet faulted and takes the handler.
int RecordCrash(EXCEPTION_POINTERS* ep, int stage) {
    fwd_last_exception_stage = stage;
    fwd_last_exception_code = 0;
    fwd_last_exception_addr = 0;
    fwd_last_exception_module_base = 0;
    fwd_last_exception_module[0] = 0;
    if (ep && ep->ExceptionRecord) {
        fwd_last_exception_code = ep->ExceptionRecord->ExceptionCode;
        fwd_last_exception_addr =
            reinterpret_cast<unsigned long long>(ep->ExceptionRecord->ExceptionAddress);
        HMODULE m = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(ep->ExceptionRecord->ExceptionAddress),
                               &m) &&
            m) {
            fwd_last_exception_module_base = reinterpret_cast<unsigned long long>(m);
            GetModuleFileNameW(m, fwd_last_exception_module, MAX_PATH);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

// One guarded helper per snippet entry point. These functions must stay free of C++ objects with
// destructors: under /EHsc a function that needs unwinding cannot contain __try (C2712). The
// result goes through a volatile so the call is not turned into a tail jump - the snippet resolves
// its caller from the return address, and that has to be this module.
int GuardedInit(PFN_NrInitExt fn, unsigned long long app, const wchar_t* data, ID3D12Device* dev,
                int version, const void* params) {
    __try {
        volatile int r = fn(app, data, dev, version, params);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 1)) {
        return kFwdCrashed;
    }
}
int GuardedCreate(PFN_NrCreate fn, ID3D12GraphicsCommandList* cmd, const void* params, void** h) {
    __try {
        volatile int r = fn(cmd, 18, params, h);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 2)) {
        return kFwdCrashed;
    }
}
int GuardedEvaluate(PFN_NrEvaluate fn, ID3D12GraphicsCommandList* cmd, const void* feature,
                    const void* params) {
    __try {
        volatile int r = fn(cmd, feature, params, nullptr);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 3)) {
        return kFwdCrashed;
    }
}
int GuardedRelease(PFN_NrRelease fn, void* feature) {
    __try {
        volatile int r = fn(feature);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 4)) {
        return kFwdCrashed;
    }
}

}  // namespace

extern "C" {

// Create a Neural Rendering feature by driving the snippet's own Init_Ext + CreateFeature from this
// (accepted) module. `params` is a block from NVSDK_NGX_D3D12_AllocateParameters, owned by the host.
// Returns nullptr on failure; fwd_last_init / fwd_last_create and the fwd_last_exception_* globals
// say why.
__declspec(dllexport) void* fwd_create(const wchar_t* snippetPath, const wchar_t* dataPath,
                                       ID3D12Device* device, ID3D12GraphicsCommandList* cmd,
                                       NVSDK_NGX_Parameter* params, unsigned int width,
                                       unsigned int height, unsigned int outWidth,
                                       unsigned int outHeight, const NrModelParams* mp) {
    fwd_last_exception_stage = 0;
    if (!load(snippetPath) || !params || !mp) return nullptr;
    if (!g.initialised) {
        // Generic application id; SDK version 0x15 (21).
        fwd_last_init = GuardedInit(g.init, 0x24480451ull, dataPath, device, 0x0000015, params);
        g.initialised = (fwd_last_init == 1);
        if (!g.initialised) return nullptr;
    }

    // Everything the model reads is latched here, at create; values written only at evaluate are
    // ignored. Only keys the snippet carries are written: a scan of nvngx_dlssnr.dll 310.8 for
    // "DLSSNR." yields 61 names.
    params->Reset();
    params->Set("Width", width);
    params->Set("Height", height);
    params->Set("OutWidth", outWidth);
    params->Set("OutHeight", outHeight);
    params->Set("PerfQualityValue", 2u);
    params->Set("CreationNodeMask", 1u);
    params->Set("VisibilityNodeMask", 1u);
    params->Set("DLSSNR.Enabled", 1u);
    params->Set("DLSSNR.Width", width);
    params->Set("DLSSNR.Height", height);
    params->Set("DLSSNR.Hint.Render.Preset", mp->preset);
    params->Set("DLSSNR.Intensity", mp->intensity);
    params->Set("DLSSNR.Style", mp->style);
    params->Set("DLSSNR.LocalStructureStrength", mp->localStructure);
    params->Set("DLSSNR.LocalToneStrength", mp->localTone);
    if (mp->skinStructure >= 0.0f) params->Set("DLSSNR.SkinStructureStrength", mp->skinStructure);
    params->Set("DLSSNR.UseAutoMask", mp->autoMask);
    params->Set("DLSSNR.UICorrection", mp->uiCorrection);

    // "Upscaling" does not make the model enlarge anything: it is a same-resolution network.
    // Colour and Output are both the display-size frame; Width/Height above name the (smaller)
    // network resolution and ScalingRatio is network/display, the NGX render-over-output
    // convention (see DLSSNRComputeScalingRatioCallback and the snippet's "Invalid network
    // dimensions %ux%u from output %ux%u scale %.3f"). The tool itself never uses it - DLSS SR
    // does the enlarge, NR runs 1:1 - so this only matters to --probe-nr.
    const bool upscaling = (outWidth != width || outHeight != height);
    if (upscaling && width > 0) {
        params->Set("DLSSNR.ScalingRatio", (float) width / (float) outWidth);
    }

    void* handle = nullptr;
    fwd_last_create = GuardedCreate(g.create, cmd, params, &handle);
    return (fwd_last_create == 1) ? handle : nullptr;
}

__declspec(dllexport) int fwd_evaluate(ID3D12GraphicsCommandList* cmd, void* feature,
                                       NVSDK_NGX_Parameter* params, ID3D12Resource* color,
                                       ID3D12Resource* depth, ID3D12Resource* motion,
                                       ID3D12Resource* output, unsigned int width,
                                       unsigned int height, unsigned int outW, unsigned int outH,
                                       int reset) {
    if (!feature || !params || !g.evaluate) return 0;
    fwd_last_exception_stage = 0;
    params->Set("DLSSNR.Color", color);
    params->Set("DLSSNR.Depth", depth);
    params->Set("DLSSNR.MVec", motion);
    params->Set("DLSSNR.Output", output);
    params->Set("DLSSNR.Enabled", 1u);
    params->Set("DLSSNR.Reset", (unsigned int) reset);
    params->Set("DLSSNR.DepthInverted", 0u);
    params->Set("DLSSNR.ColorSubrectBaseX", 0u);
    params->Set("DLSSNR.ColorSubrectBaseY", 0u);
    params->Set("DLSSNR.ColorSubrectWidth", width);
    params->Set("DLSSNR.ColorSubrectHeight", height);
    params->Set("DLSSNR.OutputSubrectBaseX", 0u);
    params->Set("DLSSNR.OutputSubrectBaseY", 0u);
    params->Set("DLSSNR.OutputSubrectWidth", outW);
    params->Set("DLSSNR.OutputSubrectHeight", outH);
    params->Set("DLSSNR.DepthSubrectBaseX", 0u);
    params->Set("DLSSNR.DepthSubrectBaseY", 0u);
    params->Set("DLSSNR.DepthSubrectWidth", width);
    params->Set("DLSSNR.DepthSubrectHeight", height);
    params->Set("DLSSNR.MVecSubrectBaseX", 0u);
    params->Set("DLSSNR.MVecSubrectBaseY", 0u);
    params->Set("DLSSNR.MVecSubrectWidth", width);
    params->Set("DLSSNR.MVecSubrectHeight", height);
    params->Set("DLSSNR.MVecScaleX", 1.0f);
    params->Set("DLSSNR.MVecScaleY", 1.0f);
    return GuardedEvaluate(g.evaluate, cmd, feature, params);
}

__declspec(dllexport) void fwd_release(void* feature) {
    if (feature && g.release) {
        fwd_last_exception_stage = 0;
        (void) GuardedRelease(g.release, feature);
    }
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
