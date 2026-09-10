// DLSS Neural Rendering caller-gate shim.
//
// The snippet (nvngx_dlssnr.dll) resolves the module owning its caller's return address via
// RtlPcToFileHeader and rejects any whose path does not contain "nvngx.dll" (the driver core is
// _nvngx.dll), returning FAIL_PlatformError before it looks at a single argument. video2dlssnr.exe fails
// that test. This DLL exists only to be named "nvngx.dll_dlssnr.dll", so calls into the snippet
// originate from a module the snippet accepts. It contains no NVIDIA code.
//
// The parameter block passed in is the driver core's capability block; its setters are driven by
// raw vtable slot because the core exports no Set/Get helpers and its block does not match the SDK
// header's layout (uint=3, resources via the ULL setter at 0, float probed on this driver).
//
// Call order:
//   PopulateParameters_Impl(caps)  -> Init_Ext(appId, dataPath, device, 0x15, caps)  -> CreateFeature(18)
// The driver core is never asked to create feature 18: from driver 616.64 on its loader routes that
// request into the snippet itself, and snippet 310.8.0.0 faults inside D3D12 on that route. Driving
// the snippet's own exports from here sidesteps the core's route entirely.
//
// Every snippet entry point is wrapped in SEH so a fault inside NVIDIA's code is reported to the host
// (stage, code, address, module) instead of killing the process silently.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include "nr_params.h"

namespace {

using PFN_SetULL = void(__thiscall*)(void*, const char*, unsigned long long);
using PFN_SetFloat = void(__thiscall*)(void*, const char*, float);
using PFN_SetUInt = void(__thiscall*)(void*, const char*, unsigned int);

int g_uintSlot = 3;
int g_floatSlot = 6;  // host confirms via probe; 6 on driver 616.56 and 616.64

void setUInt(void* p, const char* n, unsigned int v) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PFN_SetUInt>(vt[g_uintSlot])(p, n, v);
}
void setFloat(void* p, const char* n, float v) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PFN_SetFloat>(vt[g_floatSlot])(p, n, v);
}
void setResource(void* p, const char* n, ID3D12Resource* r) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PFN_SetULL>(vt[0])(p, n, reinterpret_cast<unsigned long long>(r));
}

using PFN_NrPopulate = int(__cdecl*)(const void*);
using PFN_NrInitExt = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int,
                                    const void*);
using PFN_NrCreate = int(__cdecl*)(ID3D12GraphicsCommandList*, int, const void*, void**);
using PFN_NrEvaluate = int(__cdecl*)(ID3D12GraphicsCommandList*, const void*, const void*, void*);
using PFN_NrRelease = int(__cdecl*)(void*);

struct Snippet {
    HMODULE module = nullptr;
    PFN_NrPopulate populate = nullptr;
    PFN_NrInitExt init = nullptr;
    PFN_NrCreate create = nullptr;
    PFN_NrEvaluate evaluate = nullptr;
    PFN_NrRelease release = nullptr;
    bool populated = false;
    bool initialised = false;
};
Snippet g;

bool load(const wchar_t* path) {
    if (g.module) return g.create != nullptr;
    g.module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g.module) return false;
    g.populate = (PFN_NrPopulate) GetProcAddress(g.module, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    g.init = (PFN_NrInitExt) GetProcAddress(g.module, "NVSDK_NGX_D3D12_Init_Ext");
    g.create = (PFN_NrCreate) GetProcAddress(g.module, "NVSDK_NGX_D3D12_CreateFeature");
    g.evaluate = (PFN_NrEvaluate) GetProcAddress(g.module, "NVSDK_NGX_D3D12_EvaluateFeature");
    g.release = (PFN_NrRelease) GetProcAddress(g.module, "NVSDK_NGX_D3D12_ReleaseFeature");
    return g.create && g.evaluate;
}

// Returned by the guarded calls when the snippet raised a structured exception. Not
// NVSDK_NGX_Result_Fail (0xBAD00000), so a caught fault stays distinguishable from a genuine
// generic failure; still matches NVSDK_NGX_FAILED()'s 0xFFF00000 mask.
constexpr int kFwdCrashed = (int) 0xBAD000FF;

}  // namespace

extern "C" {

__declspec(dllexport) int fwd_last_populate = 0;  // -2: the snippet has no PopulateParameters_Impl
__declspec(dllexport) int fwd_last_init = 0;
__declspec(dllexport) int fwd_last_create = 0;

// Filled by the __except filter below. Stage: 1 populate, 2 init, 3 create, 4 evaluate, 5 release.
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
int GuardedPopulate(PFN_NrPopulate fn, const void* caps) {
    __try {
        volatile int r = fn(caps);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 1)) {
        return kFwdCrashed;
    }
}
int GuardedInit(PFN_NrInitExt fn, unsigned long long app, const wchar_t* data, ID3D12Device* dev,
                int version, const void* caps) {
    __try {
        volatile int r = fn(app, data, dev, version, caps);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 2)) {
        return kFwdCrashed;
    }
}
int GuardedCreate(PFN_NrCreate fn, ID3D12GraphicsCommandList* cmd, const void* caps, void** h) {
    __try {
        volatile int r = fn(cmd, 18, caps, h);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 3)) {
        return kFwdCrashed;
    }
}
int GuardedEvaluate(PFN_NrEvaluate fn, ID3D12GraphicsCommandList* cmd, const void* feature,
                    const void* caps) {
    __try {
        volatile int r = fn(cmd, feature, caps, nullptr);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 4)) {
        return kFwdCrashed;
    }
}
int GuardedRelease(PFN_NrRelease fn, void* feature) {
    __try {
        volatile int r = fn(feature);
        return r;
    } __except (RecordCrash(GetExceptionInformation(), 5)) {
        return kFwdCrashed;
    }
}

}  // namespace

extern "C" {

__declspec(dllexport) void fwd_set_slots(int uintSlot, int floatSlot) {
    if (uintSlot >= 0 && uintSlot < 8) g_uintSlot = uintSlot;
    if (floatSlot >= 0 && floatSlot < 8) g_floatSlot = floatSlot;
}

// Create a Neural Rendering feature by driving the snippet's own PopulateParameters_Impl + Init_Ext +
// CreateFeature, from this (accepted) module. capabilityParams is the driver core's capability block.
// Returns nullptr on failure; fwd_last_populate / fwd_last_init / fwd_last_create and the
// fwd_last_exception_* globals say why.
__declspec(dllexport) void* fwd_create(const wchar_t* snippetPath, const wchar_t* dataPath,
                                       ID3D12Device* device, ID3D12GraphicsCommandList* cmd,
                                       void* caps, unsigned int width, unsigned int height,
                                       unsigned int outWidth, unsigned int outHeight,
                                       const NrModelParams* mp) {
    fwd_last_exception_stage = 0;
    if (!load(snippetPath) || !caps || !mp) return nullptr;
    if (!g.initialised) {
        // First let the snippet publish its callbacks into the core's block (this is what a core
        // CreateFeature(18) used to do for us on drivers before 616.64), then the snippet's own
        // Init_Ext. Populate is advisory: a snippet without the export still initialises.
        if (!g.populated) {
            fwd_last_populate = g.populate ? GuardedPopulate(g.populate, caps) : -2;
            if (fwd_last_populate == kFwdCrashed) return nullptr;
            g.populated = true;
        }
        if (!g.init) return nullptr;
        // Generic application id; SDK version 0x15 (21). The fifth argument is the capability
        // block, not a FeatureCommonInfo - that is what the working path passes.
        fwd_last_init = GuardedInit(g.init, 0x24480451ull, dataPath, device, 0x0000015, caps);
        g.initialised = (fwd_last_init == 1);
        if (!g.initialised) return nullptr;
    }

    // Generic NGX size params the create path reads, plus every DLSSNR model param.
    // All are latched now, at create - values written only at evaluate are ignored.
    // Only keys the snippet actually carries are written: a scan of nvngx_dlssnr.dll 310.8 for
    // "DLSSNR." yields 61 names, and GlobalToneStrength / Scale / Upscaling are not among them
    // (the block is string-keyed, so writing them was harmless but did nothing).
    setUInt(caps, "Width", width);
    setUInt(caps, "Height", height);
    setUInt(caps, "OutWidth", outWidth);
    setUInt(caps, "OutHeight", outHeight);
    setUInt(caps, "PerfQualityValue", 2);
    setUInt(caps, "CreationNodeMask", 1);
    setUInt(caps, "VisibilityNodeMask", 1);
    setUInt(caps, "DLSSNR.Enabled", 1);
    setUInt(caps, "DLSSNR.Width", width);
    setUInt(caps, "DLSSNR.Height", height);
    // Written unconditionally, including 0: the block belongs to the driver and outlives the
    // feature, so a skipped "default" would leave the previous preset in place.
    setUInt(caps, "DLSSNR.Hint.Render.Preset", mp->preset);
    setFloat(caps, "DLSSNR.Intensity", mp->intensity);
    setUInt(caps, "DLSSNR.Style", mp->style);
    setFloat(caps, "DLSSNR.LocalStructureStrength", mp->localStructure);
    setFloat(caps, "DLSSNR.LocalToneStrength", mp->localTone);
    if (mp->skinStructure >= 0.0f) setFloat(caps, "DLSSNR.SkinStructureStrength", mp->skinStructure);
    setUInt(caps, "DLSSNR.UseAutoMask", mp->autoMask);
    setUInt(caps, "DLSSNR.UICorrection", mp->uiCorrection);

    // Upscaling: when the output is larger than the input, the model does the
    // super-resolution itself. Width/Height above are the render (input) size,
    // OutWidth/OutHeight the display (target) size.
    const bool upscaling = (outWidth != width || outHeight != height);
    if (upscaling && width > 0) {
        setFloat(caps, "DLSSNR.ScalingRatio", (float) outWidth / (float) width);
    }

    void* handle = nullptr;
    fwd_last_create = GuardedCreate(g.create, cmd, caps, &handle);
    return (fwd_last_create == 1) ? handle : nullptr;
}

__declspec(dllexport) int fwd_evaluate(ID3D12GraphicsCommandList* cmd, void* feature, void* caps,
                                       ID3D12Resource* color, ID3D12Resource* depth,
                                       ID3D12Resource* motion, ID3D12Resource* output,
                                       unsigned int width, unsigned int height, unsigned int outW,
                                       unsigned int outH, int reset) {
    if (!feature || !caps || !g.evaluate) return 0;
    fwd_last_exception_stage = 0;
    setResource(caps, "DLSSNR.Color", color);
    setResource(caps, "DLSSNR.Depth", depth);
    setResource(caps, "DLSSNR.MVec", motion);
    setResource(caps, "DLSSNR.Output", output);
    setUInt(caps, "DLSSNR.Enabled", 1);
    setUInt(caps, "DLSSNR.Width", width);
    setUInt(caps, "DLSSNR.Height", height);
    setUInt(caps, "DLSSNR.Reset", (unsigned int) reset);
    setUInt(caps, "DLSSNR.DepthInverted", 0);
    setUInt(caps, "DLSSNR.ColorSubrectBaseX", 0);
    setUInt(caps, "DLSSNR.ColorSubrectBaseY", 0);
    setUInt(caps, "DLSSNR.ColorSubrectWidth", width);
    setUInt(caps, "DLSSNR.ColorSubrectHeight", height);
    setUInt(caps, "DLSSNR.OutputSubrectBaseX", 0);
    setUInt(caps, "DLSSNR.OutputSubrectBaseY", 0);
    setUInt(caps, "DLSSNR.OutputSubrectWidth", outW);
    setUInt(caps, "DLSSNR.OutputSubrectHeight", outH);
    setUInt(caps, "DLSSNR.DepthSubrectBaseX", 0);
    setUInt(caps, "DLSSNR.DepthSubrectBaseY", 0);
    setUInt(caps, "DLSSNR.DepthSubrectWidth", width);
    setUInt(caps, "DLSSNR.DepthSubrectHeight", height);
    setUInt(caps, "DLSSNR.MVecSubrectBaseX", 0);
    setUInt(caps, "DLSSNR.MVecSubrectBaseY", 0);
    setUInt(caps, "DLSSNR.MVecSubrectWidth", width);
    setUInt(caps, "DLSSNR.MVecSubrectHeight", height);
    setFloat(caps, "DLSSNR.MVecScaleX", 1.0f);
    setFloat(caps, "DLSSNR.MVecScaleY", 1.0f);
    return GuardedEvaluate(g.evaluate, cmd, feature, caps);
}

__declspec(dllexport) void fwd_release(void* feature) {
    if (feature && g.release) {
        fwd_last_exception_stage = 0;
        (void) GuardedRelease(g.release, feature);
    }
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
