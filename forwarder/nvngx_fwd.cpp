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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include "nr_params.h"

namespace {

using PFN_SetULL = void(__thiscall*)(void*, const char*, unsigned long long);
using PFN_SetFloat = void(__thiscall*)(void*, const char*, float);
using PFN_SetUInt = void(__thiscall*)(void*, const char*, unsigned int);

int g_uintSlot = 3;
int g_floatSlot = 6;  // host confirms via probe; 6 on driver 616.56

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
    return g.create && g.evaluate;
}

}  // namespace

extern "C" {

__declspec(dllexport) int fwd_last_init = 0;
__declspec(dllexport) int fwd_last_create = 0;

__declspec(dllexport) void fwd_set_slots(int uintSlot, int floatSlot) {
    if (uintSlot >= 0 && uintSlot < 8) g_uintSlot = uintSlot;
    if (floatSlot >= 0 && floatSlot < 8) g_floatSlot = floatSlot;
}

// Create a Neural Rendering feature by driving the snippet's own Init_Ext + CreateFeature, from this
// (accepted) module. capabilityParams is the driver core's capability block.
__declspec(dllexport) void* fwd_create(const wchar_t* snippetPath, const wchar_t* dataPath,
                                       ID3D12Device* device, ID3D12GraphicsCommandList* cmd,
                                       void* caps, unsigned int width, unsigned int height,
                                       unsigned int outWidth, unsigned int outHeight,
                                       const NrModelParams* mp) {
    if (!load(snippetPath) || !caps || !mp) return nullptr;
    if (!g.initialised && g.init) {
        // Generic application id; SDK version 0x15 (21).
        fwd_last_init = g.init(0x24480451ull, dataPath, device, 0x0000015, caps);
        g.initialised = (fwd_last_init == 1);
        if (!g.initialised) return nullptr;
    }

    // Generic NGX size params the create path reads, plus every DLSSNR model param.
    // All are latched now, at create - values written only at evaluate are ignored.
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
    setUInt(caps, "DLSSNR.Hint.Render.Preset", mp->preset);
    setFloat(caps, "DLSSNR.Intensity", mp->intensity);
    setUInt(caps, "DLSSNR.Style", mp->style);
    setFloat(caps, "DLSSNR.LocalStructureStrength", mp->localStructure);
    setFloat(caps, "DLSSNR.LocalToneStrength", mp->localTone);
    if (mp->skinStructure >= 0.0f) setFloat(caps, "DLSSNR.SkinStructureStrength", mp->skinStructure);
    if (mp->globalTone >= 0.0f) setFloat(caps, "DLSSNR.GlobalToneStrength", mp->globalTone);
    setUInt(caps, "DLSSNR.UseAutoMask", mp->autoMask);
    setUInt(caps, "DLSSNR.UICorrection", mp->uiCorrection);

    // Upscaling: when the output is larger than the input, the model does the
    // super-resolution itself. Width/Height above are the render (input) size,
    // OutWidth/OutHeight the display (target) size.
    const bool upscaling = (outWidth != width || outHeight != height);
    setUInt(caps, "DLSSNR.Upscaling", upscaling ? 1u : 0u);
    if (upscaling && width > 0) {
        const float ratio = (float) outWidth / (float) width;
        setFloat(caps, "DLSSNR.Scale", ratio);
        setFloat(caps, "DLSSNR.ScalingRatio", ratio);
    }

    void* handle = nullptr;
    fwd_last_create = g.create(cmd, 18, caps, &handle);
    return handle;
}

__declspec(dllexport) int fwd_evaluate(ID3D12GraphicsCommandList* cmd, void* feature, void* caps,
                                       ID3D12Resource* color, ID3D12Resource* depth,
                                       ID3D12Resource* motion, ID3D12Resource* output,
                                       unsigned int width, unsigned int height, unsigned int outW,
                                       unsigned int outH, int reset) {
    if (!feature || !caps || !g.evaluate) return 0;
    setResource(caps, "DLSSNR.Color", color);
    setResource(caps, "DLSSNR.Depth", depth);
    setResource(caps, "DLSSNR.MVec", motion);
    setResource(caps, "DLSSNR.Output", output);
    setUInt(caps, "DLSSNR.Enabled", 1);
    setUInt(caps, "DLSSNR.Width", width);
    setUInt(caps, "DLSSNR.Height", height);
    setUInt(caps, "DLSSNR.Reset", (unsigned int) reset);
    setUInt(caps, "DLSSNR.DepthInverted", 0);
    setUInt(caps, "DLSSNR.ColorSubrectWidth", width);
    setUInt(caps, "DLSSNR.ColorSubrectHeight", height);
    setUInt(caps, "DLSSNR.OutputSubrectWidth", outW);
    setUInt(caps, "DLSSNR.OutputSubrectHeight", outH);
    setUInt(caps, "DLSSNR.DepthSubrectWidth", width);
    setUInt(caps, "DLSSNR.DepthSubrectHeight", height);
    setUInt(caps, "DLSSNR.MVecSubrectWidth", width);
    setUInt(caps, "DLSSNR.MVecSubrectHeight", height);
    setFloat(caps, "DLSSNR.MVecScaleX", 1.0f);
    setFloat(caps, "DLSSNR.MVecScaleY", 1.0f);
    // Not a tail call: keep the frame so the snippet resolves THIS module as caller.
    volatile int r = g.evaluate(cmd, feature, caps, nullptr);
    return r;
}

__declspec(dllexport) void fwd_release(void* feature) {
    if (feature && g.release) {
        volatile int r = g.release(feature);
        (void) r;
    }
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
