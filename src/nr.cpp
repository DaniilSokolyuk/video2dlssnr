#include "nr.h"

#include "image.h"
#include "nr_params.h"
#include "optflow.h"
#include "optflow_nvof.h"
#include "pipeline.h"

#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Direct snippet entry points.
//
// Going through the SDK import library routes every call into the driver's NGX
// core (_nvngx.dll), which validates the feature id before the snippet is ever
// consulted. On driver 610.74 that validator rejects feature 18 outright:
//
//   NVSDK_NGX_CreateFeature_Validate: required feature is not supported by NGX
//   runtime, please update display driver
//
// But nvngx_dlssnr.dll exports the same D3D12 entry points itself, so the
// snippet can be driven directly and the core's whitelist never runs. This is
// the same route the in-game path takes to create feature 18.
// ---------------------------------------------------------------------------

namespace {

using PfnInitExt = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long appId,
                                                 const wchar_t* dataPath, ID3D12Device* device,
                                                 NVSDK_NGX_Version version,
                                                 const NVSDK_NGX_Parameter* params);
using PfnPopulateParams = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter* params);
using PfnCreateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList* cmdList,
                                                       NVSDK_NGX_Feature featureId,
                                                       NVSDK_NGX_Parameter* params,
                                                       NVSDK_NGX_Handle** outHandle);
using PfnEvaluateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList* cmdList,
                                                         const NVSDK_NGX_Handle* handle,
                                                         const NVSDK_NGX_Parameter* params,
                                                         void* progressCallback);
using PfnReleaseFeature = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle* handle);
using PfnShutdown1 = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device* device);

struct SnippetApi {
    HMODULE mod = nullptr;
    PfnInitExt InitExt = nullptr;
    PfnPopulateParams PopulateParams = nullptr;
    PfnCreateFeature CreateFeature = nullptr;
    PfnEvaluateFeature EvaluateFeature = nullptr;
    PfnReleaseFeature ReleaseFeature = nullptr;
    PfnShutdown1 Shutdown1 = nullptr;
    std::string path;

    bool Load(const std::string& dir) {
        const std::wstring full =
            Narrow2Widen(dir.empty() ? "nvngx_dlssnr.dll" : dir + "\\nvngx_dlssnr.dll");
        mod = LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!mod) return false;
        wchar_t resolved[MAX_PATH]{};
        if (GetModuleFileNameW(mod, resolved, MAX_PATH)) path = Widen2Narrow(resolved);

        InitExt = reinterpret_cast<PfnInitExt>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_Init_Ext"));
        PopulateParams = reinterpret_cast<PfnPopulateParams>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_PopulateParameters_Impl"));
        CreateFeature = reinterpret_cast<PfnCreateFeature>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_CreateFeature"));
        EvaluateFeature = reinterpret_cast<PfnEvaluateFeature>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_EvaluateFeature"));
        ReleaseFeature = reinterpret_cast<PfnReleaseFeature>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_ReleaseFeature"));
        Shutdown1 = reinterpret_cast<PfnShutdown1>(
            GetProcAddress(mod, "NVSDK_NGX_D3D12_Shutdown1"));
        return InitExt && CreateFeature && EvaluateFeature;
    }

    void Unload() {
        if (mod) {
            FreeLibrary(mod);
            mod = nullptr;
        }
    }
};

}  // namespace

// NGX's own parameter setters. The DLSSNR keys are plain strings, so nothing
// here needs an SDK helper that does not exist.
static void SetU(NVSDK_NGX_Parameter* p, const char* k, unsigned v) {
    NVSDK_NGX_Parameter_SetUI(p, k, v);
}
static void SetI(NVSDK_NGX_Parameter* p, const char* k, int v) {
    NVSDK_NGX_Parameter_SetI(p, k, v);
}
static void SetF(NVSDK_NGX_Parameter* p, const char* k, float v) {
    NVSDK_NGX_Parameter_SetF(p, k, v);
}
static void SetRes(NVSDK_NGX_Parameter* p, const char* k, GpuTexture* t) {
    NVSDK_NGX_Parameter_SetD3d12Resource(p, k, t ? t->res.Get() : nullptr);
}

// ---------------------------------------------------------------------------
// Driving the driver core's capability block by raw vtable slot.
//
// Feature 18 must be created on the CORE's
// capability parameters (GetCapabilityParameters), not a freshly allocated
// block - the capability block carries the snippet/preset callbacks a feature
// needs at create time; a fresh one has none, so CreateFeature answers
// UnableToInitializeFeature (0xBAD0000B). And that block does not lay its
// setters out the way the SDK header declares: uint is slot 3, resources go
// through slot 0 (the ULL setter), and the float setter is not at the header's
// slot 1 - it is found by round-tripping a value. The typed Get() works, which
// is what makes probing possible.
namespace {
using PfnSetULL = void(NVSDK_CONV*)(void*, const char*, unsigned long long);
using PfnSetFloatVt = void(NVSDK_CONV*)(void*, const char*, float);
using PfnSetUIntVt = void(NVSDK_CONV*)(void*, const char*, unsigned int);
constexpr int kVtSetULL = 0;
int g_floatSlot = -1;
int g_uintSlot = 3;

void VSetUInt(void* p, const char* n, unsigned v) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PfnSetUIntVt>(vt[g_uintSlot])(p, n, v);
}
void DiscoverUIntSlot(NVSDK_NGX_Parameter* p) {
    void** vt = *reinterpret_cast<void***>(p);
    for (int slot = 0; slot < 8; ++slot) {
        const unsigned probe = 0x1234u;
        unsigned rb = 0;
        reinterpret_cast<PfnSetUIntVt>(vt[slot])(p, "DLSSNR.UProbe", probe);
        if (NVSDK_NGX_Parameter_GetUI(p, "DLSSNR.UProbe", &rb) == NVSDK_NGX_Result_Success &&
            rb == probe) {
            g_uintSlot = slot;
            return;
        }
    }
}
void VSetRes(void* p, const char* n, ID3D12Resource* r) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PfnSetULL>(vt[kVtSetULL])(p, n, reinterpret_cast<unsigned long long>(r));
}
void VSetFloat(void* p, const char* n, float v) {
    void** vt = *reinterpret_cast<void***>(p);
    reinterpret_cast<PfnSetFloatVt>(vt[g_floatSlot < 0 ? 1 : g_floatSlot])(p, n, v);
}
void DiscoverFloatSlot(NVSDK_NGX_Parameter* p) {
    if (g_floatSlot >= 0) return;
    void** vt = *reinterpret_cast<void***>(p);
    for (int slot = 0; slot < 8; ++slot) {
        const float probe = 0.3125f;  // exact in binary
        float rb = 0.0f;
        reinterpret_cast<PfnSetFloatVt>(vt[slot])(p, "DLSSNR.Probe", probe);
        if (NVSDK_NGX_Parameter_GetF(p, "DLSSNR.Probe", &rb) == NVSDK_NGX_Result_Success &&
            rb == probe) {
            g_floatSlot = slot;
            return;
        }
    }
    g_floatSlot = 1;
}
}  // namespace

void NrFeature::FillCreateParams(NVSDK_NGX_Parameter* p, const NrFeatureDesc& desc) {
    // Node masks, as every NGX feature expects.
    SetU(p, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    SetU(p, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);

    // Both the generic and the DLSSNR-prefixed size keys are set: the DLL reads
    // one pair, and which one is not documented anywhere.
    SetU(p, NVSDK_NGX_Parameter_Width, desc.inputW);
    SetU(p, NVSDK_NGX_Parameter_Height, desc.inputH);
    SetU(p, NVSDK_NGX_Parameter_OutWidth, desc.outputW);
    SetU(p, NVSDK_NGX_Parameter_OutHeight, desc.outputH);

    SetU(p, NrParam::kInputWidth, desc.inputW);
    SetU(p, NrParam::kInputHeight, desc.inputH);
    SetU(p, NrParam::kOutputWidth, desc.outputW);
    SetU(p, NrParam::kOutputHeight, desc.outputH);
    SetU(p, NrParam::kWidth, desc.inputW);
    SetU(p, NrParam::kHeight, desc.inputH);

    SetI(p, NrParam::kEnabled, 1);
    SetI(p, NrParam::kUpscaling, desc.upscaling ? 1 : 0);
    SetU(p, NrParam::kPreset, desc.preset);
    SetI(p, NrParam::kDepthInverted, desc.depthInverted ? 1 : 0);

    if (desc.outputW && desc.inputW) {
        const float ratio = static_cast<float>(desc.outputW) / static_cast<float>(desc.inputW);
        SetF(p, NrParam::kScale, ratio);
        SetF(p, NrParam::kScalingRatio, ratio);
    }
    if (desc.intensity >= 0.0f) SetF(p, NrParam::kIntensity, desc.intensity);
    if (desc.localStructureStrength >= 0.0f) {
        SetF(p, NrParam::kLocalStructureStrength, desc.localStructureStrength);
    }
    if (desc.localToneStrength >= 0.0f) {
        SetF(p, NrParam::kLocalToneStrength, desc.localToneStrength);
    }
}

void NrFeature::FillEvalParams(NVSDK_NGX_Parameter* p, const NrEvalInputs& in) {
    SetRes(p, NrParam::kColor, in.color);
    SetRes(p, NrParam::kDepth, in.depth);
    SetRes(p, NrParam::kMVec, in.motion);
    SetRes(p, NrParam::kOutput, in.output);

    SetI(p, NrParam::kReset, in.reset ? 1 : 0);
    SetF(p, NrParam::kMVecScaleX, in.mvecScaleX);
    SetF(p, NrParam::kMVecScaleY, in.mvecScaleY);

    if (in.color) {
        SetU(p, NrParam::kColorSubrectWidth, static_cast<unsigned>(in.color->w));
        SetU(p, NrParam::kColorSubrectHeight, static_cast<unsigned>(in.color->h));
    }
    SetU(p, NrParam::kColorSubrectBaseX, 0);
    SetU(p, NrParam::kColorSubrectBaseY, 0);
    SetU(p, NrParam::kDepthSubrectBaseX, 0);
    SetU(p, NrParam::kDepthSubrectBaseY, 0);
    SetU(p, NrParam::kMVecSubrectBaseX, 0);
    SetU(p, NrParam::kMVecSubrectBaseY, 0);
    SetU(p, NrParam::kOutputSubrectBaseX, 0);
    SetU(p, NrParam::kOutputSubrectBaseY, 0);
}

NrFeature::~NrFeature() { Release(); }

NVSDK_NGX_Result NrFeature::TryCreate(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                                      const NrFeatureDesc& desc) {
    CHECK(session.Initialized());
    Release();
    m_desc = desc;

    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_AllocateParameters(&m_params);
    if (NVSDK_NGX_FAILED(r)) return r;

    FillCreateParams(m_params, desc);

    r = NVSDK_NGX_D3D12_CreateFeature(
        cmdList, static_cast<NVSDK_NGX_Feature>(kNgxFeatureNeuralRendering), m_params, &m_handle);
    if (NVSDK_NGX_FAILED(r)) {
        m_handle = nullptr;
    }
    return r;
}

void NrFeature::Create(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                       const NrFeatureDesc& desc) {
    const NVSDK_NGX_Result r = TryCreate(session, cmdList, desc);
    if (NVSDK_NGX_FAILED(r)) {
        char buf[64];
        snprintf(buf, sizeof(buf), " (0x%08X)", static_cast<unsigned>(r));
        throw ToolError(std::string("DLSS Neural Rendering CreateFeature failed: ") +
                        NgxResultToString(r) + buf);
    }
}

NVSDK_NGX_Result NrFeature::TryEvaluate(ID3D12GraphicsCommandList* cmdList,
                                        const NrEvalInputs& in) {
    if (!m_handle || !m_params) return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    CHECK(in.color && in.output);

    FillEvalParams(m_params, in);
    return NVSDK_NGX_D3D12_EvaluateFeature(cmdList, m_handle, m_params, nullptr);
}

void NrFeature::Evaluate(ID3D12GraphicsCommandList* cmdList, const NrEvalInputs& in) {
    const NVSDK_NGX_Result r = TryEvaluate(cmdList, in);
    if (NVSDK_NGX_FAILED(r)) {
        char buf[64];
        snprintf(buf, sizeof(buf), " (0x%08X)", static_cast<unsigned>(r));
        throw ToolError(std::string("DLSS Neural Rendering EvaluateFeature failed: ") +
                        NgxResultToString(r) + buf);
    }
}

void NrFeature::Release() {
    if (m_handle) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_handle);
        m_handle = nullptr;
    }
    if (m_params) {
        NVSDK_NGX_D3D12_DestroyParameters(m_params);
        m_params = nullptr;
    }
}

// ---------------------------------------------------------------------------

int ProbeNeuralRendering(const std::string& dllDir, int adapter, unsigned inputW,
                         unsigned inputH, unsigned outputW, unsigned outputH,
                         unsigned preset, bool verbose) {
    GpuContext gpu;
    gpu.Initialize(false, adapter);
    LogInfo("  gpu:    %s (%zu MB)", gpu.AdapterName().c_str(), gpu.AdapterVramMB());

    const std::vector<std::wstring> paths = DefaultDllSearchPaths(dllDir);
    for (const std::wstring& p : paths) LogDebug("dll search path: %s", Widen2Narrow(p).c_str());

    NgxSession ngx;
    ngx.Init(gpu.Device(), paths, verbose);

    std::string srPath = "(not found)", srVersion = "unknown";
    if (NgxSession::FindLoadedDlssModule(&srPath, &srVersion)) {
        LogInfo("  dlss:   %s  %s", srVersion.c_str(), srPath.c_str());
    }
    LogInfo("");

    // Inputs sized as a renderer would hand them over.
    GpuTexture color = gpu.CreateTexture(static_cast<int>(inputW), static_cast<int>(inputH),
                                         DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"nrColor");
    GpuTexture depth = gpu.CreateTexture(static_cast<int>(inputW), static_cast<int>(inputH),
                                         DXGI_FORMAT_R32_FLOAT, true, L"nrDepth");
    GpuTexture motion = gpu.CreateTexture(static_cast<int>(inputW), static_cast<int>(inputH),
                                          DXGI_FORMAT_R16G16_FLOAT, true, L"nrMotion");
    GpuTexture out = gpu.CreateTexture(static_cast<int>(outputW), static_cast<int>(outputH),
                                       DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"nrOutput");
    gpu.Begin();
    gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();

    NrFeatureDesc desc;
    desc.inputW = inputW;
    desc.inputH = inputH;
    desc.outputW = outputW;
    desc.outputH = outputH;
    desc.preset = preset;
    desc.upscaling = (outputW != inputW || outputH != inputH);

    LogInfo("NGX feature %d (Neural Rendering)", kNgxFeatureNeuralRendering);
    LogInfo("  %ux%u -> %ux%u, upscaling %s, preset %u", inputW, inputH, outputW, outputH,
            desc.upscaling ? "on" : "off", preset);
    LogInfo("");

    char code[32];
    auto report = [&](const char* what, NVSDK_NGX_Result r) {
        snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(r));
        LogInfo("  %-28s %s  %s %s", what, NVSDK_NGX_FAILED(r) ? "FAILED" : "OK    ",
                NgxResultToString(r), code);
    };

    // ---- route A: through the driver's NGX core, as the SDK normally does ---
    LogInfo("route A - via driver NGX core (_nvngx.dll):");
    NVSDK_NGX_Result coreResult;
    {
        NrFeature feature;
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        coreResult = feature.TryCreate(ngx, cl, desc);
        gpu.EndAndWait();
        report("CreateFeature", coreResult);
        if (!NVSDK_NGX_FAILED(coreResult)) {
            ID3D12GraphicsCommandList* cl2 = gpu.Begin();
            NrEvalInputs in;
            in.color = &color;
            in.depth = &depth;
            in.motion = &motion;
            in.output = &out;
            in.reset = true;
            const NVSDK_NGX_Result er = feature.TryEvaluate(cl2, in);
            gpu.EndAndWait();
            report("EvaluateFeature", er);
            feature.Release();
            ngx.Shutdown();
            return NVSDK_NGX_FAILED(er) ? 1 : 0;
        }
        feature.Release();
    }
    if (coreResult == NVSDK_NGX_Result_FAIL_OutOfDate) {
        LogInfo("    the core rejected the feature id before reaching the snippet");
    }

    // ---- route F: the forwarder ----------------------------------------------
    //      A shim DLL named nvngx.dll_dlssnr.dll makes the NGX calls, so the
    //      snippet's caller check (path must contain "nvngx.dll") passes, and we
    //      drive the snippet's OWN Init_Ext + CreateFeature(18) on the core's
    //      capability block. This is the route that works.
    LogInfo("");
    LogInfo("route F - forwarder shim -> snippet's own CreateFeature 18:");
    {
        using PfnFwdSetSlots = void(*)(int, int);
        using PfnFwdCreate = void*(*)(const wchar_t*, const wchar_t*, ID3D12Device*,
                                      ID3D12GraphicsCommandList*, void*, unsigned, unsigned,
                                      unsigned, unsigned, const NrModelParams*);
        using PfnFwdEval = int(*)(ID3D12GraphicsCommandList*, void*, void*, ID3D12Resource*,
                                  ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, unsigned,
                                  unsigned, unsigned, unsigned, int);
        using PfnFwdRelease = void(*)(void*);

        // The forwarder sits next to the exe.
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::wstring fwdPath = exePath;
        const size_t slash = fwdPath.find_last_of(L"\\/");
        fwdPath = (slash == std::wstring::npos ? std::wstring() : fwdPath.substr(0, slash + 1)) +
                  L"nvngx.dll_dlssnr.dll";

        HMODULE fwd = LoadLibraryExW(fwdPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!fwd) {
            LogInfo("    could not load forwarder %s (err %lu)", Widen2Narrow(fwdPath).c_str(),
                    GetLastError());
        } else {
            auto fSet = reinterpret_cast<PfnFwdSetSlots>(GetProcAddress(fwd, "fwd_set_slots"));
            auto fCreate = reinterpret_cast<PfnFwdCreate>(GetProcAddress(fwd, "fwd_create"));
            auto fEval = reinterpret_cast<PfnFwdEval>(GetProcAddress(fwd, "fwd_evaluate"));
            auto fRel = reinterpret_cast<PfnFwdRelease>(GetProcAddress(fwd, "fwd_release"));
            NVSDK_NGX_Parameter* caps = ngx.CapabilityParams();
            if (fCreate && fEval && caps) {
                DiscoverFloatSlot(caps);
                DiscoverUIntSlot(caps);
                if (fSet) fSet(g_uintSlot, g_floatSlot);
                LogInfo("    forwarder loaded; vtable slots uint=%d float=%d", g_uintSlot,
                        g_floatSlot);

                // Absolute path to the snippet for the forwarder's own LoadLibrary.
                std::wstring snippetPath = Narrow2Widen(
                    dllDir.empty() ? "nvngx_dlssnr.dll" : dllDir + "\\nvngx_dlssnr.dll");
                wchar_t abs[MAX_PATH]{};
                if (GetFullPathNameW(snippetPath.c_str(), MAX_PATH, abs, nullptr)) snippetPath = abs;

                wchar_t appData[MAX_PATH]{};
                const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", appData, MAX_PATH);
                const std::wstring dataPath =
                    (n > 0 && n < MAX_PATH) ? std::wstring(appData) + L"\\video2dlssnr"
                                            : std::wstring(L".");

                ID3D12GraphicsCommandList* cl = gpu.Begin();
                NrModelParams pm{};
                pm.preset = preset;
                pm.intensity = 1.0f;
                pm.style = 0;
                pm.localStructure = 1.0f;
                pm.localTone = 1.0f;
                pm.skinStructure = -1.0f;
                pm.globalTone = -1.0f;
                pm.autoMask = 0;
                pm.uiCorrection = 1;
                void* handle = fCreate(snippetPath.c_str(), dataPath.c_str(), gpu.Device(), cl,
                                       caps, inputW, inputH, outputW, outputH, &pm);
                gpu.EndAndWait();

                int fInit = 0, fCreateRc = 0;
                if (auto pInit = reinterpret_cast<int*>(GetProcAddress(fwd, "fwd_last_init")))
                    fInit = *pInit;
                if (auto pCreate = reinterpret_cast<int*>(GetProcAddress(fwd, "fwd_last_create")))
                    fCreateRc = *pCreate;
                LogInfo("    snippet Init_Ext -> %d, CreateFeature 18 -> 0x%08X, handle=%p", fInit,
                        static_cast<unsigned>(fCreateRc), handle);

                if (handle) {
                    ID3D12GraphicsCommandList* cl2 = gpu.Begin();
                    const int er = fEval(cl2, handle, caps, color.res.Get(), depth.res.Get(),
                                         motion.res.Get(), out.res.Get(), inputW, inputH, outputW,
                                         outputH, 1);
                    gpu.EndAndWait();
                    LogInfo("    snippet EvaluateFeature 18 -> 0x%08X", static_cast<unsigned>(er));
                    if (fRel) fRel(handle);
                    if (!NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(er))) {
                        ngx.Shutdown();
                        LogInfo("");
                        LogInfo("Neural Rendering RAN via route F (forwarder).");
                        return 0;
                    }
                }
            } else {
                LogInfo("    forwarder missing exports or no caps");
            }
        }
    }

    // ---- route P: core CreateFeature 18 on the driver's CAPABILITY block -----
    //      (not a fresh one), setters by raw vtable slot.
    LogInfo("");
    LogInfo("route P - core CreateFeature 18 on capability params:");
    {
        NVSDK_NGX_Parameter* caps = ngx.CapabilityParams();
        if (!caps) {
            LogInfo("    no capability params from the core");
        } else {
            DiscoverFloatSlot(caps);
            DiscoverUIntSlot(caps);
            LogInfo("    vtable slots: uint=%d float=%d", g_uintSlot, g_floatSlot);
            // verify the uint setter actually lands
            VSetUInt(caps, "Width", inputW);
            unsigned wcheck = 0;
            NVSDK_NGX_Parameter_GetUI(caps, "Width", &wcheck);
            LogInfo("    readback Width = %u (set %u)", wcheck, inputW);

            // Generic NGX size params - the create path (shared with the DLSS/DLAA
            // framework NR is built on) reads "Width"/"Height", not only the
            // DLSSNR-prefixed ones. A fresh capability block has neither until we set
            // them; a game's live block already carried them.
            VSetUInt(caps, "Width", inputW);
            VSetUInt(caps, "Height", inputH);
            VSetUInt(caps, "OutWidth", outputW);
            VSetUInt(caps, "OutHeight", outputH);
            VSetUInt(caps, "DLSS.Render.Subrect.Dimensions.Width", inputW);
            VSetUInt(caps, "DLSS.Render.Subrect.Dimensions.Height", inputH);
            VSetUInt(caps, "PerfQualityValue", 2);  // MaxQuality/DLAA

            VSetUInt(caps, "DLSSNR.Enabled", 1);
            VSetUInt(caps, "DLSSNR.Width", inputW);
            VSetUInt(caps, "DLSSNR.Height", inputH);
            VSetUInt(caps, "CreationNodeMask", 1);
            VSetUInt(caps, "VisibilityNodeMask", 1);
            VSetUInt(caps, "DLSSNR.Hint.Render.Preset", preset);
            VSetFloat(caps, "DLSSNR.Intensity", 1.0f);
            VSetUInt(caps, "DLSSNR.Style", 0);
            VSetFloat(caps, "DLSSNR.LocalStructureStrength", 1.0f);
            VSetFloat(caps, "DLSSNR.LocalToneStrength", 1.0f);
            VSetFloat(caps, "DLSSNR.SkinStructureStrength", 1.0f);
            VSetUInt(caps, "DLSSNR.UseAutoMask", 0);
            VSetUInt(caps, "DLSSNR.UICorrection", 1);

            NVSDK_NGX_Handle* h = nullptr;
            ID3D12GraphicsCommandList* cl = gpu.Begin();
            NVSDK_NGX_Result cr =
                NVSDK_NGX_D3D12_CreateFeature(cl, static_cast<NVSDK_NGX_Feature>(18), caps, &h);
            gpu.EndAndWait();
            report("core CreateFeature 18 (caps)", cr);

            if (!NVSDK_NGX_FAILED(cr) && h) {
                VSetRes(caps, "DLSSNR.Color", color.res.Get());
                VSetRes(caps, "DLSSNR.Depth", depth.res.Get());
                VSetRes(caps, "DLSSNR.MVec", motion.res.Get());
                VSetRes(caps, "DLSSNR.Output", out.res.Get());
                VSetUInt(caps, "DLSSNR.Enabled", 1);
                VSetUInt(caps, "DLSSNR.Width", inputW);
                VSetUInt(caps, "DLSSNR.Height", inputH);
                VSetUInt(caps, "DLSSNR.DepthInverted", 0);
                VSetUInt(caps, "DLSSNR.Reset", 1);
                VSetUInt(caps, "DLSSNR.ColorSubrectWidth", inputW);
                VSetUInt(caps, "DLSSNR.ColorSubrectHeight", inputH);
                VSetUInt(caps, "DLSSNR.OutputSubrectWidth", outputW);
                VSetUInt(caps, "DLSSNR.OutputSubrectHeight", outputH);
                VSetUInt(caps, "DLSSNR.DepthSubrectWidth", inputW);
                VSetUInt(caps, "DLSSNR.DepthSubrectHeight", inputH);
                VSetUInt(caps, "DLSSNR.MVecSubrectWidth", inputW);
                VSetUInt(caps, "DLSSNR.MVecSubrectHeight", inputH);
                VSetFloat(caps, "DLSSNR.MVecScaleX", 1.0f);
                VSetFloat(caps, "DLSSNR.MVecScaleY", 1.0f);

                ID3D12GraphicsCommandList* cl2 = gpu.Begin();
                NVSDK_NGX_Result er = NVSDK_NGX_D3D12_EvaluateFeature(cl2, h, caps, nullptr);
                gpu.EndAndWait();
                report("core EvaluateFeature 18 (caps)", er);
                NVSDK_NGX_D3D12_ReleaseFeature(h);
                if (!NVSDK_NGX_FAILED(er)) {
                    ngx.Shutdown();
                    LogInfo("");
                    LogInfo("Neural Rendering RAN via route P.");
                    return 0;
                }
            }
        }
    }

    // ---- which feature ids does THIS driver core even admit? ----------------
    //
    // The core validator (NVSDK_NGX_CreateFeature_Validate) returns OutOfDate
    // for any feature id the installed driver does not know. Anything else means
    // the id got PAST the gate and the core went on to load+wire the snippet.
    // RayReconstruction (13) lives in nvngx_dlssnr.dll too, so if the core
    // admits 13 it would adopt that snippet and hand it the cubin interface -
    // the exact bootstrap an in-game addon rides. This tells us whether
    // that door exists on driver 610.74 at all.
    LogInfo("");
    LogInfo("core feature-id gate (does _nvngx.dll admit the id at all):");
    struct { int id; const char* name; } probes[] = {
        {NVSDK_NGX_Feature_SuperSampling,    "1  SuperSampling  "},
        {NVSDK_NGX_Feature_RayReconstruction,"13 RayReconstruct."},
        {kNgxFeatureNeuralRendering,         "18 NeuralRender.  "},
    };
    for (const auto& pr : probes) {
        NVSDK_NGX_Parameter* pp = nullptr;
        NVSDK_NGX_Result ar = NVSDK_NGX_D3D12_AllocateParameters(&pp);
        if (NVSDK_NGX_FAILED(ar) || !pp) { LogInfo("    %s alloc failed", pr.name); continue; }
        NVSDK_NGX_Handle* h = nullptr;
        ID3D12GraphicsCommandList* clp = gpu.Begin();
        NVSDK_NGX_Result cr = NVSDK_NGX_D3D12_CreateFeature(
            clp, static_cast<NVSDK_NGX_Feature>(pr.id), pp, &h);
        gpu.EndAndWait();
        const bool admitted = (cr != NVSDK_NGX_Result_FAIL_OutOfDate);
        snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(cr));
        LogInfo("    %s  %s  (%s %s)", pr.name,
                admitted ? "ADMITTED past gate" : "rejected OutOfDate",
                NgxResultToString(cr), code);
        if (h) NVSDK_NGX_D3D12_ReleaseFeature(h);
        NVSDK_NGX_D3D12_DestroyParameters(pp);
    }

    // ---- warm the cubin backend with a feature that does work ---------------
    //
    // NGXCG2R::Init does two things: a generic snippet init, which succeeds, and
    // a Cubin Init through a function pointer, which is what fails. That cubin
    // backend is set up by the NGX core, and in a game it is already live by the
    // time anything asks for Neural Rendering, because the game's own DLSS has
    // long since initialised it. That is the difference between this probe and
    // an in-game addon, which only ever runs inside a process where DLSS is up.
    //
    // So bring Super Resolution up first — it works on any supported GPU — and
    // leave it alive across the Neural Rendering attempt.
    LogInfo("");
    LogInfo("warm-up - create a DLSS Super Resolution feature first:");
    DlssFeature warm;
    {
        OptimalSettings s;
        bool warmed = false;
        try {
            s = ngx.GetOptimal(outputW, outputH, NVSDK_NGX_PerfQuality_Value_MaxQuality);
            DlssFeatureDesc d;
            d.renderW = s.renderW;
            d.renderH = s.renderH;
            d.targetW = outputW;
            d.targetH = outputH;
            d.quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            ID3D12GraphicsCommandList* cl = gpu.Begin();
            warm.Create(ngx, cl, d);
            gpu.EndAndWait();
            warmed = true;
        } catch (const ToolError& e) {
            LogInfo("    SR warm-up failed: %s", e.what());
        }
        LogInfo("    %s", warmed ? "SR feature alive; cubin backend should now be initialised"
                                 : "no SR feature, cubin backend probably still cold");
    }

    // ---- route B: straight into nvngx_dlssnr.dll, bypassing that validator --
    LogInfo("");
    LogInfo("route B - direct into nvngx_dlssnr.dll exports:");
    SnippetApi snip;
    if (!snip.Load(dllDir)) {
        LogInfo("    could not load nvngx_dlssnr.dll from %s",
                dllDir.empty() ? "(search path)" : dllDir.c_str());
        ngx.Shutdown();
        return 1;
    }
    LogInfo("    loaded %s", snip.path.c_str());
    LogInfo("    exports: Init_Ext=%s Populate=%s Create=%s Evaluate=%s",
            snip.InitExt ? "y" : "n", snip.PopulateParams ? "y" : "n",
            snip.CreateFeature ? "y" : "n", snip.EvaluateFeature ? "y" : "n");

    wchar_t appData[MAX_PATH]{};
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", appData, MAX_PATH);
    const std::wstring dataPath =
        (n > 0 && n < MAX_PATH) ? std::wstring(appData) + L"\\video2dlssnr" : std::wstring(L".");

    NVSDK_NGX_Result r =
        snip.InitExt(0x0, dataPath.c_str(), gpu.Device(), NVSDK_NGX_Version_API, nullptr);
    report("snippet Init_Ext", r);

    NVSDK_NGX_Parameter* params = nullptr;
    r = NVSDK_NGX_D3D12_AllocateParameters(&params);
    report("AllocateParameters", r);
    if (NVSDK_NGX_FAILED(r) || !params) {
        snip.Unload();
        ngx.Shutdown();
        return 1;
    }
    if (snip.PopulateParams) {
        r = snip.PopulateParams(params);
        report("snippet PopulateParams", r);
    }

    NrFeature::FillCreateParams(params, desc);

    NVSDK_NGX_Handle* handle = nullptr;
    {
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        r = snip.CreateFeature(cl, static_cast<NVSDK_NGX_Feature>(kNgxFeatureNeuralRendering),
                               params, &handle);
        gpu.EndAndWait();
    }
    report("snippet CreateFeature", r);

    int rc = 1;
    if (!NVSDK_NGX_FAILED(r) && handle) {
        NrEvalInputs in;
        in.color = &color;
        in.depth = &depth;
        in.motion = &motion;
        in.output = &out;
        in.reset = true;
        NrFeature::FillEvalParams(params, in);
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        r = snip.EvaluateFeature(cl, handle, params, nullptr);
        gpu.EndAndWait();
        report("snippet EvaluateFeature", r);
        rc = NVSDK_NGX_FAILED(r) ? 1 : 0;
        if (snip.ReleaseFeature) snip.ReleaseFeature(handle);
    }

    NVSDK_NGX_D3D12_DestroyParameters(params);

    // ---- route C: patch the driver core's feature-id gate --------------------
    //
    // Feature 18 can be forced by making the "signed feature" validation writable
    // and jumping the gate. Route A proved the core rejects id 18 at
    // NVSDK_NGX_CreateFeature_Validate; disassembly located that gate exactly:
    //
    //   0x18000e645: 83 fd 12   cmp ebp, 0x12   ; featureId vs 18
    //   0x18000e648: 7c 29      jl  0x18000e673 ; id < 18 -> accept path
    //
    // So id 18 falls through to the OutOfDate reject. Patch the compare constant
    // 0x12 -> 0x13 in the loaded module (feature 18 now counts as "< 19" and
    // takes the accept path) and try creating feature 18 through the core for
    // real, instead of inferring what happens next. Signature-checked before any
    // write; original byte restored after.
    LogInfo("");
    LogInfo("route C - patch _nvngx.dll feature-id gates (addon-style) and retry core:");
    // Every NGX validate path (CUDA/D3D11/D3D12/Vulkan x CreateFeature /
    // CreateFeature1 / GetFeatureRequirements) has its own copy of the same gate:
    //   cmp <reg>, 0x12   ; featureId vs 18
    //   jl  <accept>      ; id < 18 -> proceed; 18 falls through to OutOfDate
    // The reject-string ("error: required feature is not supported") is loaded in
    // each. Their immediate-0x12 byte offsets, found by scanning the module for
    // those reject sites, are below. Patch every one 0x12 -> 0x13 so feature 18
    // counts as "< 19" and takes the accept path, whichever path the D3D12
    // CreateFeature call happens to run. Each byte is verified 0x12 before the
    // write and restored afterwards.
    static const uint32_t kGateImmRVAs[] = {
        0x0658f, 0x06943, 0x0858f, 0x087f7, 0x0d4ff, 0x0d953, 0x0dc2a, 0x0e3d7,
        0x0e647, 0x641cf, 0x64546, 0x650eb, 0x66d4f, 0x670f6, 0x67c8b,
    };
    HMODULE core = GetModuleHandleW(L"_nvngx.dll");
    if (!core) {
        LogInfo("    _nvngx.dll not loaded in this process");
    } else {
        uint8_t* cbase = reinterpret_cast<uint8_t*>(core);
        struct Patched { uint8_t* at; DWORD prot; };
        std::vector<Patched> patched;
        int mism = 0;
        for (uint32_t rva : kGateImmRVAs) {
            uint8_t* at = cbase + rva;
            if (*at != 0x12) { ++mism; continue; }
            DWORD prot = 0;
            if (!VirtualProtect(at, 1, PAGE_EXECUTE_READWRITE, &prot)) continue;
            *at = 0x13;
            FlushInstructionCache(GetCurrentProcess(), at, 1);
            patched.push_back({at, prot});
        }
        LogInfo("    opened %zu/%d feature-id gates (0x12 -> 0x13)%s", patched.size(),
                (int)(sizeof(kGateImmRVAs) / sizeof(kGateImmRVAs[0])),
                mism ? " [some bytes were not 0x12 - core build differs]" : "");

        if (!patched.empty()) {
            NrFeature feature;
            ID3D12GraphicsCommandList* cl = gpu.Begin();
            NVSDK_NGX_Result cr = feature.TryCreate(ngx, cl, desc);
            gpu.EndAndWait();
            report("core CreateFeature 18 (gates patched)", cr);
            if (!NVSDK_NGX_FAILED(cr)) {
                NrEvalInputs in;
                in.color = &color;
                in.depth = &depth;
                in.motion = &motion;
                in.output = &out;
                in.reset = true;
                ID3D12GraphicsCommandList* cl2 = gpu.Begin();
                NVSDK_NGX_Result er = feature.TryEvaluate(cl2, in);
                gpu.EndAndWait();
                report("core EvaluateFeature 18 (gates patched)", er);
                if (!NVSDK_NGX_FAILED(er)) rc = 0;
            }
            feature.Release();
        }

        for (const Patched& p : patched) {  // restore the core byte-for-byte
            *p.at = 0x12;
            FlushInstructionCache(GetCurrentProcess(), p.at, 1);
            DWORD tmp = 0;
            VirtualProtect(p.at, 1, p.prot, &tmp);
        }
        LogInfo("    %zu gates restored", patched.size());
    }

    if (snip.Shutdown1) snip.Shutdown1(gpu.Device());
    snip.Unload();
    ngx.Shutdown();
    LogInfo("");
    LogInfo(rc == 0 ? "Neural Rendering ran on this GPU."
                    : "Neural Rendering did not run; -v shows the NGX log.");
    return rc;
}

// ---------------------------------------------------------------------------
// The working tool: run Neural Rendering over a real image via the forwarder.
// ---------------------------------------------------------------------------
namespace {
struct ForwarderApi {
    HMODULE mod = nullptr;
    void (*setSlots)(int, int) = nullptr;
    void* (*create)(const wchar_t*, const wchar_t*, ID3D12Device*, ID3D12GraphicsCommandList*,
                    void*, unsigned, unsigned, unsigned, unsigned, const NrModelParams*) = nullptr;
    int (*evaluate)(ID3D12GraphicsCommandList*, void*, void*, ID3D12Resource*, ID3D12Resource*,
                    ID3D12Resource*, ID3D12Resource*, unsigned, unsigned, unsigned, unsigned,
                    int) = nullptr;
    void (*release)(void*) = nullptr;

    bool Load() {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::wstring p = exePath;
        const size_t s = p.find_last_of(L"\\/");
        p = (s == std::wstring::npos ? std::wstring() : p.substr(0, s + 1)) + L"nvngx.dll_dlssnr.dll";
        mod = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!mod) return false;
        setSlots = reinterpret_cast<decltype(setSlots)>(GetProcAddress(mod, "fwd_set_slots"));
        create = reinterpret_cast<decltype(create)>(GetProcAddress(mod, "fwd_create"));
        evaluate = reinterpret_cast<decltype(evaluate)>(GetProcAddress(mod, "fwd_evaluate"));
        release = reinterpret_cast<decltype(release)>(GetProcAddress(mod, "fwd_release"));
        return create && evaluate;
    }
};
}  // namespace

namespace {
bool HasImageExt(const std::string& p) {
    const size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string e = p.substr(dot);
    for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".tga";
}

float Luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

// Compose the model's linear output over the linear original (luma-ratio):
//  colour = 0 keeps the original hue (only the model's luma/detail), = 1 adopts the
//  model's own colour; detail is the overall strength (0 = the original, untouched).
ImageF Composite(const ImageF& orig, const std::vector<float>& nr, float detail, float colour) {
    ImageF out;
    out.w = orig.w;
    out.h = orig.h;
    out.px.resize(orig.px.size());
    const size_t n = orig.pixels();
    for (size_t i = 0; i < n; ++i) {
        const float orr = orig.px[i * 4 + 0], org = orig.px[i * 4 + 1], orb = orig.px[i * 4 + 2];
        const float nrr = nr[i * 4 + 0], nrg = nr[i * 4 + 1], nrb = nr[i * 4 + 2];
        const float lo = Luma(orr, org, orb), ln = Luma(nrr, nrg, nrb);
        const float scale = ln / (lo > 1e-4f ? lo : 1e-4f);
        // luma-only: the original colour carried to the model's luminance
        const float loR = orr * scale, loG = org * scale, loB = orb * scale;
        // blend luma-only <-> the model's own colour by `colour`
        const float cR = loR + (nrr - loR) * colour;
        const float cG = loG + (nrg - loG) * colour;
        const float cB = loB + (nrb - loB) * colour;
        // overall strength
        out.px[i * 4 + 0] = orr + (cR - orr) * detail;
        out.px[i * 4 + 1] = org + (cG - org) * detail;
        out.px[i * 4 + 2] = orb + (cB - orb) * detail;
        out.px[i * 4 + 3] = orig.px[i * 4 + 3];
    }
    return out;
}

// Bilinear resize of a linear RGBA image, for lining the original up with an upscaled
// NR output (composition and the A/B pair).
ImageF ResizeBilinear(const ImageF& src, unsigned dw, unsigned dh) {
    ImageF d;
    d.w = static_cast<int>(dw);
    d.h = static_cast<int>(dh);
    d.px.resize(static_cast<size_t>(dw) * dh * 4);
    const float sx = src.w > 1 ? static_cast<float>(src.w - 1) / (dw > 1 ? dw - 1 : 1) : 0.0f;
    const float sy = src.h > 1 ? static_cast<float>(src.h - 1) / (dh > 1 ? dh - 1 : 1) : 0.0f;
    for (unsigned y = 0; y < dh; ++y) {
        const float fy = y * sy;
        const int y0 = static_cast<int>(fy);
        const int y1 = y0 + 1 < src.h ? y0 + 1 : y0;
        const float wy = fy - y0;
        for (unsigned x = 0; x < dw; ++x) {
            const float fx = x * sx;
            const int x0 = static_cast<int>(fx);
            const int x1 = x0 + 1 < src.w ? x0 + 1 : x0;
            const float wx = fx - x0;
            for (int c = 0; c < 4; ++c) {
                const float a = src.px[(static_cast<size_t>(y0) * src.w + x0) * 4 + c];
                const float b = src.px[(static_cast<size_t>(y0) * src.w + x1) * 4 + c];
                const float e = src.px[(static_cast<size_t>(y1) * src.w + x0) * 4 + c];
                const float f = src.px[(static_cast<size_t>(y1) * src.w + x1) * 4 + c];
                const float top = a + (b - a) * wx;
                const float bot = e + (f - e) * wx;
                d.px[(static_cast<size_t>(y) * dw + x) * 4 + c] = top + (bot - top) * wy;
            }
        }
    }
    return d;
}

// The DLSS mode whose native ratio matches a pass. The feature is created with that mode's
// PerfQuality value so the driver picks the network / preset tuned for the ratio - creating
// "Quality" (a 1.5x mode) and then feeding it a 2x or 3x ratio ran the wrong model on the work.
// Ultra Quality is skipped (drivers commonly report it unsupported); DLAA covers no-upscale.
static NVSDK_NGX_PerfQuality_Value SrModeForRatio(unsigned rW, unsigned rH, unsigned tW,
                                                  unsigned tH) {
    const double r = std::max(static_cast<double>(tW) / rW, static_cast<double>(tH) / rH);
    if (r <= 1.01) return NVSDK_NGX_PerfQuality_Value_DLAA;
    if (r <= 1.6) return NVSDK_NGX_PerfQuality_Value_MaxQuality;        // 1.5x
    if (r <= 1.85) return NVSDK_NGX_PerfQuality_Value_Balanced;         // 1.72x
    if (r <= 2.5) return NVSDK_NGX_PerfQuality_Value_MaxPerf;           // 2x
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;                // 3x
}

// A cached DLSS SR pass: the feature and its textures, built once per (render, target) and
// reused for every image of that size. Recreating the feature per image was the batch's main
// cost - creation builds the network; a game creates once and evaluates every frame.
struct SrStage {
    unsigned rW = 0, rH = 0, tW = 0, tH = 0, preset = 0;
    DlssFeature feat;
    GpuTexture color, depth, motion, output;
};

ImageF RunSrPass(GpuContext& gpu, NgxSession& ngx, std::vector<std::unique_ptr<SrStage>>& cache,
                 const ImageF& in, unsigned tW, unsigned tH, bool hdr, unsigned preset) {
    const unsigned rW = static_cast<unsigned>(in.w), rH = static_cast<unsigned>(in.h);
    SrStage* st = nullptr;
    for (auto& s : cache)
        if (s->rW == rW && s->rH == rH && s->tW == tW && s->tH == tH && s->preset == preset) {
            st = s.get();
            break;
        }
    if (!st) {  // first image of this size: build the feature + textures once
        auto s = std::make_unique<SrStage>();
        s->rW = rW;
        s->rH = rH;
        s->tW = tW;
        s->tH = tH;
        s->preset = preset;
        s->color = gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                     L"srColor");
        s->depth = gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R32_FLOAT, true, L"srDepth");
        s->motion = gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R16G16_FLOAT, true, L"srMotion");
        s->output = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                      L"srOutput");
        std::vector<float> flat(static_cast<size_t>(rW) * rH, 0.0f);
        gpu.UploadR32Float(s->depth, flat);
        gpu.Begin();
        gpu.Transition(s->depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(s->motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.EndAndWait();
        DlssFeatureDesc d;
        d.renderW = rW;
        d.renderH = rH;
        d.targetW = tW;
        d.targetH = tH;
        d.quality = SrModeForRatio(rW, rH, tW, tH);
        d.preset = preset;
        d.hdr = hdr;
        d.autoExposure = true;
        ID3D12GraphicsCommandList* clc = gpu.Begin();
        s->feat.Create(ngx, clc, d);
        gpu.EndAndWait();
        st = s.get();
        cache.push_back(std::move(s));
    }

    gpu.UploadRgbaFloat(st->color, in.px);
    gpu.Begin();
    gpu.Transition(st->color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(st->output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();
    DlssEvalInputs ev;
    ev.color = &st->color;
    ev.depth = &st->depth;
    ev.motion = &st->motion;
    ev.output = &st->output;
    ev.reset = true;
    ID3D12GraphicsCommandList* cle = gpu.Begin();
    st->feat.Evaluate(cle, ev);
    gpu.EndAndWait();
    gpu.Begin();
    gpu.Transition(st->output, D3D12_RESOURCE_STATE_COMMON);
    gpu.EndAndWait();
    ImageF r;
    r.w = static_cast<int>(tW);
    r.h = static_cast<int>(tH);
    r.px = gpu.ReadbackRgbaFloat(st->output);
    return r;
}

// Reach any factor by chaining DLSS SR passes (each caps near 3x), the last pass sized to
// land exactly on the target. Only if a pass is refused do we finish the remainder bilinear.
// e.g. 768 -> 3840 (5x): pass 1 at 3x -> 2304, pass 2 at 1.67x -> 3840. Two DLSS passes.
ImageF UpscaleTo(GpuContext& gpu, NgxSession& ngx, std::vector<std::unique_ptr<SrStage>>& cache,
                 const ImageF& in, unsigned tW, unsigned tH, bool hdr, unsigned preset) {
    ImageF cur = in;
    int guard = 0;
    while ((static_cast<unsigned>(cur.w) < tW || static_cast<unsigned>(cur.h) < tH) &&
           guard++ < 8) {
        const double need =
            std::max(static_cast<double>(tW) / cur.w, static_cast<double>(tH) / cur.h);
        const double f = std::min(3.0, need);
        if (f <= 1.001) break;
        const unsigned nW = static_cast<unsigned>(std::llround(cur.w * f));
        const unsigned nH = static_cast<unsigned>(std::llround(cur.h * f));
        try {
            cur = RunSrPass(gpu, ngx, cache, cur, nW, nH, hdr, preset);
        } catch (const std::exception& e) {
            LogWarn("  DLSS pass refused (%s); finishing the rest bilinear", e.what());
            break;
        }
    }
    if (static_cast<unsigned>(cur.w) != tW || static_cast<unsigned>(cur.h) != tH)
        cur = ResizeBilinear(cur, tW, tH);
    return cur;
}

// Get-or-create a cached DLSS SR stage without evaluating it, so the video path can drive
// the pass itself and keep the upscaled output on the GPU (no readback between DLSS and NR).
SrStage* GetSrStage(GpuContext& gpu, NgxSession& ngx, std::vector<std::unique_ptr<SrStage>>& cache,
                    unsigned rW, unsigned rH, unsigned tW, unsigned tH, bool hdr, unsigned preset) {
    for (auto& s : cache)
        if (s->rW == rW && s->rH == rH && s->tW == tW && s->tH == tH && s->preset == preset)
            return s.get();
    auto s = std::make_unique<SrStage>();
    s->rW = rW;
    s->rH = rH;
    s->tW = tW;
    s->tH = tH;
    s->preset = preset;
    s->color =
        gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"srColor");
    s->depth = gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R32_FLOAT, true, L"srDepth");
    s->motion = gpu.CreateTexture((int) rW, (int) rH, DXGI_FORMAT_R16G16_FLOAT, true, L"srMotion");
    s->output =
        gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true, L"srOutput");
    std::vector<float> flat(static_cast<size_t>(rW) * rH, 0.0f);
    gpu.UploadR32Float(s->depth, flat);
    gpu.Begin();
    gpu.Transition(s->depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(s->motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();
    DlssFeatureDesc d;
    d.renderW = rW;
    d.renderH = rH;
    d.targetW = tW;
    d.targetH = tH;
    d.quality = SrModeForRatio(rW, rH, tW, tH);
    d.preset = preset;
    d.hdr = hdr;
    d.autoExposure = true;
    ID3D12GraphicsCommandList* cl = gpu.Begin();
    s->feat.Create(ngx, cl, d);
    gpu.EndAndWait();
    SrStage* p = s.get();
    cache.push_back(std::move(s));
    return p;
}
}  // namespace

int RunNeuralRendering(const std::string& dllDir, int adapter, const std::string& inPath,
                       const std::string& outDir, const NrModelParams& model, float detail,
                       float colour, bool hdr, float scale, unsigned outWReq, unsigned outHReq,
                       unsigned srPreset, bool writeDiff, bool writeOrig, bool verbose) {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double tLoad = 0, tUpscale = 0, tNrCreate = 0, tNrEval = 0, tReadback = 0, tCompose = 0,
           tSave = 0, tInit = 0;
    const auto tInit0 = Clock::now();

    // Input: a folder (every image in it) or a single image.
    std::vector<std::string> images;
    std::error_code ec;
    // Paths are UTF-8; widen for the filesystem (fs::path(std::string) would use the ANSI code
    // page and mangle non-ASCII paths), and hand back UTF-8 via the wide form.
    if (fs::is_directory(Narrow2Widen(inPath), ec)) {
        for (const auto& e : fs::recursive_directory_iterator(Narrow2Widen(inPath), ec)) {
            const std::string p = Widen2Narrow(e.path().wstring());
            if (e.is_regular_file() && HasImageExt(p)) images.push_back(p);
        }
        std::sort(images.begin(), images.end());
    } else {
        images.push_back(inPath);
    }
    if (images.empty()) {
        LogErr("no images found at %s", inPath.c_str());
        return 1;
    }
    LogInfo("Neural Rendering: %zu image(s), preset %u, style %u, intensity %.2f, "
            "detail %.2f, colour %.2f, hdr %s",
            images.size(), model.preset, model.style, model.intensity, detail, colour,
            hdr ? "on" : "off");

    GpuContext gpu;
    gpu.Initialize(false, adapter);
    LogInfo("  gpu:    %s", gpu.AdapterName().c_str());

    NgxSession ngx;
    ngx.Init(gpu.Device(), DefaultDllSearchPaths(dllDir), verbose);
    NVSDK_NGX_Parameter* caps = ngx.CapabilityParams();
    if (!caps) {
        LogErr("no capability parameters from the core");
        return 1;
    }
    DiscoverFloatSlot(caps);
    DiscoverUIntSlot(caps);

    ForwarderApi fwd;
    if (!fwd.Load()) {
        LogErr("could not load forwarder nvngx.dll_dlssnr.dll (err %lu)", GetLastError());
        return 1;
    }
    if (fwd.setSlots) fwd.setSlots(g_uintSlot, g_floatSlot);

    const std::string snippetNarrow =
        dllDir.empty() ? std::string("nvngx_dlssnr.dll") : dllDir + "/nvngx_dlssnr.dll";
    std::wstring snippetPath = Narrow2Widen(snippetNarrow);
    wchar_t absSnip[MAX_PATH]{};
    if (GetFullPathNameW(snippetPath.c_str(), MAX_PATH, absSnip, nullptr)) snippetPath = absSnip;
    const std::wstring dataPath = Narrow2Widen(outDir);

    fs::create_directories(Narrow2Widen(outDir), ec);

    // The feature and its GPU textures are built once per resolution and reused across
    // every image of that size - this is why a game hits 60 fps: create is the expensive
    // step (it builds the network), evaluate is cheap. We only rebuild when the size changes.
    void* feature = nullptr;
    unsigned curW = 0, curH = 0, curOW = 0, curOH = 0;
    bool primed = false;
    GpuTexture color, depth, motion, out;
    std::vector<std::unique_ptr<SrStage>> srCache;  // DLSS SR passes, reused across images

    tInit = ms(tInit0, Clock::now());
    int ok = 0, fail = 0;
    double sumPsnr = 0.0, sumSsim = 0.0;

    for (const std::string& imgPath : images) {
        auto t0 = Clock::now();
        ImageF img = LoadImageLinear(imgPath);
        tLoad += ms(t0, Clock::now());
        if (img.empty()) {
            LogWarn("  skip (could not load): %s", imgPath.c_str());
            ++fail;
            continue;
        }
        const unsigned w = static_cast<unsigned>(img.w);
        const unsigned h = static_cast<unsigned>(img.h);

        // Target (output) resolution. --nr-out W x H wins (0 on a side = keep aspect);
        // otherwise --nr-scale; otherwise 1:1 (native, DLAA).
        unsigned tW, tH;
        if (outWReq || outHReq) {
            if (outWReq && outHReq) {
                tW = outWReq;
                tH = outHReq;
            } else if (outWReq) {
                tW = outWReq;
                tH = static_cast<unsigned>(std::llround(static_cast<double>(h) * outWReq / w));
            } else {
                tH = outHReq;
                tW = static_cast<unsigned>(std::llround(static_cast<double>(w) * outHReq / h));
            }
        } else {
            tW = static_cast<unsigned>(std::llround(static_cast<double>(w) * scale));
            tH = static_cast<unsigned>(std::llround(static_cast<double>(h) * scale));
        }
        if (tW < 1) tW = 1;
        if (tH < 1) tH = 1;

        const bool isUpscale = (tW != w || tH != h);
        // Upscale with real DLSS Super Resolution first (chaining passes for any ratio), then
        // run NR at the target resolution (1:1) on that frame - the game order: DLSS upscales,
        // NR posts.
        t0 = Clock::now();
        ImageF work = isUpscale ? UpscaleTo(gpu, ngx, srCache, img, tW, tH, hdr, srPreset) : img;
        tUpscale += ms(t0, Clock::now());

        t0 = Clock::now();
        if (feature == nullptr || tW != curOW || tH != curOH) {
            if (feature) {
                fwd.release(feature);
                feature = nullptr;
            }
            color = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                      L"nrColor");
            depth = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R32_FLOAT, true, L"nrDepth");
            motion = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16_FLOAT, true, L"nrMotion");
            out = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                    L"nrOutput");
            std::vector<float> flatDepth(static_cast<size_t>(tW) * tH, 0.0f);
            gpu.UploadR32Float(depth, flatDepth);
            gpu.Begin();
            gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.Transition(depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.EndAndWait();

            // Prime the snippet through the core once - a core CreateFeature(18) loads and
            // wires nvngx_dlssnr.dll (hands it the NVAPI cubin interface); after that the
            // snippet's own Init_Ext via the forwarder succeeds. In a game the live DLSS
            // does this wiring; standalone, we do it here, once.
            if (!primed) {
                NrFeatureDesc pdesc;
                pdesc.inputW = tW;
                pdesc.inputH = tH;
                pdesc.outputW = tW;
                pdesc.outputH = tH;
                pdesc.preset = model.preset;
                pdesc.upscaling = false;
                NrFeature primeFeature;
                ID3D12GraphicsCommandList* clp = gpu.Begin();
                primeFeature.TryCreate(ngx, clp, pdesc);
                gpu.EndAndWait();
                primeFeature.Release();
                primed = true;
            }

            ID3D12GraphicsCommandList* clc = gpu.Begin();
            feature = fwd.create(snippetPath.c_str(), dataPath.c_str(), gpu.Device(), clc, caps, tW,
                                 tH, tW, tH, &model);
            gpu.EndAndWait();
            if (!feature) {
                int ic = 0, cc = 0;
                if (auto p = reinterpret_cast<int*>(GetProcAddress(fwd.mod, "fwd_last_init"))) ic = *p;
                if (auto p = reinterpret_cast<int*>(GetProcAddress(fwd.mod, "fwd_last_create")))
                    cc = *p;
                LogErr("  CreateFeature 18 failed at %ux%u (Init_Ext=%d, create=0x%08X)", tW, tH, ic,
                       static_cast<unsigned>(cc));
                ++fail;
                continue;
            }
            curW = tW;
            curH = tH;
            curOW = tW;
            curOH = tH;
            LogInfo("  feature created at %ux%u%s", tW, tH,
                    isUpscale ? " (input DLSS-upscaled to this)" : "");
        }
        tNrCreate += ms(t0, Clock::now());

        const size_t tpix = static_cast<size_t>(tW) * tH;
        t0 = Clock::now();
        // Colour fed to the model: display-referred (sRGB) for SDR, linear for HDR.
        std::vector<float> upv = work.px;
        if (!hdr)
            for (size_t i = 0; i < tpix; ++i)
                for (int c = 0; c < 3; ++c) upv[i * 4 + c] = LinearToSrgb(upv[i * 4 + c]);
        gpu.UploadRgbaFloat(color, upv);
        gpu.Begin();
        gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.EndAndWait();

        ID3D12GraphicsCommandList* cle = gpu.Begin();
        const int er = fwd.evaluate(cle, feature, caps, color.res.Get(), depth.res.Get(),
                                    motion.res.Get(), out.res.Get(), tW, tH, tW, tH, 1);
        gpu.EndAndWait();
        tNrEval += ms(t0, Clock::now());
        if (NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(er))) {
            LogWarn("  evaluate failed for %s: 0x%08X", imgPath.c_str(), static_cast<unsigned>(er));
            ++fail;
            continue;
        }

        t0 = Clock::now();
        gpu.Begin();
        gpu.Transition(out, D3D12_RESOURCE_STATE_COMMON);
        gpu.EndAndWait();
        std::vector<float> nrPx = gpu.ReadbackRgbaFloat(out);
        tReadback += ms(t0, Clock::now());

        t0 = Clock::now();
        if (!hdr)
            for (size_t i = 0; i < tpix; ++i)
                for (int c = 0; c < 3; ++c) nrPx[i * 4 + c] = SrgbToLinear(nrPx[i * 4 + c]);
        ImageF result = Composite(work, nrPx, detail, colour);
        tCompose += ms(t0, Clock::now());

        const std::string stem = Widen2Narrow(fs::path(Narrow2Widen(imgPath)).filename().wstring());
        const std::string base = outDir + "/" + stem;
        t0 = Clock::now();
        SavePng8(base + "_nr.png", result);
        // A/B reference (naive bilinear enlarge on an upscale, else the input). Only built when
        // its outputs are wanted - the diff/orig and the metric all use it.
        double psnr = 0, ssim = 0, maxd = 0;
        if (writeDiff || writeOrig) {
            ImageF ref = isUpscale ? ResizeBilinear(img, tW, tH) : img;
            if (writeOrig) SavePng8(base + "_orig.png", ref);
            if (writeDiff) {
                const DiffStats dstat = SaveDiffPng(base + "_nr_diff.png", ref, result, 8.0f);
                const Metrics m = ComputeMetrics(ref, result);
                psnr = m.psnrRgb;
                ssim = m.ssimLuma;
                maxd = dstat.maxAbs;
                sumPsnr += psnr;
                sumSsim += ssim;
            }
        }
        tSave += ms(t0, Clock::now());
        ++ok;
        if (writeDiff)
            LogInfo("  %-22s %ux%u -> %ux%u  %.2f dB  %.4f SSIM  (max %.3f)", stem.c_str(), w, h, tW,
                    tH, psnr, ssim, maxd);
        else
            LogInfo("  %-22s %ux%u -> %ux%u", stem.c_str(), w, h, tW, tH);
    }

    if (feature) fwd.release(feature);
    srCache.clear();  // release the DLSS SR features before NGX shuts down
    ngx.Shutdown();

    LogInfo("");
    LogInfo("Neural Rendering done: %d ok, %d failed -> %s", ok, fail, outDir.c_str());
    if (writeDiff && ok > 0)
        LogInfo("  mean vs input: %.2f dB PSNR, %.4f SSIM", sumPsnr / ok, sumSsim / ok);
    const int n = ok > 0 ? ok : 1;
    LogInfo("");
    LogInfo("timing (ms; total, and avg over %d image(s)):", ok);
    LogInfo("  init/load-once   %8.0f", tInit);
    LogInfo("  image load       %8.0f   %6.1f avg", tLoad, tLoad / n);
    LogInfo("  DLSS upscale     %8.0f   %6.1f avg", tUpscale, tUpscale / n);
    LogInfo("  NR feature build %8.0f   %6.1f avg", tNrCreate, tNrCreate / n);
    LogInfo("  NR evaluate      %8.0f   %6.1f avg", tNrEval, tNrEval / n);
    LogInfo("  readback         %8.0f   %6.1f avg", tReadback, tReadback / n);
    LogInfo("  sRGB+composite   %8.0f   %6.1f avg", tCompose, tCompose / n);
    LogInfo("  save PNGs        %8.0f   %6.1f avg", tSave, tSave / n);
    return ok > 0 ? 0 : 1;
}


// A bounded blocking queue for the frame conveyor.
namespace {
template <class T>
class FrameQueue {
public:
    explicit FrameQueue(size_t cap) : cap_(cap) {}
    bool push(T v) {
        std::unique_lock<std::mutex> l(m_);
        notFull_.wait(l, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        notEmpty_.notify_one();
        return true;
    }
    bool pop(T& out) {
        std::unique_lock<std::mutex> l(m_);
        notEmpty_.wait(l, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        notFull_.notify_one();
        return true;
    }
    void close() {
        std::unique_lock<std::mutex> l(m_);
        closed_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    size_t cap_;
    std::deque<T> q_;
    std::mutex m_;
    std::condition_variable notEmpty_, notFull_;
    bool closed_ = false;
};
}  // namespace

// Streaming video filter with a CPU/GPU conveyor: a reader thread pulls the next frame from
// stdin and does the sRGB->linear unpack while the GPU is still busy on the current one; a
// writer thread packs linear->sRGB and pushes to stdout. The GPU main loop never blocks on
// the ffmpeg pipes, so it stays saturated. The feature is built once for the whole stream.
int RunNeuralRenderingVideo(const std::string& dllDir, int adapter, unsigned inW, unsigned inH,
                            const NrModelParams& model, float detail, float colour, bool hdr,
                            float scale, unsigned outWReq, unsigned outHReq, bool motionOn,
                            bool motionVis, int motionEngine, unsigned srPreset, bool verbose) {
    SetLogToStderr(true);
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    unsigned tW, tH;
    if (outWReq || outHReq) {
        if (outWReq && outHReq) {
            tW = outWReq;
            tH = outHReq;
        } else if (outWReq) {
            tW = outWReq;
            tH = static_cast<unsigned>(std::llround(static_cast<double>(inH) * outWReq / inW));
        } else {
            tH = outHReq;
            tW = static_cast<unsigned>(std::llround(static_cast<double>(inW) * outHReq / inH));
        }
    } else {
        tW = static_cast<unsigned>(std::llround(static_cast<double>(inW) * scale));
        tH = static_cast<unsigned>(std::llround(static_cast<double>(inH) * scale));
    }
    if (tW < 1) tW = 1;
    if (tH < 1) tH = 1;
    const bool isUpscale = (tW != inW || tH != inH);
    LogInfo("video: %ux%u -> %ux%u, style %u, intensity %.2f, detail %.2f (pipelined)", inW, inH,
            tW, tH, model.style, model.intensity, detail);

    GpuContext gpu;
    gpu.Initialize(false, adapter);
    NgxSession ngx;
    ngx.Init(gpu.Device(), DefaultDllSearchPaths(dllDir), verbose);
    NVSDK_NGX_Parameter* caps = ngx.CapabilityParams();
    if (!caps) {
        LogErr("no capability parameters from the core");
        return 1;
    }
    DiscoverFloatSlot(caps);
    DiscoverUIntSlot(caps);

    ForwarderApi fwd;
    if (!fwd.Load()) {
        LogErr("could not load forwarder nvngx.dll_dlssnr.dll (err %lu)", GetLastError());
        return 1;
    }
    if (fwd.setSlots) fwd.setSlots(g_uintSlot, g_floatSlot);

    std::string snippetNarrow =
        dllDir.empty() ? std::string("nvngx_dlssnr.dll") : dllDir + "/nvngx_dlssnr.dll";
    std::wstring snippetPath = Narrow2Widen(snippetNarrow);
    wchar_t absSnip[MAX_PATH]{};
    if (GetFullPathNameW(snippetPath.c_str(), MAX_PATH, absSnip, nullptr)) snippetPath = absSnip;
    wchar_t appData[MAX_PATH]{};
    const DWORD envn = GetEnvironmentVariableW(L"LOCALAPPDATA", appData, MAX_PATH);
    const std::wstring dataPath =
        (envn > 0 && envn < MAX_PATH) ? std::wstring(appData) + L"\\video2dlssnr" : std::wstring(L".");

    GpuTexture color = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                         L"nrColor");
    GpuTexture depth = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R32_FLOAT, true, L"nrDepth");
    GpuTexture motion = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16_FLOAT, true,
                                          L"nrMotion");
    GpuTexture out = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                       L"nrOutput");
    std::vector<float> flatDepth(static_cast<size_t>(tW) * tH, 0.0f);
    gpu.UploadR32Float(depth, flatDepth);
    gpu.Begin();
    gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.EndAndWait();
    {
        NrFeatureDesc pdesc;
        pdesc.inputW = tW;
        pdesc.inputH = tH;
        pdesc.outputW = tW;
        pdesc.outputH = tH;
        pdesc.preset = model.preset;
        NrFeature primeFeature;
        ID3D12GraphicsCommandList* clp = gpu.Begin();
        primeFeature.TryCreate(ngx, clp, pdesc);
        gpu.EndAndWait();
        primeFeature.Release();
    }
    ID3D12GraphicsCommandList* clc = gpu.Begin();
    void* feature = fwd.create(snippetPath.c_str(), dataPath.c_str(), gpu.Device(), clc, caps, tW,
                               tH, tW, tH, &model);
    gpu.EndAndWait();
    if (!feature) {
        LogErr("CreateFeature 18 failed for the video stream");
        ngx.Shutdown();
        return 1;
    }
    LogInfo("feature ready; streaming (conveyor: read+unpack | GPU | pack+write)...");

    std::vector<std::unique_ptr<SrStage>> srCache;
    const size_t inBytes = static_cast<size_t>(inW) * inH * 4;
    const size_t inPix = static_cast<size_t>(inW) * inH;

    // A single DLSS pass reaches the target for any upscale <=3x (the usual video case); its
    // output stays on the GPU and the whole colour path - sRGB encode, NR, composite, 8-bit
    // pack - runs in compute shaders. No frame is copied back to the CPU mid-pipeline.
    const double needRatio =
        std::max(static_cast<double>(tW) / inW, static_cast<double>(tH) / inH);
    const bool singlePass = isUpscale && needRatio <= 3.0001;
    SrStage* sst =
        singlePass ? GetSrStage(gpu, ngx, srCache, inW, inH, tW, tH, hdr, srPreset) : nullptr;
    if (sst) LogInfo("DLSS SR preset: %s", PresetName(srPreset).c_str());

    GpuTexture origTex = gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                           L"nrOrig");
    GpuTexture finalU8 =
        gpu.CreateTexture((int) tW, (int) tH, DXGI_FORMAT_R8G8B8A8_UNORM, true, L"nrFinal");

    // Optical-flow motion vectors for NR's temporal reprojection (all on the GPU). Off unless
    // requested; --nr-motion-vis colour-codes the flow instead of running NR, to check it.
    // Motion-vector backend: prefer NVIDIA's hardware optical-flow engine (NVOFA), fall back to
    // the GPU Lucas-Kanade. NVOFA needs the Optical Flow SDK D3D12 header (gated, not yet
    // vendored), so it reports unavailable for now and we use LK. The visualisation and NR read
    // the same `motion` texture, so both work whichever backend fills it.
    // Motion backend: 0 auto (NVOFA, else LK), 1 nvof (forced, LK if unavailable), 2 lk (forced).
    const bool flowEnabled = motionOn || motionVis;
    OpticalFlow flow;
    NvofFlow nvof;
    bool nvofActive = false;
    if (flowEnabled) {
        const bool wantNvof = (motionEngine != 2);
        if (wantNvof) nvofActive = nvof.Init(gpu, (int) inW, (int) inH, (int) tW, (int) tH);
        if (!nvofActive) flow.Init(gpu, (int) tW, (int) tH);
        const char* label =
            nvofActive ? "NVIDIA NVOFA (hardware)"
                       : (wantNvof ? "Lucas-Kanade (GPU compute, NVOFA unavailable)"
                                   : "Lucas-Kanade (GPU compute)");
        LogInfo("optical flow: %s%s", label, motionVis ? "  [visualisation]" : "");
    }
    const float motionVisMax = std::max(8.0f, static_cast<float>(tW) / 120.0f);

    // A frame plus the scene-cut flag the reader computes for it (drives NR reset / flow restart).
    struct VidFrame {
        ImageF img;
        bool reset;
    };

    // Conveyor: the reader unpacks sRGB->linear off-thread, the GPU thread keeps everything on
    // the GPU, and the writer just fwrites the packed bytes the composite shader produced.
    FrameQueue<VidFrame> qIn(4);
    FrameQueue<std::vector<uint8_t>> qOut(4);

    std::thread reader([&] {
        std::vector<uint8_t> buf(inBytes);
        // Scene-cut detection on a tiny grey thumbnail: a big mean abs-diff between consecutive
        // frames (over 0.24 of full scale) means a cut, so NR history is reset rather than
        // dragged across it.
        const int gw = 64, gh = std::max(1, 64 * static_cast<int>(inH) / static_cast<int>(inW));
        std::vector<float> curGray(static_cast<size_t>(gw) * gh), prevGray;
        long long fno = 0;
        while (true) {
            size_t got = std::fread(buf.data(), 1, inBytes, stdin);
            if (got < inBytes) break;
            VidFrame vf;
            vf.img.w = static_cast<int>(inW);
            vf.img.h = static_cast<int>(inH);
            vf.img.px.resize(inPix * 4);
            for (size_t i = 0; i < inPix; ++i) {
                for (int c = 0; c < 3; ++c) {
                    float v = buf[i * 4 + c] / 255.0f;
                    vf.img.px[i * 4 + c] = hdr ? v : SrgbToLinear(v);
                }
                vf.img.px[i * 4 + 3] = buf[i * 4 + 3] / 255.0f;
            }
            for (int y = 0; y < gh; ++y)
                for (int x = 0; x < gw; ++x) {
                    const int sx = x * static_cast<int>(inW) / gw, sy = y * static_cast<int>(inH) / gh;
                    const uint8_t* p = &buf[(static_cast<size_t>(sy) * inW + sx) * 4];
                    curGray[static_cast<size_t>(y) * gw + x] =
                        (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.0f;
                }
            bool reset = (fno == 0);
            if (!prevGray.empty()) {
                double d = 0;
                for (size_t i = 0; i < curGray.size(); ++i) d += std::fabs(curGray[i] - prevGray[i]);
                if (d / curGray.size() > 0.24) reset = true;
            }
            prevGray = curGray;
            vf.reset = reset;
            ++fno;
            if (!qIn.push(std::move(vf))) break;
        }
        qIn.close();
    });

    std::thread writer([&] {
        std::vector<uint8_t> buf;
        while (qOut.pop(buf)) std::fwrite(buf.data(), 1, buf.size(), stdout);
        std::fflush(stdout);
    });

    const auto t0 = std::chrono::steady_clock::now();
    long long frames = 0;
    bool failed = false;
    VidFrame vf;
    while (qIn.pop(vf)) {
        ImageF& img = vf.img;
        GpuTexture* orig = nullptr;
        if (singlePass) {
            // DLSS upscale, output kept on the GPU, then sRGB-encode straight into the NR input.
            gpu.UploadRgbaFloat(sst->color, img.px);
            ID3D12GraphicsCommandList* cl = gpu.Begin();
            gpu.Transition(sst->color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.Transition(sst->output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            DlssEvalInputs ev;
            ev.color = &sst->color;
            ev.depth = &sst->depth;
            ev.motion = &sst->motion;
            ev.output = &sst->output;
            ev.reset = true;
            sst->feat.Evaluate(cl, ev);
            gpu.Transition(sst->output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.RecordEncodeSrgb(sst->output, color);
            gpu.EndAndWait();
            orig = &sst->output;
        } else {
            // No upscale, or an extreme ratio that needs the chained (CPU) path: land the
            // upscaled frame in a GPU texture and sRGB-encode it into the NR input.
            ImageF up = isUpscale ? UpscaleTo(gpu, ngx, srCache, img, tW, tH, hdr, srPreset) : img;
            gpu.UploadRgbaFloat(origTex, up.px);
            gpu.Begin();
            gpu.Transition(origTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.RecordEncodeSrgb(origTex, color);
            gpu.EndAndWait();
            orig = &origTex;
        }

        // Optical flow between this frame and the previous one -> motion vectors for NR.
        if (nvofActive) {
            nvof.Compute(gpu, *orig, motion, vf.reset);
        } else if (flowEnabled) {
            ID3D12GraphicsCommandList* clf = gpu.Begin();
            flow.Record(clf, *orig, motion, vf.reset);
            gpu.EndAndWait();
        }

        std::vector<uint8_t> bytes;
        if (motionVis) {
            // Debug: emit the colour-coded flow instead of running NR.
            gpu.Begin();
            gpu.Transition(motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(finalU8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.RecordFlowVis(motion, finalU8, motionVisMax);
            gpu.EndAndWait();
            gpu.Begin();
            gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.EndAndWait();
            bytes = gpu.ReadbackRgba8(finalU8);
        } else {
            ID3D12GraphicsCommandList* cle = gpu.Begin();
            gpu.Transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const int er = fwd.evaluate(cle, feature, caps, color.res.Get(), depth.res.Get(),
                                        motion.res.Get(), out.res.Get(), tW, tH, tW, tH,
                                        vf.reset ? 1 : 0);
            gpu.EndAndWait();
            if (NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(er))) {
                LogErr("evaluate failed on frame %lld: 0x%08X", frames, static_cast<unsigned>(er));
                failed = true;
                break;
            }

            // Composite the model output over the upscaled original and pack to 8-bit, all on GPU.
            gpu.Begin();
            gpu.Transition(out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(finalU8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.RecordComposite(*orig, out, finalU8, detail, colour);
            gpu.EndAndWait();
            bytes = gpu.ReadbackRgba8(finalU8);
        }
        if (!qOut.push(std::move(bytes))) break;
        ++frames;
        if ((frames % 5) == 0) {
            const double sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "NRPROG %lld %.3f\n", frames, frames / (sec > 0 ? sec : 1));
            std::fflush(stderr);
        }
    }
    qOut.close();
    reader.join();
    writer.join();

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fwd.release(feature);
    if (nvofActive)
        nvof.Shutdown();
    else if (flowEnabled)
        flow.Shutdown();
    srCache.clear();
    ngx.Shutdown();
    LogInfo("done: %lld frames in %.1f s (%.1f fps)", frames, sec, frames / (sec > 0 ? sec : 1));
    return (frames > 0 && !failed) ? 0 : 1;
}
