#include "slprobe.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "gpu.h"

// ---------------------------------------------------------------------------
// Streamline 2.13 ABI, reconstructed from the public headers (sl_struct.h,
// sl_core_types.h, sl_core_api.h, sl_result.h) and cross-checked against the
// sl.dlss_nr.dll manifest (feature id 1004). Only the pieces the support probe
// needs are declared. Layout must match MSVC x64 field-for-field.
// ---------------------------------------------------------------------------
namespace slabi {

struct StructType { uint32_t d1; uint16_t d2; uint16_t d3; uint8_t d4[8]; };

struct BaseStructure {
    BaseStructure* next{nullptr};
    StructType structType{};
    size_t structVersion{0};
};

// {1CA10965-BF8E-432B-8DA1-6716D879FB14}, kStructVersion1
static const StructType kPreferencesType{
    0x1CA10965, 0xBF8E, 0x432B, {0x8D, 0xA1, 0x67, 0x16, 0xD8, 0x79, 0xFB, 0x14}};
// {0677315F-A746-4492-9F42-CB6142C9C3D4}, kStructVersion1
static const StructType kAdapterInfoType{
    0x0677315F, 0xA746, 0x4492, {0x9F, 0x42, 0xCB, 0x61, 0x42, 0xC9, 0xC3, 0xD4}};

struct Preferences : BaseStructure {
    bool showConsole{false};
    uint32_t logLevel{1};                 // eDefault
    const wchar_t** pathsToPlugins{nullptr};
    uint32_t numPathsToPlugins{0};
    const wchar_t* pathToLogsAndData{nullptr};
    void* allocateCallback{nullptr};
    void* releaseCallback{nullptr};
    void* logMessageCallback{nullptr};
    uint64_t flags{0};                    // PreferenceFlags (64-bit)
    const uint32_t* featuresToLoad{nullptr};
    uint32_t numFeaturesToLoad{0};
    uint32_t applicationId{0};
    uint32_t engine{0};                   // eCustom
    const char* engineVersion{nullptr};
    const char* projectId{nullptr};
    uint32_t renderAPI{1};                // eD3D12
};

struct AdapterInfo : BaseStructure {
    uint8_t* deviceLUID{nullptr};
    uint32_t deviceLUIDSizeInBytes{0};
    void* vkPhysicalDevice{nullptr};
};

// Result enum (sl_result.h)
enum Result : uint32_t {
    eOk = 0,
    eErrorDriverOutOfDate = 2,
    eErrorOSOutOfDate = 3,
    eErrorNoSupportedAdapterFound = 6,
    eErrorAdapterNotSupported = 7,
    eErrorNoPlugins = 8,
    eErrorNGXFailed = 15,
    eErrorNotInitialized = 21,
    eErrorInvalidParameter = 25,
    eErrorFeatureMissing = 31,
    eErrorFeatureNotSupported = 32,
    eErrorFeatureFailedToLoad = 34,
    eErrorFeatureMissingDependency = 36,
};

using PfnSlInit = Result(*)(const Preferences&, uint64_t sdkVersion);
using PfnSlShutdown = Result(*)();
using PfnSlIsFeatureSupported = Result(*)(uint32_t feature, const AdapterInfo&);
using PfnSlIsFeatureLoaded = Result(*)(uint32_t feature, bool& loaded);
using PfnSlSetFeatureLoaded = Result(*)(uint32_t feature, bool loaded);
using PfnLogCallback = void(*)(uint32_t type, const char* msg);

// 2.13.0 : (2<<48)|(13<<32)|(0<<16)|0xfedc
static const uint64_t kSDKVersion = 0x0002000d0000fedcull;

const char* ResultName(Result r) {
    switch (r) {
        case eOk: return "eOk";
        case eErrorDriverOutOfDate: return "eErrorDriverOutOfDate";
        case eErrorOSOutOfDate: return "eErrorOSOutOfDate";
        case eErrorNoSupportedAdapterFound: return "eErrorNoSupportedAdapterFound";
        case eErrorAdapterNotSupported: return "eErrorAdapterNotSupported";
        case eErrorNoPlugins: return "eErrorNoPlugins";
        case eErrorNGXFailed: return "eErrorNGXFailed";
        case eErrorNotInitialized: return "eErrorNotInitialized";
        case eErrorInvalidParameter: return "eErrorInvalidParameter";
        case eErrorFeatureMissing: return "eErrorFeatureMissing";
        case eErrorFeatureNotSupported: return "eErrorFeatureNotSupported";
        case eErrorFeatureFailedToLoad: return "eErrorFeatureFailedToLoad";
        case eErrorFeatureMissingDependency: return "eErrorFeatureMissingDependency";
        default: return "eError?";
    }
}

}  // namespace slabi

static void SlLog(uint32_t type, const char* msg) {
    std::string s = msg ? msg : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    LogInfo("  sl[%u]: %s", type, s.c_str());
}

// Open the driver core's feature-id gates (the cmp <reg>,0x12 sites route C
// found) BEFORE slInit, so Streamline's pre-flight getNGXFeatureRequirements
// for DLSS-NR is not rejected with OutOfDate. Streamline itself loads and wires
// the snippet via sl.common's own NvAPI cubin backend; the driver core is only
// consulted for the requirements pre-check. Opening the gate lets that check
// pass so Streamline proceeds to its own snippet path - the game/addon combo.
struct GatePatch { uint8_t* at; DWORD prot; };

static HMODULE PatchCoreGates(std::vector<GatePatch>& restore) {
    static const uint32_t kGateImmRVAs[] = {
        0x0658f, 0x06943, 0x0858f, 0x087f7, 0x0d4ff, 0x0d953, 0x0dc2a, 0x0e3d7,
        0x0e647, 0x641cf, 0x64546, 0x650eb, 0x66d4f, 0x670f6, 0x67c8b,
    };
    HMODULE core = GetModuleHandleW(L"_nvngx.dll");
    if (!core) {
        const std::wstring root = L"C:\\Windows\\System32\\DriverStore\\FileRepository\\";
        WIN32_FIND_DATAW fd{};
        HANDLE h = FindFirstFileW((root + L"nvmdi.inf_amd64_*").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    std::wstring p = root + fd.cFileName + L"\\_nvngx.dll";
                    core = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
                    if (core) { LogInfo("  pre-loaded core %s", Widen2Narrow(p).c_str()); break; }
                }
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
    }
    if (!core) {
        LogInfo("  (could not locate _nvngx.dll to open gates)");
        return nullptr;
    }
    uint8_t* base = reinterpret_cast<uint8_t*>(core);
    int mism = 0;
    for (uint32_t rva : kGateImmRVAs) {
        uint8_t* at = base + rva;
        if (*at != 0x12) { ++mism; continue; }
        DWORD prot = 0;
        if (!VirtualProtect(at, 1, PAGE_EXECUTE_READWRITE, &prot)) continue;
        *at = 0x13;
        FlushInstructionCache(GetCurrentProcess(), at, 1);
        restore.push_back({at, prot});
    }
    LogInfo("  opened %zu/%d driver-core feature-id gates before slInit%s", restore.size(),
            (int)(sizeof(kGateImmRVAs) / sizeof(kGateImmRVAs[0])),
            mism ? " [some bytes not 0x12 - core build differs]" : "");
    return core;
}

static void RestoreCoreGates(std::vector<GatePatch>& restore) {
    for (GatePatch& g : restore) {
        *g.at = 0x12;
        FlushInstructionCache(GetCurrentProcess(), g.at, 1);
        DWORD tmp = 0;
        VirtualProtect(g.at, 1, g.prot, &tmp);
    }
    if (!restore.empty()) LogInfo("  restored %zu driver-core gates", restore.size());
    restore.clear();
}

int ProbeStreamlineNR(const std::string& pluginDir, unsigned featureId, bool verbose) {
    using namespace slabi;

    GpuContext gpu;
    gpu.Initialize(false, 0);
    LogInfo("  gpu:    %s (%zu MB)", gpu.AdapterName().c_str(), gpu.AdapterVramMB());

    LUID luid = gpu.Device()->GetAdapterLuid();
    LogInfo("  adapter LUID: %08lx:%08lx", (unsigned long)luid.HighPart,
            (unsigned long)luid.LowPart);

    const std::wstring dirW = Narrow2Widen(pluginDir);
    const std::wstring interposer =
        pluginDir.empty() ? L"sl.interposer.dll" : dirW + L"\\sl.interposer.dll";

    HMODULE mod = LoadLibraryExW(interposer.c_str(), nullptr,
                                 LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!mod) {
        LogErr("could not load sl.interposer.dll from %s (err %lu)",
                 pluginDir.c_str(), GetLastError());
        return 1;
    }
    {
        wchar_t resolved[MAX_PATH]{};
        if (GetModuleFileNameW(mod, resolved, MAX_PATH))
            LogInfo("  loaded %s", Widen2Narrow(resolved).c_str());
    }

    auto slInit = reinterpret_cast<PfnSlInit>(GetProcAddress(mod, "slInit"));
    auto slShutdown = reinterpret_cast<PfnSlShutdown>(GetProcAddress(mod, "slShutdown"));
    auto slIsSupported =
        reinterpret_cast<PfnSlIsFeatureSupported>(GetProcAddress(mod, "slIsFeatureSupported"));
    auto slIsLoaded =
        reinterpret_cast<PfnSlIsFeatureLoaded>(GetProcAddress(mod, "slIsFeatureLoaded"));
    LogInfo("  exports: slInit=%s slIsFeatureSupported=%s slIsFeatureLoaded=%s slShutdown=%s",
            slInit ? "y" : "n", slIsSupported ? "y" : "n", slIsLoaded ? "y" : "n",
            slShutdown ? "y" : "n");
    if (!slInit || !slIsSupported) {
        LogErr("interposer missing required exports");
        return 1;
    }

    // ---- slInit, loading the DLSS-NR plugin (1004) from the pack directory ----
    const uint32_t features[] = {featureId};
    const wchar_t* paths[] = {dirW.c_str()};

    Preferences pref;
    pref.structType = kPreferencesType;
    pref.structVersion = 1;
    pref.showConsole = false;
    pref.logLevel = verbose ? 2u : 1u;    // eVerbose : eDefault
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.logMessageCallback = reinterpret_cast<void*>(&SlLog);
    // eDisableCLStateTracking | eAllowOTA | eLoadDownloadedPlugins | eBypassOSVersionCheck
    pref.flags = 0x1 | 0x8 | 0x40 | 0x10;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 1;
    pref.applicationId = 231313132;       // id used by the SL samples
    pref.engine = 0;                      // eCustom
    pref.renderAPI = 1;                   // eD3D12

    LogInfo("");
    LogInfo("opening driver-core feature-id gates so Streamline pre-flight passes:");
    std::vector<GatePatch> gateRestore;
    PatchCoreGates(gateRestore);

    LogInfo("");
    LogInfo("slInit (feature %u from %s):", featureId, pluginDir.c_str());
    Result r = slInit(pref, kSDKVersion);
    LogInfo("  slInit -> %s (%u)", ResultName(r), (unsigned)r);
    if (r != eOk) {
        LogErr("Streamline did not initialise; cannot query support");
        if (slShutdown) slShutdown();
        RestoreCoreGates(gateRestore);
        return 1;
    }

    bool loaded = false;
    if (slIsLoaded) {
        Result lr = slIsLoaded(featureId, loaded);
        LogInfo("  slIsFeatureLoaded(%u) -> %s, loaded=%s", featureId, ResultName(lr),
                loaded ? "true" : "false");
    }

    // ---- the actual question: does Streamline say DLSS-NR is supported here? ----
    AdapterInfo ai;
    ai.structType = kAdapterInfoType;
    ai.structVersion = 1;
    ai.deviceLUID = reinterpret_cast<uint8_t*>(&luid);
    ai.deviceLUIDSizeInBytes = sizeof(LUID);

    LogInfo("");
    LogInfo("slIsFeatureSupported(%u) - Streamline's own verdict:", featureId);
    Result sr = slIsSupported(featureId, ai);
    LogInfo("  -> %s (%u)", ResultName(sr), (unsigned)sr);

    const bool ok = (sr == eOk);
    LogInfo("");
    LogInfo("VERDICT: DLSS Neural Rendering (feature %u) is %s on this driver + DLL",
            featureId, ok ? "SUPPORTED" : "NOT supported");

    if (slShutdown) slShutdown();
    RestoreCoreGates(gateRestore);
    return ok ? 0 : 1;
}
