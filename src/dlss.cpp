#include "dlss.h"

#include <psapi.h>

#include <algorithm>
#include <cctype>

#pragma comment(lib, "version.lib")

const char* NgxResultToString(NVSDK_NGX_Result r) {
    switch (r) {
        case NVSDK_NGX_Result_Success: return "Success";
        case NVSDK_NGX_Result_Fail: return "Fail";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "UnableToInitializeFeature";
        case NVSDK_NGX_Result_FAIL_OutOfDate: return "OutOfDate";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied: return "Denied";
        case NVSDK_NGX_Result_FAIL_NotImplemented: return "NotImplemented";
        default: return "Unknown";
    }
}

static void NgxCheck(NVSDK_NGX_Result r, const char* what) {
    if (NVSDK_NGX_SUCCEED(r)) return;
    char buf[64];
    snprintf(buf, sizeof(buf), " (0x%08X)", static_cast<unsigned>(r));
    throw ToolError(std::string(what) + " failed: " + NgxResultToString(r) + buf);
}

const std::vector<QualityMode>& AllQualityModes() {
    static const std::vector<QualityMode> modes = {
        {NVSDK_NGX_PerfQuality_Value_DLAA, "dlaa", "DLAA"},
        {NVSDK_NGX_PerfQuality_Value_UltraQuality, "ultraquality", "UltraQuality"},
        {NVSDK_NGX_PerfQuality_Value_MaxQuality, "quality", "Quality"},
        {NVSDK_NGX_PerfQuality_Value_Balanced, "balanced", "Balanced"},
        {NVSDK_NGX_PerfQuality_Value_MaxPerf, "performance", "Performance"},
        {NVSDK_NGX_PerfQuality_Value_UltraPerformance, "ultraperformance", "UltraPerformance"},
    };
    return modes;
}

bool ParseQualityMode(const std::string& s, QualityMode* out) {
    std::string k;
    for (char c : s) {
        if (c != '-' && c != '_' && c != ' ') k += static_cast<char>(std::tolower(c));
    }
    if (k == "perf") k = "performance";
    if (k == "ultraperf") k = "ultraperformance";
    if (k == "ultrapref") k = "ultraperformance";
    if (k == "maxquality") k = "quality";
    if (k == "maxperf") k = "performance";
    for (const QualityMode& m : AllQualityModes()) {
        if (k == m.name) {
            *out = m;
            return true;
        }
    }
    return false;
}

bool ParsePreset(const std::string& s, unsigned* out) {
    std::string k;
    for (char c : s) k += static_cast<char>(std::tolower(c));
    if (k == "default" || k == "0") {
        *out = 0;
        return true;
    }
    if (k.size() == 1 && k[0] >= 'a' && k[0] <= 'o') {
        *out = static_cast<unsigned>(k[0] - 'a' + 1);
        return true;
    }
    // Numeric form, so future presets work without a tool update.
    if (!k.empty() && k.find_first_not_of("0123456789") == std::string::npos) {
        const unsigned v = static_cast<unsigned>(std::stoul(k));
        if (v <= 31) {
            *out = v;
            return true;
        }
    }
    return false;
}

std::string PresetName(unsigned preset) {
    if (preset == 0) return "default";
    if (preset >= 1 && preset <= 15) return std::string(1, static_cast<char>('A' + preset - 1));
    return std::to_string(preset);
}

// ---------------------------------------------------------------------------

static void NVSDK_CONV NgxLogSink(const char* message, NVSDK_NGX_Logging_Level level,
                                  NVSDK_NGX_Feature /*component*/) {
    if (!message) return;
    std::string text = message;
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    if (text.empty()) return;
    if (level == NVSDK_NGX_LOGGING_LEVEL_OFF) return;
    LogDebug("ngx: %s", text.c_str());
}

NgxSession::~NgxSession() { Shutdown(); }

void NgxSession::Init(ID3D12Device* device, const std::vector<std::wstring>& dllSearchPaths,
                      bool verboseLogging) {
    CHECK(device != nullptr);
    m_device = device;

    std::vector<const wchar_t*> paths;
    paths.reserve(dllSearchPaths.size());
    for (const std::wstring& p : dllSearchPaths) paths.push_back(p.c_str());

    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths.empty() ? nullptr : paths.data();
    info.PathListInfo.Length = static_cast<unsigned>(paths.size());
    info.LoggingInfo.LoggingCallback = &NgxLogSink;
    info.LoggingInfo.MinimumLoggingLevel =
        verboseLogging ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE : NVSDK_NGX_LOGGING_LEVEL_OFF;
    info.LoggingInfo.DisableOtherLoggingSinks = true;

    wchar_t appData[MAX_PATH]{};
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", appData, MAX_PATH);
    std::wstring dataPath =
        (n > 0 && n < MAX_PATH) ? std::wstring(appData) + L"\\video2dlssnr" : std::wstring(L".");
    CreateDirectoryW(dataPath.c_str(), nullptr);

    NgxCheck(NVSDK_NGX_D3D12_Init_with_ProjectID(
                 "b5a5e4a1-9a52-4d1c-9d3f-6d0b7f2c1e84", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                 dataPath.c_str(), device, &info, NVSDK_NGX_Version_API),
             "NVSDK_NGX_D3D12_Init_with_ProjectID");
    m_initialized = true;

    NgxCheck(NVSDK_NGX_D3D12_GetCapabilityParameters(&m_capParams),
             "NVSDK_NGX_D3D12_GetCapabilityParameters");
}

void NgxSession::Shutdown() {
    if (m_capParams) {
        NVSDK_NGX_D3D12_DestroyParameters(m_capParams);
        m_capParams = nullptr;
    }
    if (m_initialized && m_device) {
        NVSDK_NGX_D3D12_Shutdown1(m_device);
        m_initialized = false;
    }
    m_device = nullptr;
}

DlssCaps NgxSession::QueryCaps() const {
    CHECK(m_capParams != nullptr);
    DlssCaps c;
    int available = 0;
    NVSDK_NGX_Parameter_GetI(m_capParams, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    c.available = available != 0;

    int needsDriver = 0;
    NVSDK_NGX_Parameter_GetI(m_capParams, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                             &needsDriver);
    c.needsUpdatedDriver = needsDriver != 0;

    NVSDK_NGX_Parameter_GetUI(m_capParams, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                              &c.minDriverMajor);
    NVSDK_NGX_Parameter_GetUI(m_capParams, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                              &c.minDriverMinor);

    unsigned initResult = 0;
    NVSDK_NGX_Parameter_GetUI(m_capParams, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                              &initResult);
    c.initResult = static_cast<NVSDK_NGX_Result>(initResult);
    return c;
}

OptimalSettings NgxSession::GetOptimal(unsigned targetW, unsigned targetH,
                                       NVSDK_NGX_PerfQuality_Value quality) const {
    CHECK(m_capParams != nullptr);
    OptimalSettings s;
    NgxCheck(NGX_DLSS_GET_OPTIMAL_SETTINGS(m_capParams, targetW, targetH, quality, &s.renderW,
                                           &s.renderH, &s.maxW, &s.maxH, &s.minW, &s.minH,
                                           &s.sharpness),
             "NGX_DLSS_GET_OPTIMAL_SETTINGS");
    if (s.renderW == 0 || s.renderH == 0) {
        throw ToolError("DLSS reported render resolution 0x0 for this quality mode — the mode is "
                        "not supported at this output size");
    }
    return s;
}

bool NgxSession::FindLoadedDlssModule(std::string* path, std::string* version) {
    HMODULE mods[512];
    DWORD needed = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return false;
    const size_t count = std::min<size_t>(needed / sizeof(HMODULE), 512);

    for (size_t i = 0; i < count; ++i) {
        wchar_t name[MAX_PATH]{};
        if (!GetModuleFileNameW(mods[i], name, MAX_PATH)) continue;
        std::wstring w = name;
        std::wstring lower = w;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::towlower);
        if (lower.find(L"nvngx_dlss.dll") == std::wstring::npos) continue;

        if (path) *path = Widen2Narrow(w);
        if (version) {
            *version = "unknown";
            DWORD dummy = 0;
            const DWORD size = GetFileVersionInfoSizeW(w.c_str(), &dummy);
            if (size) {
                std::vector<uint8_t> buf(size);
                if (GetFileVersionInfoW(w.c_str(), 0, size, buf.data())) {
                    VS_FIXEDFILEINFO* ffi = nullptr;
                    UINT len = 0;
                    if (VerQueryValueW(buf.data(), L"\\", reinterpret_cast<void**>(&ffi), &len) &&
                        ffi) {
                        char v[64];
                        snprintf(v, sizeof(v), "%u.%u.%u.%u", HIWORD(ffi->dwFileVersionMS),
                                 LOWORD(ffi->dwFileVersionMS), HIWORD(ffi->dwFileVersionLS),
                                 LOWORD(ffi->dwFileVersionLS));
                        *version = v;
                    }
                }
            }
        }
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------

DlssFeature::~DlssFeature() { Release(); }

void DlssFeature::Create(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                         const DlssFeatureDesc& desc) {
    CHECK(session.Initialized());
    Release();
    m_desc = desc;

    NgxCheck(NVSDK_NGX_D3D12_AllocateParameters(&m_params),
             "NVSDK_NGX_D3D12_AllocateParameters");

    // The preset hint is consumed at feature-creation time, so it must be set on
    // the same parameter block before CreateFeature. Setting every quality slot
    // keeps the hint effective whichever mode this feature is built for.
    const char* presetKeys[] = {
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
    };
    for (const char* key : presetKeys) NVSDK_NGX_Parameter_SetUI(m_params, key, desc.preset);

    int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (desc.hdr) flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (desc.autoExposure) flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (desc.alphaUpscaling) flags |= NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling;

    NVSDK_NGX_DLSS_Create_Params create{};
    create.Feature.InWidth = desc.renderW;
    create.Feature.InHeight = desc.renderH;
    create.Feature.InTargetWidth = desc.targetW;
    create.Feature.InTargetHeight = desc.targetH;
    create.Feature.InPerfQualityValue = desc.quality;
    create.InFeatureCreateFlags = flags;
    create.InEnableOutputSubrects = false;

    NgxCheck(NGX_D3D12_CREATE_DLSS_EXT(cmdList, 1, 1, &m_handle, m_params, &create),
             "NGX_D3D12_CREATE_DLSS_EXT");
}

void DlssFeature::Evaluate(ID3D12GraphicsCommandList* cmdList, const DlssEvalInputs& in) {
    CHECK(m_handle && m_params);
    CHECK(in.color && in.depth && in.motion && in.output);

    NVSDK_NGX_D3D12_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = in.color->res.Get();
    eval.Feature.pInOutput = in.output->res.Get();
    eval.Feature.InSharpness = 0.0f;
    eval.pInDepth = in.depth->res.Get();
    eval.pInMotionVectors = in.motion->res.Get();
    eval.InJitterOffsetX = in.jitterX;
    eval.InJitterOffsetY = in.jitterY;
    eval.InRenderSubrectDimensions.Width = static_cast<unsigned>(in.color->w);
    eval.InRenderSubrectDimensions.Height = static_cast<unsigned>(in.color->h);
    eval.InReset = in.reset ? 1 : 0;
    // Motion vectors are authored directly in render-pixel units, so no rescale.
    eval.InMVScaleX = 1.0f;
    eval.InMVScaleY = 1.0f;
    eval.InPreExposure = 1.0f;
    eval.InExposureScale = 1.0f;
    eval.InFrameTimeDeltaInMsec = in.frameTimeMs;
    if (in.exposure) eval.pInExposureTexture = in.exposure->res.Get();

    NgxCheck(NGX_D3D12_EVALUATE_DLSS_EXT(cmdList, m_handle, m_params, &eval),
             "NGX_D3D12_EVALUATE_DLSS_EXT");
}

void DlssFeature::Release() {
    if (m_handle) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_handle);
        m_handle = nullptr;
    }
    if (m_params) {
        NVSDK_NGX_D3D12_DestroyParameters(m_params);
        m_params = nullptr;
    }
}
