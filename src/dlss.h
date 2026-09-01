// dlss.h — NGX session and a single DLSS Super Resolution feature instance.
#pragma once

#include "common.h"
#include "gpu.h"

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"

const char* NgxResultToString(NVSDK_NGX_Result r);

// Quality modes, in the order the tool prints them.
struct QualityMode {
    NVSDK_NGX_PerfQuality_Value value;
    const char* name;   // canonical lowercase name accepted on the command line
    const char* label;  // pretty name for tables
};
const std::vector<QualityMode>& AllQualityModes();
bool ParseQualityMode(const std::string& s, QualityMode* out);

// Presets are letters A..O mapping to NVSDK_NGX_DLSS_Hint_Render_Preset 1..15,
// plus "default" (0, meaning "whatever the driver picks, may change via OTA").
bool ParsePreset(const std::string& s, unsigned* out);
std::string PresetName(unsigned preset);

struct DlssCaps {
    bool available = false;
    bool needsUpdatedDriver = false;
    unsigned minDriverMajor = 0;
    unsigned minDriverMinor = 0;
    NVSDK_NGX_Result initResult = NVSDK_NGX_Result_Success;
};

struct OptimalSettings {
    unsigned renderW = 0;
    unsigned renderH = 0;
    unsigned maxW = 0;
    unsigned maxH = 0;
    unsigned minW = 0;
    unsigned minH = 0;
    float sharpness = 0.0f;
};

class NgxSession {
public:
    ~NgxSession();

    // dllSearchPaths are handed to NGX so it prefers the DLSS build sitting next
    // to the tool over whatever the driver ships.
    void Init(ID3D12Device* device, const std::vector<std::wstring>& dllSearchPaths,
              bool verboseLogging);
    void Shutdown();

    DlssCaps QueryCaps() const;
    OptimalSettings GetOptimal(unsigned targetW, unsigned targetH,
                               NVSDK_NGX_PerfQuality_Value quality) const;

    NVSDK_NGX_Parameter* CapabilityParams() const { return m_capParams; }
    ID3D12Device* Device() const { return m_device; }
    bool Initialized() const { return m_initialized; }

    // Path and file version of the nvngx_dlss.dll actually mapped into this
    // process. This is the only trustworthy answer to "which DLSS am I running".
    static bool FindLoadedDlssModule(std::string* path, std::string* version);

private:
    ID3D12Device* m_device = nullptr;
    NVSDK_NGX_Parameter* m_capParams = nullptr;
    bool m_initialized = false;
};

struct DlssFeatureDesc {
    unsigned renderW = 0;
    unsigned renderH = 0;
    unsigned targetW = 0;
    unsigned targetH = 0;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    unsigned preset = 0;
    bool hdr = false;
    bool autoExposure = true;
    bool alphaUpscaling = false;
};

struct DlssEvalInputs {
    GpuTexture* color = nullptr;
    GpuTexture* depth = nullptr;
    GpuTexture* motion = nullptr;
    GpuTexture* output = nullptr;
    GpuTexture* exposure = nullptr;  // optional; null when auto-exposure is on
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    bool reset = false;
    float frameTimeMs = 16.667f;
};

class DlssFeature {
public:
    ~DlssFeature();

    void Create(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                const DlssFeatureDesc& desc);
    void Evaluate(ID3D12GraphicsCommandList* cmdList, const DlssEvalInputs& in);
    void Release();

    const DlssFeatureDesc& Desc() const { return m_desc; }

private:
    NVSDK_NGX_Handle* m_handle = nullptr;
    NVSDK_NGX_Parameter* m_params = nullptr;
    DlssFeatureDesc m_desc;
};
