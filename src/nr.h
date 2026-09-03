// nr.h — DLSS Neural Rendering (NGX feature 18).
//
// The public NGX SDK does not describe this feature: the enum slot is just
// `NVSDK_NGX_Feature_Reserved18`, there are no helper structs, and no parameter
// names are published. Everything below was recovered from the shipping DLL —
// its assertion strings, its RTTI class names, and the `DLSSNR.*` parameter keys
// it consumes. Treat the contract as provisional; NGX's verbose
// log is the authority when something is rejected.
#pragma once

#include "common.h"
#include "dlss.h"
#include "gpu.h"
#include "nr_params.h"

// NGX feature id. Reserved and undocumented in the public headers.
constexpr int kNgxFeatureNeuralRendering = 18;

// Parameter keys, taken verbatim from the strings the DLL and the addon share.
namespace NrParam {
constexpr const char* kEnabled = "DLSSNR.Enabled";
constexpr const char* kUpscaling = "DLSSNR.Upscaling";
constexpr const char* kScale = "DLSSNR.Scale";
constexpr const char* kScalingRatio = "DLSSNR.ScalingRatio";
constexpr const char* kPreset = "DLSSNR.Hint.Render.Preset";
constexpr const char* kReset = "DLSSNR.Reset";

constexpr const char* kInputWidth = "DLSSNR.InputWidth";
constexpr const char* kInputHeight = "DLSSNR.InputHeight";
constexpr const char* kOutputWidth = "DLSSNR.OutputWidth";
constexpr const char* kOutputHeight = "DLSSNR.OutputHeight";
constexpr const char* kWidth = "DLSSNR.Width";
constexpr const char* kHeight = "DLSSNR.Height";

constexpr const char* kColor = "DLSSNR.Color";
constexpr const char* kDepth = "DLSSNR.Depth";
constexpr const char* kMVec = "DLSSNR.MVec";
constexpr const char* kOutput = "DLSSNR.Output";

constexpr const char* kDepthInverted = "DLSSNR.DepthInverted";
constexpr const char* kMVecScaleX = "DLSSNR.MVecScaleX";
constexpr const char* kMVecScaleY = "DLSSNR.MVecScaleY";
constexpr const char* kUseAutoMask = "DLSSNR.UseAutoMask";

// Quality knobs the addon exposes in its UI.
constexpr const char* kIntensity = "DLSSNR.Intensity";
constexpr const char* kLocalStructureStrength = "DLSSNR.LocalStructureStrength";
constexpr const char* kLocalToneStrength = "DLSSNR.LocalToneStrength";
constexpr const char* kSkinStructureStrength = "DLSSNR.SkinStructureStrength";
constexpr const char* kStyle = "DLSSNR.Style";
constexpr const char* kUICorrection = "DLSSNR.UICorrection";

// Subrect bases, mirroring the DLSS ones.
constexpr const char* kColorSubrectBaseX = "DLSSNR.ColorSubrectBaseX";
constexpr const char* kColorSubrectBaseY = "DLSSNR.ColorSubrectBaseY";
constexpr const char* kColorSubrectWidth = "DLSSNR.ColorSubrectWidth";
constexpr const char* kColorSubrectHeight = "DLSSNR.ColorSubrectHeight";
constexpr const char* kDepthSubrectBaseX = "DLSSNR.DepthSubrectBaseX";
constexpr const char* kDepthSubrectBaseY = "DLSSNR.DepthSubrectBaseY";
constexpr const char* kMVecSubrectBaseX = "DLSSNR.MVecSubrectBaseX";
constexpr const char* kMVecSubrectBaseY = "DLSSNR.MVecSubrectBaseY";
constexpr const char* kOutputSubrectBaseX = "DLSSNR.OutputSubrectBaseX";
constexpr const char* kOutputSubrectBaseY = "DLSSNR.OutputSubrectBaseY";
}  // namespace NrParam

struct NrFeatureDesc {
    unsigned inputW = 0;
    unsigned inputH = 0;
    unsigned outputW = 0;
    unsigned outputH = 0;
    unsigned preset = 0;
    bool upscaling = true;
    bool depthInverted = false;
    float intensity = -1.0f;               // < 0 leaves the DLL default alone
    float localStructureStrength = -1.0f;
    float localToneStrength = -1.0f;
};

struct NrEvalInputs {
    GpuTexture* color = nullptr;
    GpuTexture* depth = nullptr;
    GpuTexture* motion = nullptr;
    GpuTexture* output = nullptr;
    bool reset = false;
    float mvecScaleX = 1.0f;
    float mvecScaleY = 1.0f;
};

class NrFeature {
public:
    ~NrFeature();

    // Returns the raw NGX result rather than throwing, because probing which
    // parameters this build accepts is the normal way to use this.
    NVSDK_NGX_Result TryCreate(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                               const NrFeatureDesc& desc);
    void Create(NgxSession& session, ID3D12GraphicsCommandList* cmdList,
                const NrFeatureDesc& desc);

    NVSDK_NGX_Result TryEvaluate(ID3D12GraphicsCommandList* cmdList, const NrEvalInputs& in);
    void Evaluate(ID3D12GraphicsCommandList* cmdList, const NrEvalInputs& in);

    void Release();
    bool Valid() const { return m_handle != nullptr; }

    // Shared by both call routes — through the driver core, and straight into
    // the snippet's own exports.
    static void FillCreateParams(NVSDK_NGX_Parameter* params, const NrFeatureDesc& desc);
    static void FillEvalParams(NVSDK_NGX_Parameter* params, const NrEvalInputs& in);

private:
    NVSDK_NGX_Handle* m_handle = nullptr;
    NVSDK_NGX_Parameter* m_params = nullptr;
    NrFeatureDesc m_desc;
};

// One-shot capability probe: init, try to create feature 18, report what NGX
// said. Prints the loaded DLL and its version so the answer is attributable.
int ProbeNeuralRendering(const std::string& dllDir, int adapter, unsigned inputW,
                         unsigned inputH, unsigned outputW, unsigned outputH,
                         unsigned preset, bool verbose);

// Runs DLSS Neural Rendering (feature 18) over an image or a whole folder via the
// forwarder shim. inPath may be a single image or a directory (every image in it).
// The feature is created once per resolution and reused (create is the expensive
// step; evaluate is cheap - which is why a game runs at 60 fps). Writes
// <name>_nr.png, <name>_orig.png and <name>_nr_diff.png per image into outDir.
// `model` are the model's own params (latched at create); detail/colour/hdr control
// how the model's output is composed back over the original.
int RunNeuralRendering(const std::string& dllDir, int adapter, const std::string& inPath,
                       const std::string& outDir, const NrModelParams& model, float detail,
                       float colour, bool hdr, float scale, unsigned outWReq, unsigned outHReq,
                       unsigned srPreset, bool writeDiff, bool writeOrig, bool verbose);

// Streaming video filter: raw RGBA frames of size inW x inH on stdin, DLSS SR + NR, raw RGBA
// frames on stdout. Driven by ffmpeg on both ends (decode / NVENC encode). The feature is
// built once for the whole stream. Logs go to stderr.
int RunNeuralRenderingVideo(const std::string& dllDir, int adapter, unsigned inW, unsigned inH,
                            const NrModelParams& model, float detail, float colour, bool hdr,
                            float scale, unsigned outWReq, unsigned outHReq, bool motionOn,
                            bool motionVis, int motionEngine, unsigned srPreset, bool verbose);
