#include "gpu.h"

#include <DirectXPackedVector.h>

#include <algorithm>
#include <cstring>

using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;

// ---------------------------------------------------------------------------
// The jittered downsample. One thread per render-resolution pixel: find where
// that pixel's (jittered) centre lands in the high-res source, filter there,
// and also fill the constant depth and zero motion-vector targets DLSS needs.
// ---------------------------------------------------------------------------
static const char* kDownsampleHlsl = R"HLSL(
cbuffer Constants : register(b0)
{
    float2 gJitterHR;   // frame jitter, already converted to source pixels and signed
    float2 gScale;      // sourceSize / renderSize
    float2 gHrSize;
    float2 gInvHrSize;
    uint2  gLrSize;
    uint   gFilterMode; // 0 point, 1 bilinear, 2 tent, 3 lanczos2
    uint   gSrgbOut;
    float  gDepthValue;
    uint3  gPad;
};

Texture2D<float4>   gSrc    : register(t0);
RWTexture2D<float4> gColor  : register(u0);
RWTexture2D<float>  gDepth  : register(u1);
RWTexture2D<float2> gMotion : register(u2);

SamplerState gPointClamp  : register(s0);
SamplerState gLinearClamp : register(s1);

float3 LinearToSrgb3(float3 c)
{
    c = max(c, 0.0f);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(max(c, 1e-8f), 1.0f / 2.4f) - 0.055f;
    return lerp(hi, lo, step(c, 0.0031308f));
}

float Lanczos2(float x)
{
    x = abs(x);
    if (x < 1e-5f)  return 1.0f;
    if (x >= 2.0f)  return 0.0f;
    const float PI = 3.14159265358979f;
    float px = PI * x;
    return (sin(px) / px) * (sin(px * 0.5f) / (px * 0.5f));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gLrSize.x || tid.y >= gLrSize.y) return;

    float2 hrPos = (float2(tid.xy) + 0.5f) * gScale + gJitterHR;

    float4 c;
    if (gFilterMode == 0)
    {
        c = gSrc.SampleLevel(gPointClamp, hrPos * gInvHrSize, 0);
    }
    else if (gFilterMode == 1)
    {
        c = gSrc.SampleLevel(gLinearClamp, hrPos * gInvHrSize, 0);
    }
    else
    {
        // Both wide kernels use a 4x4 support: a half-width-2 triangle, or
        // Lanczos2. A half-width-1 triangle would just reproduce bilinear.
        int radius = 2;
        float2 base = floor(hrPos - 0.5f);
        float4 acc = 0.0f;
        float wsum = 0.0f;
        for (int dy = -radius + 1; dy <= radius; ++dy)
        {
            for (int dx = -radius + 1; dx <= radius; ++dx)
            {
                float2 tap = base + float2(dx, dy);
                float2 d = (tap + 0.5f) - hrPos;
                float w;
                if (gFilterMode == 2)
                    w = max(0.0f, 1.0f - abs(d.x) * 0.5f) * max(0.0f, 1.0f - abs(d.y) * 0.5f);
                else
                    w = Lanczos2(d.x) * Lanczos2(d.y);
                int2 texel = int2(clamp(tap, float2(0.0f, 0.0f), gHrSize - 1.0f));
                acc += w * gSrc.Load(int3(texel, 0));
                wsum += w;
            }
        }
        c = (abs(wsum) > 1e-6f) ? acc / wsum
                                : gSrc.SampleLevel(gLinearClamp, hrPos * gInvHrSize, 0);
    }

    c.rgb = max(c.rgb, 0.0f);
    if (gSrgbOut != 0) c.rgb = LinearToSrgb3(c.rgb);

    gColor[tid.xy]  = c;
    gDepth[tid.xy]  = gDepthValue;
    gMotion[tid.xy] = float2(0.0f, 0.0f);
}
)HLSL";

struct DownsampleConstants {
    float jitterHR[2];
    float scale[2];
    float hrSize[2];
    float invHrSize[2];
    uint32_t lrSize[2];
    uint32_t filterMode;
    uint32_t srgbOut;
    float depthValue;
    uint32_t pad[3];
};
static_assert(sizeof(DownsampleConstants) == 64, "constants must stay 16 x 32-bit");

const char* DownFilterName(DownFilter f) {
    switch (f) {
        case DownFilter::Point: return "point";
        case DownFilter::Bilinear: return "bilinear";
        case DownFilter::Tent: return "tent";
        case DownFilter::Lanczos: return "lanczos";
    }
    return "?";
}

bool ParseDownFilter(const std::string& s, DownFilter* out) {
    if (s == "point") { *out = DownFilter::Point; return true; }
    if (s == "bilinear") { *out = DownFilter::Bilinear; return true; }
    if (s == "tent") { *out = DownFilter::Tent; return true; }
    if (s == "lanczos") { *out = DownFilter::Lanczos; return true; }
    return false;
}

// ---------------------------------------------------------------------------
// Video post passes. Two compute shaders that keep the whole colour path on the
// GPU: EncodeSrgb turns the linear DLSS output into the display-referred sRGB the
// NR model expects, and Composite blends the model output back over the upscaled
// original (luma-ratio blend) and writes the final frame straight to an 8-bit target -
// so the only CPU touch is the single uint8 readback the encoder pipe needs.
// ---------------------------------------------------------------------------
static const char* kEncodeHlsl = R"HLSL(
cbuffer C : register(b0) { uint2 gSize; uint2 gPad; };
Texture2D<float4>   gSrc : register(t0);   // linear
Texture2D<float4>   gAux : register(t1);   // unused here
RWTexture2D<float4> gDst : register(u0);   // sRGB
float3 LinToSrgb(float3 c) {
    c = max(c, 0.0f);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(max(c, 1e-8f), 1.0f / 2.4f) - 0.055f;
    return lerp(hi, lo, step(c, 0.0031308f));
}
[numthreads(8, 8, 1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gSize.x || t.y >= gSize.y) return;
    float4 c = gSrc[t.xy];
    gDst[t.xy] = float4(LinToSrgb(c.rgb), c.a);
}
)HLSL";

static const char* kCompositeHlsl = R"HLSL(
cbuffer C : register(b0) { uint2 gSize; float gDetail; float gColour; };
Texture2D<float4>   gOrig : register(t0);  // linear upscaled original
Texture2D<float4>   gNr   : register(t1);  // sRGB model output
RWTexture2D<float4> gDst  : register(u0);  // R8G8B8A8_UNORM (sRGB-encoded)
float3 SrgbToLin(float3 c) {
    c = max(c, 0.0f);
    float3 lo = c / 12.92f;
    float3 hi = pow((c + 0.055f) / 1.055f, 2.4f);
    return lerp(lo, hi, step(0.04045f, c));
}
float3 LinToSrgb(float3 c) {
    c = max(c, 0.0f);
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(max(c, 1e-8f), 1.0f / 2.4f) - 0.055f;
    return lerp(hi, lo, step(c, 0.0031308f));
}
float Luma(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
[numthreads(8, 8, 1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gSize.x || t.y >= gSize.y) return;
    float4 o = gOrig[t.xy];                 // linear
    float3 nr = SrgbToLin(gNr[t.xy].rgb);   // model output -> linear
    float lo = Luma(o.rgb), ln = Luma(nr);
    float scale = ln / max(lo, 1e-4f);
    float3 loC = o.rgb * scale;             // original hue at model luminance
    float3 cC = loC + (nr - loC) * gColour; // blend toward model colour
    float3 res = o.rgb + (cC - o.rgb) * gDetail;  // overall strength, in linear
    gDst[t.xy] = float4(saturate(LinToSrgb(res)), saturate(o.a));
}
)HLSL";

// Debug: colour-code a motion-vector field (hue = direction, brightness = magnitude), the
// standard optical-flow visualisation, so a --nr-motion-vis pass can show what the flow found.
static const char* kFlowVisHlsl = R"HLSL(
cbuffer C : register(b0) { uint2 gSize; float gMaxMag; float gPad; };
Texture2D<float2>   gMotion : register(t0);
Texture2D<float2>   gAux    : register(t1);
RWTexture2D<float4> gDst    : register(u0);
float3 Hsv2Rgb(float3 c) {
    float3 p = abs(frac(c.xxx + float3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * lerp(float3(1,1,1), saturate(p - 1.0), c.y);
}
[numthreads(8, 8, 1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gSize.x || t.y >= gSize.y) return;
    float2 mv = gMotion[t.xy];
    float hue = frac(atan2(mv.y, mv.x) / 6.28318530718 + 1.0);
    float val = saturate(length(mv) / max(gMaxMag, 1e-3));
    gDst[t.xy] = float4(Hsv2Rgb(float3(hue, 1.0, val)), 1.0);
}
)HLSL";

struct PostConstants {
    uint32_t size[2];
    float detail;
    float colour;
};
static_assert(sizeof(PostConstants) == 16, "post constants are 4 x 32-bit");

// ---------------------------------------------------------------------------

GpuContext::~GpuContext() { Shutdown(); }

void GpuContext::Initialize(bool enableDebugLayer, int adapterIndex) {
    UINT factoryFlags = 0;
    if (enableDebugLayer) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            LogDebug("D3D12 debug layer enabled");
        } else {
            LogWarn("D3D12 debug layer requested but unavailable (install the Graphics Tools "
                    "optional feature)");
        }
    }

    CHECK_HR(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory)));

    ComPtr<IDXGIAdapter1> chosen;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                  IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                                     nullptr))) {
            continue;
        }
        if (adapterIndex >= 0 && static_cast<int>(i) != adapterIndex) continue;
        chosen = adapter;
        m_adapterName = Widen2Narrow(desc.Description);
        m_vramMB = desc.DedicatedVideoMemory / (1024 * 1024);
        break;
    }
    if (!chosen) throw ToolError("no Direct3D 12 capable adapter found");

    CHECK_HR(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    CHECK_HR(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)));

    CHECK_HR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&m_alloc)));
    CHECK_HR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&m_list)));
    CHECK_HR(m_list->Close());

    CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(m_fenceEvent != nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kHeapSize;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_HR(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)));
    m_descSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_QUERY_HEAP_DESC qh{};
    qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh.Count = 2;
    CHECK_HR(m_device->CreateQueryHeap(&qh, IID_PPV_ARGS(&m_queryHeap)));

    D3D12_HEAP_PROPERTIES rbHeap{};
    rbHeap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rbDesc{};
    rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rbDesc.Width = 2 * sizeof(uint64_t);
    rbDesc.Height = 1;
    rbDesc.DepthOrArraySize = 1;
    rbDesc.MipLevels = 1;
    rbDesc.Format = DXGI_FORMAT_UNKNOWN;
    rbDesc.SampleDesc.Count = 1;
    rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK_HR(m_device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_queryReadback)));
    CHECK_HR(m_queue->GetTimestampFrequency(&m_timestampFreq));

    CreateDownsamplePipeline();
    CreatePostPipeline();

    LogDebug("D3D12 device ready on %s (%zu MB VRAM)", m_adapterName.c_str(), m_vramMB);
}

void GpuContext::Shutdown() {
    if (m_device && m_fence) WaitForGpu();
    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }
    m_pso.Reset();
    m_rootSig.Reset();
    m_psoEncode.Reset();
    m_psoComposite.Reset();
    m_psoFlowVis.Reset();
    m_postSig.Reset();
    m_queryReadback.Reset();
    m_queryHeap.Reset();
    m_heap.Reset();
    m_fence.Reset();
    m_list.Reset();
    m_alloc.Reset();
    m_queue.Reset();
    m_device.Reset();
    m_factory.Reset();
}

void GpuContext::CreateDownsamplePipeline() {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = sizeof(DownsampleConstants) / 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[2]{};
    for (int i = 0; i < 2; ++i) {
        samplers[i].Filter = (i == 0) ? D3D12_FILTER_MIN_MAG_MIP_POINT
                                      : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplers[i].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = 2;
    rs.pParameters = params;
    rs.NumStaticSamplers = 2;
    rs.pStaticSamplers = samplers;
    rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig, err;
    HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (FAILED(hr)) {
        throw ToolError(std::string("root signature: ") +
                        (err ? static_cast<const char*>(err->GetBufferPointer()) : HrToString(hr)));
    }
    CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                           IID_PPV_ARGS(&m_rootSig)));

    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    ComPtr<ID3DBlob> cs, csErr;
    hr = D3DCompile(kDownsampleHlsl, strlen(kDownsampleHlsl), "downsample.hlsl", nullptr, nullptr,
                    "CSMain", "cs_5_0", flags, 0, &cs, &csErr);
    if (FAILED(hr)) {
        throw ToolError(std::string("downsample shader: ") +
                        (csErr ? static_cast<const char*>(csErr->GetBufferPointer())
                               : HrToString(hr)));
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = m_rootSig.Get();
    pd.CS.pShaderBytecode = cs->GetBufferPointer();
    pd.CS.BytecodeLength = cs->GetBufferSize();
    CHECK_HR(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&m_pso)));
}

void GpuContext::CreatePostPipeline() {
    // One root signature shared by both post shaders: root constants (b0) plus a
    // table of two SRVs (t0..t1) and one UAV (u0). No samplers - the shaders Load().
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = sizeof(PostConstants) / 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = 2;
    rs.pParameters = params;
    rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig, err;
    HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (FAILED(hr)) {
        throw ToolError(std::string("post root signature: ") +
                        (err ? static_cast<const char*>(err->GetBufferPointer()) : HrToString(hr)));
    }
    CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                           IID_PPV_ARGS(&m_postSig)));

    auto compile = [&](const char* src, const char* name, ComPtr<ID3D12PipelineState>* outPso) {
        ComPtr<ID3DBlob> cs, csErr;
        HRESULT h = D3DCompile(src, strlen(src), name, nullptr, nullptr, "CSMain", "cs_5_0",
                               D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &csErr);
        if (FAILED(h)) {
            throw ToolError(std::string(name) + ": " +
                            (csErr ? static_cast<const char*>(csErr->GetBufferPointer())
                                   : HrToString(h)));
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = m_postSig.Get();
        pd.CS.pShaderBytecode = cs->GetBufferPointer();
        pd.CS.BytecodeLength = cs->GetBufferSize();
        CHECK_HR(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(outPso->GetAddressOf())));
    };
    compile(kEncodeHlsl, "encode_srgb.hlsl", &m_psoEncode);
    compile(kCompositeHlsl, "composite.hlsl", &m_psoComposite);
    compile(kFlowVisHlsl, "flowvis.hlsl", &m_psoFlowVis);
}

// Records one post pass into the open list: SRV0 (+ optional SRV1) -> UAV0. NGX
// rebinds descriptor heaps during its evaluates, so re-bind ours every time.
void GpuContext::RecordPostPass(ID3D12PipelineState* pso, const GpuTexture& srv0,
                                const GpuTexture& srv1, const GpuTexture& uav,
                                const PostConstants& c) {
    CHECK(m_listOpen);
    const int base = AllocDescriptor();
    const int s1 = AllocDescriptor();
    const int u0 = AllocDescriptor();
    CHECK(s1 == base + 1 && u0 == base + 2);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = srv0.fmt;
    m_device->CreateShaderResourceView(srv0.res.Get(), &srv, CpuHandle(base));
    srv.Format = srv1.fmt;
    m_device->CreateShaderResourceView(srv1.res.Get(), &srv, CpuHandle(base + 1));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Format = uav.fmt;
    m_device->CreateUnorderedAccessView(uav.res.Get(), nullptr, &uavDesc, CpuHandle(base + 2));

    ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
    m_list->SetDescriptorHeaps(1, heaps);
    m_list->SetPipelineState(pso);
    m_list->SetComputeRootSignature(m_postSig.Get());
    m_list->SetComputeRoot32BitConstants(0, sizeof(c) / 4, &c, 0);
    m_list->SetComputeRootDescriptorTable(1, GpuHandle(base));
    m_list->Dispatch(static_cast<UINT>((uav.w + 7) / 8), static_cast<UINT>((uav.h + 7) / 8), 1);
}

void GpuContext::RecordEncodeSrgb(GpuTexture& srcLinear, GpuTexture& dstSrgb) {
    PostConstants c{};
    c.size[0] = static_cast<uint32_t>(dstSrgb.w);
    c.size[1] = static_cast<uint32_t>(dstSrgb.h);
    RecordPostPass(m_psoEncode.Get(), srcLinear, srcLinear, dstSrgb, c);
}

void GpuContext::RecordComposite(GpuTexture& origLinear, GpuTexture& nrSrgb, GpuTexture& dstU8,
                                 float detail, float colour) {
    PostConstants c{};
    c.size[0] = static_cast<uint32_t>(dstU8.w);
    c.size[1] = static_cast<uint32_t>(dstU8.h);
    c.detail = detail;
    c.colour = colour;
    RecordPostPass(m_psoComposite.Get(), origLinear, nrSrgb, dstU8, c);
}

void GpuContext::RecordFlowVis(GpuTexture& motion, GpuTexture& dstU8, float maxMag) {
    PostConstants c{};
    c.size[0] = static_cast<uint32_t>(dstU8.w);
    c.size[1] = static_cast<uint32_t>(dstU8.h);
    c.detail = maxMag;
    RecordPostPass(m_psoFlowVis.Get(), motion, motion, dstU8, c);
}

int GpuContext::AllocDescriptor() {
    if (m_descNext >= kHeapSize) throw ToolError("descriptor heap exhausted");
    return m_descNext++;
}

D3D12_CPU_DESCRIPTOR_HANDLE GpuContext::CpuHandle(int index) const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<SIZE_T>(index) * m_descSize;
    return h;
}

D3D12_GPU_DESCRIPTOR_HANDLE GpuContext::GpuHandle(int index) const {
    D3D12_GPU_DESCRIPTOR_HANDLE h = m_heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<UINT64>(index) * m_descSize;
    return h;
}

GpuTexture GpuContext::CreateTexture(int w, int h, DXGI_FORMAT fmt, bool allowUav,
                                     const wchar_t* name) {
    GpuTexture t;
    t.fmt = fmt;
    t.w = w;
    t.h = h;
    t.state = D3D12_RESOURCE_STATE_COMMON;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = static_cast<UINT64>(w);
    d.Height = static_cast<UINT>(h);
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = allowUav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

    CHECK_HR(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, t.state, nullptr,
                                               IID_PPV_ARGS(&t.res)));
    if (name) t.res->SetName(name);
    return t;
}

ID3D12GraphicsCommandList* GpuContext::Begin() {
    CHECK(!m_listOpen);
    CHECK_HR(m_alloc->Reset());
    CHECK_HR(m_list->Reset(m_alloc.Get(), nullptr));
    m_listOpen = true;
    // Every Begin/EndAndWait pair is a full GPU sync, so descriptors written for
    // the previous submission are dead and their slots can be handed out again.
    m_descNext = 0;
    ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
    m_list->SetDescriptorHeaps(1, heaps);
    return m_list.Get();
}

void GpuContext::EndAndWait() {
    CHECK(m_listOpen);
    if (m_timestampPending) {
        m_list->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                 m_queryReadback.Get(), 0);
    }
    CHECK_HR(m_list->Close());
    ID3D12CommandList* lists[] = {m_list.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    m_listOpen = false;
    WaitForGpu();

    if (m_timestampPending) {
        m_timestampPending = false;
        void* mapped = nullptr;
        D3D12_RANGE range{0, 2 * sizeof(uint64_t)};
        CHECK_HR(m_queryReadback->Map(0, &range, &mapped));
        const uint64_t* ts = static_cast<const uint64_t*>(mapped);
        const uint64_t delta = (ts[1] > ts[0]) ? (ts[1] - ts[0]) : 0;
        m_lastGpuMs = m_timestampFreq ? (1000.0 * static_cast<double>(delta) /
                                         static_cast<double>(m_timestampFreq))
                                      : 0.0;
        D3D12_RANGE empty{0, 0};
        m_queryReadback->Unmap(0, &empty);
    }
}

void GpuContext::WaitForGpu() {
    const uint64_t target = ++m_fenceValue;
    CHECK_HR(m_queue->Signal(m_fence.Get(), target));
    if (m_fence->GetCompletedValue() < target) {
        CHECK_HR(m_fence->SetEventOnCompletion(target, m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void GpuContext::Transition(GpuTexture& t, D3D12_RESOURCE_STATES to) {
    CHECK(m_listOpen);
    if (t.state == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = t.res.Get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = t.state;
    b.Transition.StateAfter = to;
    m_list->ResourceBarrier(1, &b);
    t.state = to;
}

void GpuContext::UavBarrier(GpuTexture& t) {
    CHECK(m_listOpen);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = t.res.Get();
    m_list->ResourceBarrier(1, &b);
}

void GpuContext::RecordTimestampBegin() {
    CHECK(m_listOpen);
    m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    m_timestampPending = true;
}

void GpuContext::RecordTimestampEnd() {
    CHECK(m_listOpen);
    m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
}

void GpuContext::UploadRgbaFloat(GpuTexture& tex, const std::vector<float>& rgba) {
    CHECK(tex.fmt == DXGI_FORMAT_R16G16B16A16_FLOAT);
    CHECK(rgba.size() == static_cast<size_t>(tex.w) * tex.h * 4);

    D3D12_RESOURCE_DESC desc = tex.res->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT numRows = 0;
    UINT64 rowBytes = 0, totalBytes = 0;
    m_device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &numRows, &rowBytes, &totalBytes);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = totalBytes;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> staging;
    CHECK_HR(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&staging)));

    uint8_t* mapped = nullptr;
    D3D12_RANGE noRead{0, 0};
    CHECK_HR(staging->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
    for (int y = 0; y < tex.h; ++y) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(mapped + fp.Offset +
                                                    static_cast<size_t>(y) * fp.Footprint.RowPitch);
        const float* src = rgba.data() + static_cast<size_t>(y) * tex.w * 4;
        for (int i = 0; i < tex.w * 4; ++i) dst[i] = XMConvertFloatToHalf(src[i]);
    }
    staging->Unmap(0, nullptr);

    ID3D12GraphicsCommandList* cl = Begin();
    Transition(tex, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = tex.res.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = staging.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = fp;
    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    Transition(tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    EndAndWait();
}

void GpuContext::UploadR32Float(GpuTexture& tex, const std::vector<float>& data) {
    CHECK(tex.fmt == DXGI_FORMAT_R32_FLOAT);
    CHECK(data.size() == static_cast<size_t>(tex.w) * tex.h);

    D3D12_RESOURCE_DESC desc = tex.res->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT numRows = 0;
    UINT64 rowBytes = 0, totalBytes = 0;
    m_device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &numRows, &rowBytes, &totalBytes);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = totalBytes;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> staging;
    CHECK_HR(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&staging)));

    uint8_t* mapped = nullptr;
    D3D12_RANGE noRead{0, 0};
    CHECK_HR(staging->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
    for (int y = 0; y < tex.h; ++y) {
        memcpy(mapped + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch,
               data.data() + static_cast<size_t>(y) * tex.w, static_cast<size_t>(tex.w) * 4);
    }
    staging->Unmap(0, nullptr);

    ID3D12GraphicsCommandList* cl = Begin();
    Transition(tex, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = tex.res.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = staging.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = fp;
    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    Transition(tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    EndAndWait();
}

std::vector<uint8_t> GpuContext::ReadbackBytes(GpuTexture& tex,
                                               D3D12_PLACED_SUBRESOURCE_FOOTPRINT* outFp) {
    D3D12_RESOURCE_DESC desc = tex.res->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT numRows = 0;
    UINT64 rowBytes = 0, totalBytes = 0;
    m_device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &numRows, &rowBytes, &totalBytes);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = totalBytes;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> staging;
    CHECK_HR(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&staging)));

    const D3D12_RESOURCE_STATES prev = tex.state;
    ID3D12GraphicsCommandList* cl = Begin();
    Transition(tex, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = staging.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = tex.res.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;
    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    Transition(tex, prev);
    EndAndWait();

    std::vector<uint8_t> bytes(static_cast<size_t>(totalBytes));
    uint8_t* mapped = nullptr;
    D3D12_RANGE readAll{0, static_cast<SIZE_T>(totalBytes)};
    CHECK_HR(staging->Map(0, &readAll, reinterpret_cast<void**>(&mapped)));
    memcpy(bytes.data(), mapped, static_cast<size_t>(totalBytes));
    D3D12_RANGE noWrite{0, 0};
    staging->Unmap(0, &noWrite);

    if (outFp) *outFp = fp;
    return bytes;
}

std::vector<float> GpuContext::ReadbackRgbaFloat(GpuTexture& tex) {
    CHECK(tex.fmt == DXGI_FORMAT_R16G16B16A16_FLOAT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    const std::vector<uint8_t> bytes = ReadbackBytes(tex, &fp);

    std::vector<float> out(static_cast<size_t>(tex.w) * tex.h * 4);
    for (int y = 0; y < tex.h; ++y) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            bytes.data() + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch);
        float* dst = out.data() + static_cast<size_t>(y) * tex.w * 4;
        for (int i = 0; i < tex.w * 4; ++i) dst[i] = XMConvertHalfToFloat(src[i]);
    }
    return out;
}

std::vector<float> GpuContext::ReadbackR32Float(GpuTexture& tex) {
    CHECK(tex.fmt == DXGI_FORMAT_R32_FLOAT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    const std::vector<uint8_t> bytes = ReadbackBytes(tex, &fp);

    std::vector<float> out(static_cast<size_t>(tex.w) * tex.h);
    for (int y = 0; y < tex.h; ++y) {
        const float* src = reinterpret_cast<const float*>(
            bytes.data() + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch);
        memcpy(out.data() + static_cast<size_t>(y) * tex.w, src, static_cast<size_t>(tex.w) * 4);
    }
    return out;
}

std::vector<float> GpuContext::ReadbackRg16Float(GpuTexture& tex) {
    CHECK(tex.fmt == DXGI_FORMAT_R16G16_FLOAT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    const std::vector<uint8_t> bytes = ReadbackBytes(tex, &fp);

    std::vector<float> out(static_cast<size_t>(tex.w) * tex.h * 2);
    for (int y = 0; y < tex.h; ++y) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            bytes.data() + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch);
        float* dst = out.data() + static_cast<size_t>(y) * tex.w * 2;
        for (int i = 0; i < tex.w * 2; ++i) dst[i] = XMConvertHalfToFloat(src[i]);
    }
    return out;
}

std::vector<uint8_t> GpuContext::ReadbackRgba8(GpuTexture& tex) {
    CHECK(tex.fmt == DXGI_FORMAT_R8G8B8A8_UNORM);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    const std::vector<uint8_t> bytes = ReadbackBytes(tex, &fp);

    std::vector<uint8_t> out(static_cast<size_t>(tex.w) * tex.h * 4);
    for (int y = 0; y < tex.h; ++y) {
        memcpy(out.data() + static_cast<size_t>(y) * tex.w * 4,
               bytes.data() + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch,
               static_cast<size_t>(tex.w) * 4);
    }
    return out;
}

void GpuContext::RecordDownsample(const DownsampleArgs& a) {
    CHECK(m_listOpen);
    CHECK(a.src && a.dstColor && a.dstDepth && a.dstMotion);

    // The root descriptor table expects SRV then three UAVs, contiguous.
    const int base = AllocDescriptor();
    const int uav0 = AllocDescriptor();
    const int uav1 = AllocDescriptor();
    const int uav2 = AllocDescriptor();
    CHECK(uav0 == base + 1 && uav1 == base + 2 && uav2 == base + 3);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = a.src->fmt;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(a.src->res.Get(), &srv, CpuHandle(base));

    const GpuTexture* uavs[3] = {a.dstColor, a.dstDepth, a.dstMotion};
    for (int i = 0; i < 3; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = uavs[i]->fmt;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(uavs[i]->res.Get(), nullptr, &uav,
                                            CpuHandle(base + 1 + i));
    }

    DownsampleConstants c{};
    const float scaleX = static_cast<float>(a.src->w) / static_cast<float>(a.dstColor->w);
    const float scaleY = static_cast<float>(a.src->h) / static_cast<float>(a.dstColor->h);
    c.jitterHR[0] = a.jitterX * scaleX;
    c.jitterHR[1] = a.jitterY * scaleY;
    c.scale[0] = scaleX;
    c.scale[1] = scaleY;
    c.hrSize[0] = static_cast<float>(a.src->w);
    c.hrSize[1] = static_cast<float>(a.src->h);
    c.invHrSize[0] = 1.0f / static_cast<float>(a.src->w);
    c.invHrSize[1] = 1.0f / static_cast<float>(a.src->h);
    c.lrSize[0] = static_cast<uint32_t>(a.dstColor->w);
    c.lrSize[1] = static_cast<uint32_t>(a.dstColor->h);
    c.filterMode = static_cast<uint32_t>(a.filter);
    c.srgbOut = a.encodeSrgb ? 1u : 0u;
    c.depthValue = a.depthValue;

    m_list->SetPipelineState(m_pso.Get());
    m_list->SetComputeRootSignature(m_rootSig.Get());
    m_list->SetComputeRoot32BitConstants(0, sizeof(c) / 4, &c, 0);
    m_list->SetComputeRootDescriptorTable(1, GpuHandle(base));

    const UINT gx = static_cast<UINT>((a.dstColor->w + 7) / 8);
    const UINT gy = static_cast<UINT>((a.dstColor->h + 7) / 8);
    m_list->Dispatch(gx, gy, 1);
}
