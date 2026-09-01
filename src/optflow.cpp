#include "optflow.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// All flow buffers stay in UNORDERED_ACCESS for their whole life and are read as
// RWTexture2D (with a global UAV barrier between passes), so no pass needs a state
// transition or a sampler - bilinear taps are done by hand. The one exception is the
// source colour, read through an SRV (t0). Root signature: 8 root constants (b0), then
// a table of one SRV (t0) and four UAVs (u0..u3).
// ---------------------------------------------------------------------------

namespace {

struct FlowConst {
    uint32_t dstW, dstH;
    float sx, sy;  // scale factors (ToMotion)
    int32_t radius;
    float lambda;  // Tikhonov regularization: flat regions -> zero flow instead of noise
    int32_t pad1, pad2;
};

const char* kHeader = R"(
cbuffer C : register(b0) { uint2 gDst; float2 gScale; int gRadius; float gLambda; int2 gPad; };
)";

// current colour (SRV) -> base-level luma
const char* kLuma = R"(
Texture2D<float4> gColor : register(t0);
RWTexture2D<float>  gOut : register(u0);
float3 tap(int2 p, int2 mx) { return gColor.Load(int3(clamp(p, int2(0,0), mx), 0)).rgb; }
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gColor.GetDimensions(sd.x, sd.y);
    int2 mx = int2(sd) - 1;
    float2 pos = (float2(t.xy) + 0.5) * float2(sd) / float2(gDst) - 0.5;
    int2 i = int2(floor(pos)); float2 f = pos - float2(i);
    float3 c = lerp(lerp(tap(i,mx), tap(i+int2(1,0),mx), f.x),
                    lerp(tap(i+int2(0,1),mx), tap(i+int2(1,1),mx), f.x), f.y);
    gOut[t.xy] = dot(c, float3(0.2126, 0.7152, 0.0722));
}
)";

// 2x box downsample of a luma level
const char* kDown = R"(
RWTexture2D<float> gOut : register(u0);
RWTexture2D<float> gSrc : register(u1);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1;
    int2 s = int2(t.xy) * 2;
    float v = gSrc[min(s, mx)] + gSrc[min(s+int2(1,0), mx)] +
              gSrc[min(s+int2(0,1), mx)] + gSrc[min(s+int2(1,1), mx)];
    gOut[t.xy] = v * 0.25;
}
)";

const char* kClear = R"(
RWTexture2D<float2> gOut : register(u0);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    gOut[t.xy] = float2(0.0, 0.0);
}
)";

// bilinear upsample of a flow level to the next finer level, doubling the magnitude
const char* kUp = R"(
RWTexture2D<float2> gOut : register(u0);
RWTexture2D<float2> gSrc : register(u1);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1;
    float2 pos = (float2(t.xy) + 0.5) * float2(sd) / float2(gDst) - 0.5;
    int2 i = int2(floor(pos)); float2 f = pos - float2(i);
    float2 a = lerp(gSrc[clamp(i,int2(0,0),mx)],        gSrc[clamp(i+int2(1,0),int2(0,0),mx)], f.x);
    float2 b = lerp(gSrc[clamp(i+int2(0,1),int2(0,0),mx)], gSrc[clamp(i+int2(1,1),int2(0,0),mx)], f.x);
    gOut[t.xy] = lerp(a, b, f.y) * 2.0;
}
)";

// spatial gradient (Ix,Iy) of the current-frame luma
const char* kGrad = R"(
RWTexture2D<float2> gOut : register(u0);
RWTexture2D<float>  gSrc : register(u1);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1; int2 p = int2(t.xy);
    float l = gSrc[max(p-int2(1,0), int2(0,0))], r = gSrc[min(p+int2(1,0), mx)];
    float u = gSrc[max(p-int2(0,1), int2(0,0))], d = gSrc[min(p+int2(0,1), mx)];
    gOut[t.xy] = float2((r - l) * 0.5, (d - u) * 0.5);
}
)";

// temporal residual It = prev(p + flow) - cur(p), warping prev by the current flow
const char* kWarp = R"(
RWTexture2D<float>  gOut  : register(u0);
RWTexture2D<float>  gPrev : register(u1);
RWTexture2D<float2> gFlow : register(u2);
RWTexture2D<float>  gCur  : register(u3);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gPrev.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1; int2 p = int2(t.xy);
    float2 s = float2(p) + gFlow[p];
    int2 i = int2(floor(s)); float2 f = s - float2(i);
    float a = lerp(gPrev[clamp(i,int2(0,0),mx)],        gPrev[clamp(i+int2(1,0),int2(0,0),mx)], f.x);
    float b = lerp(gPrev[clamp(i+int2(0,1),int2(0,0),mx)], gPrev[clamp(i+int2(1,1),int2(0,0),mx)], f.x);
    gOut[t.xy] = lerp(a, b, f.y) - gCur[p];
}
)";

// one Lucas-Kanade refinement: windowed least squares for the flow increment
const char* kSolve = R"(
RWTexture2D<float2> gOut  : register(u0);
RWTexture2D<float2> gGrad : register(u1);
RWTexture2D<float>  gIt   : register(u2);
RWTexture2D<float2> gFlow : register(u3);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    int2 mx = int2(gDst) - 1; int2 p = int2(t.xy);
    float sxx=0,sxy=0,syy=0,bx=0,by=0;
    for (int dy=-gRadius; dy<=gRadius; ++dy)
    for (int dx=-gRadius; dx<=gRadius; ++dx) {
        int2 q = clamp(p+int2(dx,dy), int2(0,0), mx);
        float2 g = gGrad[q]; float it = gIt[q];
        sxx+=g.x*g.x; sxy+=g.x*g.y; syy+=g.y*g.y; bx+=g.x*it; by+=g.y*it;
    }
    // Tikhonov regularization on the normal equations: adding lambda to the diagonal makes the
    // solve degrade gracefully to zero in low-texture regions instead of blowing up into noise.
    sxx += gLambda; syy += gLambda;
    float det = sxx*syy - sxy*sxy;
    float2 df = float2(0,0);
    if (det > 1e-8) {
        df.x = -(syy*bx - sxy*by) / det;
        df.y = -(-sxy*bx + sxx*by) / det;
    }
    df = clamp(df, -2.0, 2.0);
    gOut[t.xy] = gFlow[p] + df;
}
)";

// 3x3 median of the flow field. Median (not blur) is the standard flow denoiser (DIS/Farneback):
// it removes outlier vectors - the speckle/flicker in ambiguous regions - without smearing edges.
const char* kSmooth = R"(
RWTexture2D<float2> gOut : register(u0);
RWTexture2D<float2> gSrc : register(u1);
#define SORT(a,b) { float t = min(v[a], v[b]); v[b] = max(v[a], v[b]); v[a] = t; }
float median9(float v[9]) {
    SORT(1,2) SORT(4,5) SORT(7,8) SORT(0,1) SORT(3,4) SORT(6,7) SORT(1,2) SORT(4,5) SORT(7,8)
    SORT(0,3) SORT(5,8) SORT(4,7) SORT(3,6) SORT(1,4) SORT(2,5) SORT(4,7) SORT(4,2) SORT(6,4)
    SORT(4,2)
    return v[4];
}
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    int2 mx = int2(gDst) - 1; int2 p = int2(t.xy);
    float vx[9]; float vy[9]; int k = 0;
    [unroll] for (int dy=-1; dy<=1; ++dy)
    [unroll] for (int dx=-1; dx<=1; ++dx) {
        float2 s = gSrc[clamp(p+int2(dx,dy), int2(0,0), mx)];
        vx[k] = s.x; vy[k] = s.y; ++k;
    }
    gOut[t.xy] = float2(median9(vx), median9(vy));
}
)";

// upsample the finest flow to target resolution and scale to target pixels
const char* kMotion = R"(
RWTexture2D<float2> gOut : register(u0);
RWTexture2D<float2> gSrc : register(u1);
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDst.x || t.y >= gDst.y) return;
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1;
    float2 pos = (float2(t.xy) + 0.5) * float2(sd) / float2(gDst) - 0.5;
    int2 i = int2(floor(pos)); float2 f = pos - float2(i);
    float2 a = lerp(gSrc[clamp(i,int2(0,0),mx)],        gSrc[clamp(i+int2(1,0),int2(0,0),mx)], f.x);
    float2 b = lerp(gSrc[clamp(i+int2(0,1),int2(0,0),mx)], gSrc[clamp(i+int2(1,1),int2(0,0),mx)], f.x);
    float2 fl = lerp(a, b, f.y);
    gOut[t.xy] = float2(fl.x * gScale.x, fl.y * gScale.y);
}
)";

ComPtr<ID3D12PipelineState> CompilePso(ID3D12Device* dev, ID3D12RootSignature* sig,
                                       const std::string& src, const char* name) {
    ComPtr<ID3DBlob> cs, err;
    const std::string full = std::string(kHeader) + src;
    HRESULT hr = D3DCompile(full.c_str(), full.size(), name, nullptr, nullptr, "CSMain", "cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &err);
    if (FAILED(hr))
        throw ToolError(std::string("optflow ") + name + ": " +
                        (err ? static_cast<const char*>(err->GetBufferPointer()) : HrToString(hr)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = sig;
    pd.CS.pShaderBytecode = cs->GetBufferPointer();
    pd.CS.BytecodeLength = cs->GetBufferSize();
    ComPtr<ID3D12PipelineState> pso;
    CHECK_HR(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    return pso;
}

}  // namespace

void OpticalFlow::Init(GpuContext& gpu, int targetW, int targetH) {
    m_device = gpu.Device();
    m_tW = targetW;
    m_tH = targetH;

    // Work at up to ~960px wide (Merserk uses 640 on the CPU; the GPU can afford more), with a
    // four-level pyramid so a handful of pixels of motion at the base reaches large motion at
    // full res.
    m_baseW = std::min(targetW, 960);
    m_baseW -= m_baseW & 1;
    m_baseH = std::max(2, static_cast<int>(std::lround(static_cast<double>(targetH) * m_baseW / targetW)));
    m_baseH -= m_baseH & 1;
    m_levels = 4;

    // Root signature: root constants + table [1 SRV, 4 UAV].
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 4;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(FlowConst) / 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = 2;
    rs.pParameters = params;
    ComPtr<ID3DBlob> sig, err;
    HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (FAILED(hr))
        throw ToolError(std::string("optflow root sig: ") +
                        (err ? static_cast<const char*>(err->GetBufferPointer()) : HrToString(hr)));
    CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                           IID_PPV_ARGS(&m_sig)));

    m_psoLuma = CompilePso(m_device, m_sig.Get(), kLuma, "luma");
    m_psoDown = CompilePso(m_device, m_sig.Get(), kDown, "down");
    m_psoClear = CompilePso(m_device, m_sig.Get(), kClear, "clear");
    m_psoUp = CompilePso(m_device, m_sig.Get(), kUp, "up");
    m_psoGrad = CompilePso(m_device, m_sig.Get(), kGrad, "grad");
    m_psoWarp = CompilePso(m_device, m_sig.Get(), kWarp, "warp");
    m_psoSolve = CompilePso(m_device, m_sig.Get(), kSolve, "solve");
    m_psoSmooth = CompilePso(m_device, m_sig.Get(), kSmooth, "smooth");
    m_psoMotion = CompilePso(m_device, m_sig.Get(), kMotion, "motion");

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 512;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_HR(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)));
    m_descSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Per-level buffers, all created in UNORDERED_ACCESS.
    m_lumaA.resize(m_levels);
    m_lumaB.resize(m_levels);
    m_grad.resize(m_levels);
    m_it.resize(m_levels);
    m_flow.resize(m_levels);
    m_flow2.resize(m_levels);
    for (int l = 0; l < m_levels; ++l) {
        const int w = std::max(1, m_baseW >> l);
        const int h = std::max(1, m_baseH >> l);
        m_lumaA[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16_FLOAT, true, L"ofLumaA");
        m_lumaB[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16_FLOAT, true, L"ofLumaB");
        m_grad[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16G16_FLOAT, true, L"ofGrad");
        m_it[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16_FLOAT, true, L"ofIt");
        m_flow[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16G16_FLOAT, true, L"ofFlow");
        m_flow2[l] = gpu.CreateTexture(w, h, DXGI_FORMAT_R16G16_FLOAT, true, L"ofFlow2");
    }
    gpu.Begin();
    for (int l = 0; l < m_levels; ++l) {
        gpu.Transition(m_lumaA[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(m_lumaB[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(m_grad[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(m_it[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(m_flow[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Transition(m_flow2[l], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    gpu.EndAndWait();
}

void OpticalFlow::Shutdown() {
    m_lumaA.clear();
    m_lumaB.clear();
    m_grad.clear();
    m_it.clear();
    m_flow.clear();
    m_flow2.clear();
    m_psoLuma.Reset();
    m_psoDown.Reset();
    m_psoClear.Reset();
    m_psoUp.Reset();
    m_psoGrad.Reset();
    m_psoWarp.Reset();
    m_psoSolve.Reset();
    m_psoSmooth.Reset();
    m_psoMotion.Reset();
    m_heap.Reset();
    m_sig.Reset();
}

int OpticalFlow::Alloc() { return m_descNext++; }

D3D12_CPU_DESCRIPTOR_HANDLE OpticalFlow::Cpu(int i) const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<SIZE_T>(i) * m_descSize;
    return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE OpticalFlow::Gpu(int i) const {
    D3D12_GPU_DESCRIPTOR_HANDLE h = m_heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<UINT64>(i) * m_descSize;
    return h;
}

void OpticalFlow::Srv(const GpuTexture& t, int slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = t.fmt;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(t.res.Get(), &d, Cpu(slot));
}
void OpticalFlow::Uav(const GpuTexture& t, int slot) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = t.fmt;
    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(t.res.Get(), nullptr, &d, Cpu(slot));
}

void OpticalFlow::Barrier(ID3D12GraphicsCommandList* cl) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = nullptr;  // all UAV accesses
    cl->ResourceBarrier(1, &b);
}

void OpticalFlow::Pass(ID3D12GraphicsCommandList* cl, ID3D12PipelineState* pso,
                       const GpuTexture* srv, const GpuTexture* u0, const GpuTexture* u1,
                       const GpuTexture* u2, const GpuTexture* u3, const GpuTexture& dst,
                       const void* constants) {
    // A fresh contiguous table: SRV t0 then UAV u0..u3. Any null slot is filled with dst so the
    // descriptor is valid even though that shader does not read it.
    const int base = Alloc();
    Alloc();
    Alloc();
    Alloc();
    Alloc();
    Srv(srv ? *srv : dst, base);
    Uav(u0 ? *u0 : dst, base + 1);
    Uav(u1 ? *u1 : dst, base + 2);
    Uav(u2 ? *u2 : dst, base + 3);
    Uav(u3 ? *u3 : dst, base + 4);
    cl->SetPipelineState(pso);
    cl->SetComputeRootSignature(m_sig.Get());
    cl->SetComputeRoot32BitConstants(0, sizeof(FlowConst) / 4, constants, 0);
    cl->SetComputeRootDescriptorTable(1, Gpu(base));
    cl->Dispatch(static_cast<UINT>((dst.w + 7) / 8), static_cast<UINT>((dst.h + 7) / 8), 1);
    Barrier(cl);
}

#define OF_PASS(PSO, SRV, U0, U1, U2, U3, DST, C) \
    Pass(cl, (PSO), (SRV), (U0), (U1), (U2), (U3), (DST), &(C))

void OpticalFlow::Record(ID3D12GraphicsCommandList* cl, GpuTexture& curColor, GpuTexture& motion,
                         bool reset) {
    m_descNext = 0;
    ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
    cl->SetDescriptorHeaps(1, heaps);

    std::vector<GpuTexture>& cur = m_curIsA ? m_lumaA : m_lumaB;
    std::vector<GpuTexture>& prev = m_curIsA ? m_lumaB : m_lumaA;

    auto sized = [&](int l) {
        FlowConst c{};
        c.dstW = static_cast<uint32_t>(cur[l].w);
        c.dstH = static_cast<uint32_t>(cur[l].h);
        c.radius = 2;
        c.lambda = 0.02f;  // regularization strength (luma is 0..1)
        return c;
    };

    // Build the current-frame luma pyramid.
    {
        FlowConst c = sized(0);
        const GpuTexture* srv = &curColor;
        OF_PASS(m_psoLuma.Get(), srv, &cur[0], nullptr, nullptr, nullptr, cur[0], c);
    }
    for (int l = 1; l < m_levels; ++l) {
        FlowConst c = sized(l);
        OF_PASS(m_psoDown.Get(), (const GpuTexture*) nullptr, &cur[l], &cur[l - 1], nullptr, nullptr,
                cur[l], c);
    }

    if (reset || !m_hasPrev) {
        FlowConst c{};
        c.dstW = static_cast<uint32_t>(motion.w);
        c.dstH = static_cast<uint32_t>(motion.h);
        OF_PASS(m_psoClear.Get(), (const GpuTexture*) nullptr, &motion, nullptr, nullptr, nullptr,
                motion, c);
        m_curIsA = !m_curIsA;
        m_hasPrev = true;
        return;
    }

    // Coarsest flow starts at zero.
    {
        FlowConst c = sized(m_levels - 1);
        OF_PASS(m_psoClear.Get(), (const GpuTexture*) nullptr, &m_flow[m_levels - 1], nullptr,
                nullptr, nullptr, m_flow[m_levels - 1], c);
    }

    const int iters = 3;
    for (int l = m_levels - 1; l >= 0; --l) {
        FlowConst c = sized(l);
        // Gradient of the current luma at this level.
        OF_PASS(m_psoGrad.Get(), (const GpuTexture*) nullptr, &m_grad[l], &cur[l], nullptr, nullptr,
                m_grad[l], c);
        if (l < m_levels - 1) {
            OF_PASS(m_psoUp.Get(), (const GpuTexture*) nullptr, &m_flow[l], &m_flow[l + 1], nullptr,
                    nullptr, m_flow[l], c);
        }
        for (int it = 0; it < iters; ++it) {
            OF_PASS(m_psoWarp.Get(), (const GpuTexture*) nullptr, &m_it[l], &prev[l], &m_flow[l],
                    &cur[l], m_it[l], c);
            OF_PASS(m_psoSolve.Get(), (const GpuTexture*) nullptr, &m_flow2[l], &m_grad[l], &m_it[l],
                    &m_flow[l], m_flow2[l], c);
            std::swap(m_flow[l], m_flow2[l]);
        }
        // Median-filter the level's flow to drop outlier vectors before it seeds the next level.
        OF_PASS(m_psoSmooth.Get(), (const GpuTexture*) nullptr, &m_flow2[l], &m_flow[l], nullptr,
                nullptr, m_flow2[l], c);
        std::swap(m_flow[l], m_flow2[l]);
    }

    // Upsample the finest flow (base pixels) to target-resolution motion vectors.
    FlowConst c{};
    c.dstW = static_cast<uint32_t>(motion.w);
    c.dstH = static_cast<uint32_t>(motion.h);
    c.sx = static_cast<float>(m_tW) / m_baseW;
    c.sy = static_cast<float>(m_tH) / m_baseH;
    OF_PASS(m_psoMotion.Get(), (const GpuTexture*) nullptr, &motion, &m_flow[0], nullptr, nullptr,
            motion, c);

    m_curIsA = !m_curIsA;
}
