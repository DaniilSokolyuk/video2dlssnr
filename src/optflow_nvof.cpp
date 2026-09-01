#include "optflow_nvof.h"

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Two small compute passes bracket the hardware call: colour(cur) is downscaled and packed to
// the ABGR8 surface NVOFA consumes, and the S10.5 flow grid NVOFA produces is upsampled and
// scaled into the RG16F target-resolution motion texture NR reads. NVOFA itself runs on the
// dedicated optical-flow engine, synchronised with our queue through two D3D12 fences.
// ---------------------------------------------------------------------------

namespace {

struct Consts {
    uint32_t dstW, dstH;
    float sx, sy;
    uint32_t reset;
    float costLo, costHi;  // confidence gate: keep vectors with cost<=lo, drop cost>=hi
    uint32_t pad;
};

// linear target-res colour -> sRGB, downscaled, packed BGRA (B8G8R8A8_UNORM stores B,G,R,A)
const char* kColorHlsl = R"(
cbuffer C : register(b0) { uint2 gDim; float2 gS; uint gReset; uint3 gPad; };
Texture2D<float4>   gSrc : register(t0);
RWTexture2D<float4> gDst : register(u0);
float3 LinToSrgb(float3 c) {
    c = max(c, 0.0f);
    return lerp(1.055f * pow(max(c, 1e-8f), 1.0f/2.4f) - 0.055f, c * 12.92f, step(c, 0.0031308f));
}
float3 tap(int2 p, int2 mx) { return gSrc.Load(int3(clamp(p, int2(0,0), mx), 0)).rgb; }
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDim.x || t.y >= gDim.y) return;
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1;
    float2 pos = (float2(t.xy) + 0.5) * float2(sd) / float2(gDim) - 0.5;
    int2 i = int2(floor(pos)); float2 f = pos - float2(i);
    float3 c = lerp(lerp(tap(i,mx), tap(i+int2(1,0),mx), f.x),
                    lerp(tap(i+int2(0,1),mx), tap(i+int2(1,1),mx), f.x), f.y);
    c = LinToSrgb(c);
    gDst[t.xy] = float4(c.b, c.g, c.r, 1.0);
}
)";

// S10.5 flow grid (input res) -> RG16F motion (target res): value/32 = input px, then to target
const char* kMotionHlsl = R"(
cbuffer C : register(b0) { uint2 gDim; float2 gS; uint gReset; float gCostLo; float gCostHi; uint gPad; };
Texture2D<int2>     gSrc  : register(t0);
Texture2D<uint>     gCost : register(t1);
RWTexture2D<float2> gDst  : register(u0);
float2 tap(int2 p, int2 mx) { return float2(gSrc.Load(int3(clamp(p, int2(0,0), mx), 0))); }
[numthreads(8,8,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    if (t.x >= gDim.x || t.y >= gDim.y) return;
    if (gReset != 0) { gDst[t.xy] = float2(0.0, 0.0); return; }
    uint2 sd; gSrc.GetDimensions(sd.x, sd.y); int2 mx = int2(sd) - 1;
    float2 pos = (float2(t.xy) + 0.5) * float2(sd) / float2(gDim) - 0.5;
    int2 i = int2(floor(pos)); float2 f = pos - float2(i);
    float2 a = lerp(tap(i,mx), tap(i+int2(1,0),mx), f.x);
    float2 b = lerp(tap(i+int2(0,1),mx), tap(i+int2(1,1),mx), f.x);
    float2 fl = lerp(a, b, f.y);
    float w = 1.0;
    if (gCostHi > gCostLo) {  // confidence gate: drop low-confidence (high-cost) vectors
        int2 ci = clamp(int2(pos + 0.5), int2(0,0), mx);
        float cost = float(gCost.Load(int3(ci, 0)));
        w = saturate((gCostHi - cost) / (gCostHi - gCostLo));
    }
    gDst[t.xy] = float2(fl.x * gS.x, fl.y * gS.y) * w;
}
)";

ComPtr<ID3D12PipelineState> CompilePso(ID3D12Device* dev, ID3D12RootSignature* sig,
                                       const char* src, const char* name) {
    ComPtr<ID3DBlob> cs, err;
    HRESULT hr = D3DCompile(src, strlen(src), name, nullptr, nullptr, "CSMain", "cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &err);
    if (FAILED(hr))
        throw ToolError(std::string("nvof ") + name + ": " +
                        (err ? static_cast<const char*>(err->GetBufferPointer()) : HrToString(hr)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = sig;
    pd.CS.pShaderBytecode = cs->GetBufferPointer();
    pd.CS.BytecodeLength = cs->GetBufferSize();
    ComPtr<ID3D12PipelineState> pso;
    CHECK_HR(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    return pso;
}

typedef NV_OF_STATUS(NVOFAPI* PfnCreateInstanceD3D12)(uint32_t, NV_OF_D3D12_API_FUNCTION_LIST*);

}  // namespace

bool NvofFlow::RegisterTex(GpuTexture& tex, NvOFGPUBufferHandle* outHandle) {
    NV_OF_REGISTER_RESOURCE_PARAMS_D3D12 rp{};
    rp.resource = tex.res.Get();
    rp.inputFencePoint.fence = m_appFence.Get();
    rp.inputFencePoint.value = m_appVal;
    rp.hOFGpuBuffer = outHandle;
    rp.outputFencePoint.fence = m_ofaFence.Get();
    rp.outputFencePoint.value = ++m_ofaVal;
    NV_OF_STATUS s = m_api.nvOFRegisterResourceD3D12(m_hOF, &rp);
    return s == NV_OF_SUCCESS && *outHandle != nullptr;
}

void NvofFlow::WaitFence(ID3D12Fence* fence, uint64_t value) {
    if (fence->GetCompletedValue() < value) {
        fence->SetEventOnCompletion(value, m_event);
        WaitForSingleObject(m_event, INFINITE);
    }
}

bool NvofFlow::Init(GpuContext& gpu, int inW, int inH, int targetW, int targetH) {
    m_device = gpu.Device();
    m_queue = gpu.Queue();
    m_inW = inW;
    m_inH = inH;
    m_tW = targetW;
    m_tH = targetH;

    m_dll = LoadLibraryW(L"nvofapi64.dll");
    if (!m_dll) return false;
    auto create = reinterpret_cast<PfnCreateInstanceD3D12>(
        GetProcAddress(m_dll, "NvOFAPICreateInstanceD3D12"));
    if (!create) return false;
    if (create(NV_OF_API_VERSION, &m_api) != NV_OF_SUCCESS) return false;
    if (!m_api.nvCreateOpticalFlowD3D12 || !m_api.nvOFInit || !m_api.nvOFExecuteD3D12 ||
        !m_api.nvOFRegisterResourceD3D12)
        return false;
    if (m_api.nvCreateOpticalFlowD3D12(m_device, &m_hOF) != NV_OF_SUCCESS || !m_hOF) return false;

    try {
        // Confirm ABGR8 is an accepted input surface format.
        uint32_t cnt = 0;
        if (m_api.nvOFGetSurfaceFormatCountD3D12(m_hOF, NV_OF_BUFFER_USAGE_INPUT,
                                                 NV_OF_MODE_OPTICALFLOW, &cnt) != NV_OF_SUCCESS ||
            cnt == 0)
            return false;
        std::vector<DXGI_FORMAT> fmts(cnt);
        m_api.nvOFGetSurfaceFormatD3D12(m_hOF, NV_OF_BUFFER_USAGE_INPUT, NV_OF_MODE_OPTICALFLOW,
                                        fmts.data());
        if (std::find(fmts.begin(), fmts.end(), DXGI_FORMAT_B8G8R8A8_UNORM) == fmts.end())
            return false;

        // Pick the finest supported output grid size.
        uint32_t gsCount = 0;
        if (m_api.nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &gsCount) !=
                NV_OF_SUCCESS ||
            gsCount == 0)
            return false;
        std::vector<uint32_t> grids(gsCount);
        m_api.nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, grids.data(), &gsCount);
        m_grid = static_cast<int>(*std::min_element(grids.begin(), grids.end()));
        if (m_grid < 1) m_grid = 1;
        {
            std::string gl;
            for (uint32_t g : grids) gl += (gl.empty() ? "" : ",") + std::to_string(g);
            LogInfo("NVOFA output grid: %dx%d (supported: %s)", m_grid, m_grid, gl.c_str());
        }

        // Confidence (cost) output, used to gate low-confidence vectors (kills flicker).
        uint32_t costCount = 0;
        if (m_api.nvOFGetSurfaceFormatCountD3D12(m_hOF, NV_OF_BUFFER_USAGE_COST,
                                                 NV_OF_MODE_OPTICALFLOW, &costCount) ==
                NV_OF_SUCCESS &&
            costCount > 0) {
            std::vector<DXGI_FORMAT> cf(costCount);
            m_api.nvOFGetSurfaceFormatD3D12(m_hOF, NV_OF_BUFFER_USAGE_COST, NV_OF_MODE_OPTICALFLOW,
                                            cf.data());
            if (std::find(cf.begin(), cf.end(), DXGI_FORMAT_R8_UINT) != cf.end())
                m_costFmt = DXGI_FORMAT_R8_UINT;
            else
                m_costFmt = cf[0];
        }

        NV_OF_INIT_PARAMS ip{};
        ip.width = static_cast<uint32_t>(inW);
        ip.height = static_cast<uint32_t>(inH);
        ip.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(m_grid);
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
        ip.enableExternalHints = NV_OF_FALSE;
        ip.enableOutputCost = (m_costFmt != DXGI_FORMAT_UNKNOWN) ? NV_OF_TRUE : NV_OF_FALSE;
        ip.hPrivData = nullptr;
        ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        ip.enableRoi = NV_OF_FALSE;
        ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
        ip.enableGlobalFlow = NV_OF_FALSE;
        ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
        if (m_api.nvOFInit(m_hOF, &ip) != NV_OF_SUCCESS) return false;

        const int outW = (inW + m_grid - 1) / m_grid;
        const int outH = (inH + m_grid - 1) / m_grid;
        m_in[0] = gpu.CreateTexture(inW, inH, DXGI_FORMAT_B8G8R8A8_UNORM, true, L"nvofIn0");
        m_in[1] = gpu.CreateTexture(inW, inH, DXGI_FORMAT_B8G8R8A8_UNORM, true, L"nvofIn1");
        m_out = gpu.CreateTexture(outW, outH, DXGI_FORMAT_R16G16_SINT, true, L"nvofOut");
        if (m_costFmt != DXGI_FORMAT_UNKNOWN)
            m_cost = gpu.CreateTexture(outW, outH, m_costFmt, true, L"nvofCost");

        CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_appFence)));
        CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_ofaFence)));
        m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_event) return false;

        if (!RegisterTex(m_in[0], &m_hIn[0]) || !RegisterTex(m_in[1], &m_hIn[1]) ||
            !RegisterTex(m_out, &m_hOut))
            return false;
        if (m_cost.res && !RegisterTex(m_cost, &m_hCost)) return false;

        // Compute plumbing: root constants + table [1 SRV, 1 UAV].
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 2;
        ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = sizeof(Consts) / 4;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 2;
        params[1].DescriptorTable.pDescriptorRanges = ranges;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rs{};
        rs.NumParameters = 2;
        rs.pParameters = params;
        ComPtr<ID3DBlob> sig, err;
        CHECK_HR(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
        CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                               IID_PPV_ARGS(&m_sig)));
        m_psoColor = CompilePso(m_device, m_sig.Get(), kColorHlsl, "color");
        m_psoMotion = CompilePso(m_device, m_sig.Get(), kMotionHlsl, "motion");

        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 8;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_HR(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)));
        m_descSize =
            m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    } catch (const std::exception& e) {
        LogWarn("NVOFA init failed: %s", e.what());
        return false;
    }

    m_ok = true;
    return true;
}

void NvofFlow::RunPass(ID3D12GraphicsCommandList* cl, ID3D12PipelineState* pso,
                       const GpuTexture& srv0, const GpuTexture& srv1, const GpuTexture& uav,
                       const void* constants) {
    // Fixed table: SRV t0 @0, SRV t1 @1, UAV u0 @2.
    auto cpu = [&](int i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(i) * m_descSize;
        return h;
    };
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Texture2D.MipLevels = 1;
    sd.Format = srv0.fmt;
    m_device->CreateShaderResourceView(srv0.res.Get(), &sd, cpu(0));
    sd.Format = srv1.fmt;
    m_device->CreateShaderResourceView(srv1.res.Get(), &sd, cpu(1));
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = uav.fmt;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(uav.res.Get(), nullptr, &ud, cpu(2));

    ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetPipelineState(pso);
    cl->SetComputeRootSignature(m_sig.Get());
    cl->SetComputeRoot32BitConstants(0, sizeof(Consts) / 4, constants, 0);
    cl->SetComputeRootDescriptorTable(1, m_heap->GetGPUDescriptorHandleForHeapStart());
    cl->Dispatch(static_cast<UINT>((uav.w + 7) / 8), static_cast<UINT>((uav.h + 7) / 8), 1);
}

void NvofFlow::Compute(GpuContext& gpu, GpuTexture& curColor, GpuTexture& motion, bool reset) {
    GpuTexture& in = m_in[m_cur];

    // 1) current colour -> OF input surface (downscaled, ABGR8), left in COMMON for the engine.
    {
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        gpu.Transition(curColor, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Transition(in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Consts c{};
        c.dstW = static_cast<uint32_t>(m_inW);
        c.dstH = static_cast<uint32_t>(m_inH);
        RunPass(cl, m_psoColor.Get(), curColor, curColor, in, &c);
        gpu.Transition(in, D3D12_RESOURCE_STATE_COMMON);
        gpu.EndAndWait();
    }
    m_queue->Signal(m_appFence.Get(), ++m_appVal);

    if (reset || !m_hasPrev) {
        // No valid history: zero the motion and start accumulating from this frame.
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Consts c{};
        c.dstW = static_cast<uint32_t>(m_tW);
        c.dstH = static_cast<uint32_t>(m_tH);
        c.reset = 1;
        RunPass(cl, m_psoMotion.Get(), m_out, m_cost.res ? m_cost : m_out, motion, &c);
        gpu.EndAndWait();
        m_cur ^= 1;
        m_hasPrev = true;
        return;
    }

    // 2) hardware optical flow: cur (input) vs prev (reference) -> S10.5 grid in m_out.
    NV_OF_EXECUTE_INPUT_PARAMS_D3D12 ei{};
    ei.inputFrame = m_hIn[m_cur];
    ei.referenceFrame = m_hIn[m_cur ^ 1];
    ei.disableTemporalHints = NV_OF_FALSE;
    NV_OF_FENCE_POINT inFp{m_appFence.Get(), m_appVal};
    ei.numFencePoints = 1;
    ei.fencePoint = &inFp;

    NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12 eo{};
    eo.outputBuffer = m_hOut;
    eo.outputCostBuffer = m_hCost;  // null when cost is unavailable
    NV_OF_FENCE_POINT outFp{m_ofaFence.Get(), ++m_ofaVal};
    eo.fencePoint = &outFp;

    NV_OF_STATUS s = m_api.nvOFExecuteD3D12(m_hOF, &ei, &eo);
    if (s != NV_OF_SUCCESS) {
        LogWarn("NVOFA execute failed (status %d); zeroing motion this frame", static_cast<int>(s));
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Consts c{};
        c.dstW = static_cast<uint32_t>(m_tW);
        c.dstH = static_cast<uint32_t>(m_tH);
        c.reset = 1;
        RunPass(cl, m_psoMotion.Get(), m_out, m_cost.res ? m_cost : m_out, motion, &c);
        gpu.EndAndWait();
        m_cur ^= 1;
        return;
    }

    // 3) wait for the engine (GPU-side), then flow grid -> target-resolution motion.
    m_queue->Wait(m_ofaFence.Get(), m_ofaVal);
    {
        ID3D12GraphicsCommandList* cl = gpu.Begin();
        gpu.Transition(m_out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Transition(motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Consts c{};
        c.dstW = static_cast<uint32_t>(m_tW);
        c.dstH = static_cast<uint32_t>(m_tH);
        c.sx = static_cast<float>(m_tW) / m_inW / 32.0f;
        c.sy = static_cast<float>(m_tH) / m_inH / 32.0f;
        // Confidence gate disabled (costLo==costHi==0 -> shader keeps every vector). The cost
        // buffer is still produced by the engine; re-enable by setting c.costLo/c.costHi.
        RunPass(cl, m_psoMotion.Get(), m_out, m_cost.res ? m_cost : m_out, motion, &c);
        gpu.Transition(m_out, D3D12_RESOURCE_STATE_COMMON);
        gpu.EndAndWait();
    }
    m_cur ^= 1;
    m_hasPrev = true;
}

void NvofFlow::Shutdown() {
    // Drain the engine, unregister and release the resources it referenced, THEN destroy the
    // instance. Releasing a still-registered resource, or one released after nvOFDestroy, access-
    // violates in the driver - hence this exact order.
    if (m_ofaFence && m_event) WaitFence(m_ofaFence.Get(), m_ofaVal);
    if (m_hOF && m_api.nvOFUnregisterResourceD3D12) {
        for (int i = 0; i < 2; ++i)
            if (m_hIn[i]) {
                NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 up{};
                up.hOFGpuBuffer = m_hIn[i];
                m_api.nvOFUnregisterResourceD3D12(&up);
                m_hIn[i] = nullptr;
            }
        if (m_hOut) {
            NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 up{};
            up.hOFGpuBuffer = m_hOut;
            m_api.nvOFUnregisterResourceD3D12(&up);
            m_hOut = nullptr;
        }
        if (m_hCost) {
            NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 up{};
            up.hOFGpuBuffer = m_hCost;
            m_api.nvOFUnregisterResourceD3D12(&up);
            m_hCost = nullptr;
        }
    }
    m_in[0] = GpuTexture{};
    m_in[1] = GpuTexture{};
    m_out = GpuTexture{};
    m_cost = GpuTexture{};
    if (m_hOF && m_api.nvOFDestroy) m_api.nvOFDestroy(m_hOF);
    m_hOF = nullptr;
    m_psoColor.Reset();
    m_psoMotion.Reset();
    m_sig.Reset();
    m_heap.Reset();
    m_appFence.Reset();
    m_ofaFence.Reset();
    if (m_event) {
        CloseHandle(m_event);
        m_event = nullptr;
    }
    // Do not FreeLibrary(nvofapi64.dll): the driver module keeps worker threads and unloading it
    // mid-process access-violates. Leaking the handle for the life of the process is harmless.
    m_dll = nullptr;
    m_ok = false;
}
