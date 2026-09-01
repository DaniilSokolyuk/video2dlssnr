// optflow_nvof.h - NVIDIA hardware optical flow (NVOFA) backend.
//
// Estimates cur->prev motion vectors on the dedicated optical-flow engine (Turing and above)
// via the Optical Flow SDK D3D12 interface, and writes target-resolution RG16F motion for NR -
// the same convention and output texture the Lucas-Kanade backend (optflow.cpp) fills, so the
// two are interchangeable. This is a separate file on purpose: the LK path never touches it.
//
// Init() returns false if NVOFA is unavailable (no nvofapi64.dll, unsupported GPU/driver, or
// any init failure), so the caller falls back to Lucas-Kanade.
#pragma once

#include "gpu.h"

#include "nvOpticalFlowD3D12.h"

class NvofFlow {
public:
    bool Init(GpuContext& gpu, int inW, int inH, int targetW, int targetH);
    void Shutdown();

    // Colour(cur) -> OF input, run NVOFA against the previous frame, flow grid -> target motion.
    // On reset (first frame / scene cut) the motion is zeroed and the history restarts.
    void Compute(GpuContext& gpu, GpuTexture& curColor, GpuTexture& motion, bool reset);

    bool Ok() const { return m_ok; }

private:
    void RunPass(ID3D12GraphicsCommandList* cl, ID3D12PipelineState* pso, const GpuTexture& srv0,
                 const GpuTexture& srv1, const GpuTexture& uav, const void* constants);
    bool RegisterTex(GpuTexture& tex, NvOFGPUBufferHandle* outHandle);
    void WaitFence(ID3D12Fence* fence, uint64_t value);

    bool m_ok = false;
    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_queue = nullptr;
    HMODULE m_dll = nullptr;
    NV_OF_D3D12_API_FUNCTION_LIST m_api{};
    NvOFHandle m_hOF = nullptr;

    int m_inW = 0, m_inH = 0, m_tW = 0, m_tH = 0;
    int m_grid = 1;

    GpuTexture m_in[2];
    NvOFGPUBufferHandle m_hIn[2]{nullptr, nullptr};
    GpuTexture m_out;
    NvOFGPUBufferHandle m_hOut = nullptr;
    GpuTexture m_cost;  // per-cell confidence (higher = less reliable); gates flicker
    NvOFGPUBufferHandle m_hCost = nullptr;
    DXGI_FORMAT m_costFmt = DXGI_FORMAT_UNKNOWN;

    ComPtr<ID3D12Fence> m_appFence;  // app signals after writing the OF input
    ComPtr<ID3D12Fence> m_ofaFence;  // OF signals after the flow is ready
    uint64_t m_appVal = 0;
    uint64_t m_ofaVal = 0;
    HANDLE m_event = nullptr;

    ComPtr<ID3D12RootSignature> m_sig;
    ComPtr<ID3D12PipelineState> m_psoColor;   // colour -> ABGR8 OF input (downscaled)
    ComPtr<ID3D12PipelineState> m_psoMotion;  // S10.5 flow grid -> RG16F target motion
    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT m_descSize = 0;
    int m_descNext = 0;

    int m_cur = 0;
    bool m_hasPrev = false;
};
