// optflow.h - dense pyramidal Lucas-Kanade optical flow, entirely on the GPU.
//
// Estimates, for every pixel of the current frame, where it was in the previous frame
// (screen-space displacement, in target pixels) - the motion vectors DLSS/NR use for
// temporal reprojection. The frames it compares are the colour textures already living
// in VRAM, so no pixel is read back to the CPU. All intermediate buffers stay in the
// UNORDERED_ACCESS state and passes are separated by UAV barriers, so there is no per-
// frame state juggling - the module just records its dispatches into a caller's list.
#pragma once

#include "gpu.h"

class OpticalFlow {
public:
    void Init(GpuContext& gpu, int targetW, int targetH);
    void Shutdown();

    // Records the whole estimate (luma pyramid -> coarse-to-fine LK -> upsample) into `cl`,
    // writing target-resolution motion vectors into `motion` (RG16F, target size). `curColor`
    // is the current linear RGBA frame at target size. On `reset` (first frame or a scene cut)
    // the motion is zeroed and the history restarts from this frame.
    void Record(ID3D12GraphicsCommandList* cl, GpuTexture& curColor, GpuTexture& motion,
                bool reset);

private:
    int Alloc();
    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(int i) const;
    D3D12_GPU_DESCRIPTOR_HANDLE Gpu(int i) const;
    void Srv(const GpuTexture& t, int slot);
    void Uav(const GpuTexture& t, int slot);
    void Barrier(ID3D12GraphicsCommandList* cl);  // global UAV barrier between passes
    // One compute pass: binds a fresh table (SRV t0 + UAV u0..u3, null slots filled with dst)
    // and dispatches over dst, followed by a UAV barrier. `constants` points at a FlowConst.
    void Pass(ID3D12GraphicsCommandList* cl, ID3D12PipelineState* pso, const GpuTexture* srv,
              const GpuTexture* u0, const GpuTexture* u1, const GpuTexture* u2,
              const GpuTexture* u3, const GpuTexture& dst, const void* constants);

    ID3D12Device* m_device = nullptr;
    int m_tW = 0, m_tH = 0;
    int m_baseW = 0, m_baseH = 0;
    int m_levels = 0;
    bool m_hasPrev = false;

    ComPtr<ID3D12RootSignature> m_sig;
    ComPtr<ID3D12PipelineState> m_psoLuma, m_psoDown, m_psoClear, m_psoUp, m_psoGrad, m_psoWarp,
        m_psoSolve, m_psoSmooth, m_psoMotion;

    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT m_descSize = 0;
    int m_descNext = 0;
    int m_tableBase = 0;

    // Two luma pyramids (current + previous), swapped each frame; plus per-level scratch.
    std::vector<GpuTexture> m_lumaA, m_lumaB;  // pyramids, m_lumaA/B alternate as cur/prev
    std::vector<GpuTexture> m_grad;            // (Ix,Iy) per level, RG16F
    std::vector<GpuTexture> m_it;              // temporal residual per level, R16F
    std::vector<GpuTexture> m_flow, m_flow2;   // flow ping-pong per level, RG16F
    bool m_curIsA = true;
};
