// gpu.h — minimal headless D3D12 context: device, one direct queue, one command
// list, a shader-visible descriptor heap, and the jittered downsample pass that
// turns the high-res source into a DLSS-sized "rendered frame".
#pragma once

#include "common.h"

struct GpuTexture {
    ComPtr<ID3D12Resource> res;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    int w = 0;
    int h = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct PostConstants;  // defined in gpu.cpp

enum class DownFilter { Point, Bilinear, Tent, Lanczos };

const char* DownFilterName(DownFilter f);
bool ParseDownFilter(const std::string& s, DownFilter* out);

struct DownsampleArgs {
    GpuTexture* src = nullptr;      // high-res source, linear RGBA
    GpuTexture* dstColor = nullptr; // low-res colour handed to DLSS
    GpuTexture* dstDepth = nullptr; // low-res constant depth
    GpuTexture* dstMotion = nullptr;// low-res zero motion vectors
    float jitterX = 0.0f;           // subpixel offset in *render* pixel space
    float jitterY = 0.0f;
    DownFilter filter = DownFilter::Bilinear;
    bool encodeSrgb = false;        // true when feeding DLSS display-referred LDR
    float depthValue = 0.5f;
};

class GpuContext {
public:
    ~GpuContext();

    void Initialize(bool enableDebugLayer, int adapterIndex);
    void Shutdown();

    ID3D12Device* Device() const { return m_device.Get(); }
    ID3D12CommandQueue* Queue() const { return m_queue.Get(); }
    const std::string& AdapterName() const { return m_adapterName; }
    size_t AdapterVramMB() const { return m_vramMB; }

    GpuTexture CreateTexture(int w, int h, DXGI_FORMAT fmt, bool allowUav, const wchar_t* name);

    // Uploads linear RGBA float data, converting to the texture's half format.
    void UploadRgbaFloat(GpuTexture& tex, const std::vector<float>& rgba);
    // Reads a half-float RGBA texture back as linear RGBA float.
    std::vector<float> ReadbackRgbaFloat(GpuTexture& tex);
    // Fills an R32_FLOAT texture (used for the constant exposure input).
    void UploadR32Float(GpuTexture& tex, const std::vector<float>& data);
    // Readback for the depth and motion-vector targets, mainly so tests can
    // assert on what the downsample pass actually wrote.
    std::vector<float> ReadbackR32Float(GpuTexture& tex);
    std::vector<float> ReadbackRg16Float(GpuTexture& tex);  // two floats per pixel
    std::vector<uint8_t> ReadbackRgba8(GpuTexture& tex);    // R8G8B8A8_UNORM, 4 bytes/pixel

    ID3D12GraphicsCommandList* Begin();
    void EndAndWait();

    void Transition(GpuTexture& t, D3D12_RESOURCE_STATES to);
    void UavBarrier(GpuTexture& t);

    void RecordDownsample(const DownsampleArgs& args);

    // Video post passes (record into an already-open command list). EncodeSrgb turns
    // the linear DLSS output into the sRGB the NR model wants; Composite blends the
    // model output over the linear original and writes an 8-bit sRGB frame.
    void RecordEncodeSrgb(GpuTexture& srcLinear, GpuTexture& dstSrgb);
    void RecordComposite(GpuTexture& origLinear, GpuTexture& nrSrgb, GpuTexture& dstU8,
                         float detail, float colour);
    // Debug: colour-code a motion field into an 8-bit frame (hue = direction, brightness = mag).
    void RecordFlowVis(GpuTexture& motion, GpuTexture& dstU8, float maxMag);

    // GPU timestamps around whatever is recorded between the two calls.
    void RecordTimestampBegin();
    void RecordTimestampEnd();
    double LastGpuMs() const { return m_lastGpuMs; }

private:
    void CreateDownsamplePipeline();
    void CreatePostPipeline();
    void RecordPostPass(ID3D12PipelineState* pso, const GpuTexture& srv0, const GpuTexture& srv1,
                        const GpuTexture& uav, const PostConstants& c);
    // Copies a texture into a readback buffer and returns its raw bytes together
    // with the footprint needed to walk the rows.
    std::vector<uint8_t> ReadbackBytes(GpuTexture& tex, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* fp);
    int AllocDescriptor();
    D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(int index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(int index) const;
    void WaitForGpu();

    ComPtr<IDXGIFactory6> m_factory;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12CommandAllocator> m_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;

    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT m_descSize = 0;
    int m_descNext = 0;
    static constexpr int kHeapSize = 64;

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_pso;

    ComPtr<ID3D12RootSignature> m_postSig;
    ComPtr<ID3D12PipelineState> m_psoEncode;
    ComPtr<ID3D12PipelineState> m_psoComposite;
    ComPtr<ID3D12PipelineState> m_psoFlowVis;

    ComPtr<ID3D12QueryHeap> m_queryHeap;
    ComPtr<ID3D12Resource> m_queryReadback;
    uint64_t m_timestampFreq = 0;
    bool m_timestampPending = false;
    double m_lastGpuMs = 0.0;

    std::string m_adapterName;
    size_t m_vramMB = 0;
    bool m_listOpen = false;
};
