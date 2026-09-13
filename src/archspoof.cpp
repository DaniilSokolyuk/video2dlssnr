#include "archspoof.h"

#include <windows.h>

#include <cstring>

namespace {

// nvapi_QueryInterface ids.
constexpr unsigned kNvapiInitialize = 0x0150E828u;
constexpr unsigned kNvapiEnumPhysicalGpus = 0xE5AC921Fu;
constexpr unsigned kNvapiGpuGetArchInfo = 0xD8265D24u;

// NV_GPU_ARCHITECTURE_ID groups (the low nibble carries the implementation).
constexpr unsigned kArchTuring = 0x160u;
constexpr unsigned kArchAmpere = 0x170u;
constexpr unsigned kArchAda = 0x190u;
// What the model accepts. GB2XX is what it checks for; GB1XX (0x1A0) is still refused.
constexpr unsigned kArchBlackwell = 0x1B0u;

// NV_GPU_ARCH_INFO: version | architecture | implementation | revision.
struct NvArchInfo {
    unsigned version;
    unsigned architecture;
    unsigned implementation;
    unsigned revision;
};

using PfnQueryInterface = void*(__cdecl*)(unsigned);
using PfnInitialize = int(__cdecl*)();
using PfnEnumPhysicalGpus = int(__cdecl*)(void**, unsigned*);
using PfnGetArchInfo = int(__cdecl*)(void*, NvArchInfo*);

constexpr int kMaxGpus = 64;
void* g_handles[kMaxGpus];
NvArchInfo g_cache[kMaxGpus];
int g_count = 0;
bool g_enabled = true;
bool g_done = false;
ArchSpoofResult g_result = ArchSpoofResult::Disabled;

unsigned Group(unsigned arch) { return arch & 0xFFFFFFF0u; }
bool NeedsSpoof(unsigned arch) {
    const unsigned g = Group(arch);
    return g == kArchTuring || g == kArchAmpere || g == kArchAda;
}

// Replaces NvAPI_GPU_GetArchInfo for the whole process. Answers from the cache taken before the
// redirect went in; an unknown handle gets the first card's answer, which is the same GPU reached
// through another API path in every case seen. The structure version the caller asked for is kept.
int __cdecl ArchInfoHook(void* gpu, NvArchInfo* info) {
    if (!info) return -1;  // NVAPI_ERROR
    const unsigned want = info->version;
    int idx = 0;
    for (int i = 0; i < g_count; ++i) {
        if (g_handles[i] == gpu) {
            idx = i;
            break;
        }
    }
    *info = g_cache[idx];
    info->version = want;
    if (NeedsSpoof(info->architecture)) {
        // Implementation and revision as a live GB2XX board reports them; the check may read
        // more than the group.
        info->architecture = kArchBlackwell;
        info->implementation = 0x3u;
        info->revision = 0xA1u;
    }
    return 0;  // NVAPI_OK
}

bool WriteCode(void* at, const void* src, size_t bytes) {
    DWORD old = 0;
    if (!VirtualProtect(at, bytes, PAGE_EXECUTE_READWRITE, &old)) return false;
    memcpy(at, src, bytes);
    DWORD tmp = 0;
    VirtualProtect(at, bytes, old, &tmp);
    FlushInstructionCache(GetCurrentProcess(), at, bytes);
    return true;
}

// mov rax, imm64 ; jmp rax
bool Redirect(void* target, void* to) {
    unsigned char code[12] = {0x48, 0xB8};
    memcpy(code + 2, &to, sizeof(to));
    code[10] = 0xFF;
    code[11] = 0xE0;
    return WriteCode(target, code, sizeof(code));
}

}  // namespace

void SetArchSpoofEnabled(bool enabled) { g_enabled = enabled; }
bool ArchSpoofEnabled() { return g_enabled; }

const char* NvArchName(unsigned architecture) {
    switch (Group(architecture)) {
        case kArchTuring: return "Turing";
        case kArchAmpere: return "Ampere";
        case 0x180u: return "Hopper";
        case kArchAda: return "Ada";
        case 0x1A0u:
        case kArchBlackwell: return "Blackwell";
        default: return "unknown";
    }
}

ArchSpoofResult SetupArchSpoof() {
    if (g_done) return g_result;
    g_done = true;
    if (!g_enabled) {
        LogInfo("  arch:   spoof disabled (--nr-arch-spoof 0)");
        return g_result = ArchSpoofResult::Disabled;
    }
    auto fail = [&](const char* why) {
        LogWarn("GPU architecture spoof not installed: %s; the model will refuse a pre-Blackwell "
                "card unless the DLL beside the tool is a build for it",
                why);
        return g_result = ArchSpoofResult::Failed;
    };

    HMODULE nvapi = LoadLibraryW(L"nvapi64.dll");
    if (!nvapi) return fail("nvapi64.dll did not load");
    auto qi = reinterpret_cast<PfnQueryInterface>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
    if (!qi) return fail("no nvapi_QueryInterface");
    auto init = reinterpret_cast<PfnInitialize>(qi(kNvapiInitialize));
    auto enumGpus = reinterpret_cast<PfnEnumPhysicalGpus>(qi(kNvapiEnumPhysicalGpus));
    auto getArch = reinterpret_cast<PfnGetArchInfo>(qi(kNvapiGpuGetArchInfo));
    if (!init || !enumGpus || !getArch) return fail("the NvAPI entry points did not resolve");
    if (init() != 0) return fail("NvAPI_Initialize failed");

    void* handles[kMaxGpus] = {};
    unsigned count = 0;
    if (enumGpus(handles, &count) != 0 || count == 0) return fail("no NVIDIA GPU enumerated");
    if (count > static_cast<unsigned>(kMaxGpus)) count = kMaxGpus;

    // Every card's real answer, taken while the real function is still in place.
    g_count = 0;
    for (unsigned i = 0; i < count; ++i) {
        NvArchInfo one{};
        one.version = static_cast<unsigned>(sizeof(NvArchInfo)) | (2u << 16);
        if (getArch(handles[i], &one) != 0) {
            one.version = static_cast<unsigned>(sizeof(NvArchInfo)) | (1u << 16);
            if (getArch(handles[i], &one) != 0) continue;
        }
        g_handles[g_count] = handles[i];
        g_cache[g_count] = one;
        ++g_count;
    }
    if (g_count == 0) return fail("GetArchInfo answered for no card");

    bool anyOld = false;
    for (int i = 0; i < g_count; ++i) anyOld = anyOld || NeedsSpoof(g_cache[i].architecture);
    if (!anyOld) {
        LogInfo("  arch:   0x%X (%s), accepted by the model as is", g_cache[0].architecture,
                NvArchName(g_cache[0].architecture));
        return g_result = ArchSpoofResult::NotNeeded;
    }

    if (!Redirect(reinterpret_cast<void*>(getArch), reinterpret_cast<void*>(&ArchInfoHook)))
        return fail("could not make the NvAPI entry writable");
    LogInfo("  arch:   0x%X (%s) reported to the model as 0x%X (%s); %d card(s) cached",
            g_cache[0].architecture, NvArchName(g_cache[0].architecture), kArchBlackwell,
            NvArchName(kArchBlackwell), g_count);
    return g_result = ArchSpoofResult::Installed;
}
