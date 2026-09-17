// archspoof.h — let the Neural Rendering model run on RTX 20/30/40.
//
// nvngx_dlssnr.dll 310.8 carries kernels for Turing, Ampere, Ada and Blackwell, but before it
// builds feature 18 it asks NvAPI for the GPU architecture and refuses anything below Blackwell
// ("DLSSNR: Unsupported GPU architecture"). The refusal is a policy check, not missing code.
//
// The model learns the architecture through NvAPI_GPU_GetArchInfo, resolved by id from
// nvapi64.dll's single export nvapi_QueryInterface. This module answers that one query itself,
// in this process only: the real answers for every GPU are cached first, the function's entry is
// then redirected to a handler that returns the cached data with the architecture of any
// pre-Blackwell NVIDIA card reported as Blackwell. Nothing on disk changes, and on a card the
// model already accepts nothing is patched at all.
#pragma once

#include "common.h"

enum class ArchSpoofResult { Disabled, NotNeeded, Installed, Failed };

// Process-wide switch, set from the command line before any NR entry point runs. On by default.
void SetArchSpoofEnabled(bool enabled);
bool ArchSpoofEnabled();

// Installs the redirect if the primary NVIDIA GPU is older than Blackwell. Idempotent; logs what
// it did. Call before the model DLL is loaded.
ArchSpoofResult SetupArchSpoof();

// "Turing" / "Ampere" / "Ada" / "Blackwell" / "unknown" for an NvAPI architecture id group.
const char* NvArchName(unsigned architecture);
