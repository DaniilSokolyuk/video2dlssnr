// common.h — shared includes, error handling, small utilities.
#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

// A fatal, user-facing error. main() catches this and prints it without a stack dump.
struct ToolError : std::runtime_error {
    explicit ToolError(const std::string& what) : std::runtime_error(what) {}
};

std::string HrToString(HRESULT hr);
std::string Widen2Narrow(const std::wstring& w);
std::wstring Narrow2Widen(const std::string& s);

[[noreturn]] void FailAt(const char* file, int line, const char* expr, const std::string& detail);

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) FailAt(__FILE__, __LINE__, #expr, "");                    \
    } while (0)

#define CHECK_HR(expr)                                                         \
    do {                                                                       \
        HRESULT hr_ = (expr);                                                  \
        if (FAILED(hr_)) FailAt(__FILE__, __LINE__, #expr, HrToString(hr_));   \
    } while (0)

// Console helpers: the tool prints a lot of tabular status, keep it in one place.
void LogInfo(const char* fmt, ...);
void LogWarn(const char* fmt, ...);
void LogErr(const char* fmt, ...);
void SetVerbose(bool on);
// Route LogInfo/LogWarn/LogDebug to stderr (video mode uses stdout for frame data).
void SetLogToStderr(bool on);
bool IsVerbose();
void LogDebug(const char* fmt, ...);
