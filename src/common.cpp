#include "common.h"

#include <cstdarg>

static bool g_verbose = false;

void SetVerbose(bool on) { g_verbose = on; }
bool IsVerbose() { return g_verbose; }

std::string HrToString(HRESULT hr) {
    char* msg = nullptr;
    DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, static_cast<DWORD>(hr),
                             MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                             reinterpret_cast<char*>(&msg), 0, nullptr);
    std::string text;
    if (n && msg) {
        text.assign(msg, n);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    }
    if (msg) LocalFree(msg);

    char buf[128];
    snprintf(buf, sizeof(buf), "HRESULT 0x%08X", static_cast<unsigned>(hr));
    return text.empty() ? std::string(buf) : std::string(buf) + ": " + text;
}

std::string Widen2Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), n, nullptr, nullptr);
    return out;
}

std::wstring Narrow2Widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), out.data(), n);
    return out;
}

void FailAt(const char* file, int line, const char* expr, const std::string& detail) {
    const char* base = strrchr(file, '\\');
    base = base ? base + 1 : file;
    std::string msg = std::string(base) + ":" + std::to_string(line) + "  failed: " + expr;
    if (!detail.empty()) msg += "\n    " + detail;
    throw ToolError(msg);
}

// Where the informational logs go. Normally stdout; in video mode stdout carries the raw
// frame stream, so the caller redirects these to stderr.
static bool g_logToStderr = false;
void SetLogToStderr(bool on) { g_logToStderr = on; }
static FILE* InfoStream() { return g_logToStderr ? stderr : stdout; }

static void VLog(FILE* f, const char* prefix, const char* fmt, va_list ap) {
    fputs(prefix, f);
    vfprintf(f, fmt, ap);
    fputc('\n', f);
    fflush(f);
}

void LogInfo(const char* fmt, ...) { va_list a; va_start(a, fmt); VLog(InfoStream(), "", fmt, a); va_end(a); }
void LogWarn(const char* fmt, ...) { va_list a; va_start(a, fmt); VLog(InfoStream(), "  ! ", fmt, a); va_end(a); }
void LogErr(const char* fmt, ...)  { va_list a; va_start(a, fmt); VLog(stderr, "ERROR: ", fmt, a); va_end(a); }
void LogDebug(const char* fmt, ...) {
    if (!g_verbose) return;
    va_list a; va_start(a, fmt); VLog(InfoStream(), "  . ", fmt, a); va_end(a);
}
