// video2dlssnr test suite.
//
// Two tiers:
//   CPU  — pure logic: parsing, colour conversion, metrics, PNG round-trips.
//   GPU  — needs the real device and the real nvngx_dlss.dll. These assert on
//          the downsample shader's exact output and on the behavioural
//          invariants of the accumulation loop.
//
// Run everything:      video2dlssnr_tests.exe
// CPU only:            video2dlssnr_tests.exe --no-gpu
// One group:           video2dlssnr_tests.exe --filter Downsample

#include "cli.h"
#include "common.h"
#include "dlss.h"
#include "gpu.h"
#include "image.h"
#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Tiny assertion framework
// ---------------------------------------------------------------------------

static int g_checks = 0;
static int g_failures = 0;
static int g_caseFailures = 0;

static const char* Basename(const char* path) {
    const char* slash = strrchr(path, '\\');
    return slash ? slash + 1 : path;
}

static void Failed(const char* file, int line, const std::string& msg) {
    ++g_failures;
    ++g_caseFailures;
    printf("      FAIL  %s:%d  %s\n", Basename(file), line, msg.c_str());
}

// Value formatting for assertion messages. Declared before the macros that use
// it. Note size_t and uint64_t are the same type on x64, so one overload covers
// both.
static std::string ToStr(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}
static std::string ToStr(int v) { return std::to_string(v); }
static std::string ToStr(unsigned v) { return std::to_string(v); }
static std::string ToStr(uint64_t v) { return std::to_string(v); }
static std::string ToStr(bool v) { return v ? "true" : "false"; }
static std::string ToStr(const std::string& v) { return "\"" + v + "\""; }

#define EXPECT_TRUE(cond)                                                                 \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) Failed(__FILE__, __LINE__, std::string("expected true: ") + #cond);  \
    } while (0)

#define EXPECT_FALSE(cond)                                                                \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if ((cond)) Failed(__FILE__, __LINE__, std::string("expected false: ") + #cond);  \
    } while (0)

#define EXPECT_EQ(actual, expected)                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        const auto a_ = (actual);                                                         \
        const auto e_ = (expected);                                                       \
        if (!(a_ == e_)) {                                                                \
            Failed(__FILE__, __LINE__, std::string(#actual) + " = " + ToStr(a_) +         \
                                           ", expected " + ToStr(e_));                    \
        }                                                                                 \
    } while (0)

#define EXPECT_NEAR(actual, expected, tol)                                                \
    do {                                                                                  \
        ++g_checks;                                                                       \
        const double a_ = static_cast<double>(actual);                                    \
        const double e_ = static_cast<double>(expected);                                  \
        if (!(std::fabs(a_ - e_) <= (tol))) {                                             \
            Failed(__FILE__, __LINE__,                                                    \
                   std::string(#actual) + " = " + ToStr(a_) + ", expected " + ToStr(e_) + \
                       " +/- " + ToStr(static_cast<double>(tol)));                        \
        }                                                                                 \
    } while (0)

#define EXPECT_GT(a, b)                                                                   \
    do {                                                                                  \
        ++g_checks;                                                                       \
        const double a_ = static_cast<double>(a);                                         \
        const double b_ = static_cast<double>(b);                                         \
        if (!(a_ > b_)) {                                                                 \
            Failed(__FILE__, __LINE__,                                                    \
                   std::string(#a) + " = " + ToStr(a_) + " is not > " + #b + " = " +      \
                       ToStr(b_));                                                        \
        }                                                                                 \
    } while (0)

// Variadic so the body may be a braced block containing several statements.
#define EXPECT_THROWS(...)                                                                \
    do {                                                                                  \
        ++g_checks;                                                                       \
        bool threw_ = false;                                                              \
        try {                                                                             \
            __VA_ARGS__;                                                                  \
        } catch (const ToolError&) {                                                      \
            threw_ = true;                                                                \
        }                                                                                 \
        if (!threw_)                                                                      \
            Failed(__FILE__, __LINE__, std::string("expected throw: ") + #__VA_ARGS__);   \
    } while (0)

#define EXPECT_NO_THROW(...)                                                              \
    do {                                                                                  \
        ++g_checks;                                                                       \
        try {                                                                             \
            __VA_ARGS__;                                                                  \
        } catch (const std::exception& e_) {                                              \
            Failed(__FILE__, __LINE__, std::string("unexpected throw from ") +            \
                                           #__VA_ARGS__ + ": " + e_.what());              \
        }                                                                                 \
    } while (0)

struct TestCase {
    const char* name;
    std::function<void()> fn;
    bool needsGpu;
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

static Options ParseCli(std::vector<const char*> args, bool* wantHelp = nullptr) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("video2dlssnr"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    bool help = false;
    Options o = ParseArgs(static_cast<int>(argv.size()), argv.data(), &help);
    if (wantHelp) *wantHelp = help;
    return o;
}

static ImageF MakeImage(int w, int h, const std::function<void(int, int, float*)>& fill) {
    ImageF img;
    img.w = w;
    img.h = h;
    img.px.assign(static_cast<size_t>(w) * h * 4, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            fill(x, y, &img.px[(static_cast<size_t>(y) * w + x) * 4]);
        }
    }
    return img;
}

// A miniature version of the test chart: high-frequency content that only
// temporal accumulation can resolve.
static ImageF MakeDetailChart(int w, int h) {
    return MakeImage(w, h, [&](int x, int y, float* p) {
        const float checker = ((x / 3 + y / 3) % 2) ? 1.0f : 0.0f;
        const float diag = (((x + y) % 5) == 0) ? 1.0f : 0.0f;
        const float r2 = static_cast<float>((x - w / 2) * (x - w / 2) + (y - h / 2) * (y - h / 2));
        const float zone = 0.5f + 0.5f * std::cos(r2 / (0.35f * w));
        p[0] = SrgbToLinear(std::clamp(checker * 0.8f + diag * 0.2f, 0.0f, 1.0f));
        p[1] = SrgbToLinear(std::clamp(zone, 0.0f, 1.0f));
        p[2] = SrgbToLinear(std::clamp(0.5f * checker + 0.5f * zone, 0.0f, 1.0f));
        p[3] = 1.0f;
    });
}

static fs::path TempDir() {
    static fs::path dir = [] {
        std::error_code ec;
        fs::path d = fs::temp_directory_path(ec) / "video2dlssnr_tests";
        fs::create_directories(d, ec);
        return d;
    }();
    return dir;
}

// ---------------------------------------------------------------------------
// CPU tests
// ---------------------------------------------------------------------------

static void Test_SrgbRoundTrip() {
    for (int i = 0; i <= 100; ++i) {
        const float v = i / 100.0f;
        EXPECT_NEAR(LinearToSrgb(SrgbToLinear(v)), v, 1e-5);
    }
    EXPECT_NEAR(SrgbToLinear(0.0f), 0.0, 1e-9);
    EXPECT_NEAR(SrgbToLinear(1.0f), 1.0, 1e-6);
    // Mid sRGB grey is far from mid linear — the fact the whole tool hinges on.
    EXPECT_NEAR(SrgbToLinear(0.5f), 0.21404, 1e-4);
    EXPECT_NEAR(LinearToSrgb(0.5f), 0.73536, 1e-4);
    // Below the linear-segment knee.
    EXPECT_NEAR(SrgbToLinear(0.02f), 0.02f / 12.92f, 1e-9);
    EXPECT_TRUE(LinearToSrgb(-1.0f) == 0.0f);
}

static void Test_Halton() {
    // Van der Corput base 2: 1/2, 1/4, 3/4, 1/8, 5/8 ...
    EXPECT_NEAR(Halton(1, 2), 0.5, 1e-12);
    EXPECT_NEAR(Halton(2, 2), 0.25, 1e-12);
    EXPECT_NEAR(Halton(3, 2), 0.75, 1e-12);
    EXPECT_NEAR(Halton(4, 2), 0.125, 1e-12);
    EXPECT_NEAR(Halton(5, 2), 0.625, 1e-12);
    // Base 3.
    EXPECT_NEAR(Halton(1, 3), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(Halton(2, 3), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(Halton(3, 3), 1.0 / 9.0, 1e-12);
    EXPECT_NEAR(Halton(0, 2), 0.0, 1e-12);

    // Everything must land in [0,1) so jitter stays within +/- half a pixel.
    for (int i = 0; i < 512; ++i) {
        const double h2 = Halton(i, 2), h3 = Halton(i, 3);
        EXPECT_TRUE(h2 >= 0.0 && h2 < 1.0);
        EXPECT_TRUE(h3 >= 0.0 && h3 < 1.0);
    }
}

static void Test_JitterPhaseCount() {
    EXPECT_EQ(JitterPhaseCount(1920, 1280, 42), 42);   // explicit override wins
    EXPECT_EQ(JitterPhaseCount(1920, 1920, 0), 8);     // DLAA, 1x
    EXPECT_EQ(JitterPhaseCount(1920, 960, 0), 32);     // 2x  -> 8*4
    EXPECT_EQ(JitterPhaseCount(1920, 640, 0), 72);     // 3x  -> 8*9
    EXPECT_EQ(JitterPhaseCount(1920, 1280, 0), 18);    // 1.5x -> 8*2.25
    EXPECT_EQ(JitterPhaseCount(1920, 0, 0), 8);        // guard against divide by zero
}

static void Test_ParseJitterSign() {
    JitterSign s;
    EXPECT_TRUE(ParseJitterSign("++", &s));
    EXPECT_EQ(s.x, 1.0f);
    EXPECT_EQ(s.y, 1.0f);
    EXPECT_TRUE(ParseJitterSign("--", &s));
    EXPECT_EQ(s.x, -1.0f);
    EXPECT_EQ(s.y, -1.0f);
    EXPECT_TRUE(ParseJitterSign("+-", &s));
    EXPECT_EQ(s.x, 1.0f);
    EXPECT_EQ(s.y, -1.0f);
    EXPECT_TRUE(ParseJitterSign("-+", &s));
    EXPECT_EQ(s.x, -1.0f);
    EXPECT_EQ(s.y, 1.0f);
    EXPECT_EQ(s.name, std::string("-+"));

    EXPECT_FALSE(ParseJitterSign("", &s));
    EXPECT_FALSE(ParseJitterSign("+", &s));
    EXPECT_FALSE(ParseJitterSign("+++", &s));
    EXPECT_FALSE(ParseJitterSign("ab", &s));
    EXPECT_FALSE(ParseJitterSign("auto", &s));
}

static void Test_ParsePreset() {
    unsigned v = 999;
    EXPECT_TRUE(ParsePreset("default", &v));
    EXPECT_EQ(v, 0u);
    EXPECT_TRUE(ParsePreset("0", &v));
    EXPECT_EQ(v, 0u);
    EXPECT_TRUE(ParsePreset("A", &v));
    EXPECT_EQ(v, 1u);
    EXPECT_TRUE(ParsePreset("a", &v));
    EXPECT_EQ(v, 1u);
    // The letters that matter today: J=10, K=11, L=12, M=13.
    EXPECT_TRUE(ParsePreset("J", &v));
    EXPECT_EQ(v, 10u);
    EXPECT_TRUE(ParsePreset("K", &v));
    EXPECT_EQ(v, 11u);
    EXPECT_TRUE(ParsePreset("L", &v));
    EXPECT_EQ(v, 12u);
    EXPECT_TRUE(ParsePreset("M", &v));
    EXPECT_EQ(v, 13u);
    EXPECT_TRUE(ParsePreset("O", &v));
    EXPECT_EQ(v, 15u);
    // Numeric form keeps future presets reachable without a code change.
    EXPECT_TRUE(ParsePreset("11", &v));
    EXPECT_EQ(v, 11u);
    EXPECT_TRUE(ParsePreset("31", &v));

    EXPECT_FALSE(ParsePreset("P", &v));
    EXPECT_FALSE(ParsePreset("", &v));
    EXPECT_FALSE(ParsePreset("AA", &v));
    EXPECT_FALSE(ParsePreset("32", &v));
    EXPECT_FALSE(ParsePreset("-1", &v));

    EXPECT_EQ(PresetName(0), std::string("default"));
    EXPECT_EQ(PresetName(11), std::string("K"));
    EXPECT_EQ(PresetName(1), std::string("A"));
}

static void Test_ParseQualityMode() {
    QualityMode m;
    EXPECT_TRUE(ParseQualityMode("dlaa", &m));
    EXPECT_TRUE(m.value == NVSDK_NGX_PerfQuality_Value_DLAA);
    EXPECT_TRUE(ParseQualityMode("DLAA", &m));
    EXPECT_TRUE(ParseQualityMode("quality", &m));
    EXPECT_TRUE(m.value == NVSDK_NGX_PerfQuality_Value_MaxQuality);
    EXPECT_TRUE(ParseQualityMode("Balanced", &m));
    EXPECT_TRUE(ParseQualityMode("ultra-performance", &m));
    EXPECT_TRUE(m.value == NVSDK_NGX_PerfQuality_Value_UltraPerformance);
    EXPECT_TRUE(ParseQualityMode("perf", &m));
    EXPECT_TRUE(m.value == NVSDK_NGX_PerfQuality_Value_MaxPerf);

    EXPECT_FALSE(ParseQualityMode("", &m));
    EXPECT_FALSE(ParseQualityMode("ultra", &m));
    EXPECT_FALSE(ParseQualityMode("nonsense", &m));

    // Every advertised mode must round-trip through its own canonical name.
    for (const QualityMode& q : AllQualityModes()) {
        QualityMode back;
        EXPECT_TRUE(ParseQualityMode(q.name, &back));
        EXPECT_TRUE(back.value == q.value);
    }
}

static void Test_ParseDownFilter() {
    DownFilter f;
    EXPECT_TRUE(ParseDownFilter("point", &f));
    EXPECT_TRUE(f == DownFilter::Point);
    EXPECT_TRUE(ParseDownFilter("bilinear", &f));
    EXPECT_TRUE(ParseDownFilter("tent", &f));
    EXPECT_TRUE(ParseDownFilter("lanczos", &f));
    EXPECT_FALSE(ParseDownFilter("Point", &f));  // names are lowercase by contract
    EXPECT_FALSE(ParseDownFilter("", &f));
    EXPECT_EQ(std::string(DownFilterName(DownFilter::Point)), std::string("point"));
}

static void Test_SplitList() {
    EXPECT_EQ(SplitList("a,b,c").size(), size_t(3));
    EXPECT_EQ(SplitList("a").size(), size_t(1));
    EXPECT_EQ(SplitList("a,,b").size(), size_t(2));  // empty entries dropped
    EXPECT_EQ(SplitList("").size(), size_t(0));
    EXPECT_EQ(SplitList("a;b").size(), size_t(2));
    EXPECT_EQ(SplitList("a,b,").size(), size_t(2));
    EXPECT_EQ(SplitList("x,y")[1], std::string("y"));
}

static void Test_ParseArgsDefaults() {
    const Options o = ParseCli({"--in", "photo.png"});
    EXPECT_EQ(o.input, std::string("photo.png"));
    EXPECT_EQ(o.outDir, std::string("out"));
    EXPECT_EQ(o.frames, 32);
    EXPECT_EQ(o.phases, 0);
    EXPECT_TRUE(o.filter == DownFilter::Point);
    EXPECT_EQ(o.jitterSign, std::string("auto"));
    EXPECT_FALSE(o.hdr);
    EXPECT_FALSE(o.displaySpace);
    EXPECT_TRUE(o.autoExposure);
    EXPECT_TRUE(o.saveDiff);
    EXPECT_NEAR(o.diffGain, 8.0f, 1e-6);
    EXPECT_EQ(o.presets.size(), size_t(3));
}

static void Test_ParseArgsValues() {
    const Options o = ParseCli({"--in", "a.png", "--out", "res", "--frames", "64", "--phases", "16",
                                "--filter", "lanczos", "--jitter-sign", "-+", "--quality",
                                "dlaa,quality", "--preset", "J,K", "--depth", "0.25", "--png16",
                                "--save-lr", "--no-diff", "--diff-gain", "4.5", "--alpha",
                                "--no-auto-exposure", "--adapter", "1", "-v"});
    EXPECT_EQ(o.outDir, std::string("res"));
    EXPECT_EQ(o.frames, 64);
    EXPECT_EQ(o.phases, 16);
    EXPECT_TRUE(o.filter == DownFilter::Lanczos);
    EXPECT_EQ(o.jitterSign, std::string("-+"));
    EXPECT_EQ(o.qualities.size(), size_t(2));
    EXPECT_EQ(o.presets.size(), size_t(2));
    EXPECT_NEAR(o.depthValue, 0.25f, 1e-6);
    EXPECT_TRUE(o.png16);
    EXPECT_TRUE(o.saveLowRes);
    EXPECT_FALSE(o.saveDiff);
    EXPECT_NEAR(o.diffGain, 4.5f, 1e-6);
    EXPECT_TRUE(o.alphaUpscaling);
    EXPECT_FALSE(o.autoExposure);
    EXPECT_EQ(o.adapter, 1);
    EXPECT_TRUE(o.verbose);

    bool help = false;
    ParseCli({"--help"}, &help);
    EXPECT_TRUE(help);
    // --help must short-circuit before the "--in is required" check.
    EXPECT_NO_THROW(ParseCli({"-h"}));
}

static void Test_ParseArgsErrors() {
    EXPECT_THROWS(ParseCli({}));                                    // --in missing
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--bogus"}));          // unknown flag
    EXPECT_THROWS(ParseCli({"--in"}));                              // value missing
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--frames", "0"}));    // out of range
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--frames", "-3"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--frames", "abc"}));  // not a number
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--frames", "12x"}));  // trailing junk
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--phases", "-1"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--filter", "nope"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--jitter-sign", "xx"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--filter-space", "srgb"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--diff-gain", "0"}));
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--diff-gain", "-2"}));
    // HDR needs linear colour, so display-space filtering is incompatible.
    EXPECT_THROWS(ParseCli({"--in", "a.png", "--hdr", "--filter-space", "display"}));
    EXPECT_NO_THROW(ParseCli({"--in", "a.png", "--filter-space", "linear"}));
    EXPECT_NO_THROW(ParseCli({"--in", "a.png", "--hdr"}));
}

static void Test_MetricsIdentity() {
    const ImageF a = MakeDetailChart(64, 48);
    const Metrics m = ComputeMetrics(a, a);
    EXPECT_NEAR(m.psnrRgb, 99.0, 1e-9);   // capped sentinel for "no error"
    EXPECT_NEAR(m.ssimLuma, 1.0, 1e-6);
}

static void Test_MetricsKnownError() {
    // Two flat images 0.1 apart in sRGB space: MSE = 0.01 exactly -> PSNR = 20 dB.
    const ImageF a = MakeImage(32, 32, [](int, int, float* p) {
        p[0] = p[1] = p[2] = SrgbToLinear(0.5f);
        p[3] = 1.0f;
    });
    const ImageF b = MakeImage(32, 32, [](int, int, float* p) {
        p[0] = p[1] = p[2] = SrgbToLinear(0.6f);
        p[3] = 1.0f;
    });
    const Metrics m = ComputeMetrics(a, b);
    EXPECT_NEAR(m.psnrRgb, 20.0, 0.02);

    // With zero variance SSIM collapses to the luminance term alone.
    const double ux = 0.5, uy = 0.6, C1 = 0.0001;
    EXPECT_NEAR(m.ssimLuma, (2 * ux * uy + C1) / (ux * ux + uy * uy + C1), 1e-3);
}

static void Test_MetricsSizeMismatch() {
    const ImageF a = MakeDetailChart(32, 32);
    const ImageF b = MakeDetailChart(32, 16);
    EXPECT_THROWS(ComputeMetrics(a, b));
}

static void Test_HashImage() {
    const ImageF a = MakeDetailChart(48, 32);
    ImageF b = a;
    EXPECT_EQ(HashImage(a), HashImage(b));
    EXPECT_EQ(HashImage(a), HashImage(a));  // stable across calls

    // A one-step change in a single 8-bit channel must change the hash: this is
    // what the "preset had no effect" detection relies on.
    b.px[4 * 100 + 1] = SrgbToLinear(LinearToSrgb(b.px[4 * 100 + 1]) + 1.0f / 255.0f);
    EXPECT_TRUE(HashImage(a) != HashImage(b));
}

static void Test_Png8RoundTrip() {
    const ImageF src = MakeDetailChart(61, 37);  // odd size catches stride bugs
    const std::string path = (TempDir() / "png8.png").string();
    SavePng8(path, src);
    const ImageF back = LoadImageLinear(path);

    EXPECT_EQ(back.w, src.w);
    EXPECT_EQ(back.h, src.h);
    double worst = 0.0;
    for (size_t i = 0; i < src.pixels(); ++i) {
        for (int c = 0; c < 3; ++c) {
            worst = std::max(worst, static_cast<double>(std::fabs(
                                        LinearToSrgb(src.px[i * 4 + c]) -
                                        LinearToSrgb(back.px[i * 4 + c]))));
        }
    }
    EXPECT_TRUE(worst <= 1.0 / 255.0 + 1e-6);
}

static void Test_Png16RoundTrip() {
    // The 16-bit writer is hand-rolled, so this checks it produces a PNG that a
    // third-party decoder reads back with 16-bit accuracy.
    const ImageF src = MakeDetailChart(53, 29);
    const std::string path = (TempDir() / "png16.png").string();
    SavePng16(path, src);

    EXPECT_TRUE(fs::exists(path));
    const ImageF back = LoadImageLinear(path);
    EXPECT_EQ(back.w, src.w);
    EXPECT_EQ(back.h, src.h);

    double worst = 0.0;
    for (size_t i = 0; i < src.pixels(); ++i) {
        for (int c = 0; c < 3; ++c) {
            worst = std::max(worst, static_cast<double>(std::fabs(
                                        LinearToSrgb(src.px[i * 4 + c]) -
                                        LinearToSrgb(back.px[i * 4 + c]))));
        }
    }
    EXPECT_TRUE(worst <= 1.0 / 65535.0 + 1e-5);
}

static void Test_DiffImage() {
    const ImageF a = MakeImage(16, 16, [](int, int, float* p) {
        p[0] = p[1] = p[2] = SrgbToLinear(0.5f);
        p[3] = 1.0f;
    });
    ImageF b = a;
    // One pixel off by 0.25 in sRGB, everything else identical.
    for (int c = 0; c < 3; ++c) b.px[4 * 40 + c] = SrgbToLinear(0.75f);

    const std::string path = (TempDir() / "diff.png").string();
    const DiffStats d = SaveDiffPng(path, a, b, 4.0f);
    EXPECT_NEAR(d.maxAbs, 0.25, 2e-3);
    EXPECT_EQ(d.maxX, 40 % 16);
    EXPECT_EQ(d.maxY, 40 / 16);
    EXPECT_NEAR(d.meanAbs, 0.25 / 256.0, 1e-3);
    EXPECT_TRUE(fs::exists(path));

    // Identical inputs must produce an all-black map and zero stats.
    const DiffStats zero = SaveDiffPng((TempDir() / "diff0.png").string(), a, a, 8.0f);
    EXPECT_NEAR(zero.maxAbs, 0.0, 1e-9);
    EXPECT_NEAR(zero.meanAbs, 0.0, 1e-9);

    EXPECT_THROWS(SaveDiffPng("", a, MakeDetailChart(8, 8), 4.0f));  // size mismatch
    EXPECT_THROWS(SaveDiffPng("", a, b, 0.0f));                      // bad gain
}

// ---------------------------------------------------------------------------
// GPU fixture
// ---------------------------------------------------------------------------

struct GpuFixture {
    GpuContext gpu;
    NgxSession ngx;
    bool ready = false;

    void Init() {
        if (ready) return;
        gpu.Initialize(false, -1);
        ngx.Init(gpu.Device(), DefaultDllSearchPaths(std::string()), false);
        ready = true;
    }
    void Shutdown() {
        if (!ready) return;
        ngx.Shutdown();
        gpu.Shutdown();
        ready = false;
    }
};
static GpuFixture g_fix;

// Uploads an RGBA float image into a fresh source texture.
static GpuTexture MakeSourceTexture(const ImageF& img) {
    GpuTexture t = g_fix.gpu.CreateTexture(img.w, img.h, DXGI_FORMAT_R16G16B16A16_FLOAT, false,
                                           L"testSource");
    g_fix.gpu.UploadRgbaFloat(t, img.px);
    return t;
}

static void RunDownsampleOnce(GpuTexture& src, LowResSet& dst, float jitterX, float jitterY,
                              DownFilter filter, bool encodeSrgb, float depthValue) {
    g_fix.gpu.Begin();
    DownsampleArgs da;
    da.src = &src;
    da.dstColor = &dst.color;
    da.dstDepth = &dst.depth;
    da.dstMotion = &dst.motion;
    da.jitterX = jitterX;
    da.jitterY = jitterY;
    da.filter = filter;
    da.encodeSrgb = encodeSrgb;
    da.depthValue = depthValue;
    g_fix.gpu.RecordDownsample(da);
    g_fix.gpu.EndAndWait();
}

// ---------------------------------------------------------------------------
// GPU tests
// ---------------------------------------------------------------------------

static void Test_GpuDevice() {
    EXPECT_TRUE(g_fix.gpu.Device() != nullptr);
    EXPECT_FALSE(g_fix.gpu.AdapterName().empty());
    printf("      adapter: %s (%zu MB)\n", g_fix.gpu.AdapterName().c_str(),
           g_fix.gpu.AdapterVramMB());
}

static void Test_HalfRoundTrip() {
    const ImageF src = MakeImage(37, 23, [](int x, int y, float* p) {
        p[0] = (x + y) * 0.01f;
        p[1] = x * 0.02f;
        p[2] = y * 0.03f;
        p[3] = 1.0f;
    });
    GpuTexture tex = g_fix.gpu.CreateTexture(src.w, src.h, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                             L"halfRT");
    g_fix.gpu.UploadRgbaFloat(tex, src.px);
    const std::vector<float> back = g_fix.gpu.ReadbackRgbaFloat(tex);

    EXPECT_EQ(back.size(), src.px.size());
    double worst = 0.0;
    for (size_t i = 0; i < src.px.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(back[i] - src.px[i])));
    }
    // half has ~11 bits of mantissa; values here stay under 1.0.
    EXPECT_TRUE(worst < 1e-3);
}

static void Test_DownsampleIdentity() {
    // Scale 1, no jitter, point filter: the shader must reproduce the source
    // texel for texel. Any off-by-half-a-pixel bug shows up immediately here.
    const int n = 32;
    const ImageF src = MakeImage(n, n, [&](int x, int y, float* p) {
        p[0] = static_cast<float>(x) / n;
        p[1] = static_cast<float>(y) / n;
        p[2] = 0.25f;
        p[3] = 1.0f;
    });
    GpuTexture srcTex = MakeSourceTexture(src);
    LowResSet dst = CreateLowResSet(g_fix.gpu, n, n);
    RunDownsampleOnce(srcTex, dst, 0.0f, 0.0f, DownFilter::Point, false, 0.5f);

    const std::vector<float> got = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    double worst = 0.0;
    for (size_t i = 0; i < src.px.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i] - src.px[i])));
    }
    EXPECT_TRUE(worst < 1e-3);

    // Point sampling snaps to a texel, so it cannot tell a half-pixel offset
    // from a quarter-pixel one. Bilinear can: landing anywhere but the exact
    // texel centre blends neighbours and shifts every value.
    RunDownsampleOnce(srcTex, dst, 0.0f, 0.0f, DownFilter::Bilinear, false, 0.5f);
    const std::vector<float> lerped = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    double worstLerp = 0.0;
    for (size_t i = 0; i < src.px.size(); ++i) {
        worstLerp = std::max(worstLerp, std::fabs(static_cast<double>(lerped[i] - src.px[i])));
    }
    EXPECT_TRUE(worstLerp < 1e-3);
}

static void Test_DownsampleScaleAndJitter() {
    // Source encodes its own x coordinate, so we can assert exactly which texel
    // each render pixel sampled.
    const int n = 64;
    const ImageF src = MakeImage(n, n, [&](int x, int y, float* p) {
        p[0] = static_cast<float>(x) / n;
        p[1] = static_cast<float>(y) / n;
        p[2] = 0.0f;
        p[3] = 1.0f;
    });
    GpuTexture srcTex = MakeSourceTexture(src);
    LowResSet dst = CreateLowResSet(g_fix.gpu, n / 2, n / 2);

    // Scale 2, no jitter: render pixel i sits at source position 2i+1.
    RunDownsampleOnce(srcTex, dst, 0.0f, 0.0f, DownFilter::Point, false, 0.5f);
    std::vector<float> got = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    for (int i = 0; i < n / 2; ++i) {
        const float expected = static_cast<float>(2 * i + 1) / n;
        EXPECT_NEAR(got[static_cast<size_t>(i) * 4], expected, 2e-3);
    }

    // Jitter of +0.5 render pixels equals +1 source texel at scale 2.
    RunDownsampleOnce(srcTex, dst, 0.5f, 0.0f, DownFilter::Point, false, 0.5f);
    got = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    for (int i = 0; i < n / 2 - 1; ++i) {
        const float expected = static_cast<float>(2 * i + 2) / n;
        EXPECT_NEAR(got[static_cast<size_t>(i) * 4], expected, 2e-3);
    }
}

static void Test_DownsampleSrgbEncode() {
    const int n = 16;
    const ImageF src = MakeImage(n, n, [](int, int, float* p) {
        p[0] = p[1] = p[2] = 0.5f;  // linear 0.5
        p[3] = 1.0f;
    });
    GpuTexture srcTex = MakeSourceTexture(src);
    LowResSet dst = CreateLowResSet(g_fix.gpu, n, n);

    RunDownsampleOnce(srcTex, dst, 0.0f, 0.0f, DownFilter::Point, true, 0.5f);
    const std::vector<float> got = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    // The shader's encode must agree with the CPU one, or readback would not
    // round-trip and every metric would be skewed.
    EXPECT_NEAR(got[0], LinearToSrgb(0.5f), 2e-3);

    RunDownsampleOnce(srcTex, dst, 0.0f, 0.0f, DownFilter::Point, false, 0.5f);
    const std::vector<float> raw = g_fix.gpu.ReadbackRgbaFloat(dst.color);
    EXPECT_NEAR(raw[0], 0.5f, 2e-3);
}

static void Test_DownsampleDepthAndMotion() {
    const int n = 24;
    const ImageF src = MakeDetailChart(n * 2, n * 2);
    GpuTexture srcTex = MakeSourceTexture(src);
    LowResSet dst = CreateLowResSet(g_fix.gpu, n, n);
    RunDownsampleOnce(srcTex, dst, 0.1f, -0.2f, DownFilter::Point, false, 0.375f);

    const std::vector<float> depth = g_fix.gpu.ReadbackR32Float(dst.depth);
    EXPECT_EQ(depth.size(), size_t(n) * n);
    for (float d : depth) EXPECT_NEAR(d, 0.375f, 1e-6);

    // Motion must be exactly zero: a still image has no movement, and any
    // nonzero value would make DLSS reproject its history away.
    const std::vector<float> motion = g_fix.gpu.ReadbackRg16Float(dst.motion);
    EXPECT_EQ(motion.size(), size_t(n) * n * 2);
    for (float m : motion) EXPECT_NEAR(m, 0.0f, 1e-9);
}

static void Test_DlssAvailable() {
    const DlssCaps caps = g_fix.ngx.QueryCaps();
    EXPECT_TRUE(caps.available);
    EXPECT_FALSE(caps.needsUpdatedDriver);

    std::string path, version;
    EXPECT_TRUE(NgxSession::FindLoadedDlssModule(&path, &version));
    EXPECT_FALSE(path.empty());
    EXPECT_TRUE(version != "unknown");
    printf("      dlss %s\n      %s\n", version.c_str(), path.c_str());
}

static void Test_OptimalSettings() {
    struct Expect {
        const char* name;
        NVSDK_NGX_PerfQuality_Value value;
        double ratio;  // targetW / renderW, 0 = do not check
    };
    const Expect cases[] = {
        {"DLAA", NVSDK_NGX_PerfQuality_Value_DLAA, 1.0},
        {"Quality", NVSDK_NGX_PerfQuality_Value_MaxQuality, 1.5},
        {"Balanced", NVSDK_NGX_PerfQuality_Value_Balanced, 0.0},
        {"Performance", NVSDK_NGX_PerfQuality_Value_MaxPerf, 2.0},
        {"UltraPerformance", NVSDK_NGX_PerfQuality_Value_UltraPerformance, 3.0},
    };
    for (const Expect& e : cases) {
        OptimalSettings s;
        EXPECT_NO_THROW(s = g_fix.ngx.GetOptimal(1920, 1080, e.value));
        EXPECT_TRUE(s.renderW > 0 && s.renderH > 0);
        EXPECT_TRUE(s.renderW <= 1920 && s.renderH <= 1080);
        if (e.ratio > 0.0) {
            EXPECT_NEAR(1920.0 / static_cast<double>(s.renderW), e.ratio, 0.02);
        }
    }
    // UltraQuality is not supported by this DLSS build; it must fail cleanly
    // rather than hand back a 0x0 render size the caller would then use.
    OptimalSettings uq;
    bool threw = false;
    try {
        uq = g_fix.ngx.GetOptimal(1920, 1080, NVSDK_NGX_PerfQuality_Value_UltraQuality);
    } catch (const ToolError&) {
        threw = true;
    }
    ++g_checks;
    if (!threw && (uq.renderW == 0 || uq.renderH == 0)) {
        Failed(__FILE__, __LINE__, "UltraQuality returned a 0x0 render size without throwing");
    }
}

static void Test_DlssFeatureLifecycle() {
    for (const QualityMode& q : AllQualityModes()) {
        OptimalSettings s;
        try {
            s = g_fix.ngx.GetOptimal(640, 360, q.value);
        } catch (const ToolError&) {
            continue;  // unsupported mode, covered by Test_OptimalSettings
        }
        DlssFeatureDesc d;
        d.renderW = s.renderW;
        d.renderH = s.renderH;
        d.targetW = 640;
        d.targetH = 360;
        d.quality = q.value;
        d.preset = 0;

        DlssFeature feature;
        EXPECT_NO_THROW({
            ID3D12GraphicsCommandList* cl = g_fix.gpu.Begin();
            feature.Create(g_fix.ngx, cl, d);
            g_fix.gpu.EndAndWait();
        });
        // Releasing twice must be safe — main() and the destructor both do it.
        EXPECT_NO_THROW(feature.Release());
        EXPECT_NO_THROW(feature.Release());
    }
}

// Runs the real accumulation loop and returns PSNR against the source.
static double AccumulatePsnr(const ImageF& source, GpuTexture& srcTex, GpuTexture& outTex,
                             LowResSet& lr, const DlssFeatureDesc& desc, const char* signText,
                             int frames) {
    JitterSign sign;
    CHECK(ParseJitterSign(signText, &sign));

    AccumulationParams p;
    p.source = &srcTex;
    p.output = &outTex;
    p.lowRes = &lr;
    p.feature = desc;
    p.sign = sign;
    p.frames = frames;
    p.phases = JitterPhaseCount(source.w, desc.renderW, 0);
    p.filter = DownFilter::Point;
    p.encodeSrgb = true;
    p.decodeSrgbOnReadback = true;

    const AccumulationResult r = RunAccumulation(g_fix.gpu, g_fix.ngx, p);
    return ComputeMetrics(source, r.output).psnrRgb;
}

static void Test_TemporalAccumulationConverges() {
    // The core behavioural claim of the tool: accumulating jittered samples must
    // measurably beat a single reset frame.
    const ImageF source = MakeDetailChart(512, 288);
    GpuTexture srcTex = MakeSourceTexture(source);
    GpuTexture outTex = CreateOutputTexture(g_fix.gpu, source.w, source.h);

    const OptimalSettings s =
        g_fix.ngx.GetOptimal(source.w, source.h, NVSDK_NGX_PerfQuality_Value_MaxQuality);
    LowResSet lr = CreateLowResSet(g_fix.gpu, s.renderW, s.renderH);

    DlssFeatureDesc d;
    d.renderW = s.renderW;
    d.renderH = s.renderH;
    d.targetW = static_cast<unsigned>(source.w);
    d.targetH = static_cast<unsigned>(source.h);
    d.quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    d.preset = 0;

    const double one = AccumulatePsnr(source, srcTex, outTex, lr, d, "--", 1);
    const double many = AccumulatePsnr(source, srcTex, outTex, lr, d, "--", 24);
    printf("      1 frame %.2f dB -> 24 frames %.2f dB\n", one, many);
    EXPECT_GT(many, one + 1.0);

    // And the wrong sign convention must be worse, which is what makes the
    // auto-probe in main() meaningful.
    const double wrong = AccumulatePsnr(source, srcTex, outTex, lr, d, "++", 24);
    printf("      wrong sign at 24 frames: %.2f dB\n", wrong);
    EXPECT_GT(many, wrong);
}

static void Test_PresetSelectionTakesEffect() {
    // "default" always resolves to one of the concrete presets. Which one varies
    // by DLSS version, so assert the mapping exists rather than naming a letter.
    const ImageF source = MakeDetailChart(512, 288);
    GpuTexture srcTex = MakeSourceTexture(source);
    GpuTexture outTex = CreateOutputTexture(g_fix.gpu, source.w, source.h);

    const OptimalSettings s =
        g_fix.ngx.GetOptimal(source.w, source.h, NVSDK_NGX_PerfQuality_Value_MaxQuality);
    LowResSet lr = CreateLowResSet(g_fix.gpu, s.renderW, s.renderH);

    auto hashFor = [&](unsigned preset) {
        DlssFeatureDesc d;
        d.renderW = s.renderW;
        d.renderH = s.renderH;
        d.targetW = static_cast<unsigned>(source.w);
        d.targetH = static_cast<unsigned>(source.h);
        d.quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
        d.preset = preset;

        JitterSign sign;
        ParseJitterSign("--", &sign);
        AccumulationParams p;
        p.source = &srcTex;
        p.output = &outTex;
        p.lowRes = &lr;
        p.feature = d;
        p.sign = sign;
        p.frames = 8;
        p.phases = JitterPhaseCount(source.w, s.renderW, 0);
        p.filter = DownFilter::Point;
        return HashImage(RunAccumulation(g_fix.gpu, g_fix.ngx, p).output);
    };

    const uint64_t base = hashFor(0);
    const unsigned candidates[] = {5, 6, 10, 11, 12, 13};  // E, F, J, K, L, M
    int matches = 0;
    int distinct = 0;
    for (unsigned c : candidates) {
        const uint64_t h = hashFor(c);
        if (h == base) ++matches;
        else ++distinct;
    }
    printf("      default matches %d preset(s), %d differ\n", matches, distinct);
    EXPECT_GT(matches, 0);   // default must map onto some concrete preset
    EXPECT_GT(distinct, 0);  // and the preset hint must actually change output
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    bool runGpu = true;
    std::string nameFilter;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-gpu") {
            runGpu = false;
        } else if (a == "--filter" && i + 1 < argc) {
            nameFilter = argv[++i];
        } else if (a == "-h" || a == "--help") {
            printf("Usage: video2dlssnr_tests [--no-gpu] [--filter <substring>]\n");
            return 0;
        } else {
            printf("unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }

    const std::vector<TestCase> tests = {
        {"SrgbRoundTrip", Test_SrgbRoundTrip, false},
        {"Halton", Test_Halton, false},
        {"JitterPhaseCount", Test_JitterPhaseCount, false},
        {"ParseJitterSign", Test_ParseJitterSign, false},
        {"ParsePreset", Test_ParsePreset, false},
        {"ParseQualityMode", Test_ParseQualityMode, false},
        {"ParseDownFilter", Test_ParseDownFilter, false},
        {"SplitList", Test_SplitList, false},
        {"ParseArgsDefaults", Test_ParseArgsDefaults, false},
        {"ParseArgsValues", Test_ParseArgsValues, false},
        {"ParseArgsErrors", Test_ParseArgsErrors, false},
        {"MetricsIdentity", Test_MetricsIdentity, false},
        {"MetricsKnownError", Test_MetricsKnownError, false},
        {"MetricsSizeMismatch", Test_MetricsSizeMismatch, false},
        {"HashImage", Test_HashImage, false},
        {"Png8RoundTrip", Test_Png8RoundTrip, false},
        {"Png16RoundTrip", Test_Png16RoundTrip, false},
        {"DiffImage", Test_DiffImage, false},

        {"GpuDevice", Test_GpuDevice, true},
        {"HalfRoundTrip", Test_HalfRoundTrip, true},
        {"DownsampleIdentity", Test_DownsampleIdentity, true},
        {"DownsampleScaleAndJitter", Test_DownsampleScaleAndJitter, true},
        {"DownsampleSrgbEncode", Test_DownsampleSrgbEncode, true},
        {"DownsampleDepthAndMotion", Test_DownsampleDepthAndMotion, true},
        {"DlssAvailable", Test_DlssAvailable, true},
        {"OptimalSettings", Test_OptimalSettings, true},
        {"DlssFeatureLifecycle", Test_DlssFeatureLifecycle, true},
        {"TemporalAccumulationConverges", Test_TemporalAccumulationConverges, true},
        {"PresetSelectionTakesEffect", Test_PresetSelectionTakesEffect, true},
    };

    int ran = 0, skipped = 0, failedCases = 0;
    bool gpuInitialised = false;

    for (const TestCase& t : tests) {
        if (!nameFilter.empty() && std::string(t.name).find(nameFilter) == std::string::npos) {
            continue;
        }
        if (t.needsGpu && !runGpu) {
            ++skipped;
            continue;
        }
        if (t.needsGpu && !gpuInitialised) {
            printf("\n--- initialising GPU + NGX ---\n");
            try {
                g_fix.Init();
                gpuInitialised = true;
            } catch (const std::exception& e) {
                printf("  GPU/NGX unavailable, skipping GPU tests: %s\n", e.what());
                runGpu = false;
                ++skipped;
                continue;
            }
        }

        printf("  %-32s", t.name);
        g_caseFailures = 0;
        const int before = g_failures;
        try {
            t.fn();
        } catch (const std::exception& e) {
            Failed(__FILE__, __LINE__, std::string("threw: ") + e.what());
        }
        ++ran;
        if (g_failures == before) {
            printf(" ok\n");
        } else {
            printf(" FAILED\n");
            ++failedCases;
        }
    }

    if (gpuInitialised) g_fix.Shutdown();

    printf("\n%d test(s) run, %d skipped, %d assertion(s), %d failure(s) in %d case(s)\n", ran,
           skipped, g_checks, g_failures, failedCases);
    return g_failures == 0 ? 0 : 1;
}
