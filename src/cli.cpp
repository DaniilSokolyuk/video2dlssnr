#include "cli.h"

#include <algorithm>
#include <cmath>

void PrintUsage() {
    printf(
        "video2dlssnr - DLSS Super Resolution for still images\n"
        "\n"
        "Usage: video2dlssnr --in <image> [options]\n"
        "\n"
        "  --in <file>          Source image (PNG/JPG/BMP/TGA). Treated as ground truth.\n"
        "  --out <dir>          Output directory (default: out)\n"
        "  --quality <list>     Comma list of quality modes (default: quality)\n"
        "                       dlaa, ultraquality, quality, balanced, performance,\n"
        "                       ultraperformance, or 'all'\n"
        "  --preset <list>      Comma list of presets A..O or 'default' (default: default,J,K)\n"
        "                       'all' expands to default,E,F,J,K,L,M\n"
        "  --frames <n>         Accumulation passes per run (default: 32)\n"
        "  --phases <n>         Jitter sequence length (default: 8 * upscale^2)\n"
        "  --filter <mode>      Downsample filter: point, bilinear, tent, lanczos\n"
        "                       (default: point). Point is the faithful emulation of\n"
        "                       1-sample-per-pixel rasterisation and reconstructs far\n"
        "                       better; the wider kernels pre-filter away the subpixel\n"
        "                       detail DLSS needs, and exist for comparison.\n"
        "  --jitter-sign <s>    auto, ++, +-, -+, -- (default: auto, probes for the best)\n"
        "  --hdr                Feed linear colour and set the DLSS IsHDR flag\n"
        "  --filter-space <s>   linear (default) or display. Linear is what a real\n"
        "                       renderer does; on high-contrast detail it correctly\n"
        "                       reads brighter than the source. Use display to keep\n"
        "                       the source's own sRGB levels.\n"
        "  --no-auto-exposure   Supply a constant exposure texture instead. Only has\n"
        "                       an effect together with --hdr.\n"
        "  --alpha              Enable DLSS alpha upscaling\n"
        "  --depth <v>          Constant depth written to the depth input (default: 0.5)\n"
        "  --png16              Write 16-bit PNGs instead of 8-bit\n"
        "  --save-lr            Also write the final low-res input frame per run\n"
        "  --no-diff            Skip the <name>_diff.png error map (written by default)\n"
        "  --diff-gain <f>      Error-map amplification (default: 8)\n"
        "  --metrics-only       Measure but write no images. For batch sweeps, where\n"
        "                       4K PNGs would cost gigabytes.\n"
        "  --dll-dir <dir>      Directory to load nvngx_dlss.dll from\n"
        "  --adapter <i>        DXGI adapter index (default: highest performance)\n"
        "  --debug-layer        Enable the D3D12 debug layer\n"
        "  --json <file>        Write results as JSON (default: <out>/results.json)\n"
        "\n"
        " DLSS Neural Rendering (NGX feature 18, undocumented):\n"
        "  --probe-nr           Try to create the NR feature and exit. Needs no image.\n"
        "  --nr-in <WxH>        Probe input size (default: 1920x1080)\n"
        "  --nr-out <WxH>       Probe output size (default: 3840x2160)\n"
        "  --nr-preset <n>      DLSSNR render preset hint (default: 0)\n"
        "\n"
        "  -v, --verbose        Verbose logging, including NGX messages\n"
        "  -h, --help           This message\n");
}

static void ParseSize(const std::string& v, unsigned* w, unsigned* h) {
    const size_t x = v.find_first_of("xX*");
    if (x == std::string::npos) throw ToolError("expected WxH, got '" + v + "'");
    try {
        *w = static_cast<unsigned>(std::stoul(v.substr(0, x)));
        *h = static_cast<unsigned>(std::stoul(v.substr(x + 1)));
    } catch (const std::exception&) {
        throw ToolError("expected WxH, got '" + v + "'");
    }
    if (*w == 0 || *h == 0) throw ToolError("size must be non-zero: '" + v + "'");
}

std::vector<std::string> SplitList(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ';') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

Options ParseArgs(int argc, char** argv, bool* wantHelp) {
    Options o;
    *wantHelp = false;

    auto need = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) throw ToolError(std::string("missing value after ") + flag);
        return argv[++i];
    };
    auto toInt = [](const std::string& v, const char* flag) {
        try {
            size_t used = 0;
            const int n = std::stoi(v, &used);
            if (used != v.size()) throw std::invalid_argument("trailing");
            return n;
        } catch (const std::exception&) {
            throw ToolError(std::string(flag) + " expects an integer, got '" + v + "'");
        }
    };
    auto toFloat = [](const std::string& v, const char* flag) {
        try {
            size_t used = 0;
            const float f = std::stof(v, &used);
            if (used != v.size()) throw std::invalid_argument("trailing");
            return f;
        } catch (const std::exception&) {
            throw ToolError(std::string(flag) + " expects a number, got '" + v + "'");
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            *wantHelp = true;
            return o;
        } else if (a == "--in") {
            o.input = need(i, "--in");
        } else if (a == "--out") {
            o.outDir = need(i, "--out");
        } else if (a == "--quality") {
            o.qualities = SplitList(need(i, "--quality"));
        } else if (a == "--preset") {
            o.presets = SplitList(need(i, "--preset"));
        } else if (a == "--frames") {
            o.frames = toInt(need(i, "--frames"), "--frames");
        } else if (a == "--phases") {
            o.phases = toInt(need(i, "--phases"), "--phases");
        } else if (a == "--filter") {
            const std::string v = need(i, "--filter");
            if (!ParseDownFilter(v, &o.filter)) throw ToolError("unknown filter '" + v + "'");
        } else if (a == "--jitter-sign") {
            o.jitterSign = need(i, "--jitter-sign");
        } else if (a == "--hdr") {
            o.hdr = true;
        } else if (a == "--filter-space") {
            const std::string v = need(i, "--filter-space");
            if (v == "display") {
                o.displaySpace = true;
            } else if (v == "linear") {
                o.displaySpace = false;
            } else {
                throw ToolError("--filter-space must be linear or display");
            }
        } else if (a == "--no-auto-exposure") {
            o.autoExposure = false;
        } else if (a == "--alpha") {
            o.alphaUpscaling = true;
        } else if (a == "--depth") {
            o.depthValue = toFloat(need(i, "--depth"), "--depth");
        } else if (a == "--png16") {
            o.png16 = true;
        } else if (a == "--save-lr") {
            o.saveLowRes = true;
        } else if (a == "--no-diff") {
            o.saveDiff = false;
        } else if (a == "--metrics-only") {
            o.metricsOnly = true;
            o.saveDiff = false;
        } else if (a == "--diff-gain") {
            o.diffGain = toFloat(need(i, "--diff-gain"), "--diff-gain");
        } else if (a == "--dll-dir") {
            o.dllDir = need(i, "--dll-dir");
        } else if (a == "--adapter") {
            o.adapter = toInt(need(i, "--adapter"), "--adapter");
        } else if (a == "--debug-layer") {
            o.debugLayer = true;
        } else if (a == "--json") {
            o.jsonPath = need(i, "--json");
        } else if (a == "--probe-nr") {
            o.probeNr = true;
        } else if (a == "--probe-sl") {
            o.probeSl = true;
        } else if (a == "--nr-run") {
            o.nrRun = true;
        } else if (a == "--nr-video") {
            o.nrVideo = true;
        } else if (a == "--nr-intensity") {
            o.nrIntensity = toFloat(need(i, "--nr-intensity"), "--nr-intensity");
        } else if (a == "--nr-style") {
            o.nrStyle = static_cast<unsigned>(toInt(need(i, "--nr-style"), "--nr-style"));
        } else if (a == "--nr-local-structure") {
            o.nrLocalStructure = toFloat(need(i, "--nr-local-structure"), "--nr-local-structure");
        } else if (a == "--nr-local-tone") {
            o.nrLocalTone = toFloat(need(i, "--nr-local-tone"), "--nr-local-tone");
        } else if (a == "--nr-skin") {
            o.nrSkin = toFloat(need(i, "--nr-skin"), "--nr-skin");
        } else if (a == "--nr-global-tone") {
            o.nrGlobalTone = toFloat(need(i, "--nr-global-tone"), "--nr-global-tone");
        } else if (a == "--nr-auto-mask") {
            o.nrAutoMask = true;
        } else if (a == "--nr-ui-correction") {
            o.nrUiCorrection = toInt(need(i, "--nr-ui-correction"), "--nr-ui-correction") != 0;
        } else if (a == "--nr-detail") {
            o.nrDetail = toFloat(need(i, "--nr-detail"), "--nr-detail");
        } else if (a == "--nr-color" || a == "--nr-colour") {
            o.nrColour = toFloat(need(i, "--nr-color"), "--nr-color");
        } else if (a == "--nr-hdr") {
            o.nrHdr = true;
        } else if (a == "--nr-diff") {
            o.nrDiff = true;
        } else if (a == "--nr-orig") {
            o.nrOrig = true;
        } else if (a == "--nr-motion") {
            o.nrMotion = toInt(need(i, "--nr-motion"), "--nr-motion") != 0;
        } else if (a == "--nr-motion-vis") {
            o.nrMotionVis = true;
        } else if (a == "--nr-motion-engine") {
            const std::string e = need(i, "--nr-motion-engine");
            if (e == "auto") o.nrMotionEngine = 0;
            else if (e == "nvof") o.nrMotionEngine = 1;
            else if (e == "lk") o.nrMotionEngine = 2;
            else throw ToolError("--nr-motion-engine must be auto, nvof or lk");
        } else if (a == "--nr-scale") {
            o.nrScale = toFloat(need(i, "--nr-scale"), "--nr-scale");
        } else if (a == "--nr-width") {
            o.nrTargetW = static_cast<unsigned>(toInt(need(i, "--nr-width"), "--nr-width"));
        } else if (a == "--nr-height") {
            o.nrTargetH = static_cast<unsigned>(toInt(need(i, "--nr-height"), "--nr-height"));
        } else if (a == "--sl-feature") {
            o.slFeature = static_cast<unsigned>(toInt(need(i, "--sl-feature"), "--sl-feature"));
        } else if (a == "--nr-in") {
            ParseSize(need(i, "--nr-in"), &o.nrInW, &o.nrInH);
        } else if (a == "--nr-out") {
            ParseSize(need(i, "--nr-out"), &o.nrOutW, &o.nrOutH);
        } else if (a == "--nr-preset") {
            o.nrPreset = static_cast<unsigned>(toInt(need(i, "--nr-preset"), "--nr-preset"));
        } else if (a == "-v" || a == "--verbose") {
            o.verbose = true;
        } else {
            throw ToolError("unknown argument '" + a + "' (try --help)");
        }
    }

    if (o.probeNr || o.probeSl || o.nrVideo) return o;  // these need no --in image
    if (o.input.empty()) throw ToolError("--in is required (try --help)");
    if (o.frames < 1) throw ToolError("--frames must be at least 1");
    if (o.phases < 0) throw ToolError("--phases must not be negative");
    if (o.diffGain <= 0.0f) throw ToolError("--diff-gain must be positive");
    if (o.qualities.empty()) throw ToolError("--quality needs at least one mode");
    if (o.presets.empty()) throw ToolError("--preset needs at least one preset");
    if (o.hdr && o.displaySpace) {
        throw ToolError("--hdr and --filter-space display are mutually exclusive: HDR mode "
                        "requires linear colour");
    }
    if (o.jitterSign != "auto") {
        JitterSign probe;
        if (!ParseJitterSign(o.jitterSign, &probe)) {
            throw ToolError("--jitter-sign must be auto, ++, +-, -+ or --");
        }
    }
    return o;
}

double Halton(int index, int base) {
    double f = 1.0, r = 0.0;
    while (index > 0) {
        f /= base;
        r += f * (index % base);
        index /= base;
    }
    return r;
}

bool ParseJitterSign(const std::string& s, JitterSign* out) {
    if (s.size() != 2) return false;
    if ((s[0] != '+' && s[0] != '-') || (s[1] != '+' && s[1] != '-')) return false;
    out->x = (s[0] == '+') ? 1.0f : -1.0f;
    out->y = (s[1] == '+') ? 1.0f : -1.0f;
    out->name = s;
    return true;
}

int JitterPhaseCount(int sourceWidth, unsigned renderWidth, int overrideValue) {
    if (overrideValue > 0) return overrideValue;
    if (renderWidth == 0) return 8;
    const double scale = static_cast<double>(sourceWidth) / static_cast<double>(renderWidth);
    return std::max(8, static_cast<int>(std::lround(8.0 * scale * scale)));
}
