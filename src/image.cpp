#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>

float SrgbToLinear(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float c) {
    if (c <= 0.0f) return 0.0f;
    if (c <= 0.0031308f) return c * 12.92f;
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// A 256-entry LUT is exact for 8-bit input and removes a pow() per channel.
static const float* Srgb8Lut() {
    static float lut[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) lut[i] = SrgbToLinear(i / 255.0f);
        init = true;
    }
    return lut;
}

ImageF LoadImageLinear(const std::string& path) {
    int w = 0, h = 0, comp = 0;
    if (!stbi_info(path.c_str(), &w, &h, &comp)) {
        throw ToolError("cannot read image '" + path + "': " + stbi_failure_reason());
    }

    ImageF img;
    if (stbi_is_16_bit(path.c_str())) {
        stbi_us* data = stbi_load_16(path.c_str(), &w, &h, &comp, 4);
        if (!data) throw ToolError("cannot decode '" + path + "': " + stbi_failure_reason());
        img.w = w;
        img.h = h;
        img.px.resize(img.pixels() * 4);
        for (size_t i = 0; i < img.pixels(); ++i) {
            for (int c = 0; c < 3; ++c) img.px[i * 4 + c] = SrgbToLinear(data[i * 4 + c] / 65535.0f);
            img.px[i * 4 + 3] = data[i * 4 + 3] / 65535.0f;
        }
        stbi_image_free(data);
    } else {
        stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 4);
        if (!data) throw ToolError("cannot decode '" + path + "': " + stbi_failure_reason());
        const float* lut = Srgb8Lut();
        img.w = w;
        img.h = h;
        img.px.resize(img.pixels() * 4);
        for (size_t i = 0; i < img.pixels(); ++i) {
            for (int c = 0; c < 3; ++c) img.px[i * 4 + c] = lut[data[i * 4 + c]];
            img.px[i * 4 + 3] = data[i * 4 + 3] / 255.0f;
        }
        stbi_image_free(data);
    }
    return img;
}

static std::vector<uint8_t> EncodeSrgb8(const ImageF& img) {
    std::vector<uint8_t> out(img.pixels() * 4);
    for (size_t i = 0; i < img.pixels(); ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = LinearToSrgb(img.px[i * 4 + c]);
            out[i * 4 + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
        out[i * 4 + 3] =
            static_cast<uint8_t>(std::clamp(img.px[i * 4 + 3], 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    return out;
}

void SavePng8(const std::string& path, const ImageF& img) {
    if (img.empty()) throw ToolError("SavePng8: empty image");
    const std::vector<uint8_t> bytes = EncodeSrgb8(img);
    if (!stbi_write_png(path.c_str(), img.w, img.h, 4, bytes.data(), img.w * 4)) {
        throw ToolError("cannot write '" + path + "'");
    }
}

// ---------------------------------------------------------------------------
// Minimal 16-bit PNG writer. stb_image_write only does 8-bit, and 8 bits clips
// the shadow detail that shows up when comparing presets, so emit our own.
// Stored (uncompressed) deflate keeps this dependency-free; files are large but
// this path is opt-in via --png16.
// ---------------------------------------------------------------------------

static const uint32_t* Crc32Table() {
    static uint32_t t[256];
    static bool init = false;
    if (!init) {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[n] = c;
        }
        init = true;
    }
    return t;
}

static uint32_t Crc32(const uint8_t* p, size_t n) {
    const uint32_t* t = Crc32Table();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = t[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static void PushBE32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

static void PushChunk(std::vector<uint8_t>& file, const char* tag,
                      const std::vector<uint8_t>& data) {
    PushBE32(file, static_cast<uint32_t>(data.size()));
    std::vector<uint8_t> body(tag, tag + 4);
    body.insert(body.end(), data.begin(), data.end());
    file.insert(file.end(), body.begin(), body.end());
    PushBE32(file, Crc32(body.data(), body.size()));
}

void SavePng16(const std::string& path, const ImageF& img) {
    if (img.empty()) throw ToolError("SavePng16: empty image");
    const int w = img.w, h = img.h;

    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (1 + static_cast<size_t>(w) * 8));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);  // per-row filter type: None
        for (int x = 0; x < w; ++x) {
            const size_t i = (static_cast<size_t>(y) * w + x) * 4;
            for (int c = 0; c < 4; ++c) {
                const float v = (c < 3) ? LinearToSrgb(img.px[i + c]) : img.px[i + c];
                const uint16_t s =
                    static_cast<uint16_t>(std::clamp(v, 0.0f, 1.0f) * 65535.0f + 0.5f);
                raw.push_back(static_cast<uint8_t>(s >> 8));
                raw.push_back(static_cast<uint8_t>(s & 0xFFu));
            }
        }
    }

    std::vector<uint8_t> file;
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    file.insert(file.end(), sig, sig + 8);

    std::vector<uint8_t> ihdr;
    PushBE32(ihdr, static_cast<uint32_t>(w));
    PushBE32(ihdr, static_cast<uint32_t>(h));
    ihdr.push_back(16);  // bit depth
    ihdr.push_back(6);   // colour type: truecolour + alpha
    ihdr.push_back(0);   // deflate
    ihdr.push_back(0);   // adaptive filtering
    ihdr.push_back(0);   // no interlace
    PushChunk(file, "IHDR", ihdr);

    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t off = 0;
    do {
        const size_t n = std::min<size_t>(65535, raw.size() - off);
        const bool last = (off + n >= raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFFu));
        z.push_back(static_cast<uint8_t>((n >> 8) & 0xFFu));
        const uint16_t nn = static_cast<uint16_t>(~n);
        z.push_back(static_cast<uint8_t>(nn & 0xFFu));
        z.push_back(static_cast<uint8_t>((nn >> 8) & 0xFFu));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    } while (off < raw.size());

    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    PushBE32(z, (b << 16) | a);
    PushChunk(file, "IDAT", z);
    PushChunk(file, "IEND", {});

    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) throw ToolError("cannot write '" + path + "'");
    fwrite(file.data(), 1, file.size(), f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Error visualisation
// ---------------------------------------------------------------------------

// Black -> blue -> green -> yellow -> red. Deliberately starts at black so that
// "no error" reads as empty and the eye is drawn only to real differences.
static void HeatColor(float t, float* rgb) {
    static const float stops[5][3] = {
        {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
    };
    t = std::clamp(t, 0.0f, 1.0f) * 4.0f;
    const int i = std::min(3, static_cast<int>(t));
    const float f = t - static_cast<float>(i);
    for (int c = 0; c < 3; ++c) rgb[c] = stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f;
}

DiffStats SaveDiffPng(const std::string& path, const ImageF& reference, const ImageF& test,
                      float gain) {
    if (reference.w != test.w || reference.h != test.h) {
        throw ToolError("SaveDiffPng: size mismatch");
    }
    if (gain <= 0.0f) throw ToolError("SaveDiffPng: gain must be positive");

    DiffStats stats;
    ImageF vis;
    vis.w = reference.w;
    vis.h = reference.h;
    vis.px.resize(vis.pixels() * 4);

    double total = 0.0;
    for (size_t i = 0; i < reference.pixels(); ++i) {
        float worst = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float a = std::clamp(LinearToSrgb(reference.px[i * 4 + c]), 0.0f, 1.0f);
            const float b = std::clamp(LinearToSrgb(test.px[i * 4 + c]), 0.0f, 1.0f);
            worst = std::max(worst, std::abs(a - b));
        }
        total += worst;
        if (worst > stats.maxAbs) {
            stats.maxAbs = worst;
            stats.maxX = static_cast<int>(i % static_cast<size_t>(reference.w));
            stats.maxY = static_cast<int>(i / static_cast<size_t>(reference.w));
        }

        float rgb[3];
        HeatColor(worst * gain, rgb);
        // Store linear so SavePng8's encode reproduces the ramp colour exactly.
        for (int c = 0; c < 3; ++c) vis.px[i * 4 + c] = SrgbToLinear(rgb[c]);
        vis.px[i * 4 + 3] = 1.0f;
    }
    stats.meanAbs = total / static_cast<double>(reference.pixels());

    if (!path.empty()) SavePng8(path, vis);
    return stats;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

static std::vector<float> SrgbLuma(const ImageF& img) {
    std::vector<float> y(img.pixels());
    for (size_t i = 0; i < img.pixels(); ++i) {
        const float r = LinearToSrgb(img.px[i * 4 + 0]);
        const float g = LinearToSrgb(img.px[i * 4 + 1]);
        const float b = LinearToSrgb(img.px[i * 4 + 2]);
        y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    return y;
}

// Separable 11-tap Gaussian, sigma 1.5 — the window the SSIM paper specifies.
static void GaussBlur(const std::vector<float>& src, std::vector<float>& dst, int w, int h) {
    static float k[11];
    static bool init = false;
    if (!init) {
        float sum = 0.0f;
        for (int i = 0; i < 11; ++i) {
            const float x = static_cast<float>(i - 5);
            k[i] = std::exp(-(x * x) / (2.0f * 1.5f * 1.5f));
            sum += k[i];
        }
        for (float& v : k) v /= sum;
        init = true;
    }

    std::vector<float> tmp(src.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = 0; i < 11; ++i) {
                const int sx = std::clamp(x + i - 5, 0, w - 1);
                acc += k[i] * src[static_cast<size_t>(y) * w + sx];
            }
            tmp[static_cast<size_t>(y) * w + x] = acc;
        }
    }
    dst.resize(src.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = 0; i < 11; ++i) {
                const int sy = std::clamp(y + i - 5, 0, h - 1);
                acc += k[i] * tmp[static_cast<size_t>(sy) * w + x];
            }
            dst[static_cast<size_t>(y) * w + x] = acc;
        }
    }
}

Metrics ComputeMetrics(const ImageF& reference, const ImageF& test) {
    if (reference.w != test.w || reference.h != test.h) {
        throw ToolError("ComputeMetrics: size mismatch " + std::to_string(reference.w) + "x" +
                        std::to_string(reference.h) + " vs " + std::to_string(test.w) + "x" +
                        std::to_string(test.h));
    }
    Metrics m;

    double mse = 0.0;
    for (size_t i = 0; i < reference.pixels(); ++i) {
        for (int c = 0; c < 3; ++c) {
            const double a = std::clamp(LinearToSrgb(reference.px[i * 4 + c]), 0.0f, 1.0f);
            const double b = std::clamp(LinearToSrgb(test.px[i * 4 + c]), 0.0f, 1.0f);
            const double d = a - b;
            mse += d * d;
        }
    }
    mse /= static_cast<double>(reference.pixels() * 3);
    m.psnrRgb = (mse <= 1e-20) ? 99.0 : 10.0 * std::log10(1.0 / mse);

    const int w = reference.w, h = reference.h;
    std::vector<float> x = SrgbLuma(reference), y = SrgbLuma(test);
    std::vector<float> xx(x.size()), yy(x.size()), xy(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xx[i] = x[i] * x[i];
        yy[i] = y[i] * y[i];
        xy[i] = x[i] * y[i];
    }
    std::vector<float> mx, my, mxx, myy, mxy;
    GaussBlur(x, mx, w, h);
    GaussBlur(y, my, w, h);
    GaussBlur(xx, mxx, w, h);
    GaussBlur(yy, myy, w, h);
    GaussBlur(xy, mxy, w, h);

    const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
    double acc = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double ux = mx[i], uy = my[i];
        const double vx = static_cast<double>(mxx[i]) - ux * ux;
        const double vy = static_cast<double>(myy[i]) - uy * uy;
        const double vxy = static_cast<double>(mxy[i]) - ux * uy;
        acc += ((2 * ux * uy + C1) * (2 * vxy + C2)) /
               ((ux * ux + uy * uy + C1) * (vx + vy + C2));
    }
    m.ssimLuma = acc / static_cast<double>(x.size());
    return m;
}

uint64_t HashImage(const ImageF& img) {
    const std::vector<uint8_t> bytes = EncodeSrgb8(img);
    uint64_t h = 1469598103934665603ull;  // FNV-1a
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}
