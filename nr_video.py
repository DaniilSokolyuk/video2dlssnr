"""DLSS 5 video: Super Resolution + Neural Rendering over a whole clip, streaming.

This is the one entry point for video - it drives the optimised, fully-on-GPU flow:

    ffmpeg (decode) --raw rgba--> video2dlssnr --nr-video --raw rgba--> ffmpeg (encode + audio)

The heavy work (upscale, sRGB encode, neural rendering, composite, 8-bit pack) all happens on
the GPU inside video2dlssnr; nothing is shuttled back to the CPU between DLSS and NR. ffmpeg only
decodes and encodes. A live progress bar reports the running fps (parsed from the tool's NRPROG
lines) with an ETA.

    python nr_video.py --in clip.mp4 --out clip_4k.mp4 --nr-width 3840 --nr-style 2

Output quality is constant-quality by default (--cq 19, 10-bit HEVC, colour tagged bt709), so the
encoder is never the weak link: the old fixed ~2 Mbit/s NVENC default made every 4K result blocky.
The container comes from the --out extension (.mp4 / .mkv / .mov / .webm), the codec from --codec.
The NR / scale flags are named exactly like video2dlssnr.exe (--nr-*).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time


# ----------------------------------------------------------------------------- encoders

# name -> (ffmpeg encoder, family). Families: nvenc (GPU), sw (CPU), prores, ffv1 (lossless).
CODECS = {
    "hevc_nvenc": ("hevc_nvenc", "nvenc"),
    "h264_nvenc": ("h264_nvenc", "nvenc"),
    "av1_nvenc": ("av1_nvenc", "nvenc"),
    "av1_svt": ("libsvtav1", "sw"),
    "prores": ("prores_ks", "prores"),
    "ffv1": ("ffv1", "ffv1"),
}
CODEC_ALIASES = {
    "hevc": "hevc_nvenc", "h265": "hevc_nvenc", "h.265": "hevc_nvenc", "x265": "hevc_nvenc",
    "h264": "h264_nvenc", "h.264": "h264_nvenc", "avc": "h264_nvenc", "x264": "h264_nvenc",
    "av1": "av1_nvenc", "libsvtav1": "av1_svt", "svtav1": "av1_svt", "svt-av1": "av1_svt",
    "prores_ks": "prores", "lossless": "ffv1",
}
# Which codecs each container can legally hold (what mainstream players will actually open).
CONTAINERS = {
    ".mp4": {"hevc_nvenc", "h264_nvenc", "av1_nvenc", "av1_svt"},
    ".mov": {"hevc_nvenc", "h264_nvenc", "prores"},
    ".mkv": set(CODECS),
    ".webm": {"av1_nvenc", "av1_svt"},
}
# Audio codecs a container can carry unchanged (else the track is re-encoded).
COPYABLE_AUDIO = {
    ".mp4": {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"},
    ".mov": {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_s24be"},
    ".mkv": None,  # anything
    ".webm": {"opus", "vorbis"},
}
PRORES_PROFILES = ["proxy", "lt", "standard", "hq", "4444", "4444xq"]
# NVENC hardware limits (Ada / Blackwell): H.264 stops at 4096x4096, HEVC / AV1 at 8192x8192.
NVENC_MAX = {"h264_nvenc": 4096, "hevc_nvenc": 8192, "av1_nvenc": 8192}
# Chroma / scaling flags for every swscale conversion: proper 4:2:0 <-> RGB resampling instead of
# the default nearest-neighbour chroma, and lanczos for any size change.
SWS = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"


def find_tool(name):
    # Prefer a copy bundled next to the tool (out\), then PATH, then a winget install.
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "out", name + ".exe"), os.path.join(here, name + ".exe")):
        if os.path.isfile(cand):
            return cand
    p = shutil.which(name)
    if p:
        return p
    root = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    for hit in glob.glob(os.path.join(root, "Gyan.FFmpeg*", "**", name + ".exe"), recursive=True):
        return hit
    return None


def probe(ffprobe, path):
    """Video geometry / rate / colour tags and the first audio codec, via ffprobe JSON."""
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", path], capture_output=True, text=True).stdout
    info = json.loads(out or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        sys.exit(f"no video stream in {path}")
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    rate = v.get("r_frame_rate") or v.get("avg_frame_rate") or "25/1"
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    frames = int(v.get("nb_frames") or 0)
    if frames <= 0:  # some containers don't store a frame count; estimate from duration
        try:
            dur = float(v.get("duration") or info.get("format", {}).get("duration") or 0)
            frames = round(dur * fps)
        except ValueError:
            frames = 0
    unk = lambda s: (s or "unknown") if s not in ("", None, "unspecified", "reserved") else "unknown"
    return {
        "w": int(v["width"]), "h": int(v["height"]), "fps": fps, "rate": rate, "frames": frames,
        "pix_fmt": v.get("pix_fmt", ""), "range": unk(v.get("color_range")),
        "matrix": unk(v.get("color_space")), "primaries": unk(v.get("color_primaries")),
        "trc": unk(v.get("color_transfer")),
        "audio": a.get("codec_name") if a else None,
    }


def colour_tags(src):
    """The (matrix, primaries, trc) the clip is in. Unknown tags default to bt709 (SD: bt601).

    Known tags are carried through so an HDR (bt2020 / PQ) clip keeps its signalling: the NR
    model then works on the PQ-coded signal, which is what the other DLSS 5 video tools do too.
    """
    sd = src["h"] <= 576 and src["w"] <= 1024
    # Only Y'CbCr matrices can be carried into a YUV output; an RGB source reports "gbr".
    yuv_matrices = {"bt709", "bt470bg", "smpte170m", "smpte240m", "bt2020nc", "bt2020c", "fcc"}
    m = src["matrix"] if src["matrix"] in yuv_matrices else ("smpte170m" if sd else "bt709")
    p = src["primaries"] if src["primaries"] != "unknown" else ("smpte170m" if sd else "bt709")
    t = src["trc"] if src["trc"] != "unknown" else ("smpte170m" if sd else "bt709")
    return m, p, t


def even(v):
    v = int(round(v))
    return max(2, v - v % 2)


def output_size(src, width, height, scale, fit=""):
    """(outW, outH): --nr-fit fits inside a WxH box (turned to match a portrait source), aspect kept;
    width and height together pin the exact size; one of them keeps the aspect; else scale."""
    w, h = src["w"], src["h"]
    if fit:
        bw, bh = (int(x) for x in fit.lower().split("x"))
        if (h > w) != (bh > bw):
            bw, bh = bh, bw
        f = min(bw / w, bh / h)
        ow, oh = w * f, h * f
    elif width and height:
        ow, oh = width, height
    elif width:
        ow, oh = width, h * width / w
    elif height:
        ow, oh = w * height / h, height
    else:
        ow, oh = w * scale, h * scale
    return even(ow), even(oh)


def tool_size(src, outW, outH):
    """What the tool renders at. DLSS only enlarges, so any shrink happens in ffmpeg afterwards:
    f = the larger of the two ratios; f <= 1 means NR at native size, then ffmpeg downsizes."""
    f = max(outW / src["w"], outH / src["h"])
    if f <= 1.0:
        return src["w"], src["h"]
    if (even(src["w"] * f), even(src["h"] * f)) == (outW, outH):
        return outW, outH
    return even(src["w"] * f), even(src["h"] * f)


def resolve_codec(name):
    key = CODEC_ALIASES.get(name.lower(), name.lower())
    if key not in CODECS:
        sys.exit(f"unknown --codec {name!r}; one of: {', '.join(CODECS)} (aliases: hevc, h264, av1, "
                 f"svtav1, lossless)")
    return key


def audio_args(src, ext, mode, kbps):
    """-map / -c:a for the audio track. auto = copy when the container allows it, else re-encode."""
    if mode == "none" or not src["audio"]:
        return ["-an"]
    args = ["-map", "1:a:0?"]
    if mode == "auto":
        ok = COPYABLE_AUDIO[ext]
        mode = "copy" if (ok is None or src["audio"] in ok) else ("opus" if ext == ".webm" else "aac")
    if mode == "copy":
        return args + ["-c:a", "copy"]
    if mode == "aac":
        return args + ["-c:a", "aac", "-b:a", f"{kbps}k"]
    if mode == "opus":
        return args + ["-c:a", "libopus", "-b:a", f"{kbps}k"]
    if mode == "flac":
        return args + ["-c:a", "flac"]
    if mode == "pcm":
        return args + ["-c:a", "pcm_s24le"]
    sys.exit(f"unknown --audio {mode!r}")


def video_args(codec, args, outW, outH, tags):
    """Encoder flags + the pixel format the filter chain must deliver."""
    enc, family = CODECS[codec]
    ten = args.bit_depth == 10
    matrix, primaries, trc = tags
    colour = ["-colorspace", matrix, "-color_primaries", primaries, "-color_trc", trc,
              "-color_range", "tv"]
    if family == "nvenc":
        lim = NVENC_MAX[enc]
        if outW > lim or outH > lim:
            sys.exit(f"{codec} cannot encode {outW}x{outH} (hardware limit {lim}); use hevc_nvenc / "
                     f"av1_nvenc for above-4K, or a smaller output")
        if enc == "h264_nvenc" and ten:
            print("h264_nvenc is 8-bit only; encoding 8-bit (use hevc_nvenc / av1_nvenc for 10-bit)",
                  file=sys.stderr)
            ten = False
        pix = "p010le" if ten else "yuv420p"
        v = ["-c:v", enc, "-preset", args.enc_preset, "-tune", "hq", "-multipass", args.multipass,
             "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8", "-rc-lookahead", "32",
             "-bf", "3", "-b_ref_mode", "middle"]
        if args.bitrate > 0:  # explicit target bitrate (VBR, 2x headroom for peaks)
            v += ["-rc", "vbr", "-b:v", f"{args.bitrate}k", "-maxrate", f"{2 * args.bitrate}k",
                  "-bufsize", f"{4 * args.bitrate}k"]
        else:  # constant quality: bits follow the content, the number is the quality target
            v += ["-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
        if enc == "hevc_nvenc":
            v += ["-profile:v", "main10" if ten else "main", "-tier", "high"]
        elif enc == "h264_nvenc":
            v += ["-profile:v", "high"]
        return v + colour, pix
    if family == "sw":  # libsvtav1
        pix = "yuv420p10le" if ten else "yuv420p"
        v = ["-c:v", enc, "-preset", str(args.sw_preset), "-svtav1-params", "tune=0"]
        v += ["-b:v", f"{args.bitrate}k"] if args.bitrate > 0 else ["-crf", str(args.cq)]
        return v + colour, pix
    if family == "prores":
        prof = args.prores_profile
        pix = "yuv444p10le" if prof.startswith("4444") else "yuv422p10le"
        return ["-c:v", enc, "-profile:v", prof, "-vendor", "apl0"] + colour, pix
    if family == "ffv1":  # mathematically lossless, RGB exactly as the tool produced it
        return ["-c:v", enc, "-level", "3", "-coder", "1", "-context", "1", "-g", "1",
                "-slices", "16", "-slicecrc", "1"], "bgr0"
    raise AssertionError(family)


# ----------------------------------------------------------------------------- progress

def fmt_eta(seconds):
    if seconds < 0 or seconds != seconds:  # negative / NaN
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_progress(cur, total, fps):
    """One-line progress bar with running fps and ETA, on stdout."""
    if total > 0:
        frac = min(1.0, cur / total)
        bar_w = 28
        fill = int(bar_w * frac)
        bar = "=" * fill + (">" if fill < bar_w else "") + " " * (bar_w - fill - 1)
        eta = fmt_eta((total - cur) / fps) if fps > 0 else "--:--"
        sys.stdout.write(f"\r  [{bar}] {cur:>5}/{total} ({frac*100:4.0f}%)  {fps:5.1f} fps  ETA {eta}   ")
    else:
        sys.stdout.write(f"\r  {cur:>6} frames  {fps:5.1f} fps   ")
    sys.stdout.flush()


def pump_progress(stderr, total, state):
    """Read the tool's stderr; turn NRPROG lines into a live bar, pass everything else through."""
    prog = re.compile(r"^NRPROG (\d+) ([\d.]+)")
    for raw in iter(stderr.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        m = prog.match(line)
        if m:
            state["frames"] = int(m.group(1))
            state["fps"] = float(m.group(2))
            render_progress(state["frames"], total, state["fps"])
        elif line.startswith("done:"):
            state["done"] = line
        elif line:
            # a real log/error line from the tool - break the bar and show it
            sys.stdout.write("\n")
            sys.stdout.flush()
            print(line, file=sys.stderr)


# ----------------------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(
        description="DLSS Super Resolution + Neural Rendering over a clip (ffmpeg <-> video2dlssnr).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="output file; .mp4 / .mkv / .mov / .webm picks the container")
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(__file__), "out", "video2dlssnr.exe"))
    ap.add_argument("--dll-dir", default="", help="override nvngx dir (default: next to video2dlssnr.exe)")
    ap.add_argument("--adapter", type=int, default=-1, help="DXGI adapter index (default: fastest)")

    g = ap.add_argument_group("output size")
    g.add_argument("--nr-width", type=int, default=0, help="output width (height by aspect)")
    g.add_argument("--nr-height", type=int, default=0, help="output height (width by aspect); with "
                                                            "--nr-width: the exact size")
    g.add_argument("--nr-scale", type=float, default=1.0, help="scale factor when width/height unset")
    g.add_argument("--nr-sr-preset", default="default",
                   help="DLSS Super Resolution model preset for the upscale: default (driver picks "
                        "per mode) or a letter A..O. E = CNN (DLSS 3.7 default), F = CNN (Ultra "
                        "Performance / DLAA), J = DLSS 4 transformer (first), K = DLSS 4 transformer "
                        "(default), L / M = DLSS 4.5 transformer (newest)")
    g.add_argument("--nr-fit", default="", metavar="WxH",
                   help="fit inside this box keeping the aspect, e.g. 3840x2160 (a portrait clip "
                        "gets the box turned); overrides width / height / scale")

    # NR / scale flags are named exactly like video2dlssnr.exe (--nr-*), so the same knobs carry over.
    g = ap.add_argument_group("neural rendering (same flags as video2dlssnr.exe)")
    g.add_argument("--nr-style", type=int, default=0, help="0 default / 1 natural / 2 cinematic")
    g.add_argument("--nr-preset", type=int, default=0)
    g.add_argument("--nr-intensity", type=float, default=1.0, help="DLSSNR.Intensity (0-2)")
    g.add_argument("--nr-local-structure", type=float, default=1.0, help="DLSSNR.LocalStructureStrength (0-2)")
    g.add_argument("--nr-local-tone", type=float, default=1.0, help="DLSSNR.LocalToneStrength (0-2)")
    g.add_argument("--nr-skin", type=float, default=-1.0, help="DLSSNR.SkinStructureStrength (-1 = model default)")
    g.add_argument("--nr-global-tone", type=float, default=-1.0, help="DLSSNR.GlobalToneStrength (<0 = model default)")
    g.add_argument("--nr-detail", type=float, default=1.0, help="composite strength (0 = original, 1 = full NR)")
    g.add_argument("--nr-color", type=float, default=1.0, help="0 = keep original hue, 1 = NR colour")
    g.add_argument("--nr-hdr", action="store_true", help="feed linear (HDR) instead of the sRGB proxy")
    g.add_argument("--nr-ui-correction", type=int, default=0, help="DLSSNR.UICorrection (0/1)")
    g.add_argument("--nr-auto-mask", action="store_true", help="enable DLSSNR.UseAutoMask")
    g.add_argument("--nr-motion", type=int, default=1, help="optical-flow motion vectors for NR (0/1)")
    g.add_argument("--nr-motion-engine", choices=["auto", "nvof", "lk"], default="auto",
                   help="flow backend: auto (NVOFA else LK) / nvof (hardware) / lk (Lucas-Kanade)")
    g.add_argument("--nr-motion-vis", action="store_true",
                   help="output the flow visualisation instead of the NR result (debug)")
    g.add_argument("--frames", type=int, default=0, help="cap frames (0 = whole clip)")

    g = ap.add_argument_group("encoding")
    g.add_argument("--codec", default="hevc_nvenc",
                   help="hevc_nvenc (default) / h264_nvenc / av1_nvenc (GPU); av1_svt (CPU AV1); "
                        "prores (CPU, .mov/.mkv); ffv1 (lossless RGB, .mkv)")
    g.add_argument("--cq", type=int, default=19,
                   help="constant-quality target, lower = better: 15 near-transparent, 19 high "
                        "(default), 23 medium, 28 small. crf for av1_svt; ignored by prores / ffv1")
    g.add_argument("--bitrate", type=int, default=0, help="target bitrate in kbit/s instead of --cq (0 = use --cq)")
    g.add_argument("--bit-depth", type=int, choices=[8, 10], default=10,
                   help="10 (default; less banding, better compression) or 8 (max compatibility). "
                        "h264_nvenc is always 8")
    g.add_argument("--enc-preset", default="p5", help="NVENC preset p1(fast)..p7(quality), default p5")
    g.add_argument("--multipass", choices=["disabled", "qres", "fullres"], default="qres",
                   help="NVENC two-pass mode (default qres)")
    g.add_argument("--sw-preset", type=int, default=6, help="av1_svt speed preset 0(slow)..13(fast), default 6")
    g.add_argument("--prores-profile", choices=PRORES_PROFILES, default="hq")
    g.add_argument("--audio", choices=["auto", "copy", "aac", "opus", "flac", "pcm", "none"], default="auto",
                   help="auto (default) copies the track when the container allows, else re-encodes")
    g.add_argument("--audio-bitrate", type=int, default=192, help="kbit/s for aac / opus (default 192)")
    g.add_argument("--dry-run", action="store_true", help="print the three commands and exit")
    return ap


def main():
    args = build_parser().parse_args()

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        sys.exit("ffmpeg/ffprobe not found. Install with:  winget install Gyan.FFmpeg")

    ext = os.path.splitext(args.out)[1].lower()
    if ext not in CONTAINERS:
        sys.exit(f"--out must end in one of {', '.join(CONTAINERS)} (got {ext or 'no extension'!r})")
    codec = resolve_codec(args.codec)
    if codec not in CONTAINERS[ext]:
        sys.exit(f"{codec} does not go into {ext}; containers for it: "
                 + ", ".join(e for e, ok in CONTAINERS.items() if codec in ok))

    src = probe(ffprobe, args.inp)
    inW, inH, fps, total = src["w"], src["h"], src["fps"], src["frames"]
    if args.frames:
        total = min(total, args.frames) if total else args.frames
    outW, outH = output_size(src, args.nr_width, args.nr_height, args.nr_scale, args.nr_fit)
    toolW, toolH = tool_size(src, outW, outH)
    tags = colour_tags(src)
    hdr_in = src["trc"] in ("smpte2084", "arib-std-b67")
    if hdr_in:
        print(f"HDR source ({src['trc']}): colour tags carried through; NR runs on the PQ/HLG-coded "
              f"signal", file=sys.stderr)

    upscale = (toolW, toolH) != (inW, inH)

    steps = []
    if upscale:
        steps.append(f"DLSS {inW}x{inH}->{toolW}x{toolH}")
    if (toolW, toolH) != (outW, outH):
        steps.append(f"ffmpeg resize {toolW}x{toolH}->{outW}x{outH}")
    print(f"{inW}x{inH} @ {fps:.3f} fps, {total or '?'} frames  ->  {outW}x{outH}  "
          f"[{', '.join(steps) or 'native'}]  style={args.nr_style} intensity={args.nr_intensity} "
          f"detail={args.nr_detail}", file=sys.stderr)

    # ---- decode: tag unknown colour metadata, proper chroma upsampling, rgba out.
    setp = []
    for key, val, cur in (("colorspace", tags[0], src["matrix"]),
                          ("color_primaries", tags[1], src["primaries"]),
                          ("color_trc", tags[2], src["trc"])):
        if cur == "unknown":
            setp.append(f"{key}={val}")
    vf = ([f"setparams={':'.join(setp)}"] if setp else []) + [f"scale=flags={SWS}", "format=rgba"]
    dec = [ffmpeg, "-v", "error", "-i", args.inp]
    if args.frames:
        dec += ["-frames:v", str(args.frames)]
    dec += ["-vf", ",".join(vf), "-f", "rawvideo", "-"]

    # ---- the tool
    tool = [args.exe, "--nr-video", "--nr-in", f"{inW}x{inH}",
            "--nr-style", str(args.nr_style), "--nr-preset", str(args.nr_preset),
            "--nr-intensity", str(args.nr_intensity),
            "--nr-local-structure", str(args.nr_local_structure),
            "--nr-local-tone", str(args.nr_local_tone), "--nr-skin", str(args.nr_skin),
            "--nr-global-tone", str(args.nr_global_tone), "--nr-detail", str(args.nr_detail),
            "--nr-color", str(args.nr_color), "--nr-ui-correction", str(args.nr_ui_correction),
            "--nr-motion", str(args.nr_motion), "--nr-motion-engine", args.nr_motion_engine,
            "--nr-sr-preset", args.nr_sr_preset]
    if args.nr_hdr:
        tool += ["--nr-hdr"]
    if args.nr_auto_mask:
        tool += ["--nr-auto-mask"]
    if args.nr_motion_vis:
        tool += ["--nr-motion-vis"]
    if upscale:  # pass the exact even dims so tool and encoder agree
        tool += ["--nr-width", str(toolW), "--nr-height", str(toolH)]
    if args.dll_dir:  # empty => the tool looks for nvngx_dlssnr.dll next to video2dlssnr.exe
        tool += ["--dll-dir", args.dll_dir]
    if args.adapter >= 0:
        tool += ["--adapter", str(args.adapter)]

    # ---- encode: RGB -> YUV with the right matrix and full-chroma resampling, then the codec.
    vargs, pix = video_args(codec, args, outW, outH, tags)
    evf = []
    if (toolW, toolH) != (outW, outH):
        evf.append(f"scale={outW}:{outH}:flags={SWS}")
    if pix != "bgr0":
        evf.append(f"scale=out_color_matrix={tags[0]}:out_range=tv:flags={SWS}")
    evf.append(f"format={pix}")
    # ffmpeg takes the stream's colour signalling from the frames, so tag them here (the encoder
    # -colorspace/-color_* flags alone leave primaries / transfer "unknown" in the file).
    if pix != "bgr0":
        evf.append(f"setparams=colorspace={tags[0]}:color_primaries={tags[1]}:color_trc={tags[2]}"
                   ":range=tv")
    else:  # RGB output: no Y'CbCr matrix to signal, full range
        evf.append(f"setparams=color_primaries={tags[1]}:color_trc={tags[2]}:range=pc")
    enc = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{toolW}x{toolH}", "-r", src["rate"], "-i", "-",
           "-i", args.inp, "-map", "0:v:0"]
    enc += audio_args(src, ext, args.audio, args.audio_bitrate)
    enc += ["-vf", ",".join(evf)] + vargs
    if ext in (".mp4", ".mov"):
        enc += ["-movflags", "+faststart"]
    enc += ["-shortest", args.out]

    if args.dry_run:
        for name, cmd in (("decode", dec), ("tool", tool), ("encode", enc)):
            print(f"{name}: " + subprocess.list2cmdline(cmd))
        return 0

    started = time.perf_counter()
    p1 = subprocess.Popen(dec, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(tool, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()
    p3 = subprocess.Popen(enc, stdin=p2.stdout)
    p2.stdout.close()

    state = {"frames": 0, "fps": 0.0, "done": None}
    pump = threading.Thread(target=pump_progress, args=(p2.stderr, total, state), daemon=True)
    pump.start()

    rc3 = p3.wait()
    rc2 = p2.wait()
    rc1 = p1.wait()
    pump.join(timeout=2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    elapsed = time.perf_counter() - started
    rc = rc3 or rc2 or rc1
    if rc == 0:
        # Prefer the tool's own final tally (exact frame count + steady-state GPU fps).
        frames, steady = state["frames"], state["fps"]
        m = re.search(r"done: (\d+) frames in [\d.]+ s \(([\d.]+) fps\)", state["done"] or "")
        if m:
            frames, steady = int(m.group(1)), float(m.group(2))
        wall = frames / elapsed if elapsed > 0 else 0.0
        size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
        kbps = size * 8 / 1000 / (frames / fps) if frames and fps else 0
        print(f"wrote {args.out}  ({frames} frames, {steady:.1f} fps GPU / {wall:.1f} fps wall, "
              f"{elapsed:.1f}s, {size / 1e6:.1f} MB, {kbps / 1000:.1f} Mbit/s)", file=sys.stderr)
    else:
        print(f"failed (decode={rc1} tool={rc2} encode={rc3})", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
