"""DLSS 5 video: Super Resolution + Neural Rendering over a whole clip, streaming.

This is the one entry point for video - it drives the optimised, fully-on-GPU flow:

    ffmpeg (decode) --raw rgba--> video2dlssnr --nr-video --raw rgba--> ffmpeg (NVENC + audio)

The heavy work (DLSS upscale, sRGB encode, neural rendering, composite, 8-bit
pack) all happens on the GPU inside video2dlssnr; nothing is shuttled back to the CPU
between DLSS and NR. ffmpeg only decodes and encodes. A live progress bar reports the
running fps (parsed from the tool's NRPROG lines) with an ETA.

    python nr_video.py --in clip.mp4 --out clip_4k.mp4 \
        --nr-width 3840 --nr-style 2 --nr-intensity 1.0

The NR / scale flags are named exactly like video2dlssnr.exe (--nr-*).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time


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
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True).stdout
    d = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    num, den = (d["r_frame_rate"].split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    frames = int(d.get("nb_frames") or 0)
    if frames <= 0:  # some containers don't store a frame count; estimate from duration
        try:
            frames = round(float(d.get("duration") or 0) * fps)
        except ValueError:
            frames = 0
    return int(d["width"]), int(d["height"]), fps, frames


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(__file__), "out", "video2dlssnr.exe"))
    ap.add_argument("--dll-dir", default="", help="override nvngx dir (default: next to video2dlssnr.exe)")
    # NR / scale flags are named exactly like video2dlssnr.exe (--nr-*), so the same knobs carry over.
    ap.add_argument("--nr-width", type=int, default=0, help="output width (height by aspect)")
    ap.add_argument("--nr-height", type=int, default=0, help="output height (width by aspect)")
    ap.add_argument("--nr-scale", type=float, default=1.0, help="upscale factor (if width/height unset)")
    ap.add_argument("--nr-style", type=int, default=0, help="0 default / 1 natural / 2 cinematic")
    ap.add_argument("--nr-preset", type=int, default=0)
    ap.add_argument("--nr-intensity", type=float, default=1.0, help="DLSSNR.Intensity (0-2)")
    ap.add_argument("--nr-local-structure", type=float, default=1.0, help="DLSSNR.LocalStructureStrength (0-2)")
    ap.add_argument("--nr-local-tone", type=float, default=1.0, help="DLSSNR.LocalToneStrength (0-2)")
    ap.add_argument("--nr-skin", type=float, default=-1.0, help="DLSSNR.SkinStructureStrength (-1 = model default)")
    ap.add_argument("--nr-global-tone", type=float, default=-1.0, help="DLSSNR.GlobalToneStrength (<0 = model default)")
    ap.add_argument("--nr-detail", type=float, default=1.0, help="composite strength (0 = original, 1 = full NR)")
    ap.add_argument("--nr-color", type=float, default=1.0, help="0 = keep original hue, 1 = NR colour")
    ap.add_argument("--nr-hdr", action="store_true", help="feed linear (HDR) instead of the sRGB proxy")
    ap.add_argument("--nr-ui-correction", type=int, default=0, help="DLSSNR.UICorrection (0/1)")
    ap.add_argument("--nr-auto-mask", action="store_true", help="enable DLSSNR.UseAutoMask")
    ap.add_argument("--nr-motion", type=int, default=1, help="optical-flow motion vectors for NR (0/1)")
    ap.add_argument("--nr-motion-engine", choices=["auto", "nvof", "lk"], default="auto",
                    help="flow backend: auto (NVOFA else LK) / nvof (hardware) / lk (Lucas-Kanade)")
    ap.add_argument("--nr-motion-vis", action="store_true",
                    help="output the flow visualisation instead of the NR result (debug)")
    ap.add_argument("--frames", type=int, default=0, help="cap frames (0 = whole clip)")
    ap.add_argument("--codec", default="hevc_nvenc", help="hevc_nvenc / h264_nvenc / av1_nvenc")
    ap.add_argument("--enc-preset", default="p5", help="NVENC preset p1(fast)..p7(quality)")
    args = ap.parse_args()

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        sys.exit("ffmpeg/ffprobe not found. Install with:  winget install Gyan.FFmpeg")

    inW, inH, fps, total = probe(ffprobe, args.inp)
    if args.frames:
        total = min(total, args.frames) if total else args.frames
    if args.nr_width and args.nr_height:
        outW, outH = args.nr_width, args.nr_height
    elif args.nr_width:
        outW, outH = args.nr_width, round(inH * args.nr_width / inW)
    elif args.nr_height:
        outW, outH = round(inW * args.nr_height / inH), args.nr_height
    else:
        outW, outH = round(inW * args.nr_scale), round(inH * args.nr_scale)
    outW -= outW % 2  # even dims for the encoder
    outH -= outH % 2
    print(f"{inW}x{inH} @ {fps:.3f} fps, {total or '?'} frames  ->  {outW}x{outH}   "
          f"style={args.nr_style} intensity={args.nr_intensity} detail={args.nr_detail}",
          file=sys.stderr)

    # ProRes and friends carry unspecified colour metadata; tag bt709 so swscale gives rgba.
    vf = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=rgba"
    dec = [ffmpeg, "-v", "error", "-i", args.inp]
    if args.frames:
        dec += ["-frames:v", str(args.frames)]
    dec += ["-vf", vf, "-f", "rawvideo", "-"]

    tool = [args.exe, "--nr-video", "--nr-in", f"{inW}x{inH}",
            "--nr-style", str(args.nr_style), "--nr-preset", str(args.nr_preset),
            "--nr-intensity", str(args.nr_intensity),
            "--nr-local-structure", str(args.nr_local_structure),
            "--nr-local-tone", str(args.nr_local_tone), "--nr-skin", str(args.nr_skin),
            "--nr-global-tone", str(args.nr_global_tone), "--nr-detail", str(args.nr_detail),
            "--nr-color", str(args.nr_color), "--nr-ui-correction", str(args.nr_ui_correction),
            "--nr-motion", str(args.nr_motion), "--nr-motion-engine", args.nr_motion_engine]
    if args.nr_hdr:
        tool += ["--nr-hdr"]
    if args.nr_auto_mask:
        tool += ["--nr-auto-mask"]
    if args.nr_motion_vis:
        tool += ["--nr-motion-vis"]
    if (outW, outH) != (inW, inH):  # pass the exact even dims so tool and encoder agree
        tool += ["--nr-width", str(outW), "--nr-height", str(outH)]
    if args.dll_dir:  # empty => the tool looks for nvngx_dlssnr.dll next to video2dlssnr.exe
        tool += ["--dll-dir", args.dll_dir]

    enc = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{outW}x{outH}", "-r", f"{fps}", "-i", "-",
           "-i", args.inp, "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", args.codec, "-preset", args.enc_preset, "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", args.out]

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
        print(f"wrote {args.out}  ({frames} frames, {steady:.1f} fps GPU / {wall:.1f} fps wall, "
              f"{elapsed:.1f}s)", file=sys.stderr)
    else:
        print(f"failed (decode={rc1} tool={rc2} encode={rc3})", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
