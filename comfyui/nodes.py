"""ComfyUI nodes for video2dlssnr — DLSS Super Resolution + Neural Rendering.

Two nodes, mirroring the two tabs of the Gradio UI:

  * DLSS Neural Rendering (Image)  — every image in the batch independently
                                     (video2dlssnr.exe --nr-run, same as the Image tab).
  * DLSS Neural Rendering (Video)  — the batch is a clip: frames are streamed through
                                     --nr-video with optical-flow motion vectors, so NR stays
                                     temporally stable. Frames go to the exe as raw RGBA over a
                                     pipe and come back the same way: no ffmpeg, no temp files.
                                     HDR clips (PQ / HLG) are recognised from the Load Video
                                     source and take the tool's HDR path: 16-bit frames, an SDR
                                     proxy for the model, the edit back on the HDR frame as a
                                     linear-light residual; the VIDEO output is tagged 10-bit
                                     HDR PQ / HLG the way the core Create Video node does.

video2dlssnr.exe (with its runtime DLLs next to it) is looked up in this order:
  1. the VIDEO2DLSSNR_EXE environment variable — full path to video2dlssnr.exe;
  2. <this folder>/bin/video2dlssnr.exe.
"""

import glob
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from fractions import Fraction

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}
PRESETS = {"Default": 0, "Preset 1": 1, "Preset 2": 2, "Preset 3": 3}
ENGINES = ["auto", "nvof", "lk"]
SR_PRESETS = ["Default", "E", "F", "J", "K", "L", "M"]
# How the clip is coded. auto reads the transfer characteristic of the connected VIDEO's source
# stream (an IMAGE batch carries none and counts as SDR); the rest override it.
HDR_SOURCES = ["auto", "sdr", "pq", "hlg"]
# ffmpeg / PyAV transfer characteristic -> the tool's --nr-transfer, and the ComfyUI colour space
# name the core Create Video / Save Video nodes use for it.
AV_TRC_SMPTE2084, AV_TRC_ARIB_STD_B67 = 16, 18
TRC_TO_TRANSFER = {"smpte2084": "pq", "arib-std-b67": "hlg",
                   AV_TRC_SMPTE2084: "pq", AV_TRC_ARIB_STD_B67: "hlg"}
COMFY_COLOR_SPACE = {"pq": "HDR PQ", "hlg": "HDR"}


# ----------------------------------------------------------------------------- tool lookup

def find_exe():
    env = os.environ.get("VIDEO2DLSSNR_EXE", "").strip().strip('"')
    cands = [env] if env else []
    cands.append(os.path.join(HERE, "bin", "video2dlssnr.exe"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError(
        "video2dlssnr.exe not found. Set VIDEO2DLSSNR_EXE to its full path, or copy the out\\ "
        "folder of a video2dlssnr release into: " + os.path.join(HERE, "bin"))


def nr_args(style, preset, intensity, local_structure, local_tone, skin, detail,
            color, ui_correction, auto_mask, sr_preset="Default"):
    """NR model + composite flags, named exactly like the CLI (--nr-*)."""
    a = ["--nr-sr-preset", "default" if sr_preset == "Default" else sr_preset,
         "--nr-style", str(STYLES[style]), "--nr-preset", str(PRESETS[preset]),
         "--nr-intensity", f"{intensity}", "--nr-local-structure", f"{local_structure}",
         "--nr-local-tone", f"{local_tone}", "--nr-skin", f"{skin}", "--nr-detail", f"{detail}",
         "--nr-color", f"{color}", "--nr-ui-correction", "1" if ui_correction else "0"]
    if auto_mask:
        a.append("--nr-auto-mask")
    return a


def probe_transfer(video):
    """'pq' / 'hlg' when the VIDEO input's source stream is tagged HDR, else None.

    Load Video hands frames over as coded values (PQ / HLG code in a float tensor) without saying
    which curve they are on; the stream itself still knows. Opened read-only with PyAV, which
    ComfyUI ships; anything that fails means "treat as SDR"."""
    try:
        src = video.get_stream_source()
    except Exception:
        return None
    try:
        import av
        if hasattr(src, "seek"):
            src.seek(0)
        with av.open(src) as container:
            stream = container.streams.video[0]
            trc = getattr(stream.codec_context, "color_trc", None)
            if trc is None:
                trc = getattr(stream, "color_trc", None)
            name = getattr(trc, "name", None)
            key = str(name).lower().replace("_", "-") if name else int(trc)
            return TRC_TO_TRANSFER.get(key)
    except Exception:
        return None
    finally:
        try:
            if hasattr(src, "seek"):
                src.seek(0)
        except Exception:
            pass


def resolve_transfer(hdr_source, video):
    """The tool's --nr-transfer for this job: None for SDR."""
    if hdr_source == "auto":
        return probe_transfer(video) if video is not None else None
    return None if hdr_source == "sdr" else hdr_source


def out_dims(in_w, in_h, width, scale, height=0):
    """Output size the way nr_video.py computes it: width and height together pin the exact size,
    one of them keeps the aspect, else scale; even dims."""
    if width > 0 and height > 0:
        out_w, out_h = int(width), int(height)
    elif width > 0:
        out_w, out_h = int(width), round(in_h * width / in_w)
    elif height > 0:
        out_w, out_h = round(in_w * height / in_h), int(height)
    else:
        out_w, out_h = round(in_w * scale), round(in_h * scale)
    return max(2, out_w - out_w % 2), max(2, out_h - out_h % 2)


# ----------------------------------------------------------------------------- numpy cores

def run_image_np(img_u8, width, scale, nr, exe, adapter=0, height=0):
    """One uint8 RGB [H,W,3] image through --nr-run; returns uint8 RGB [H',W',3]."""
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="v2d_")
    try:
        src = os.path.join(tmp, "in.png")
        out_dir = os.path.join(tmp, "out")
        Image.fromarray(np.ascontiguousarray(img_u8), "RGB").save(src)
        cmd = [exe, "--nr-run", "--in", src, "--out", out_dir, "--adapter", str(adapter)] + nr
        if width > 0:
            cmd += ["--nr-width", str(int(width))]
        if height > 0:
            cmd += ["--nr-height", str(int(height))]
        if width <= 0 and height <= 0 and abs(scale - 1.0) > 1e-6:
            cmd += ["--nr-scale", f"{scale}"]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        hits = glob.glob(os.path.join(out_dir, "*_nr.png"))
        if p.returncode != 0 or not hits:
            raise RuntimeError(f"video2dlssnr failed (exit {p.returncode})\n"
                               + ((p.stdout or "") + (p.stderr or ""))[-3000:])
        return np.array(Image.open(hits[0]).convert("RGB"), dtype=np.uint8)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_video_np(frames, width, scale, nr, motion, engine, motion_vis, exe, adapter=0,
                 progress=None, log=None, height=0, transfer=None, sdr_white=203.0):
    """RGB [B,H,W,3] frames streamed through --nr-video; returns RGB [B,H',W',3] of the same dtype.

    uint8 frames are SDR (sRGB) and travel 8-bit. uint16 frames carry PQ / HLG code values and
    travel 16-bit with --nr-transfer, so the tool takes its HDR path. `progress(n)` is called with
    the frame count as the tool reports it; `log`, if a list, receives the tool's non-progress
    stderr lines (backend in use, final tally, any warnings).
    """
    b, h, w, _ = frames.shape
    wide = frames.dtype == np.uint16
    if bool(transfer) != wide:
        raise ValueError("HDR frames must be uint16 with a transfer, SDR frames uint8 without")
    out_w, out_h = out_dims(w, h, width, scale, height)
    cmd = [exe, "--nr-video", "--nr-in", f"{w}x{h}", "--adapter", str(adapter)] + nr + [
        "--nr-motion", "1" if motion else "0", "--nr-motion-engine", engine]
    if transfer:
        cmd += ["--nr-transfer", transfer, "--nr-sdr-white", f"{sdr_white:g}"]
    if motion_vis:
        cmd.append("--nr-motion-vis")
    if (out_w, out_h) != (w, h):  # exact even dims so the tool and this reader agree
        cmd += ["--nr-width", str(out_w), "--nr-height", str(out_h)]

    alpha = np.full((b, h, w, 1), 65535 if wide else 255, frames.dtype)
    rgba = np.concatenate([frames, alpha], axis=3)
    bpp = 8 if wide else 4

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    err = []

    def feed():
        try:
            for i in range(b):
                p.stdin.write(rgba[i].tobytes())
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def drain():
        prog = re.compile(rb"^NRPROG (\d+) ")
        for raw in iter(p.stderr.readline, b""):
            m = prog.match(raw)
            if m:
                if progress:
                    progress(int(m.group(1)))
            else:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    err.append(line)

    tf = threading.Thread(target=feed, daemon=True)
    td = threading.Thread(target=drain, daemon=True)
    tf.start()
    td.start()

    need = b * out_w * out_h * bpp
    buf = bytearray(need)
    view = memoryview(buf)
    got = 0
    while got < need:
        n = p.stdout.readinto(view[got:])
        if not n:
            break
        got += n
    rc = p.wait()
    tf.join(timeout=5)
    td.join(timeout=5)
    if log is not None:
        log.extend(err)
    if rc != 0 or got != need:
        raise RuntimeError(f"video2dlssnr failed (exit {rc}, got {got}/{need} bytes)\n"
                           + "\n".join(err)[-3000:])
    return np.frombuffer(buf, frames.dtype).reshape(b, out_h, out_w, 4)[..., :3]


# ----------------------------------------------------------------------------- tensor glue

def to_u8(t):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> uint8 RGB [B,H,W,3]."""
    x = t.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return np.ascontiguousarray(x[..., :3])


def to_u16(t):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 (PQ / HLG code values) -> uint16 RGB [B,H,W,3]."""
    x = t.detach().cpu().clamp(0, 1).mul(65535).round().to(torch.int32).numpy().astype(np.uint16)
    return np.ascontiguousarray(x[..., :3])


def to_tensor(arr):
    full = 65535.0 if arr.dtype == np.uint16 else 255.0
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32) / full)


def make_video(images, audio, frame_rate, transfer=None):
    """Wrap frames into ComfyUI's VIDEO type the way the core Create Video node does.

    An HDR clip is tagged 10-bit "HDR PQ" / "HDR" (HLG) so Save Video writes BT.2020 / PQ or HLG
    at 10 bits; the frames already hold the code values. Returns None on ComfyUI builds that
    predate the VIDEO type, so the IMAGE output still works.
    """
    try:
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.util import VideoComponents
    except Exception:
        return None
    comps = VideoComponents(images=images, audio=audio, frame_rate=frame_rate)
    if transfer:
        try:
            return VideoFromComponents(comps, bit_depth=10, color_space=COMFY_COLOR_SPACE[transfer])
        except TypeError:  # an older ComfyUI without HDR video: the frames are still right
            pass
    return VideoFromComponents(comps)


def _nr_inputs():
    f = lambda d, lo, hi: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": 0.05})
    return {
        "style": (list(STYLES), {"default": "Cinematic"}),
        "preset": (list(PRESETS), {"default": "Default"}),
        "intensity": f(1.0, 0.0, 2.0),
        "local_structure": f(1.0, 0.0, 2.0),
        "local_tone": f(1.0, 0.0, 2.0),
        "skin": f(-1.0, -1.0, 2.0),          # -1 = model default
        "detail": f(1.0, 0.0, 2.0),          # composite: 0 = original, 1 = full NR
        "color": f(1.0, 0.0, 1.0),           # 0 = keep original hue, 1 = NR colour
        "ui_correction": ("BOOLEAN", {"default": False}),
        "auto_mask": ("BOOLEAN", {"default": False}),
        "scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.05}),
        "width": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 2,
                          "tooltip": "Output width (height by aspect unless height is set too). "
                                     "0 = use height, else scale."}),
        "height": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 2,
                           "tooltip": "Output height (width by aspect unless width is set too). "
                                      "0 = use width, else scale."}),
        "adapter": ("INT", {"default": 0, "min": 0, "max": 15,
                            "tooltip": "DXGI adapter index (which GPU)."}),
        "sr_preset": (SR_PRESETS, {"default": "Default",
                                   "tooltip": "DLSS Super Resolution model preset for the upscale. "
                                              "Default = driver picks per mode; E = CNN (DLSS 3.7 "
                                              "default), F = CNN (Ultra Performance / DLAA), J = "
                                              "DLSS 4 transformer (first), K = DLSS 4 transformer "
                                              "(default), L / M = DLSS 4.5 transformer (newest)."}),
    }


# ----------------------------------------------------------------------------- nodes

class DLSSNRImage:
    """Each image in the batch through DLSS SR + Neural Rendering independently."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **_nr_inputs()}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "video2dlssnr"
    DESCRIPTION = ("DLSS Super Resolution + Neural Rendering on each image independently "
                   "(same as the Image tab). width pins the output width, else scale.")

    def run(self, image, style, preset, intensity, local_structure, local_tone, skin,
            detail, color, ui_correction, auto_mask, scale, width, height, adapter, sr_preset):
        exe = find_exe()
        nr = nr_args(style, preset, intensity, local_structure, local_tone, skin,
                     detail, color, ui_correction, auto_mask, sr_preset)
        src = to_u8(image)
        outs = [run_image_np(src[i], width, scale, nr, exe, adapter, height)
                for i in range(src.shape[0])]
        return (to_tensor(np.stack(outs)),)


class DLSSNRVideo:
    """A clip streamed through --nr-video with motion vectors for temporal NR.

    Takes either the core Load Video output (VIDEO — audio and frame rate carried over) or a plain
    IMAGE batch (e.g. VideoHelperSuite), and returns both the frames and a ready VIDEO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_nr_inputs(),
                "motion": ("BOOLEAN", {"default": True,
                                       "tooltip": "Optical-flow motion vectors for temporal stability."}),
                "motion_engine": (ENGINES, {"default": "auto"}),
                "motion_vis": ("BOOLEAN", {"default": False,
                                           "tooltip": "Debug: output the flow field instead of NR."}),
                "hdr_source": (HDR_SOURCES, {"default": "auto",
                                             "tooltip": "auto: read the connected VIDEO's transfer "
                                                        "tag (PQ / HLG clips take the HDR path; an IMAGE "
                                                        "batch counts as SDR). sdr / pq / hlg override "
                                                        "it for a mistagged clip. HDR output: feed the "
                                                        "VIDEO to Save Video (10-bit, HDR PQ / HDR)."}),
                "sdr_white": ("FLOAT", {"default": 203.0, "min": 50.0, "max": 1000.0, "step": 1.0,
                                        "tooltip": "HDR PQ: the luminance in nits the model is shown as "
                                                   "SDR white (203 = BT.2408 graphics white; lower = "
                                                   "brighter proxy, more detail in the shadows)."}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "From the core Load Video node. Its audio and frame "
                                               "rate are carried over to the video output."}),
                "images": ("IMAGE", {"tooltip": "Frame batch (e.g. VideoHelperSuite Load Video). "
                                                "Used when no video is connected."}),
                "images_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                         "tooltip": "Only with an IMAGE input: frame-rate metadata for "
                                                    "the VIDEO output (the batch carries none). Does NOT "
                                                    "add frames — the frame count stays the same. Ignored "
                                                    "when a VIDEO is connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("images", "video")
    FUNCTION = "run"
    CATEGORY = "video2dlssnr"
    DESCRIPTION = ("A clip through DLSS SR + Neural Rendering with optical-flow motion vectors for "
                   "temporal stability (same as the Video tab). Connect the core Load Video (VIDEO) "
                   "or an IMAGE batch; get frames and a ready VIDEO (audio and fps kept) back. HDR "
                   "clips (PQ / HLG) are handled as HDR and the VIDEO output is tagged accordingly.")

    def run(self, style, preset, intensity, local_structure, local_tone, skin,
            detail, color, ui_correction, auto_mask, scale, width, height, adapter, sr_preset,
            motion, motion_engine, motion_vis, hdr_source="auto", sdr_white=203.0, video=None,
            images=None, images_fps=24.0):
        audio, frame_rate = None, None
        transfer = resolve_transfer(hdr_source, video)
        if video is not None:
            comps = video.get_components()
            images, audio, frame_rate = comps.images, comps.audio, comps.frame_rate
        if images is None:
            raise RuntimeError("Connect a VIDEO (Load Video) or an IMAGE batch (e.g. VHS Load Video).")
        if frame_rate is None:  # an IMAGE batch carries no rate; this only labels the VIDEO output
            frame_rate = Fraction(images_fps).limit_denominator(1000)

        exe = find_exe()
        nr = nr_args(style, preset, intensity, local_structure, local_tone, skin,
                     detail, color, ui_correction, auto_mask, sr_preset)
        # HDR frames keep their PQ / HLG code values and travel 16-bit; SDR frames travel 8-bit.
        src = to_u16(images) if transfer else to_u8(images)
        total = src.shape[0]
        pbar = None
        try:
            from comfy.utils import ProgressBar  # only exists inside ComfyUI
            pbar = ProgressBar(total)
        except Exception:
            pass

        def progress(done):
            if pbar is not None:
                pbar.update_absolute(min(done, total))

        out = run_video_np(src, width, scale, nr, motion, motion_engine, motion_vis, exe, adapter,
                           progress, height=height, transfer=transfer, sdr_white=sdr_white)
        out_t = to_tensor(out)
        return (out_t, make_video(out_t, audio, frame_rate, transfer))


class DLSSNRRuntimeInfo:
    """Diagnostics: pushes two small frames through the real pipeline and reports the outcome."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"adapter": ("INT", {"default": 0, "min": 0, "max": 15})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    CATEGORY = "video2dlssnr"
    OUTPUT_NODE = True
    DESCRIPTION = ("Runs a tiny test job through the tool and reports whether Neural Rendering "
                   "works on this machine, plus the optical-flow backend in use.")

    def info(self, adapter):
        lines = []
        try:
            exe = find_exe()
        except RuntimeError as e:
            return (str(e),)
        lines.append(f"exe: {exe}")
        try:
            if torch.cuda.is_available():
                lines.append(f"cuda device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

        # A real 2-frame job at 1280x720, native res (NR only), motion vectors on.
        h, w = 720, 1280
        yy, xx = np.mgrid[0:h, 0:w]
        f = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], -1).astype(np.uint8)
        frames = np.stack([f, f])
        nr = nr_args("Default", "Default", 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, False, False, False)
        log = []
        t0 = time.time()
        try:
            out = run_video_np(frames, 0, 1.0, nr, True, "auto", False, exe, adapter, log=log)
            lines.append(f"neural rendering: OK  ({w}x{h}, 2 frames, {time.time() - t0:.1f}s, "
                         f"output {out.shape[2]}x{out.shape[1]})")
        except RuntimeError as e:
            lines.append("neural rendering: FAILED")
            lines.append(str(e))
        lines.extend(log)
        return ("\n".join(lines)[-4000:],)


NODE_CLASS_MAPPINGS = {
    "DLSSNR_Image": DLSSNRImage,
    "DLSSNR_Video": DLSSNRVideo,
    "DLSSNR_RuntimeInfo": DLSSNRRuntimeInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSSNR_Image": "DLSS Neural Rendering (Image)",
    "DLSSNR_Video": "DLSS Neural Rendering (Video)",
    "DLSSNR_RuntimeInfo": "DLSS NR Runtime Check",
}
