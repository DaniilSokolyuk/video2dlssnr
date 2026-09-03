"""Gradio UI for video2dlssnr — two tabs (Image / Video) over the same DLSS NR knobs.

Image tab runs the CLI  (out/video2dlssnr.exe --nr-run);
Video tab runs the streaming script (nr_video.py) and shows a live fps/progress line.

    start.bat            # installs gradio if needed, then launches this
"""

import glob
import importlib
import os
import subprocess
import sys


def _ensure(module, pip_name=None):
    """Install a dependency on first run so the release needs no separate setup step."""
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"installing {pip_name or module} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name or module])


_ensure("gradio")

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_video import find_tool  # noqa: E402  (same ffmpeg lookup as the video script)

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "out", "video2dlssnr.exe")
NRV = os.path.join(ROOT, "nr_video.py")
OUT_DIR = os.path.join(ROOT, "ui_out")
# nvngx_dlssnr.dll is expected next to video2dlssnr.exe; the tool finds it there on its own.

STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}
PRESETS = {"Default": 0, "Preset 1": 1, "Preset 2": 2, "Preset 3": 3}
# DLSS Super Resolution model preset for the upscale stage. Default = the driver picks per mode;
# E/F are the CNN models (DLSS 3.x), J/K the DLSS 4 transformer, L/M the DLSS 4.5 transformer.
SR_PRESETS = {
    "Default (driver picks)": "default",
    "E (CNN, DLSS 3.7 default)": "E",
    "F (CNN, Ultra Performance / DLAA)": "F",
    "J (Transformer, DLSS 4 first)": "J",
    "K (Transformer, DLSS 4 default)": "K",
    "L (Transformer, DLSS 4.5)": "L",
    "M (Transformer, DLSS 4.5 newest)": "M",
}

# Output size presets. Fixed sizes fit inside the box keeping the aspect (a portrait clip gets the
# box turned); "Custom" uses the width / height fields (height 0 = keep aspect).
SIZES = {
    "Source (no upscale)": ("scale", 1.0),
    "×1.5": ("scale", 1.5),
    "×2": ("scale", 2.0),
    "×3": ("scale", 3.0),
    "720p (1280×720)": ("fit", "1280x720"),
    "1080p (1920×1080)": ("fit", "1920x1080"),
    "1440p (2560×1440)": ("fit", "2560x1440"),
    "4K (3840×2160)": ("fit", "3840x2160"),
    "5K (5120×2880)": ("fit", "5120x2880"),
    "8K (7680×4320)": ("fit", "7680x4320"),
    "Custom": ("custom", None),
}

# Video codecs: label -> nr_video.py --codec. NVENC = GPU, the rest run on the CPU.
CODECS = {
    "HEVC / H.265 (NVENC)": "hevc_nvenc",
    "H.264 (NVENC)": "h264_nvenc",
    "AV1 (NVENC)": "av1_nvenc",
    "AV1 (CPU, SVT-AV1)": "av1_svt",
    "ProRes (CPU)": "prores",
    "FFV1 lossless (CPU)": "ffv1",
}
CONTAINERS = {  # which containers each codec can go into (first = default)
    "hevc_nvenc": ["mp4", "mkv", "mov"],
    "h264_nvenc": ["mp4", "mkv", "mov"],
    "av1_nvenc": ["mp4", "mkv", "webm"],
    "av1_svt": ["mp4", "mkv", "webm"],
    "prores": ["mov", "mkv"],
    "ffv1": ["mkv"],
}
QUALITIES = {  # constant-quality targets (lower = better); "Custom" opens the CQ / bitrate fields
    "Best (CQ 15)": 15,
    "High (CQ 19)": 19,
    "Medium (CQ 23)": 23,
    "Low (CQ 28)": 28,
    "Custom": None,
}
AUDIO = ["auto", "copy", "aac", "opus", "flac", "none"]
PRORES_PROFILES = ["proxy", "lt", "standard", "hq", "4444", "4444xq"]


def nr_model_args(style, preset, intensity, local_structure, local_tone, skin, global_tone,
                  detail, color, ui_correction, auto_mask, hdr):
    """The NR model + composite flags shared by both tabs (same names as the CLI)."""
    a = ["--nr-style", str(STYLES[style]), "--nr-preset", str(PRESETS[preset]),
         "--nr-intensity", str(intensity), "--nr-local-structure", str(local_structure),
         "--nr-local-tone", str(local_tone), "--nr-skin", str(skin),
         "--nr-global-tone", str(global_tone), "--nr-detail", str(detail),
         "--nr-color", str(color), "--nr-ui-correction", "1" if ui_correction else "0"]
    if auto_mask:
        a += ["--nr-auto-mask"]
    if hdr:
        a += ["--nr-hdr"]
    return a


def fit_box(w, h, box):
    """Largest even size with the same aspect as w x h that fits in the box (turned for portrait)."""
    bw, bh = (int(x) for x in box.split("x"))
    if (h > w) != (bh > bw):
        bw, bh = bh, bw
    f = min(bw / w, bh / h)
    ow, oh = int(round(w * f)), int(round(h * f))
    return max(2, ow - ow % 2), max(2, oh - oh % 2)


def custom_size_args(width, height, scale):
    """Custom: width / height win (one side = aspect kept, both = exact size); else the scale."""
    a = []
    if int(width or 0) > 0:
        a += ["--nr-width", str(int(width))]
    if int(height or 0) > 0:
        a += ["--nr-height", str(int(height))]
    if not a and float(scale or 1.0) != 1.0:
        a += ["--nr-scale", str(float(scale))]
    return a


def size_args_image(image, size, width, height, scale):
    """--nr-width/--nr-height/--nr-scale for the CLI, from the size preset (the CLI has no --nr-fit)."""
    kind, val = SIZES[size]
    if kind == "scale":
        return ["--nr-scale", str(val)] if val != 1.0 else []
    if kind == "fit":
        from PIL import Image
        with Image.open(image) as im:
            w, h = im.size
        ow, oh = fit_box(w, h, val)
        return ["--nr-width", str(ow), "--nr-height", str(oh)]
    return custom_size_args(width, height, scale)


def size_args_video(size, width, height, scale):
    kind, val = SIZES[size]
    if kind == "scale":
        return ["--nr-scale", str(val)] if val != 1.0 else []
    if kind == "fit":
        return ["--nr-fit", val]
    return custom_size_args(width, height, scale)


def run_image(image, size, width, height, scale, sr_preset, style, preset, intensity,
              local_structure, local_tone, skin, global_tone, detail, color, ui_correction,
              auto_mask, hdr):
    if not image:
        return None, "Drop an image first."
    os.makedirs(OUT_DIR, exist_ok=True)
    args = [EXE, "--nr-run", "--in", image, "--out", OUT_DIR, "--nr-sr-preset", SR_PRESETS[sr_preset]]
    args += nr_model_args(style, preset, intensity, local_structure, local_tone, skin,
                          global_tone, detail, color, ui_correction, auto_mask, hdr)
    args += size_args_image(image, size, width, height, scale)
    before = set(glob.glob(os.path.join(OUT_DIR, "*_nr.png")))
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        return None, f"failed (exit {p.returncode})\n\n{log[-4000:]}"
    # The CLI names results by the input file; just pick the newest _nr.png this run produced.
    cands = glob.glob(os.path.join(OUT_DIR, "*_nr.png"))
    fresh = [c for c in cands if c not in before] or cands
    if not fresh:
        return None, "no output image was produced.\n\n" + log[-4000:]
    out = max(fresh, key=os.path.getmtime)
    return out, log[-2000:]


# What a browser <video> tag plays (gradio's own list). Anything else gets a small H.264 preview
# copy so the result is visible in the UI; the real file is untouched and offered for download.
BROWSER_PLAYABLE = {("mp4", "h264_nvenc"), ("mp4", "av1_nvenc"), ("mp4", "av1_svt"),
                    ("webm", "av1_nvenc"), ("webm", "av1_svt")}


def make_preview(out):
    """A quick 8-bit H.264 copy (<=1080p) next to the result, for the browser only."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        return None
    prev = os.path.splitext(out)[0] + "_preview.mp4"
    common = [ffmpeg, "-y", "-v", "error", "-i", out, "-map", "0:v:0", "-map", "0:a:0?",
              "-vf", "scale='min(1920,iw)':-2:flags=bicubic,format=yuv420p"]
    tail = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", prev]
    for venc in (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "24", "-b:v", "0"],
                 ["-c:v", "libopenh264", "-b:v", "8M"]):
        if subprocess.run(common + venc + tail, capture_output=True).returncode == 0:
            return prev
    return None


def _stream(proc):
    """Yield output lines, splitting on both \\r (progress bar) and \\n for live updates."""
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in "\r\n":
            if buf.strip():
                yield buf
            buf = ""
        else:
            buf += ch
    if buf.strip():
        yield buf


def run_video(video, engine, motion, motion_vis, size, width, height, scale, sr_preset, style,
              preset, intensity, local_structure, local_tone, skin, global_tone, detail, color,
              ui_correction, auto_mask, hdr, codec_label, container, quality, cq, bitrate,
              bit_depth, enc_preset, prores_profile, audio, frames):
    if not video:
        yield None, None, "Drop a video first."
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    codec = CODECS[codec_label]
    if container not in CONTAINERS[codec]:
        container = CONTAINERS[codec][0]
    base = os.path.splitext(os.path.basename(video))[0]
    tag = "_flow" if motion_vis else "_nr"
    out = os.path.join(OUT_DIR, f"{base}{tag}.{container}")
    args = [sys.executable, NRV, "--in", video, "--out", out,
            "--nr-motion-engine", engine, "--nr-motion", "1" if motion else "0",
            "--codec", codec, "--enc-preset", enc_preset, "--bit-depth", str(int(bit_depth)),
            "--prores-profile", prores_profile, "--audio", audio,
            "--nr-sr-preset", SR_PRESETS[sr_preset]]
    if QUALITIES[quality] is not None:
        args += ["--cq", str(QUALITIES[quality])]
    else:
        args += ["--cq", str(int(cq)), "--bitrate", str(int(bitrate))]
    args += nr_model_args(style, preset, intensity, local_structure, local_tone, skin,
                          global_tone, detail, color, ui_correction, auto_mask, hdr)
    args += size_args_video(size, width, height, scale)
    if motion_vis:
        args += ["--nr-motion-vis"]
    if int(frames) > 0:
        args += ["--frames", str(int(frames))]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    status = ""
    for line in _stream(proc):
        status = line.strip()
        yield None, None, status
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(out):
        yield None, None, status + f"\n\nfailed (exit {proc.returncode})."
        return
    if (container, codec) in BROWSER_PLAYABLE:  # h264 is always 8-bit, av1 10-bit plays fine
        yield out, out, status + f"\n\ndone -> {out}"
        return
    yield None, out, status + f"\n\ndone -> {out}\nmaking a browser preview copy ..."
    prev = make_preview(out)
    note = "" if prev else "\n(no preview: the browser cannot play this format; download the file)"
    yield prev, out, status + f"\n\ndone -> {out}" + note


def nr_controls():
    """Build the shared NR control set; returns the list of components in argument order."""
    with gr.Row():
        style = gr.Dropdown(list(STYLES), value="Cinematic", label="Style")
        preset = gr.Dropdown(list(PRESETS), value="Default", label="NR preset")
        intensity = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Intensity")
    with gr.Row():
        local_structure = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Local structure")
        local_tone = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Local tone")
    with gr.Row():
        skin = gr.Slider(-1.0, 2.0, value=-1.0, step=0.05, label="Skin (-1 = model default)")
        global_tone = gr.Slider(-1.0, 2.0, value=-1.0, step=0.05, label="Global tone (<0 = default)")
    with gr.Row():
        detail = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Composite detail (0=original)")
        color = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Composite colour")
    with gr.Row():
        ui_correction = gr.Checkbox(value=False, label="UI correction")
        auto_mask = gr.Checkbox(value=False, label="Auto mask")
        hdr = gr.Checkbox(value=False, label="HDR (linear)")
    return [style, preset, intensity, local_structure, local_tone, skin, global_tone, detail,
            color, ui_correction, auto_mask, hdr]


def size_controls(default):
    """Resolution preset + the Custom fields (width / height / scale, shown for Custom).

    Returns [size, width, height, scale, sr_preset]. Line 1: the size preset. Line 2: the DLSS SR
    model for the upscale (always shown) and the Custom fields, hidden until Custom is chosen.
    In Custom, width / height win when set (one side keeps the aspect, both = exact size); with both
    at 0 the scale factor is used.
    """
    with gr.Row():
        size = gr.Dropdown(list(SIZES), value=default, label="Output size")
    custom = default == "Custom"
    with gr.Row():
        sr_preset = gr.Dropdown(list(SR_PRESETS), value="Default (driver picks)",
                                label="DLSS SR preset (upscale model)")
        width = gr.Number(value=0, precision=0, label="Custom width (0 = by aspect)",
                          visible=custom)
        height = gr.Number(value=0, precision=0, label="Custom height (0 = by aspect)",
                           visible=custom)
        scale = gr.Slider(1.0, 4.0, value=2.0, step=0.05, label="Custom scale (when W/H are 0)",
                          visible=custom)
    size.change(lambda s: [gr.update(visible=s == "Custom")] * 3,
                inputs=size, outputs=[width, height, scale])
    return [size, width, height, scale, sr_preset]


with gr.Blocks(title="video2dlssnr") as demo:
    gr.Markdown("# video2dlssnr — DLSS Super Resolution + Neural Rendering")

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    # image_mode=None hands the uploaded file over untouched (gradio would otherwise
                    # re-save e.g. an RGBA PNG as lossy webp before the tool sees it).
                    img_in = gr.Image(type="filepath", image_mode=None, label="Input image")
                    i_size = size_controls("×2")
                    i_nr = nr_controls()
                    i_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Result (PNG, saved in ui_out\\)", type="filepath",
                                       format="png")
                    i_log = gr.Textbox(label="Log", lines=6)
            i_btn.click(run_image, inputs=[img_in] + i_size + i_nr, outputs=[img_out, i_log])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Input video")
                    with gr.Row():
                        v_engine = gr.Dropdown(["auto", "nvof", "lk"], value="auto",
                                               label="Motion engine")
                        v_motion = gr.Checkbox(value=True, label="Motion vectors")
                        v_vis = gr.Checkbox(value=False, label="Flow visualisation (debug)")
                    v_size = size_controls("4K (3840×2160)")
                    v_nr = nr_controls()
                    with gr.Accordion("Encoding", open=True):
                        with gr.Row():
                            v_codec = gr.Dropdown(list(CODECS), value="HEVC / H.265 (NVENC)",
                                                  label="Codec", scale=2)
                            v_container = gr.Dropdown(["mp4", "mkv", "mov", "webm"], value="mp4",
                                                      label="Container")
                            v_quality = gr.Dropdown(list(QUALITIES), value="High (CQ 19)",
                                                    label="Quality")
                        with gr.Row():
                            v_cq = gr.Slider(0, 51, value=19, step=1,
                                             label="CQ (lower = better; Custom only)", visible=False)
                            v_bitrate = gr.Number(value=0, precision=0,
                                                  label="Bitrate kbit/s (0 = use CQ; Custom only)",
                                                  visible=False)
                        with gr.Row():
                            v_depth = gr.Radio([10, 8], value=10, label="Bit depth")
                            v_enc = gr.Dropdown(["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                                                value="p5", label="NVENC preset (p7 = best)")
                            v_prores = gr.Dropdown(PRORES_PROFILES, value="hq",
                                                   label="ProRes profile", visible=False)
                            v_audio = gr.Dropdown(AUDIO, value="auto", label="Audio")
                            v_frames = gr.Number(value=0, precision=0, label="Frame cap (0 = all)")
                    v_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    vid_out = gr.Video(label="Preview (a browser copy when the format needs one)")
                    v_file = gr.File(label="Result file (also saved in ui_out\\)")
                    v_log = gr.Textbox(label="Progress", lines=3)

            def on_codec(label, container):
                allowed = CONTAINERS[CODECS[label]]
                return (gr.update(choices=allowed,
                                  value=container if container in allowed else allowed[0]),
                        gr.update(visible=CODECS[label] == "prores"))

            v_codec.change(on_codec, inputs=[v_codec, v_container], outputs=[v_container, v_prores])
            v_quality.change(lambda q: (gr.update(visible=q == "Custom"),
                                        gr.update(visible=q == "Custom")),
                             inputs=v_quality, outputs=[v_cq, v_bitrate])
            v_btn.click(run_video,
                        inputs=[vid_in, v_engine, v_motion, v_vis] + v_size + v_nr
                        + [v_codec, v_container, v_quality, v_cq, v_bitrate, v_depth, v_enc,
                           v_prores, v_audio, v_frames],
                        outputs=[vid_out, v_file, v_log])


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
