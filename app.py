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

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "out", "video2dlssnr.exe")
NRV = os.path.join(ROOT, "nr_video.py")
OUT_DIR = os.path.join(ROOT, "ui_out")
# nvngx_dlssnr.dll is expected next to video2dlssnr.exe; the tool finds it there on its own.

STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}
PRESETS = {"Default": 0, "Preset 1": 1, "Preset 2": 2, "Preset 3": 3}


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


def run_image(image, style, preset, intensity, local_structure, local_tone, skin,
              global_tone, detail, color, ui_correction, auto_mask, hdr, scale, width):
    if not image:
        return None, "Drop an image first."
    os.makedirs(OUT_DIR, exist_ok=True)
    args = [EXE, "--nr-run", "--in", image, "--out", OUT_DIR]
    args += nr_model_args(style, preset, intensity, local_structure, local_tone, skin,
                          global_tone, detail, color, ui_correction, auto_mask, hdr)
    if int(width) > 0:
        args += ["--nr-width", str(int(width))]
    elif float(scale) != 1.0:
        args += ["--nr-scale", str(scale)]
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


def run_video(video, engine, motion, motion_vis, width, scale, style, preset, intensity,
              local_structure, local_tone, skin, global_tone, detail, color, ui_correction,
              auto_mask, hdr, codec, enc_preset, frames):
    if not video:
        yield None, "Drop a video first."
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    tag = "_flow" if motion_vis else "_nr"
    out = os.path.join(OUT_DIR, base + tag + ".mp4")
    args = [sys.executable, NRV, "--in", video, "--out", out,
            "--nr-motion-engine", engine, "--nr-motion", "1" if motion else "0",
            "--codec", codec, "--enc-preset", enc_preset]
    args += nr_model_args(style, preset, intensity, local_structure, local_tone, skin,
                          global_tone, detail, color, ui_correction, auto_mask, hdr)
    if motion_vis:
        args += ["--nr-motion-vis"]
    if int(width) > 0:
        args += ["--nr-width", str(int(width))]
    elif float(scale) != 1.0:
        args += ["--nr-scale", str(scale)]
    if int(frames) > 0:
        args += ["--frames", str(int(frames))]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    status = ""
    for line in _stream(proc):
        status = line.strip()
        yield None, status
    proc.wait()
    if proc.returncode == 0 and os.path.exists(out):
        yield out, status + "\n\ndone."
    else:
        yield None, status + f"\n\nfailed (exit {proc.returncode})."


def nr_controls():
    """Build the shared NR control set; returns the list of components in argument order."""
    with gr.Row():
        style = gr.Dropdown(list(STYLES), value="Cinematic", label="Style")
        preset = gr.Dropdown(list(PRESETS), value="Default", label="Preset")
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


with gr.Blocks(title="video2dlssnr") as demo:
    gr.Markdown("# video2dlssnr — DLSS Super Resolution + Neural Rendering")

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="filepath", label="Input image")
                    with gr.Row():
                        i_width = gr.Number(value=0, precision=0, label="Output width (0 = use scale)")
                        i_scale = gr.Slider(1.0, 3.0, value=1.0, step=0.05, label="Scale")
                    i_nr = nr_controls()
                    i_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Result", type="filepath")
                    i_log = gr.Textbox(label="Log", lines=6)
            i_btn.click(run_image,
                        inputs=[img_in] + i_nr + [i_scale, i_width],
                        outputs=[img_out, i_log])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Input video")
                    with gr.Row():
                        v_engine = gr.Dropdown(["auto", "nvof", "lk"], value="auto",
                                               label="Motion engine")
                        v_motion = gr.Checkbox(value=True, label="Motion vectors")
                        v_vis = gr.Checkbox(value=False, label="Flow visualisation (debug)")
                    with gr.Row():
                        v_width = gr.Number(value=3840, precision=0, label="Output width (0 = scale)")
                        v_scale = gr.Slider(1.0, 3.0, value=1.0, step=0.05, label="Scale")
                    v_nr = nr_controls()
                    with gr.Row():
                        v_codec = gr.Dropdown(["hevc_nvenc", "h264_nvenc", "av1_nvenc"],
                                              value="hevc_nvenc", label="Codec")
                        v_enc = gr.Dropdown(["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                                            value="p5", label="NVENC preset")
                        v_frames = gr.Number(value=0, precision=0, label="Frame cap (0 = all)")
                    v_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    vid_out = gr.Video(label="Result")
                    v_log = gr.Textbox(label="Progress", lines=3)
            v_btn.click(run_video,
                        inputs=[vid_in, v_engine, v_motion, v_vis, v_width, v_scale] + v_nr
                        + [v_codec, v_enc, v_frames],
                        outputs=[vid_out, v_log])


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
