import os
from pathlib import Path

import gradio as gr

DEMO_ROOT = Path(__file__).resolve().parent
ASSETS = DEMO_ROOT / "assets" / "examples"
PROJECT_ROOT = Path(os.environ.get("TRUSTDETR_ROOT", DEMO_ROOT.parent / "TrustDETR"))

INTRO = """
# TrustDETR

TrustDETR is an RGB-T object detector for aligned visible–infrared image pairs.

**Pipeline:** dual-modality input → shared backbone → cross-modal encoder → detection decoder.

**Benchmarks:** RGBTDronePerson, VTUAV, RGBT-Tiny.

> Full training code and model weights will be released after the paper is accepted.
"""

DATASETS = """
### Supported benchmarks

| Dataset | Modality pairing |
|---|---|
| RGBTDronePerson | thermal + visible |
| VTUAV | ir + rgb |
| RGBT-Tiny | 00 (visible) + 01 (infrared) |

Place result images under `assets/examples/` to populate the gallery tab.
"""


def list_gallery_images():
    if not ASSETS.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in ASSETS.rglob("*") if p.suffix.lower() in exts)
    return [str(p) for p in files]


def live_infer(ir_file, rgb_file, config_path, checkpoint_path, score_thr, device):
    if not PROJECT_ROOT.exists():
        return None, "Local model code not found. Set TRUSTDETR_ROOT or keep TrustDETR/ next to demo/."

    if ir_file is None or rgb_file is None:
        return None, "Please upload both IR and RGB images."

    if not config_path or not checkpoint_path:
        return None, "Config and checkpoint paths are required for live inference."

    if not Path(config_path).exists():
        return None, f"Config not found: {config_path}"

    if not Path(checkpoint_path).exists():
        return None, f"Checkpoint not found: {checkpoint_path}"

    try:
        from inference import run_demo_pair

        if device == "cuda" and not __import__("torch").cuda.is_available():
            device = "cpu"

        panel, n = run_demo_pair(
            config_path,
            checkpoint_path,
            ir_file,
            rgb_file,
            score_thr=score_thr,
            device=device,
        )
        return panel, f"Done. {n} boxes above threshold."
    except Exception as exc:
        return None, f"Inference failed: {exc}"


def build_app():
    gallery_files = list_gallery_images()
    with gr.Blocks(title="TrustDETR Demo") as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Results"):
            gr.Markdown(DATASETS)
            if gallery_files:
                gr.Gallery(value=gallery_files, label="Example outputs", columns=2, height=600)
            else:
                gr.Markdown(
                    "No example images yet. Add visualization images to `demo/assets/examples/`."
                )

        with gr.Tab("Live inference (local only)"):
            gr.Markdown(
                "This tab loads the full model from your local checkout. "
                "It is intended for authors / reviewers with access to weights, not for public release."
            )
            with gr.Row():
                ir_in = gr.Image(type="filepath", label="IR / Thermal")
                rgb_in = gr.Image(type="filepath", label="RGB / Visible")
            config_in = gr.Textbox(
                label="Config path",
                value=str(PROJECT_ROOT / "configs/trustdetr/trustdetr_r50vd_rgbt_droneperson.yml"),
            )
            ckpt_in = gr.Textbox(label="Checkpoint path (.pth)")
            score_in = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Score threshold")
            device_in = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")
            run_btn = gr.Button("Run")
            out_img = gr.Image(type="pil", label="Detection (IR | RGB)")
            out_msg = gr.Textbox(label="Status")

            run_btn.click(
                live_infer,
                inputs=[ir_in, rgb_in, config_in, ckpt_in, score_in, device_in],
                outputs=[out_img, out_msg],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
