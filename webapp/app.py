#!/usr/bin/env python3
"""
Interactive web app for SAM3 + LoRA inference.

Features:
- Choose LoRA model via config dropdown (no manual path entry)
- Upload image and run prompt-based inference
- Separate visualization outputs: boxes, masks, and contours
- Contour simplification with adjustable epsilon (independent of inference)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "full_lora_config.yaml"

# Keep inference imports from existing implementation.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from infer_sam import SAM3LoRAInference  # noqa: E402


MODEL_LOCK = threading.Lock()
MODEL_CACHE: Dict[Tuple[str, str, int, str], SAM3LoRAInference] = {}

BOX_COLOR = (255, 64, 64)
MASK_COLOR = (80, 170, 255)
CONTOUR_COLOR = (60, 230, 60)
ALL_CLASSES_LABEL = "All classes"


def parse_prompts(prompt_text: str) -> List[str]:
    if not prompt_text or not prompt_text.strip():
        return ["object"]

    raw_items: List[str] = []
    for line in prompt_text.splitlines():
        raw_items.extend(line.split(","))

    prompts = []
    seen = set()
    for item in raw_items:
        p = item.strip()
        if p and p not in seen:
            prompts.append(p)
            seen.add(p)
    return prompts if prompts else ["object"]


def discover_config_paths() -> List[Path]:
    paths = list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))
    uniq = sorted({p.resolve() for p in paths}, key=lambda p: p.name)
    return uniq


def config_dropdown_choices() -> List[Tuple[str, str]]:
    return [(p.name, str(p)) for p in discover_config_paths()]


def auto_detect_weights(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg.get("output", {}).get("output_dir", "outputs/sam3_lora_full"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    best_path = output_dir / "best_lora_weights.pt"
    if best_path.exists():
        return best_path
    last_path = output_dir / "last_lora_weights.pt"
    if last_path.exists():
        return last_path
    return best_path


def describe_selected_config(config_path_str: str) -> str:
    if not config_path_str:
        return "No config selected."

    config_path = Path(config_path_str).expanduser().resolve()
    if not config_path.exists():
        return f"Config not found: `{config_path}`"

    weights = auto_detect_weights(config_path)
    if weights.exists():
        return (
            f"**Selected config:** `{config_path.name}`\n\n"
            f"**Auto-detected weights:** `{weights}`"
        )
    return (
        f"**Selected config:** `{config_path.name}`\n\n"
        f"**Expected weights path (missing):** `{weights}`\n\n"
        "Train first or place `best_lora_weights.pt` / `last_lora_weights.pt` in config output dir."
    )


def get_or_load_model(
    selected_config: str,
    resolution: int,
    threshold: float,
    nms_iou: float,
    device: str,
) -> Tuple[SAM3LoRAInference, str]:
    config_path = Path(selected_config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    weights_path = auto_detect_weights(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found for config '{config_path.name}'. Expected: {weights_path}"
        )

    key = (str(config_path), str(weights_path), int(resolution), device)
    with MODEL_LOCK:
        if key in MODEL_CACHE:
            model = MODEL_CACHE[key]
            status = "Reusing loaded model."
        else:
            MODEL_CACHE.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = SAM3LoRAInference(
                config_path=str(config_path),
                weights_path=str(weights_path),
                resolution=int(resolution),
                detection_threshold=float(threshold),
                nms_iou_threshold=float(nms_iou),
                device=device,
            )
            MODEL_CACHE[key] = model
            status = "Loaded new model."

    model.detection_threshold = float(threshold)
    model.nms_iou_threshold = float(nms_iou)
    return model, f"{status} Config: {config_path.name}, Weights: {weights_path.name}"


def render_boxes_image(image: Image.Image, results: dict) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    for idx in sorted(results.keys()):
        result = results[idx]
        if result["num_detections"] == 0 or result["boxes"] is None:
            continue

        boxes = result["boxes"]
        scores = result["scores"]
        prompt = result["prompt"]
        for i in range(result["num_detections"]):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
            score = float(scores[i]) if scores is not None else 0.0
            draw.text((x1 + 4, max(0.0, y1 - 16)), f"{prompt}: {score:.2f}", fill=BOX_COLOR)

    return out


def render_masks_image(image: Image.Image, results: dict) -> Image.Image:
    final = image.convert("RGBA")
    for idx in sorted(results.keys()):
        result = results[idx]
        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        masks = result["masks"]
        for i in range(result["num_detections"]):
            mask = masks[i].astype(bool)
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[mask] = [MASK_COLOR[0], MASK_COLOR[1], MASK_COLOR[2], 100]
            final = Image.alpha_composite(final, Image.fromarray(mask_rgba, mode="RGBA"))

    return final.convert("RGB")


def simplify_contours(mask_bool: np.ndarray, eps_ratio: float) -> List[np.ndarray]:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    simplified = []
    eps_ratio = max(0.0, float(eps_ratio))

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        epsilon = eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 3:
            simplified.append(approx[:, 0, :])
    return simplified


def render_contour_image(image: Image.Image, results: dict, contour_eps: float) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    for idx in sorted(results.keys()):
        result = results[idx]
        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        prompt = result["prompt"]
        masks = result["masks"]
        for i in range(result["num_detections"]):
            simplified_contours = simplify_contours(masks[i].astype(bool), contour_eps)
            first_point = None
            for poly in simplified_contours:
                pts = [(int(x), int(y)) for x, y in poly]
                if len(pts) < 2:
                    continue
                if first_point is None:
                    first_point = pts[0]
                draw.line(pts + [pts[0]], fill=CONTOUR_COLOR, width=2)
            if first_point is not None:
                draw.text((first_point[0] + 3, max(0, first_point[1] - 14)), prompt, fill=CONTOUR_COLOR)

    return out


def class_options_from_results(results: dict) -> List[str]:
    options = [ALL_CLASSES_LABEL]
    seen = {ALL_CLASSES_LABEL}
    for idx in sorted(results.keys()):
        prompt = results[idx].get("prompt", "")
        if prompt and prompt not in seen:
            options.append(prompt)
            seen.add(prompt)
    return options


def filter_results_by_class(results: dict, selected_class: str) -> dict:
    if not selected_class or selected_class == ALL_CLASSES_LABEL:
        return results
    filtered = {}
    for idx in sorted(results.keys()):
        if results[idx].get("prompt") == selected_class:
            filtered[idx] = results[idx]
    return filtered


def summarize_results(status: str, prompts: List[str], threshold: float, nms_iou: float, results: dict) -> str:
    summary_rows = []
    for idx in sorted(results.keys()):
        r = results[idx]
        max_score = float(np.max(r["scores"])) if r["scores"] is not None else None
        summary_rows.append(
            {
                "prompt": r["prompt"],
                "detections": int(r["num_detections"]),
                "max_score": None if max_score is None else round(max_score, 4),
            }
        )

    return (
        f"**Model status:** {status}\n\n"
        f"**Prompts:** {', '.join(prompts)}\n\n"
        f"**Threshold / NMS IoU:** {threshold:.2f} / {nms_iou:.2f}\n\n"
        "```json\n"
        f"{json.dumps(summary_rows, indent=2)}\n"
        "```"
    )


def run_inference(
    input_image: Image.Image,
    prompt_text: str,
    selected_class: str,
    selected_config: str,
    resolution: int,
    threshold: float,
    nms_iou: float,
    contour_eps: float,
    device: str,
):
    if input_image is None:
        raise gr.Error("Please upload an image.")

    prompts = parse_prompts(prompt_text)
    model, status = get_or_load_model(
        selected_config=selected_config,
        resolution=resolution,
        threshold=threshold,
        nms_iou=nms_iou,
        device=device,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        input_image.save(tmp.name)
        temp_path = tmp.name

    try:
        raw_results = model.predict(temp_path, prompts)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    # Strip helper image key so state remains concise and deterministic.
    results = {k: v for k, v in raw_results.items() if k != "_image"}
    source_image = input_image.convert("RGB")
    class_options = class_options_from_results(results)
    if selected_class not in class_options:
        selected_class = ALL_CLASSES_LABEL

    filtered_results = filter_results_by_class(results, selected_class)
    boxes_img = render_boxes_image(source_image, filtered_results)
    masks_img = render_masks_image(source_image, filtered_results)
    contour_img = render_contour_image(source_image, filtered_results, contour_eps=contour_eps)

    summary = summarize_results(status, prompts, threshold, nms_iou, results)
    infer_state = {"image": source_image, "results": results}
    class_update = gr.update(choices=class_options, value=selected_class, interactive=True)

    return boxes_img, masks_img, contour_img, summary, infer_state, class_update


def update_visualizations(infer_state: dict, contour_eps: float, selected_class: str):
    if not infer_state or "image" not in infer_state or "results" not in infer_state:
        return gr.update(), gr.update(), gr.update()
    image = infer_state["image"]
    filtered_results = filter_results_by_class(infer_state["results"], selected_class)
    return (
        render_boxes_image(image, filtered_results),
        render_masks_image(image, filtered_results),
        render_contour_image(image, filtered_results, contour_eps=contour_eps),
    )


def update_contour_only(infer_state: dict, contour_eps: float, selected_class: str):
    if not infer_state or "image" not in infer_state or "results" not in infer_state:
        return gr.update()
    filtered_results = filter_results_by_class(infer_state["results"], selected_class)
    return render_contour_image(infer_state["image"], filtered_results, contour_eps=contour_eps)


def build_ui() -> gr.Blocks:
    choices = config_dropdown_choices()
    if not choices:
        raise RuntimeError("No config files found in ./configs")

    default_value = str(DEFAULT_CONFIG.resolve()) if DEFAULT_CONFIG.exists() else choices[0][1]
    default_info = describe_selected_config(default_value)

    with gr.Blocks(title="SAM3 LoRA Web Inference") as demo:
        gr.Markdown(
            "# SAM3 LoRA Web Inference\n"
            "Upload an image, set prompts, choose a LoRA config, and run interactive inference."
        )

        infer_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                prompt_text = gr.Textbox(
                    label="Text Prompts",
                    value="building with no damage, destroyed building",
                    lines=3,
                    info="Use commas or new lines for multiple prompts.",
                )
                with gr.Row():
                    run_btn = gr.Button("Run Inference", variant="primary")
                    contour_btn = gr.Button("Update Contour Only")

                with gr.Accordion("Model & Advanced Settings", open=False):
                    selected_config = gr.Dropdown(
                        label="LoRA Config",
                        choices=choices,
                        value=default_value,
                    )
                    config_info = gr.Markdown(value=default_info)
                    resolution = gr.Slider(
                        label="Input Resolution",
                        minimum=288,
                        maximum=1008,
                        step=16,
                        value=1008,
                    )
                    threshold = gr.Slider(
                        label="Detection Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                    )
                    nms_iou = gr.Slider(
                        label="NMS IoU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                    )
                    contour_eps = gr.Slider(
                        label="Contour Simplification Epsilon Ratio (PolyDP)",
                        minimum=0.0,
                        maximum=0.05,
                        step=0.001,
                        value=0.005,
                        info="Used only for contour visualization. Inference outputs are unchanged.",
                    )
                    device = gr.Dropdown(
                        label="Device",
                        choices=["cuda", "cpu"],
                        value="cuda",
                    )

            with gr.Column(scale=1):
                class_filter = gr.Dropdown(
                    label="Class Filter",
                    choices=[ALL_CLASSES_LABEL],
                    value=ALL_CLASSES_LABEL,
                    interactive=False,
                    info="After inference, select one class to isolate visualizations.",
                )
                with gr.Tab("Boxes"):
                    output_boxes = gr.Image(type="pil", label="Box Visualization")
                with gr.Tab("Masks"):
                    output_masks = gr.Image(type="pil", label="Mask Visualization")
                with gr.Tab("Contours"):
                    output_contours = gr.Image(type="pil", label="Contour Visualization")
                output_summary = gr.Markdown(label="Summary")

        selected_config.change(
            fn=describe_selected_config,
            inputs=[selected_config],
            outputs=[config_info],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                input_image,
                prompt_text,
                class_filter,
                selected_config,
                resolution,
                threshold,
                nms_iou,
                contour_eps,
                device,
            ],
            outputs=[output_boxes, output_masks, output_contours, output_summary, infer_state, class_filter],
        )

        class_filter.change(
            fn=update_visualizations,
            inputs=[infer_state, contour_eps, class_filter],
            outputs=[output_boxes, output_masks, output_contours],
        )

        contour_eps.change(
            fn=update_contour_only,
            inputs=[infer_state, contour_eps, class_filter],
            outputs=[output_contours],
        )

        contour_btn.click(
            fn=update_contour_only,
            inputs=[infer_state, contour_eps, class_filter],
            outputs=[output_contours],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA interactive web app")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable public Gradio share link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
