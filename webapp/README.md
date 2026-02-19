# Web App (Interactive LoRA Inference)

This web app lets you:

1. Choose LoRA model by config dropdown (weights auto-resolved from config output dir)
2. Upload images interactively
3. Run inference with one or multiple text prompts
4. See three separate visualization outputs:
   - Boxes
   - Masks
   - Contours (Polygon + `approxPolyDP` simplification)
5. Toggle right-panel visualizations by class using Class Filter (`All classes` or one prompt)
6. Tune threshold / NMS / contour epsilon

## Setup

From repo root:

```bash
cd /home/frank/Desktop/AGENT/SAMS/SAM3_LoRA
uv sync --extra webapp
```

## Run

```bash
uv run python webapp/app.py --server-name 0.0.0.0 --server-port 7860
```

Then open:

- `http://localhost:7860`

## Notes

- Weights are auto-detected from selected config:
  - `<output.output_dir>/best_lora_weights.pt`
  - fallback: `<output.output_dir>/last_lora_weights.pt`
- Multiple prompts are supported via commas or new lines.
- Class Filter applies to all three right-side visualizations (boxes/masks/contours).
- The app caches one loaded model instance in memory for faster repeat inference.
- Contour epsilon only affects contour visualization, not model inference.
