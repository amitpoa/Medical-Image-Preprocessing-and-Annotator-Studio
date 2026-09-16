# Medical-Image-Preprocessing-and-Annotator-Studio
Batch preprocessing for X-ray/MRI/CT scans: tilt correction, contrast enhancement, and LabelMe/.text annotation transformation.
# Working Site URL
https://medical-image-preprocessing-and-ddta.onrender.com/
# 🦴 Medical Image Preprocessing Studio

A Gradio app that batch-preprocesses medical film scans (X-ray, MRI, CT) together with
LabelMe-style JSON annotations, producing a clean, model-ready dataset.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

This launches a local Gradio server (prints a URL, e.g. `http://127.0.0.1:7860`).

## What it does

For each uploaded image (optionally paired with a matching `.json` annotation file):

1. **Deskew (optional)** — detects the film's corners against its background and
   perspective-warps a tilted scan to axis-aligned. Falls back to the original image
   if no reliable film boundary is found.
2. **Black border removal** — crops away near-black scanner background.
3. **CLAHE contrast enhancement** — brings out bone/soft-tissue detail.
4. **Aspect-ratio-preserving resize** — onto a square canvas of your chosen size,
   padded with the image's own median gray value.
5. **Annotation transform** — polygon/rectangle shapes in the JSON are carried through
   every step above so coordinates stay correct on the final image.
6. **Overlay + dataset sorting (optional)** — draws the shapes on the processed image
   and sorts annotated images into per-class folders.

## Uploading files

Match files by identical base filename:

| File | Required? | Purpose |
|---|---|---|
| `scan01.png` / `.jpg` / `.tif` | required | the image itself |
| `scan01.json` | optional | LabelMe-style shapes: `points`, `label`, `shape_type` |
| `scan01.txt` | optional | plain-text label used to fill in a shape's label *only* if the JSON entry doesn't already have one |

Images with no matching JSON are still cleaned and resized, just without annotations.

## Output

Downloaded as a single ZIP:

```
clean_images/           # preprocessed grayscale PNGs
clean_json/              # JSON with transformed coordinates matching clean_images
annotated_images/        # (if enabled) images with shapes drawn on
label_wise_dataset/      # (if enabled) annotated images + JSON sorted by class folder
processing_report.json   # run settings, per-class shape counts, any files that failed
```

## Notes on robustness

- A single corrupt image or malformed JSON does not stop the batch — it's skipped and
  reported at the end (in the UI and in `processing_report.json`).
- Class-to-color assignment is scoped to a single processing run rather than kept as
  global state, so concurrent users on a shared deployment never see each other's
  color assignments bleed together.
- Rectangle annotations are expanded to their 4 corners before any perspective warp
  and rebuilt as an axis-aligned box afterward, so the box always fully encloses the
  (possibly rotated) region instead of clipping through it.
