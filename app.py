import os
import json
import cv2
import numpy as np
import tempfile
import zipfile
import traceback
import gradio as gr

# -------------------------------------------------
# HUGGING FACE SPACES ZEROGPU COMPATIBILITY
# -------------------------------------------------
# If this Space's hardware is set to ZeroGPU, HF requires at least one
# function decorated with @spaces.GPU to be registered at startup, or the
# Space fails to build with "No @spaces.GPU function detected during
# startup". This app does no GPU work at all (pure OpenCV/CPU), so the
# decorator below is a no-op wrapper everywhere except on a ZeroGPU Space,
# where it satisfies that startup requirement. If you deploy to CPU Basic
# hardware instead, this has no effect either way.
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    spaces = None
    HAS_SPACES = False


def gpu_compatible(fn):
    if HAS_SPACES:
        return spaces.GPU(fn)
    return fn

# COLOR PALETTE FOR ANNOTATION VISUALIZATION
# -------------------------------------------------
# Light/pastel colors (BGR order, since OpenCV draws in BGR) so the mask
# fill and box stay easy to read against both dark background and bright
# bone, without needing an outline halo.
COLOR_PALETTE = [
    (255, 181, 100),  # Light Blue
    (144, 238, 144),  # Light Green
    (127, 127, 255),  # Light Red / Salmon
    (255, 255, 150),  # Light Cyan
    (255, 150, 255),  # Light Pink / Magenta
    (107, 183, 255),  # Light Orange
    (255, 160, 216),  # Light Violet
    (150, 255, 255),  # Light Yellow
    (140, 255, 185),  # Light Lime
]


def get_class_color(label_name, class_color_map):
    """Assigns a stable color to a class name, scoped to a single batch run.

    NOTE: this map is created fresh per call to batch_preprocess_xrays and
    passed through explicitly, rather than kept as module-level global state.
    A global dict would leak color assignments between unrelated uploads if
    this app is ever served to more than one user/session at a time.
    """
    if label_name not in class_color_map:
        idx = len(class_color_map) % len(COLOR_PALETTE)
        class_color_map[label_name] = COLOR_PALETTE[idx]
    return class_color_map[label_name]


def darken_color(color_bgr, factor=0.5):
    """Returns a darker shade of a light palette color, used only for the
    label background so white text has enough contrast to read clearly."""
    return tuple(int(c * factor) for c in color_bgr)


def color_to_css(color_bgr):
    b, g, r = color_bgr
    return f"rgb({r},{g},{b})"


# -------------------------------------------------
# DESKEW / CORNER DETECTION (adapted from a reference app)
# -------------------------------------------------

def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_film_corners(gray):
    """
    Try to find the 4 corners of the tilted film against its background.
    Returns a (4,2) float32 array of corners, or None if nothing reasonable
    was found (caller should then skip deskew and use the raw image).
    """
    h, w = gray.shape[:2]
    total_area = float(h * w)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    border = int(max(h, w) * 0.02) or 1
    border_pixels = np.concatenate([
        blurred[:border, :].ravel(),
        blurred[-border:, :].ravel(),
        blurred[:, :border].ravel(),
        blurred[:, -border:].ravel(),
    ])
    background_is_bright = np.median(border_pixels) > 127

    invert = background_is_bright
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)

    k = max(15, (max(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    ratio = area / total_area

    if not (0.05 < ratio < 0.97):
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    inside_mean = gray[mask == 255].mean()
    border_mean = border_pixels.mean()
    if background_is_bright and inside_mean >= border_mean:
        return None
    if not background_is_bright and inside_mean <= border_mean:
        return None

    return order_points(box.astype(np.float32))


def deskew_and_crop(gray_img, points_list):
    """
    Detect the film region, straighten it (perspective warp), and crop away
    the background. Works on a single-channel (grayscale) image.

    points_list: list of Nx2 float32 arrays (one per shape) to transform
                 the same way as the image.

    Returns: warped_gray, transformed_points_list, used_deskew (bool)
    """
    corners = find_film_corners(gray_img)

    if corners is None:
        return gray_img.copy(), [pts.copy() for pts in points_list], False

    (tl, tr, br, bl) = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = max(int(max(width_top, width_bottom)), 2)
    out_h = max(int(max(height_left, height_right)), 2)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray_img, M, (out_w, out_h))

    transformed_points = []
    for pts in points_list:
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        new_pts = cv2.perspectiveTransform(pts_reshaped, M).reshape(-1, 2)
        transformed_points.append(new_pts)

    return warped, transformed_points, True


# -------------------------------------------------
# PREPROCESSING PIPELINE HELPERS
# -------------------------------------------------

def remove_black_border(img):
    """Removes scanner black background while preserving anatomical areas."""
    mask = img > 3
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img, 0, 0

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    margin = 15
    x0 = max(x0 - margin, 0)
    y0 = max(y0 - margin, 0)
    x1 = min(x1 + margin, img.shape[1])
    y1 = min(y1 + margin, img.shape[0])

    crop = img[y0:y1, x0:x1]
    return crop, x0, y0


def resize_keep_ratio(img, target_size):
    """Resizes grayscale image keeping aspect ratio and centers it on a median-padded canvas."""
    h, w = img.shape
    scale = min(target_size / w, target_size / h)

    nw = max(int(w * scale), 1)
    nh = max(int(h * scale), 1)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    bg = np.median(img)
    canvas[:] = bg

    x = (target_size - nw) // 2
    y = (target_size - nh) // 2

    canvas[y:y + nh, x:x + nw] = resized
    return canvas, scale, x, y


def transform_points_crop_resize(points, crop_x, crop_y, scale, offset_x, offset_y, target_size):
    """Applies the crop-offset + scale + pad-offset transform to a set of points."""
    new_points = []
    for p in points:
        x, y = p
        x -= crop_x
        y -= crop_y
        x *= scale
        y *= scale
        x += offset_x
        y += offset_y
        x = max(0, min(target_size - 1, x))
        y = max(0, min(target_size - 1, y))
        new_points.append([float(x), float(y)])
    return new_points


def overlay_annotations(img, shapes, class_color_map):
    """Overlays polygons and annotated bounding boxes with labels."""
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas_size = max(vis_img.shape[0], vis_img.shape[1])

    # Keep the outline thin and the label small/subtle even on large canvases —
    # a fixed pixel thickness/font that looked fine at 640px reads as a bold,
    # heavy border on a much bigger image, so scale both from canvas size.
    line_thickness = max(1, round(canvas_size / 800))
    font_scale = max(0.4, canvas_size / 1400)
    font_thickness = 1
    label_padding = max(2, round(canvas_size / 320))

    for shape in shapes:
        shape_type = shape.get("shape_type", "polygon")
        points = np.array(shape["points"], dtype=np.int32)
        if len(points) < 2:
            continue
        label = shape.get("label", "unclassified")
        color = get_class_color(label, class_color_map)

        if shape_type != "rectangle" and len(points) > 2:
            overlay = vis_img.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.25, vis_img, 0.75, 0, vis_img)
            cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

    for shape in shapes:
        shape_type = shape.get("shape_type", "polygon")
        points = np.array(shape["points"], dtype=np.int32)
        if len(points) < 2:
            continue
        label = shape.get("label", "unclassified")
        color = get_class_color(label, class_color_map)

        if shape_type == "rectangle" or len(points) == 2:
            pt1 = tuple(points[0])
            pt2 = tuple(points[1])
            cv2.rectangle(vis_img, pt1, pt2, color, line_thickness, cv2.LINE_AA)

            display_label = str(label).lower()
            text_pos = (pt1[0], max(15, pt1[1] - 5))
            (w, h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_bg_color = darken_color(color, factor=0.5)
            cv2.rectangle(
                vis_img,
                (text_pos[0], text_pos[1] - h - label_padding),
                (text_pos[0] + w + label_padding, text_pos[1] + 2),
                label_bg_color, -1
            )
            cv2.putText(
                vis_img, display_label,
                (text_pos[0] + label_padding // 2, text_pos[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

    return vis_img


def build_color_legend_html(class_color_map):
    """Builds a small inline HTML legend mapping class name -> swatch color,
    shown in the UI so users can see which color corresponds to which class."""
    if not class_color_map:
        return ""
    swatches = ""
    for label, color in class_color_map.items():
        css_color = color_to_css(color)
        swatches += (
            f'<span style="display:inline-flex;align-items:center;margin:4px 10px 4px 0;">'
            f'<span style="width:14px;height:14px;border-radius:3px;background:{css_color};'
            f'display:inline-block;margin-right:6px;border:1px solid #4b5563;"></span>'
            f'<span style="font-size:0.9em;">{label}</span></span>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;padding:6px 0;">{swatches}</div>'


# -------------------------------------------------
# MAIN AGGREGATED PROCESSOR FUNCTION
# -------------------------------------------------

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@gpu_compatible
def batch_preprocess_xrays(files, target_size, generate_annotations, clip_limit, enable_deskew,
                            progress=gr.Progress()):
    if not files:
        return "❌ Please upload image and JSON files to proceed.", None, [], ""

    # --- Input validation / sane clamping ---
    warnings = []
    try:
        target_size = int(target_size)
    except (TypeError, ValueError):
        target_size = 640
        warnings.append("Target size was invalid — defaulted to 640.")
    if target_size < 64:
        warnings.append(f"Target size {target_size} too small — clamped to 64.")
        target_size = 64
    elif target_size > 4096:
        warnings.append(f"Target size {target_size} too large — clamped to 4096.")
        target_size = 4096

    try:
        clip_limit = float(clip_limit)
    except (TypeError, ValueError):
        clip_limit = 1.5
    clip_limit = max(0.1, min(clip_limit, 20.0))

    work_dir = tempfile.mkdtemp()
    img_out = os.path.join(work_dir, "clean_images")
    json_out = os.path.join(work_dir, "clean_json")
    vis_out = os.path.join(work_dir, "annotated_images")
    seg_out = os.path.join(work_dir, "label_wise_dataset")
    preview_dir = os.path.join(work_dir, "gallery_previews")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(json_out, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    if generate_annotations:
        os.makedirs(vis_out, exist_ok=True)
        os.makedirs(seg_out, exist_ok=True)

    image_dict = {}
    json_dict = {}
    txt_dict = {}

    for f in files:
        filename = os.path.basename(f.name)
        # Skip OS-generated junk (macOS resource forks, hidden dotfiles, etc.)
        if filename.startswith(".") or filename.startswith("._"):
            continue
        base_name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        if ext_lower in VALID_IMAGE_EXTS:
            image_dict[base_name] = f.name
        elif ext_lower == ".json":
            json_dict[base_name] = f.name
        elif ext_lower == ".txt":
            txt_dict[base_name] = f.name

    if not image_dict:
        return "❌ No valid medical images detected (.jpg, .jpeg, .png, .tif, .tiff).", None, [], ""

    preview_gallery = []
    processed_count = 0
    annotated_count = 0
    deskewed_count = 0
    failed_files = []
    class_counts = {}
    class_color_map = {}  # scoped to this single run — never a module-level global

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    items = list(image_dict.items())
    total = len(items)

    for idx, (base_name, img_path) in enumerate(items):
        progress((idx + 1) / total, desc=f"Processing {base_name} ({idx + 1}/{total})")
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed_files.append(f"{base_name}: could not read image file (corrupt or unsupported).")
                continue

            clean_img_name = f"clean_{base_name}.png"

            matching_json_path = json_dict.get(base_name)
            matching_txt_path = txt_dict.get(base_name)
            data = None
            shapes_points = []   # list of raw np.float32 point arrays, one per shape
            shape_refs = []       # parallel list of the shape dicts they came from
            shape_is_rect = []    # parallel bool: was this originally a 2-point rectangle?

            # A matching .txt file is optional and only used to fill in a label
            # when a shape's JSON entry doesn't already carry one. It never
            # overrides a label that's already present in the JSON.
            txt_label = None
            if matching_txt_path and os.path.exists(matching_txt_path):
                with open(matching_txt_path, "r") as tf:
                    txt_label = tf.read().strip()

            if matching_json_path and os.path.exists(matching_json_path):
                try:
                    with open(matching_json_path, "r") as jf:
                        data = json.load(jf)
                except (json.JSONDecodeError, OSError) as e:
                    failed_files.append(f"{base_name}: JSON could not be parsed ({e}) — image processed without annotations.")
                    data = None

                if data is not None:
                    for shape in data.get("shapes", []):
                        if txt_label and not str(shape.get("label", "")).strip():
                            shape["label"] = txt_label
                        raw_pts = np.array(shape.get("points", []), dtype=np.float32)
                        if raw_pts.ndim != 2 or len(raw_pts) < 2:
                            continue  # malformed shape entry — skip rather than crash
                        shape_type = shape.get("shape_type", "polygon")
                        is_rect = (shape_type == "rectangle" or len(raw_pts) == 2)

                        if is_rect and len(raw_pts) == 2:
                            # Expand the 2-point rectangle to all 4 corners so the
                            # perspective warp below can be applied to the full box,
                            # not just 2 of its corners (which would no longer
                            # describe an axis-aligned box after a tilt correction).
                            x1, y1 = raw_pts[0]
                            x2, y2 = raw_pts[1]
                            xmin, xmax = min(x1, x2), max(x1, x2)
                            ymin, ymax = min(y1, y2), max(y1, y2)
                            pts = np.array(
                                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                                dtype=np.float32,
                            )
                        else:
                            pts = raw_pts

                        shapes_points.append(pts)
                        shape_refs.append(shape)
                        shape_is_rect.append(is_rect)

            # --- Step 1: deskew / straighten the film + crop background ---
            used_deskew = False
            if enable_deskew:
                working_img, shapes_points, used_deskew = deskew_and_crop(img, shapes_points)
                if used_deskew:
                    deskewed_count += 1
            else:
                working_img = img

            # Rectangles that were expanded to 4 corners must be collapsed back
            # down to an axis-aligned 2-point box (min/max of the transformed
            # corners) so the box still fully encloses the warped region instead
            # of cutting through it.
            for i, is_rect in enumerate(shape_is_rect):
                if is_rect:
                    pts = shapes_points[i]
                    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
                    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
                    shapes_points[i] = np.array(
                        [[xmin, ymin], [xmax, ymax]], dtype=np.float32
                    )

            # --- Step 2: black-border crop + CLAHE + padded resize ---
            cropped, cx, cy = remove_black_border(working_img)
            enhanced = clahe.apply(cropped)
            final_img, scale, ox, oy = resize_keep_ratio(enhanced, target_size)

            out_img_path = os.path.join(img_out, clean_img_name)
            cv2.imwrite(out_img_path, final_img)

            preview_file_path = os.path.join(preview_dir, f"preview_{base_name}.png")

            if data is not None:
                for shape, pts in zip(shape_refs, shapes_points):
                    shape["points"] = transform_points_crop_resize(
                        pts, cx, cy, scale, ox, oy, target_size
                    )

                data["imageWidth"] = target_size
                data["imageHeight"] = target_size
                data["imagePath"] = clean_img_name

                with open(os.path.join(json_out, f"clean_{base_name}.json"), "w") as jf:
                    json.dump(data, jf, indent=4)

                if generate_annotations:
                    shapes = data.get("shapes", [])
                    if shapes:
                        for shape in shapes:
                            lbl = shape.get("label", "unclassified")
                            class_counts[lbl] = class_counts.get(lbl, 0) + 1

                        vis_img = overlay_annotations(final_img, shapes, class_color_map)
                        cv2.imwrite(os.path.join(vis_out, f"vis_{clean_img_name}"), vis_img)

                        primary_label = str(shapes[0].get("label", "unclassified")).strip().replace("/", "_").replace("\\", "_")
                        class_folder = os.path.join(seg_out, primary_label or "unclassified")
                        os.makedirs(class_folder, exist_ok=True)

                        cv2.imwrite(os.path.join(class_folder, f"vis_{clean_img_name}"), vis_img)
                        with open(os.path.join(class_folder, f"clean_{base_name}.json"), "w") as jf:
                            json.dump(data, jf, indent=4)

                        cv2.imwrite(preview_file_path, vis_img)
                        preview_gallery.append(preview_file_path)
                        annotated_count += 1
                    else:
                        cv2.imwrite(preview_file_path, final_img)
                        preview_gallery.append(preview_file_path)
                else:
                    cv2.imwrite(preview_file_path, final_img)
                    preview_gallery.append(preview_file_path)
            else:
                cv2.imwrite(preview_file_path, final_img)
                preview_gallery.append(preview_file_path)

            processed_count += 1

        except Exception as e:
            # A single bad file should never take down the whole batch.
            failed_files.append(f"{base_name}: unexpected error — {e}")
            continue

    # --- Write a machine-readable summary report into the dataset ---
    report = {
        "target_size": target_size,
        "clahe_clip_limit": clip_limit,
        "deskew_enabled": bool(enable_deskew),
        "total_images_uploaded": total,
        "processed_count": processed_count,
        "annotated_count": annotated_count,
        "deskewed_count": deskewed_count,
        "failed_files": failed_files,
        "class_counts": class_counts,
    }
    with open(os.path.join(work_dir, "processing_report.json"), "w") as rf:
        json.dump(report, rf, indent=2)

    zip_path = os.path.join(work_dir, "preprocessed_dataset.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(work_dir):
            for file_in_dir in files_in_dir:
                if file_in_dir.endswith(".zip") or "gallery_previews" in root:
                    continue
                full_path = os.path.join(root, file_in_dir)
                arcname = os.path.relpath(full_path, work_dir)
                zipf.write(full_path, arcname)

    # --- Build status markdown ---
    status_lines = [
        "✅ **Processing Complete!**",
        "",
        f"- Total Scans Uploaded: `{total}`",
        f"- Successfully Processed: `{processed_count}`",
        f"- Matching JSON Annotations Transformed: `{annotated_count}`",
        f"- Images Successfully Deskewed: `{deskewed_count}`",
        f"- Target Canvas Resolution: `{target_size} x {target_size}`",
        f"- CLAHE Clip Limit Used: `{clip_limit}`",
    ]

    if class_counts:
        status_lines.append("")
        status_lines.append("**Class distribution (shape count):**")
        for lbl, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
            status_lines.append(f"- `{lbl}`: {count}")

    if warnings:
        status_lines.append("")
        status_lines.append("**⚠️ Input adjustments:**")
        for w in warnings:
            status_lines.append(f"- {w}")

    if failed_files:
        status_lines.append("")
        status_lines.append(f"**⚠️ {len(failed_files)} file(s) had issues (skipped, rest processed normally):**")
        for msg in failed_files[:20]:
            status_lines.append(f"- {msg}")
        if len(failed_files) > 20:
            status_lines.append(f"- ...and {len(failed_files) - 20} more (see `processing_report.json` in the ZIP).")

    status_msg = "\n".join(status_lines)
    legend_html = build_color_legend_html(class_color_map)

    return status_msg, zip_path, preview_gallery[:12], legend_html


# -------------------------------------------------
# GRADIO UI INTERFACE
# -------------------------------------------------

custom_css = """
[data-testid="file-upload"] .file-preview,
[data-testid="file-upload"] .file-preview-holder,
[data-testid="file-upload"] div:has(> table),
div.file-preview,
div.file-preview-holder {
    max-height: 250px !important;
    overflow-y: auto !important;
    display: block !important;
}

[data-testid="file-upload"] table {
    width: 100% !important;
}

[data-testid="file-upload"] ::-webkit-scrollbar {
    width: 8px !important;
}
[data-testid="file-upload"] ::-webkit-scrollbar-track {
    background: #1f2937 !important;
}
[data-testid="file-upload"] ::-webkit-scrollbar-thumb {
    background-color: #4b5563 !important;
    border-radius: 4px !important;
}
"""

ABOUT_MD = """
## What this tool does

A batch preprocessing + annotation pipeline for medical film scans (X-ray, MRI, CT),
intended to prepare a clean, model-ready dataset from raw scans and their LabelMe-style
JSON annotations.

### Pipeline steps, in order
1. **Deskew (optional)** — detects the film's corners against its background using
   adaptive thresholding + contour analysis, then perspective-warps it to axis-aligned.
   Falls back to the untouched image if no reliable film boundary is found.
2. **Black border removal** — crops away near-black scanner background with a small margin.
3. **CLAHE contrast enhancement** — adaptive histogram equalization to bring out bone/soft
   tissue detail.
4. **Aspect-ratio-preserving resize** — scales onto a square canvas of your chosen size,
   padded with the image's own median gray value (never stretched/distorted).
5. **Annotation transform** — any polygon or rectangle shape in a matching JSON file is
   carried through every geometric step above, so coordinates always line up with the
   final processed image.
6. **Overlay + dataset sorting (optional)** — draws the transformed shapes on the final
   image and sorts annotated images into per-class folders by their first shape's label.

### File naming convention
Upload images together with their annotation files, matched by identical base filename:
- `scan01.png` — the image
- `scan01.json` — LabelMe-style shapes (`points`, `label`, `shape_type`) *(optional)*
- `scan01.txt` — plain text label, used only to fill in a label for shapes in the JSON
  that don't already have one *(optional, never overrides an existing label)*

Images without a matching JSON are still cleaned/resized and included in the output —
they just won't have annotation overlays or transformed coordinates.

### Output (zipped)
```
clean_images/          # preprocessed grayscale PNGs
clean_json/             # JSON with transformed coordinates matching clean_images
annotated_images/       # (if enabled) images with shapes drawn on
label_wise_dataset/     # (if enabled) annotated images + JSON sorted by class folder
processing_report.json  # run settings, per-class counts, any files that failed
```

### Notes
- Rectangle shapes are expanded to all 4 corners before any perspective warp and
  rebuilt as an axis-aligned box afterward, so the box always fully encloses the
  (possibly rotated) region rather than clipping through it.
- One bad/corrupt file will not stop the batch — it's reported at the end instead.
"""

with gr.Blocks(title="Medical Film Preprocessing Studio", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🦴 Medical Image Preprocessing Studio
        Batch preprocess medical film scans (X-Ray, MRI, CT) with tilt correction (deskew),
        black border cropping, CLAHE enhancement, and automatic label-coordinate transformation.
        """
    )

    with gr.Tabs():
        with gr.TabItem("🚀 Process"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Preprocessing Configurations")

                    target_size = gr.Number(
                        value=640,
                        label="Target Size (Pixels)",
                        precision=0,
                        info="Generates a square canvas [Size x Size], 64–4096px, with aspect-ratio preservation and median padding.",
                    )

                    clahe_clip = gr.Slider(
                        minimum=0.5,
                        maximum=5.0,
                        value=1.5,
                        step=0.1,
                        label="CLAHE Contrast Limit",
                        info="Enhances soft tissue and bone definition.",
                    )

                    enable_deskew = gr.Checkbox(
                        value=True,
                        label="Detect & Straighten Tilted Film (Deskew)",
                        info="Detects the film's corners against its background and perspective-warps it to axis-aligned before further processing. Falls back to the raw image if no reliable film boundary is found.",
                    )

                    generate_annotations = gr.Checkbox(
                        value=True,
                        label="Generate Annotated Overlays & Segregated Dataset",
                        info="If checked, renders bounding boxes/polygons for images with matching JSON files and sorts them into label-wise folders.",
                    )

                    uploaded_files = gr.File(
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json", ".txt"],
                        label="Upload Scans, JSON Files, and Optional .txt Label Files",
                        height=250
                    )
                    gr.Markdown(
                        "*A `.txt` file with the same base filename as an image can supply a label "
                        "for shapes in its matching JSON that don't already have one (e.g. `scan01.png`, "
                        "`scan01.json`, `scan01.txt`).*"
                    )

                    with gr.Row():
                        btn_submit = gr.Button("🚀 Run Aggregated Processing", variant="primary")
                        btn_clear = gr.ClearButton(value="🗑️ Clear")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Processing Summary & Outputs")
                    status_output = gr.Markdown(value="*Upload files and click process to view results.*")

                    file_download = gr.File(label="📦 Download Complete Dataset (.ZIP)")

                    gr.Markdown("#### Class Color Legend")
                    legend_output = gr.HTML(value="")

                    gr.Markdown("#### Preview Samples (Up to 12)")
                    gallery_preview = gr.Gallery(
                        label="Output Gallery",
                        columns=4,
                        height="450px",
                        object_fit="contain"
                    )

            btn_submit.click(
                fn=batch_preprocess_xrays,
                inputs=[uploaded_files, target_size, generate_annotations, clahe_clip, enable_deskew],
                outputs=[status_output, file_download, gallery_preview, legend_output],
            )
            btn_clear.add([
                uploaded_files, status_output, file_download, gallery_preview, legend_output
            ])

        with gr.TabItem("ℹ️ About / Help"):
            gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    # Render (and most PaaS hosts) assign a dynamic port via the PORT env var
    # and expect the process to bind 0.0.0.0. Falls back to 7860 for local runs.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
