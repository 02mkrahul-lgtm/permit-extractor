"""Layout segmentation: detect region bounding boxes on a sheet image.

Primary path: OpenCV line-detection + spatial heuristics.
Fallback path: pure heuristic bbox rules (used if opencv import fails or
               if the primary path finds nothing useful).

Output: List[Region] with bboxes in both PDF points and pixels.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from permit_extractor.models.regions import BoundingBox
from permit_extractor.models.regions import Region, RegionType, SheetInfo

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not available; using heuristic layout segmentation fallback")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_sheet(
    image: Image.Image,
    sheet_info: SheetInfo,
) -> list[Region]:
    """Return detected regions for one sheet.

    Tries the OpenCV path first; falls back to heuristics if cv2 is absent
    or if the detection yields no title block (which is always expected).
    """
    if _CV2_AVAILABLE:
        try:
            regions = _segment_opencv(image, sheet_info)
            if any(r.region_type == RegionType.TITLE_BLOCK for r in regions):
                return regions
        except Exception as exc:
            logger.warning("OpenCV segmentation failed (%s); using heuristic fallback", exc)

    return _segment_heuristic(image, sheet_info)


# ---------------------------------------------------------------------------
# OpenCV path
# ---------------------------------------------------------------------------

def _segment_opencv(image: Image.Image, sheet_info: SheetInfo) -> list[Region]:
    """Detect regions via line detection and spatial analysis."""
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h_px, w_px = gray.shape

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines (long ones only — borders between regions)
    min_line_len = int(min(w_px, h_px) * 0.15)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=10,
    )

    h_lines: list[tuple[int, int, int, int]] = []  # (x1,y1,x2,y2)
    v_lines: list[tuple[int, int, int, int]] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10:
                h_lines.append((x1, y1, x2, y2))
            elif angle > 80:
                v_lines.append((x1, y1, x2, y2))

    regions: list[Region] = []

    # --- Title block: look for a dense rectangle in the bottom-right quadrant ---
    tb = _find_title_block_opencv(gray, w_px, h_px, h_lines, v_lines)
    if tb:
        regions.append(_make_region(RegionType.TITLE_BLOCK, tb, sheet_info, image))

    # --- Schedule blocks: look for grid-like rectangles (regular line spacing) ---
    schedules = _find_schedules_opencv(gray, w_px, h_px, h_lines, v_lines, exclude=tb)
    for sched_bbox in schedules:
        regions.append(_make_region(RegionType.SCHEDULE, sched_bbox, sheet_info, image))

    # --- Notes regions: dense text column(s) at sheet margins ---
    notes = _find_notes_opencv(gray, w_px, h_px, exclude_boxes=[tb] + schedules if tb else schedules)
    for note_bbox in notes:
        regions.append(_make_region(RegionType.NOTES, note_bbox, sheet_info, image))

    # --- Everything else is drawing body ---
    body_bbox = (0, 0, w_px, h_px)  # entire sheet as fallback body
    regions.append(_make_region(RegionType.DRAWING_BODY, body_bbox, sheet_info, image, label="Drawing Body"))

    return regions


def _find_title_block_opencv(
    gray: np.ndarray,
    w_px: int,
    h_px: int,
    h_lines: list,
    v_lines: list,
) -> Optional[tuple[int, int, int, int]]:
    """Locate title block in the bottom-right quadrant."""
    # Look in the rightmost 35% and bottom 30% of the sheet
    x_start = int(w_px * 0.65)
    y_start = int(h_px * 0.70)

    # Find horizontal lines in that zone
    zone_hlines = [l for l in h_lines if l[1] > y_start or l[3] > y_start]
    zone_vlines = [l for l in v_lines if l[0] > x_start or l[2] > x_start]

    if zone_hlines and zone_vlines:
        xs = [l[0] for l in zone_vlines] + [l[2] for l in zone_vlines]
        ys = [l[1] for l in zone_hlines] + [l[3] for l in zone_hlines]
        x0 = max(x_start, min(xs))
        y0 = max(y_start, min(ys))
        x1 = min(w_px, max(xs))
        y1 = min(h_px, max(ys))
        if (x1 - x0) > w_px * 0.05 and (y1 - y0) > h_px * 0.05:
            return (x0, y0, x1, y1)

    # Fallback: fixed bottom-right corner
    return (int(w_px * 0.65), int(h_px * 0.75), w_px, h_px)


def _find_schedules_opencv(
    gray: np.ndarray,
    w_px: int,
    h_px: int,
    h_lines: list,
    v_lines: list,
    exclude: Optional[tuple] = None,
) -> list[tuple[int, int, int, int]]:
    """Find grid-like rectangular regions that look like schedules."""
    # Use morphological operations to detect table-like grid regions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2
    )

    # Detect horizontal and vertical lines morphologically
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_px * 0.05), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h_px * 0.02)))

    h_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    grid_mask = cv2.bitwise_or(h_detected, v_detected)
    grid_dilated = cv2.dilate(grid_mask, np.ones((5, 5), np.uint8), iterations=3)

    contours, _ = cv2.findContours(grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    schedules = []
    min_area = w_px * h_px * 0.01  # at least 1% of sheet
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area:
            continue
        # Aspect ratio filter: schedules are wider than tall or roughly square
        if ch > cw * 3:
            continue
        bbox = (x, y, x + cw, y + ch)
        if exclude and _boxes_overlap(bbox, exclude, iou_threshold=0.3):
            continue
        schedules.append(bbox)

    return schedules[:5]  # cap at 5 to avoid noise


def _find_notes_opencv(
    gray: np.ndarray,
    w_px: int,
    h_px: int,
    exclude_boxes: list,
) -> list[tuple[int, int, int, int]]:
    """Find dense text columns that look like general notes."""
    # Use text density: notes regions have high pixel density in horizontal runs
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_px * 0.03), 3))
    text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, text_kernel)
    text_dilated = cv2.dilate(text_mask, np.ones((10, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notes = []
    min_area = w_px * h_px * 0.02
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area:
            continue
        # Notes columns are taller than wide
        if cw > ch * 1.5:
            continue
        bbox = (x, y, x + cw, y + ch)
        if any(_boxes_overlap(bbox, ex, iou_threshold=0.3) for ex in exclude_boxes if ex):
            continue
        notes.append(bbox)

    return notes[:3]


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    iou_threshold: float = 0.3,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    iou = inter / (area_a + area_b - inter + 1e-6)
    return iou >= iou_threshold


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _segment_heuristic(image: Image.Image, sheet_info: SheetInfo) -> list[Region]:
    """Fixed-proportion bbox rules when OpenCV is unavailable or fails."""
    w = sheet_info.image_width_px
    h = sheet_info.image_height_px

    regions = [
        _make_region(RegionType.TITLE_BLOCK,
                     (int(w * 0.65), int(h * 0.75), w, h), sheet_info, image,
                     label="Title Block (heuristic)"),
        _make_region(RegionType.DRAWING_BODY,
                     (0, 0, w, h), sheet_info, image,
                     label="Drawing Body"),
    ]
    return regions


# ---------------------------------------------------------------------------
# Helper: convert pixel bbox → Region with dual coordinates
# ---------------------------------------------------------------------------

def _make_region(
    region_type: RegionType,
    bbox_px: tuple[int, int, int, int],
    sheet_info: SheetInfo,
    image: Image.Image,
    label: Optional[str] = None,
) -> Region:
    x0_px, y0_px, x1_px, y1_px = bbox_px
    # Convert pixels → PDF points
    scale = 72.0 / sheet_info.render_dpi
    bbox_pt = BoundingBox(
        x0=x0_px * scale,
        y0=y0_px * scale,
        x1=x1_px * scale,
        y1=y1_px * scale,
        page_index=sheet_info.page_index,
    )
    bbox_px_obj = BoundingBox(
        x0=float(x0_px),
        y0=float(y0_px),
        x1=float(x1_px),
        y1=float(y1_px),
        page_index=sheet_info.page_index,
    )
    return Region(
        region_type=region_type,
        bbox=bbox_pt,
        bbox_px=bbox_px_obj,
        label=label or region_type.value.replace("_", " ").title(),
    )


def crop_region(image: Image.Image, region: Region) -> Image.Image:
    """Crop a PIL Image to the region's pixel bbox."""
    if region.bbox_px is None:
        raise ValueError(f"Region {region.region_type} has no pixel bbox — cannot crop")
    b = region.bbox_px
    x0 = max(0, int(b.x0))
    y0 = max(0, int(b.y0))
    x1 = min(image.width, int(b.x1))
    y1 = min(image.height, int(b.y1))
    return image.crop((x0, y0, x1, y1))
