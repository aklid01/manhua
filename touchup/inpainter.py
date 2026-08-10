import cv2
import numpy as np
from PIL import Image


def build_text_erase_mask(
    image_size: tuple[int, int],
    text_boxes: list[tuple[int, int, int, int]],
    sam_mask: np.ndarray | None = None,
    padding_px: int = 5,
) -> np.ndarray:
    """Builds a dilated text inpainting mask from OCR text boxes, optionally constrained by SAM."""
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in text_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Apply padding
        x1 = max(0, x1 - padding_px)
        y1 = max(0, y1 - padding_px)
        x2 = min(w, x2 + padding_px)
        y2 = min(h, y2 + padding_px)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    # Dilate slightly for smooth contour blending
    if padding_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (padding_px * 2 + 1, padding_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Constrain with SAM container mask if provided
    if sam_mask is not None:
        if sam_mask.shape != (h, w):
            sam_resized = cv2.resize(
                sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            mask = np.where(sam_resized > 0, mask, 0)
        else:
            mask = np.where(sam_mask > 0, mask, 0)

    return mask


def inpaint_text_regions(
    image: Image.Image,
    text_boxes: list[tuple[int, int, int, int]],
    sam_mask: np.ndarray | None = None,
    padding_px: int = 5,
    inpaint_radius: int = 3,
) -> tuple[Image.Image, np.ndarray]:
    """Erases text regions on an image while preserving the background and container artwork.

    Returns:
        (inpainted_pil_image, binary_mask_uint8)
    """
    w, h = image.size
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    mask = build_text_erase_mask(
        (w, h), text_boxes, sam_mask=sam_mask, padding_px=padding_px
    )

    if not np.any(mask):
        return image.copy(), mask

    # Inpaint using Telea algorithm
    inpainted_bgr = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(inpainted_rgb), mask
