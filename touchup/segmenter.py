import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("touchup.segmenter")


class SAMSegmenter:
    """Lazy loader and inference wrapper for SAM 2.1 model."""

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            # Default model path resolution
            candidates = [
                Path("models/sam2.1_s.pt"),
                Path("models/sam2.1_hiera_small.pt"),
                Path(__file__).resolve().parent.parent / "models" / "sam2.1_s.pt",
                Path(r"D:\projects\yolo\models\sam2.1_hiera_small.pt"),
            ]
            self.model_path = next((p for p in candidates if p.exists()), candidates[0])
        else:
            self.model_path = Path(model_path)

        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from ultralytics import SAM

        logger.info(
            "Loading SAM 2.1 model from %s on %s...", self.model_path, self.device
        )
        self._model = SAM(str(self.model_path))

    def segment_box(
        self, image: Image.Image, box_xyxy: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Run SAM segmentation using a bounding box prompt [x1, y1, x2, y2].

        Returns a 2D boolean numpy array (H, W) where True indicates the mask.
        """
        self._ensure_loaded()
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        w, h = image.size

        # Clamp box to image dimensions
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=bool)

        results = self._model(
            image,
            bboxes=[[x1, y1, x2, y2]],
            device=self.device,
            verbose=False,
        )

        for r in results:
            if r.masks is not None and len(r.masks.data) > 0:
                mask_tensor = r.masks.data[0]
                mask_np = mask_tensor.cpu().numpy() > 0.5
                if mask_np.shape != (h, w):
                    # Resize mask if needed
                    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
                    return np.array(mask_pil) > 0
                return mask_np

        # Fallback: if SAM returns no mask, use the bounding box rectangle
        fallback = np.zeros((h, w), dtype=bool)
        fallback[y1:y2, x1:x2] = True
        return fallback

    def segment_points(
        self,
        image: Image.Image,
        points: list[tuple[int, int]],
        labels: list[int] | None = None,
    ) -> np.ndarray:
        """Run SAM segmentation using positive (1) or negative (0) point prompts."""
        self._ensure_loaded()
        w, h = image.size
        if not points:
            return np.zeros((h, w), dtype=bool)

        if labels is None:
            labels = [1] * len(points)

        results = self._model(
            image,
            points=points,
            labels=labels,
            device=self.device,
            verbose=False,
        )

        for r in results:
            if r.masks is not None and len(r.masks.data) > 0:
                mask_tensor = r.masks.data[0]
                mask_np = mask_tensor.cpu().numpy() > 0.5
                if mask_np.shape != (h, w):
                    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
                    return np.array(mask_pil) > 0
                return mask_np

        return np.zeros((h, w), dtype=bool)
