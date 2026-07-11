"""Central configuration and constants for the manhua translation pipeline."""

from pathlib import Path

# ---- Models ----
MODEL_DETECTION = "ogkalu/comic-speech-bubble-detector-yolov8m"
OCR_ENGINE = "PaddleOCR"

# ---- Detection ----
DETECTION_MODEL = MODEL_DETECTION
DETECTION_CONF = 0.35
OVERLAY_ENABLED = True
READING_ORDER_BAND_FRACTION = 0.02

# ---- Fonts ----
FONT_PATH = Path("assets/fonts/ComicNeue-Regular.ttf")  # OFL, from comicneue.com

# ---- Region IDs ----
# Format: P{page:03d}_R{idx:03d}  e.g. P002_R001
REGION_ID_FORMAT = "P{page:03d}_R{idx:03d}"

# ---- QA thresholds ----
SUCCESS_MAX = 2  # 0-2 warnings  -> SUCCESS
REVIEW_MAX = 10  # 3-10 warnings -> REVIEW ; >10 -> FAILED

# ---- OCR ----
OCR_CONFIDENCE_THRESHOLD = 0.7  # below this -> needs_correction

# ---- Detection types (v0 handles the first two) ----
TYPE_SPEECH = "speech_bubble"
TYPE_NARRATION = "narration"
TYPE_NAME_LABEL = "name_label"  # deferred past v0
TYPE_SCENE_TEXT = "scene_text"  # deferred past v0

# ---- Workspace stage folders ----
STAGE_FOLDERS = {
    "pages": "pages",
    "detection": "stage1_detection",
    "ocr": "stage2_ocr",
    "translation": "stage3_translation",
    "paraphrase": "stage4_paraphrase",
    "render": "stage5_render",
    "qa": "stage6_qa",
    "logs": "logs",
}

MANIFEST_NAME = "manifest.json"
GLOSSARY_NAME = "glossary.json"

# Ordered stage pipeline
STAGE_ORDER = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
