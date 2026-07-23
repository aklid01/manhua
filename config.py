"""Central configuration and constants for the manhua translation pipeline."""

import re

# ---- Models ----
MODEL_DETECTION = "ogkalu/comic-speech-bubble-detector-yolov8m"
OCR_ENGINE = "transformers"

# ---- Detection ----
DETECTION_MODEL = MODEL_DETECTION
DETECTION_CONF = 0.35
OVERLAY_ENABLED = True
READING_ORDER_BAND_FRACTION = 0.02

# ---- Detector backend ----
DETECTOR_BACKEND = "rtdetr"          # "yolov8" (fallback) | "rtdetr"
RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
RTDETR_CONF = 0.30                   # score threshold
RTDETR_SKIP_TEXT_FREE = True         # don't OCR text_free (SFX/watermark/credits)
# Class ids from the model card. DON'T CHANGE THESE!
RTDETR_CLASS_BUBBLE = 0
RTDETR_CLASS_TEXT_BUBBLE = 1
RTDETR_CLASS_TEXT_FREE = 2


# ---- Region IDs ----
# Format: P{page:03d}_R{idx:03d}  e.g. P002_R001
REGION_ID_FORMAT = "P{page:03d}_R{idx:03d}"

# ---- QA thresholds ----
SUCCESS_MAX = 2  # 0-2 warnings  -> SUCCESS
REVIEW_MAX = 10  # 3-10 warnings -> REVIEW ; >10 -> FAILED
QA_SUCCESS_MAX = SUCCESS_MAX
QA_REVIEW_MAX = REVIEW_MAX
OVERRIDES_NAME = "overrides.json"

# ---- OCR ----
OCR_CONFIDENCE_THRESHOLD = 0.7  # below this -> needs_correction
OCR_LANG = "ch"
OCR_USE_GPU = True # set to false if not using GPU
DETECTOR_USE_GPU = True # set to false if not using GPU
OCR_MIN_TEXT_CONF = 0.30
OCR_VERSION = "PP-OCRv6" # check https://github.com/PaddlePaddle/PaddleOCR/releases to see which version is latest
EDGE_TOUCH_EPS = 3
BATCH_SUBPROCESS = True # use subprocesses for batch flag

# ---- OCR retry ----
OCR_RETRY_ENABLED = True
OCR_RETRY_MAX = 2  # extra attempts after the base pass (up to 3 total reads)
OCR_RETRY_FLOOR = 0.30  # below this, don't retry (probably not text)

WATERMARK_PATTERNS = [
    r"w\s*w\s*w\s*\.",
    r"\.\s*c\s*o\s*m",
    r"b\s*a\s*o?\s*z\s*[i1l]\s*m\s*h",
    r"tencent|腾讯",
    r"colamanga|cola\s*manga",
    r"包?\s*子\s*漫\s*[画畫]",
    r"漫\s*[画畫]\s*屋?",
    r"最新免[费費]漫[画畫]",
    r"免[费費]漫[画畫]",
]
WATERMARK_REGEX = [re.compile(p, re.IGNORECASE) for p in WATERMARK_PATTERNS]

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
    "package": "stage7_package",
    "logs": "logs",
}

MANIFEST_NAME = "manifest.json"
GLOSSARY_NAME = "glossary.json"
MCP_SERVER_NAME = "Manhua Pipeline"

# ---- Packaging (Stage 7) ----
VALID_PACKAGE_FORMATS = ("zip", "cbz", "tar", "pdf")
PACKAGE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ---- Translation ----
TRANSLATOR_BACKEND = (
    "ollama"  # "manual" | "mcp" | "ollama"  (keep mcp default until benchmark passes)
)
TRANSLATION_PROMPT_NAME = "translation_prompt.json"
TRANSLATION_RESPONSE_NAME = "translation_response.json"

# ---- Ollama (local translation backend) ----
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TRANSLATE_MODEL = "qwen2.5:3b-instruct"
OLLAMA_BATCH_SIZE = 8
OLLAMA_TIMEOUT = 120
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_BACKOFF = 1.0
OLLAMA_MIN_COMPLETION_RATIO = 0.95
OLLAMA_MAX_MISSING = 0
OLLAMA_TRIVIAL_CJK_CHARS = 1
OLLAMA_NUM_CTX = 2048
OLLAMA_KEEP_ALIVE = "0"
OLLAMA_PROMPT_VERSION = "translation-v2"

# ---- Paraphrase ----
PARAPHRASE_BACKEND = "mcp"  # "manual" | "mcp" | "ollama"  (keep mcp default until you trust local output)
PARAPHRASE_PROMPT_NAME = "paraphrase_prompt.json"
PARAPHRASE_RESPONSE_NAME = "paraphrase_response.json"
PARAPHRASE_TONE_DIRECTIVE = "preserve crude/rude register; casual US English"
PARAPHRASE_MAX_CHARS = 90
PARAPHRASE_RUDE_MARKERS = [
    "fuck",
    "shit",
    "ass",
    "crap",
    "bitch",
    "damn",
    "screw",
    "bastard",
    "hell",
]

# ---- Ollama (local paraphrase backend) ----
OLLAMA_PARA_HOST = "http://localhost:11434"
OLLAMA_PARA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_PARA_BATCH_SIZE = 8
OLLAMA_PARA_TIMEOUT = 120
OLLAMA_PARA_TEMPERATURE = 0.7
OLLAMA_PARA_MAX_RETRIES = 3
OLLAMA_PARA_RETRY_BACKOFF = 1.0
OLLAMA_PARA_MIN_COMPLETION_RATIO = 0.80
OLLAMA_PARA_NUM_CTX = 2048
OLLAMA_PARA_KEEP_ALIVE = "0"
OLLAMA_PARA_PROMPT_VERSION = "paraphrase-v1"

# ---- Rendering ----
FONT_PATH = "assets/fonts/ComicNeue-Bold.ttf"
FONT_MAX_PT = 21
FONT_MIN_PT = 9
FONT_STEP_PT = 1
LINE_SPACING = 1.15
TEXT_PADDING_PX = 8
EMPHASIS_UPPERCASE = True
BG_FILL_DEFAULT = (255, 255, 255)
FONT_MISSING_HARD_ERROR = True
BUBBLE_WHITE_THRESHOLD = 220

# Ordered stage pipeline
STAGE_ORDER = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]

# ---- Credits page ----
CREDITS_DIR = "assets/credits"
CREDITS_TEXT_FILL = (240, 236, 225)
CREDITS_MATCH_PAGE_SIZE = True

_CREDITS_3COL = [
    ("scanlator", 0.185, 0.852, "center", 34, 0.24, 0.030),
    ("pipeline_name", 0.500, 0.852, "center", 30, 0.30, 0.030),
    ("pipeline_url", 0.815, 0.852, "center", 30, 0.30, 0.030),
]
_CREDITS_WEBTOON = [
    ("scanlator", 0.165, 0.622, "left", 28, 0.26, 0.026),
    ("pipeline_name", 0.165, 0.700, "left", 24, 0.28, 0.026),
    ("pipeline_url", 0.165, 0.780, "left", 24, 0.30, 0.026),
]
CREDITS_TEMPLATES = {
    "credits_cliff.png": _CREDITS_3COL,
    "credits_lake.png": _CREDITS_3COL,
    "credits_chibi.png": _CREDITS_3COL,
    "credits_webtoon.png": _CREDITS_WEBTOON,
}

# ---- Stitching (Stage 1 detection sub-step) ----
STITCH_ENABLED = True
STITCH_EDGE_EPS = 6  # px tolerance for "flush to edge"
STITCH_MIN_X_OVERLAP = 0.5  # min overlap as fraction of the narrower box's width
STITCH_MAX_CHAIN = 4  # Max pages a single bubble may span
STITCH_TEXT_PROBE = True  # require usable text on both halves (guard #4)
STITCH_TEXT_MIN_CONF = 0.15  # relaxed text confidence floor for split halves
