# Central configuration and constants for the manhua translation pipeline.

import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Pipeline execution and workspace directories
STAGE_ORDER = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
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
SUBPROCESS_STAGES = {"import", "detect", "ocr"}
BATCH_SUBPROCESS = True

MANIFEST_NAME = "manifest.json"
GLOSSARY_NAME = "glossary.json"
OVERRIDES_NAME = "overrides.json"
MCP_SERVER_NAME = "Manhua Pipeline"

# Stage 0: Page Import & Slicing
IMPORT_MAX_WIDTH = 1500  # Max page width in px (proportional downscale; 0 to disable)
IMPORT_SLICE_ENABLED = True
IMPORT_SLICE_TRIGGER_RATIO = 5.0  # Height-to-width ratio required to trigger slicing
IMPORT_SLICE_MIN_HEIGHT = 4000  # Minimum page height (px) required for slicing
IMPORT_SLICE_TARGET_HEIGHT = 5000  # Target height (px) for each slice
IMPORT_SLICE_SEAM_TOLERANCE = 0.20  # Search window around target height (fraction)
IMPORT_SLICE_SEAM_BAND = 24  # Pixel band inspected for cutting seams
IMPORT_SLICE_MAX_CONTENT_DENSITY = 0.02  # Max non-background pixel density on cut line
IMPORT_STITCH_SMALL_PAGES = True
IMPORT_MIN_PAGE_HEIGHT = 600
IMPORT_TARGET_STITCH_HEIGHT = 1800
IMPORT_MAX_STITCH_HEIGHT = 2200

# Stage 1: Bubble & Text Detection
MODEL_DETECTION = "ogkalu/comic-speech-bubble-detector-yolov8m"
DETECTION_MODEL = MODEL_DETECTION
DETECTION_CONF = 0.35
DETECTOR_USE_GPU = True
OVERLAY_ENABLED = False
READING_ORDER_BAND_FRACTION = 0.02
DEDUP_IOU_THRESHOLD = 0.90  # Merge same-class boxes with IoU above this threshold
DEDUP_IOS_THRESHOLD = 0.85  # Merge when smaller box is >=85% contained inside larger

# Detector backend
DETECTOR_BACKEND = "rtdetr"  # "rtdetr" | "yolov8"
RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
RTDETR_CONF = 0.30
RTDETR_SKIP_TEXT_FREE = True  # Skip OCR for floating SFX, watermarks, and credits
RTDETR_CLASS_BUBBLE = 0
RTDETR_CLASS_TEXT_BUBBLE = 1
RTDETR_CLASS_TEXT_FREE = 2

# Region types & formatting
TYPE_SPEECH = "speech_bubble"
TYPE_NARRATION = "narration"
TYPE_NAME_LABEL = "name_label"
TYPE_SCENE_TEXT = "scene_text"
REGION_ID_FORMAT = "P{page:03d}_R{idx:03d}"

# Stage 1 Sub-steps: Stitching & Segmentation
STITCH_ENABLED = True
STITCH_EDGE_EPS = 60  # Pixel distance tolerance to page border
STITCH_MIN_X_OVERLAP = 0.5  # Min horizontal overlap fraction to merge split boxes
STITCH_MAX_CHAIN = 6  # Max consecutive pages a bubble can span
STITCH_TEXT_PROBE = False  # Require valid text on both halves before merging
STITCH_TEXT_MIN_CONF = 0.15  # Text confidence floor for stitching check

YOLO_SEG_ENABLED = True
YOLO_SEG_MODEL = "models/best.pt"  # Relative to PROJECT_ROOT
YOLO_SEG_CONF = 0.5  # Minimum confidence threshold for masks
YOLO_SEG_DEVICE = "0"  # Device: "0" (GPU) or "cpu"
YOLO_SEG_PAD_FRAC = 0.20  # Crop padding fraction added around bubble box
YOLO_SEG_SAVE_FAILURES = True
YOLO_SEG_FAILURES_DIR = "debug/seg_failures"

# Stage 2: OCR & Filtering
OCR_ENGINE = "transformers"
OCR_VERSION = "PP-OCRv6"
OCR_LANG = "ch"
OCR_USE_GPU = True
OCR_CONFIDENCE_THRESHOLD = (
    0.7  # Confidence floor below which text is flagged for review
)
OCR_MIN_TEXT_CONF = 0.30
EDGE_TOUCH_EPS = 3

OCR_RETRY_ENABLED = True
OCR_RETRY_MAX = 2  # Extra retry attempts on low-confidence boxes (up to 3 total)
OCR_RETRY_FLOOR = 0.30  # Minimum confidence to attempt OCR retry passes

OCR_REVIEW_ENABLED = True
OCR_REVIEW_TYPES = ("speech_bubble", "narration")
OCR_REVIEW_DUMP_CROPS = True

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

WATERMARK_FUZZY_TOKENS = (
    # Scanlation site names and common OCR misreads
    "包子漫",
    "包子",
    "巴子",
    "baozim",
    "baozi",
    "漫画屋",
    "colamanga",
    "cola manga",
    "manhuagui",
    "腾讯",
    "tencent",
    # Promotional phrases
    "最新免费",
    "免费漫",
    "最新连载",
    "免费阅读",
    "在线漫画",
    # Domain fragments
    "www.",
    "ww.",
    ".com",
    ".net",
    ".cc",
    "http",
)

# Stage 3: Translation
TRANSLATOR_BACKEND = "ollama"  # "ollama" | "mcp" | "manual"
TRANSLATION_PROMPT_NAME = "translation_prompt.json"
TRANSLATION_RESPONSE_NAME = "translation_response.json"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TRANSLATE_MODEL = "translategemma:4b"
OLLAMA_BATCH_SIZE = 8
OLLAMA_TIMEOUT = 120
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_BACKOFF = 1.0
OLLAMA_MIN_COMPLETION_RATIO = 0.95
OLLAMA_MAX_MISSING = 2
OLLAMA_CONTEXTUAL_RETRY_ENABLED = False
OLLAMA_TRIVIAL_CJK_CHARS = 1
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 1024
OLLAMA_KEEP_ALIVE = "15m"
OLLAMA_PROMPT_VERSION = "translation-v2"

# Translation triage & filtering
TRANSLATE_NON_CJK_DIALOGUE = False  # If False, skip LLM translation for non-CJK text
TRANSLATE_SHORT_CJK_DIALOGUE = (
    True  # Translate 1-2 char CJK if containing dialogue signals
)
TRANSLATE_SHORT_SPEECH_BUBBLES = (
    True  # Only route short CJK to LLM if in a speech bubble
)
SHORT_SPEECH_BUBBLE_CJK_CHARS = 1
RENDER_PUNCT_ONLY_REGIONS = (
    True  # Normalize and render punctuation-only bubbles directly
)
TRANSLATE_TRIVIAL_WITH_LLM = (
    False  # If False, bypass LLM for single-character trivial tokens
)
TRANSLATION_FAILURE_HANDOFF_TO_PARAPHRASE = (
    True  # Forward untranslated OCR text to Stage 4 on LLM failure
)

# Stage 4: Paraphrase
PARAPHRASE_BACKEND = "mcp"  # "mcp" | "ollama" | "manual"
PARAPHRASE_PROMPT_NAME = "paraphrase_prompt.json"
PARAPHRASE_RESPONSE_NAME = "paraphrase_response.json"
PARAPHRASE_TONE_DIRECTIVE = "preserve crude/rude register; casual US English"
PARAPHRASE_MAX_CHARS = 90

OLLAMA_PARA_HOST = "http://localhost:11434"
OLLAMA_PARA_MODEL = "translategemma:4b"
OLLAMA_PARA_BATCH_SIZE = 8
OLLAMA_PARA_TIMEOUT = 120
OLLAMA_PARA_TEMPERATURE = 0.7
OLLAMA_PARA_MAX_RETRIES = 3
OLLAMA_PARA_RETRY_BACKOFF = 1.0
OLLAMA_PARA_MIN_COMPLETION_RATIO = 0.80
OLLAMA_PARA_NUM_CTX = 2048
OLLAMA_PARA_NUM_PREDICT = 1024
OLLAMA_PARA_KEEP_ALIVE = "15m"
OLLAMA_PARA_PROMPT_VERSION = "paraphrase-v1"

# Length sanity check thresholds to catch dropped text
PARAPHRASE_CHECK_TRUNCATION = None
PARAPHRASE_MIN_TEXT_LENGTH_RATIO = 0.45
PARAPHRASE_MIN_TEXT_LENGTH_CHARS = 80

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
    # Mature dialogue markers
    "pussy",
    "boobs",
    "dick",
    "cock",
    "asshole",
    "motherfucker",
    "cunt",
    "vagina",
    "penis",
    "nipples",
    "breast",
    "cum",
    "ejaculate",
    "rape",
    "blowjob",
    "handjob",
    "cumming",
]

# Stage 5: Rendering & Inpainting
FONT_PATH = ASSETS_DIR / "fonts" / "ComicNeue-Bold.ttf"
FONT_MAX_PT = 40
FONT_MAX_PAGE_FRAC = 0.045  # Max font size as a fraction of page width (4.5%)
FONT_MIN_PT = 12  # Target minimum readable font size
FONT_ABS_MIN_PT = 9  # Hard lower bound on font size
FONT_STEP_PT = 1
LINE_SPACING = 1.15
TEXT_PADDING_PX = 8
EMPHASIS_UPPERCASE = True
BG_FILL_DEFAULT = (255, 255, 255)
FONT_MISSING_HARD_ERROR = True
BUBBLE_WHITE_THRESHOLD = 220

# Bubble interior color filling and background erasing
BUBBLE_FILL_MODE = "dominant"  # "dominant" (extract background tone) | "white"
BUBBLE_FILL_TOLERANCE = 42  # Color distance tolerance for background matching
BUBBLE_FILL_MIN_COVERAGE = (
    0.40  # Fall back to rectangular erase if coverage is below this
)
TEXT_ERASE_PADDING_PX = 12
TEXT_LAYOUT_PADDING_PX = 8
TEXT_ERASE_DILATION_PX = 5
TEXT_ERASE_PARENT_MARGIN_PX = 8
TEXT_ERASE_USE_RECT_WHEN_SAFE = True
TEXT_ERASE_MIN_RETENTION = 0.75
RENDER_IMAGE_INPAINT = True
FALLBACK_ERASE_SHAPE = "oval"  # "oval" | "rounded_rect" | "rect"
FALLBACK_ERASE_FEATHER_PX = 3  # Soft Gaussian alpha feathering radius
PRESERVE_UNCONTAINED_SFX_ART = True  # Leave raw calligraphy SFX on artwork untouched

# Radiating thought bubble mask reconstruction
THOUGHT_BUBBLE_MASK_ENABLED = True
THOUGHT_BUBBLE_DARK_THRESHOLD = 130
THOUGHT_BUBBLE_ANGLE_SAMPLES = 144
THOUGHT_BUBBLE_MIN_RAY_LENGTH_PX = 5
THOUGHT_BUBBLE_MIN_RAY_SECTOR_RATIO = 0.35
THOUGHT_BUBBLE_MAX_RAY_SECTOR_RATIO = 0.95
THOUGHT_BUBBLE_GENERIC_MIN_COVERAGE = 0.92
THOUGHT_BUBBLE_INTERIOR_MIN_COVERAGE = 0.28
THOUGHT_BUBBLE_INTERIOR_MAX_COVERAGE = 0.80
THOUGHT_BUBBLE_INTERIOR_INSET_PX = 6
THOUGHT_BUBBLE_MIN_TEXT_COVERAGE = 0.85

# Layout box expansion into bubble interior
LAYOUT_EXPAND_ENABLED = True
LAYOUT_INTERIOR_ERODE_PX = 10  # Inset from bubble boundary to avoid border collision
LAYOUT_MAX_GROWTH_FACTOR = 3.0  # Max layout area expansion multiplier
LAYOUT_MIN_ANCHOR_OVERLAP = 0.5  # Required overlap ratio with original text box

# Render debug options
RENDER_DEBUG_ARTIFACTS = False
RENDER_DEBUG_REGION_IDS = []

# Output image encoding
RENDER_OUTPUT_FORMAT = "jpeg"  # "jpeg" | "webp" | "png"
RENDER_OUTPUT_QUALITY = 93  # Compression quality (1-100) for jpeg and webp

# Stage 6: Quality Assurance (QA)
SUCCESS_MAX = 2  # <= 2 warnings -> SUCCESS
REVIEW_MAX = 10  # 3-10 warnings -> REVIEW; > 10 -> FAILED
QA_SUCCESS_MAX = SUCCESS_MAX
QA_REVIEW_MAX = REVIEW_MAX

# Stage 7: Packaging & Credits
VALID_PACKAGE_FORMATS = ("zip", "cbz")
PACKAGE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

CREDITS_DIR = ASSETS_DIR / "credits"
CREDITS_TEXT_FILL = (240, 236, 225)
CREDITS_MATCH_PAGE_SIZE = False

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
