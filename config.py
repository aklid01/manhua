"""Central configuration and constants for the manhua translation pipeline."""

import re
from pathlib import Path

# Repository root, so asset paths resolve no matter the current working dir.
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# ---- Models ----
MODEL_DETECTION = "ogkalu/comic-speech-bubble-detector-yolov8m"
OCR_ENGINE = "transformers"

# ---- Detection ----
DETECTION_MODEL = MODEL_DETECTION
DETECTION_CONF = 0.35
OVERLAY_ENABLED = False
READING_ORDER_BAND_FRACTION = 0.02
DEDUP_IOU_THRESHOLD = 0.90  # merge same-class same-page boxes above this IoU
DEDUP_IOS_THRESHOLD = 0.85  # merge same-class same-page boxes when the smaller
# is >=85% contained in the other (catches offset/size dups)

# ---- Detector backend ----
DETECTOR_BACKEND = "rtdetr"  # "yolov8" (fallback) | "rtdetr"
RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
RTDETR_CONF = 0.30  # score threshold
RTDETR_SKIP_TEXT_FREE = True  # don't OCR text_free (SFX/watermark/credits)
# Class ids from the model card. DON'T CHANGE THESE!
RTDETR_CLASS_BUBBLE = 0
RTDETR_CLASS_TEXT_BUBBLE = 1
RTDETR_CLASS_TEXT_FREE = 2

# ---- YOLO-seg bubble mask refinement (optional) ----
# When enabled, Stage 1 crops each parent_bubble ROI and runs YOLO-seg on it.
# The resulting mask is stored in detection.json and used by Stage 5 instead of
# the heuristic _get_bubble_mask(). RT-DETR is explicitly unloaded before YOLO
# loads so both can safely share a GPU without VRAM spikes.
YOLO_SEG_ENABLED = True
YOLO_SEG_MODEL = "models/best.pt"  # relative to PROJECT_ROOT
YOLO_SEG_CONF = 0.5  # masks below this confidence are discarded
YOLO_SEG_DEVICE = "0"  # "cpu" | "0" for GPU (RT-DETR unloaded first)
YOLO_SEG_PAD_FRAC = 0.15  # padding fraction added around parent_bubble crop


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
OCR_USE_GPU = True  # set to false if not using GPU
DETECTOR_USE_GPU = True  # set to false if not using GPU
OCR_MIN_TEXT_CONF = 0.30
OCR_VERSION = "PP-OCRv6"  # check https://github.com/PaddlePaddle/PaddleOCR/releases to see which version is latest
EDGE_TOUCH_EPS = 3
BATCH_SUBPROCESS = True  # use subprocesses for batch flag

# ---- OCR retry ----
OCR_RETRY_ENABLED = True
OCR_RETRY_MAX = 2  # extra attempts after the base pass (up to 3 total reads)
OCR_RETRY_FLOOR = 0.30  # below this, don't retry (probably not text)

# ---- OCR review fallback (suspicious dialogue bubbles) ----
# A speech_bubble/narration region that OCRs to punctuation-only or empty
# is very likely a dropped-character failure (e.g. PaddleOCR '诶？' -> '？').
# Stage 2 forces the preprocessing ladder; if it still has no real words it
# renders the fallback AND writes ocr_review.json (+ a cropped PNG) for
# optional manual correction. Fill 'corrected_text' and re-run. Non-blocking.
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

# Fuzzy substring tokens for OCR-mangled watermark variants the regex misses.
# Matched case-insensitively against each OCR line.
WATERMARK_FUZZY_TOKENS = (
    # site names + common OCR manglings
    "包子漫",
    "包子",
    "巴子",  # OCR misread of 包子 (包→巴, 漫→量 etc.)
    "baozim",
    "baozi",
    "漫画屋",
    "colamanga",
    "cola manga",
    "manhuagui",
    "腾讯",
    "tencent",
    # promo phrases (often truncated by OCR)
    "最新免费",
    "免费漫",
    "最新连载",
    "免费阅读",
    "在线漫画",
    # url-ish fragments (OCR drops chars: www -> ww, .com -> cnn/cm/etc.)
    "www.",
    "ww.",
    ".com",
    ".net",
    ".cc",
    "http",
)

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
VALID_PACKAGE_FORMATS = ("zip", "cbz")
PACKAGE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ---- Output image format ----
RENDER_OUTPUT_FORMAT = "jpeg"  # "jpeg" | "webp" | "png"
RENDER_OUTPUT_QUALITY = 93  # quality for jpeg/webp (ignored for png)

# ---- Translation ----
TRANSLATOR_BACKEND = "ollama"  # "manual" | "mcp" | "ollama"
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
OLLAMA_MAX_MISSING = 2
# Optional local one-region retry after a batch response misses entries.
# Keep disabled when Stage 4 uses the stronger MCP recovery backend.
OLLAMA_CONTEXTUAL_RETRY_ENABLED = False
OLLAMA_TRIVIAL_CJK_CHARS = 1
# following should be tested on YOUR pc/laptop for optimal values
# current values are just for me and may not be optimal
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 1024
OLLAMA_KEEP_ALIVE = "15m"  # keep qwen resident across batches/retries/paraphrase
OLLAMA_PROMPT_VERSION = "translation-v2"

# ---- Translation triage (keep noise out of the LLM) ----
TRANSLATE_NON_CJK_DIALOGUE = False  # True restores legacy: send no-CJK regions to qwen
# Short CJK dialogue (1-2 chars): translate when it carries a dialogue signal
# (punctuation like ？！… or a known interjection). Set False to skip all short
# CJK; set OLLAMA_TRIVIAL_CJK_CHARS=0 to translate every short CJK region.
TRANSLATE_SHORT_CJK_DIALOGUE = True
# Route short, dialogue-like CJK only when detection identifies a renderable
# speech bubble. Free-text/narration regions (including background SFX) remain
# on the ordinary trivial-text path.
TRANSLATE_SHORT_SPEECH_BUBBLES = True
SHORT_SPEECH_BUBBLE_CJK_CHARS = 1
# Punctuation-only bubbles ("？", "！？", "…") are universal dialogue: normalize
# and render them without an LLM call. Set False to skip them as noise.
RENDER_PUNCT_ONLY_REGIONS = True
TRANSLATE_TRIVIAL_WITH_LLM = (
    False  # True restores legacy: send single-CJK misses to qwen
)

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
    # 18+ comic markers
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
OLLAMA_PARA_NUM_PREDICT = 1024
OLLAMA_PARA_KEEP_ALIVE = "15m"  # share the warm model with translation
OLLAMA_PARA_PROMPT_VERSION = "paraphrase-v1"
# Reject paraphrases that are suspiciously shorter than the literal translation.
# A short response can silently drop an entire sentence or factual clause.
PARAPHRASE_MIN_TEXT_LENGTH_RATIO = 0.45
PARAPHRASE_MIN_TEXT_LENGTH_CHARS = 80

# ---- Rendering ----
FONT_PATH = ASSETS_DIR / "fonts" / "ComicNeue-Bold.ttf"
FONT_MAX_PT = 40
FONT_MAX_PAGE_FRAC = 0.045  # ceiling as a fraction of page WIDTH (~4.5%)
FONT_MIN_PT = 12  # readable target floor (raised from 9)
FONT_ABS_MIN_PT = 9  # hard clamp; never render below this silently
FONT_STEP_PT = 1
LINE_SPACING = 1.15
TEXT_PADDING_PX = 8
EMPHASIS_UPPERCASE = True
BG_FILL_DEFAULT = (255, 255, 255)
FONT_MISSING_HARD_ERROR = True
BUBBLE_WHITE_THRESHOLD = 220
# --- Colored / non-white panel erasing ---
BUBBLE_FILL_MODE = "dominant"  # "white" (legacy) | "dominant"
BUBBLE_FILL_TOLERANCE = 42  # max per-channel-ish RGB distance to the fill color
BUBBLE_FILL_MIN_COVERAGE = (
    0.40  # if dominant mask covers < this frac of crop, fall back
)
TEXT_ERASE_PADDING_PX = 12
TEXT_LAYOUT_PADDING_PX = 8
TEXT_ERASE_DILATION_PX = 5
TEXT_ERASE_PARENT_MARGIN_PX = 8
TEXT_ERASE_USE_RECT_WHEN_SAFE = True
TEXT_ERASE_MIN_RETENTION = 0.75
RENDER_IMAGE_INPAINT = True
# --- Radiating thought-bubble masks ---
# Some thought bubbles are bounded by many black radial strokes. Their pale page
# background can merge with the white interior, so the generic fill mask paints
# over the rays. Detect the ray ring and reconstruct only the inner interior.
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
RENDER_DEBUG_ARTIFACTS = False
RENDER_DEBUG_REGION_IDS = []

# ---- Rendering: layout-box expansion into bubble interior ----
LAYOUT_EXPAND_ENABLED = True
LAYOUT_INTERIOR_ERODE_PX = 10  # inset from bubble interior edge (tail/rim guard)
LAYOUT_MAX_GROWTH_FACTOR = 3.0  # cap layout area growth vs original text box
LAYOUT_MIN_ANCHOR_OVERLAP = 0.5  # expanded box must still cover >=50% of text box

# ---- Import: page size cap ----
# Pages wider than this are scaled down proportionally at import time.
# Set to 0 to disable. Pillow thumbnail() only shrinks, never upscales.
# Safe range: 1200-1500 px (OCR stays reliable; detector input is 640px anyway).
IMPORT_MAX_WIDTH = 1500

# ---- Import: safe tall-page slicing ----
# Only pages meeting both the ratio and absolute-height thresholds are considered.
IMPORT_SLICE_ENABLED = True
IMPORT_SLICE_TRIGGER_RATIO = 5.0
IMPORT_SLICE_MIN_HEIGHT = 4000
IMPORT_SLICE_TARGET_HEIGHT = 5000
IMPORT_SLICE_SEAM_TOLERANCE = 0.20  # fraction of target height around each boundary
IMPORT_SLICE_SEAM_BAND = 24  # rows on either side of a candidate seam
IMPORT_SLICE_MAX_CONTENT_DENSITY = 0.02

# Ordered stage pipeline
STAGE_ORDER = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]

# Stage process isolation. If defined, overrides BATCH_SUBPROCESS: only these
# stages run in a fresh interpreter (detect/ocr need CUDA/DLL isolation).
SUBPROCESS_STAGES = {"import", "detect", "ocr"}

# ---- Credits page ----
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

# ---- Stitching (Stage 1 detection sub-step) ----
STITCH_ENABLED = True
STITCH_EDGE_EPS = 6  # px tolerance for "flush to edge"
STITCH_MIN_X_OVERLAP = 0.5  # min overlap as fraction of the narrower box's width
STITCH_MAX_CHAIN = 4  # Max pages a single bubble may span
STITCH_TEXT_PROBE = True  # require usable text on both halves (guard #4)
STITCH_TEXT_MIN_CONF = 0.15  # relaxed text confidence floor for split halves
# When the local Ollama translator cannot produce a valid result, keep the
# original OCR text and let Stage 4 translate + paraphrase it with the stronger
# backend. Existing single-character CJK/SFX passthrough is unaffected.
TRANSLATION_FAILURE_HANDOFF_TO_PARAPHRASE = True
