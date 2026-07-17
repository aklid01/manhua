"""Central configuration and constants for the manhua translation pipeline."""

import re

# ---- Models ----
MODEL_DETECTION = "ogkalu/comic-speech-bubble-detector-yolov8m"
OCR_ENGINE = "PaddleOCR"

# ---- Detection ----
DETECTION_MODEL = MODEL_DETECTION
DETECTION_CONF = 0.35
OVERLAY_ENABLED = True
READING_ORDER_BAND_FRACTION = 0.02


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
OCR_USE_GPU = False
OCR_MIN_TEXT_CONF = 0.30
EDGE_TOUCH_EPS = 3

WATERMARK_PATTERNS = [
    r"www\.",
    r"baozimh",
    r"\.com",
    r"包子漫[画畫]",
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
    "logs": "logs",
}

MANIFEST_NAME = "manifest.json"
GLOSSARY_NAME = "glossary.json"
MCP_SERVER_NAME = "Manhua Pipeline"

# ---- Translation ----
TRANSLATOR_BACKEND = "ollama"  # "manual" | "mcp" | "ollama"  (keep mcp default until benchmark passes)
TRANSLATION_PROMPT_NAME = "translation_prompt.json"
TRANSLATION_RESPONSE_NAME = "translation_response.json"

# ---- Ollama (local translation backend) ----
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TRANSLATE_MODEL = "qwen2.5:3b-instruct"
OLLAMA_BATCH_SIZE = 15
OLLAMA_TIMEOUT = 120
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_BACKOFF = 1.0
OLLAMA_MIN_COMPLETION_RATIO = 0.95
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
OLLAMA_PARA_BATCH_SIZE = 15
OLLAMA_PARA_TIMEOUT = 120
OLLAMA_PARA_TEMPERATURE = 0.7
OLLAMA_PARA_MAX_RETRIES = 3
OLLAMA_PARA_RETRY_BACKOFF = 1.0
OLLAMA_PARA_MIN_COMPLETION_RATIO = 0.80
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
