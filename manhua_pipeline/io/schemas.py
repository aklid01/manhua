"""Typed stubs mirroring the 7 JSON schemas (source of truth).

These are dataclass stubs for editor autocompletion / validation; the on-disk
JSON remains authoritative.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Region:
    region_id: str
    page_number: int
    type: str  # speech_bubble | narration | name_label | scene_text
    bbox: dict  # {x, y, w, h}
    reading_order: int
    style_hint: str  # round | spiky | narration | label | in_art
    confidence: float
    read_region: dict = field(default_factory=dict)
    erase_mask: dict = field(default_factory=dict)
    render: bool = True


@dataclass
class OcrResult:
    region_id: str
    page_number: int
    type: str
    original_text: str
    text_direction: str = "horizontal"
    ocr_confidence: float = 0.0
    ocr_confidence_min: float = 0.0
    has_usable_text: bool = False
    do_not_render: bool = False
    needs_correction: bool = False
    edge_touching: bool = False
    edge: str = "none"
    note: Optional[str] = None
    watermark_filtered: bool = False


@dataclass
class TranslationResult:
    region_id: str
    page_number: int
    original_text: str
    literal_translation: str
    translated: bool = False
    skip_reason: Optional[str] = None
    glossary_terms_applied: list = field(default_factory=list)
    glossary_conflict: bool = False


@dataclass
class ParaphraseResult:
    region_id: str
    page_number: int
    literal_translation: str
    final_text: str
    register: str = "neutral"
    char_count: int = 0
    paraphrased: bool = False
    skip_reason: Optional[str] = None
    glossary_conflict: bool = False


@dataclass
class GlossaryTerm:
    term_id: str
    source_term: str
    target_term: str
    category: str
    locked: bool = False
    auto_seeded: bool = False
    source_region: Optional[str] = None
    notes: str = ""


@dataclass
class QaReport:
    chapter_id: str
    total_warnings: int = 0
    status: str = "SUCCESS"
    warnings: list = field(default_factory=list)


@dataclass
class Manifest:
    chapter_id: str
    source_language: str = "zh"
    target_language: str = "en-US"
    input_format: str = "paginated"
    total_pages: int = 0
    current_stage: str = "import"
    completed_stages: list = field(default_factory=list)
    warning_count: int = 0
    status: str = "in_progress"
    pages: list = field(default_factory=list)
