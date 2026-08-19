# Dataclass schema definitions for pipeline artifacts and intermediate representations.

from dataclasses import dataclass, field
from typing import Optional


# Bounding box and classification metadata for detected text region
@dataclass
class Region:
    region_id: str
    page_number: int
    type: str
    bbox: dict
    reading_order: int
    style_hint: str
    confidence: float
    read_region: dict = field(default_factory=dict)
    erase_mask: dict = field(default_factory=dict)
    render: bool = True


# OCR extraction result and text usability metadata
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


# Literal translation output and glossary application metadata
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
    translation_source: Optional[str] = None


# Final dialogue paraphrase output and register classification
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
    paraphrase_source: Optional[str] = None


# Standard glossary term definition
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


# Chapter quality assurance report and warning summary
@dataclass
class QaReport:
    chapter_id: str
    total_warnings: int = 0
    status: str = "SUCCESS"
    warnings: list = field(default_factory=list)


# Chapter manifest tracking stage progress and page metadata
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
