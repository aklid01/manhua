# Region triage classification, watermark filtering, and interjection routing.

import re
from dataclasses import dataclass

from manhua_pipeline.translation_validation import (
    is_meaningful_region,
    normalize_punct,
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ESCAPE_SEQ_RE = re.compile(r"\\(?:r\\n|n|r)", re.IGNORECASE)

_FULLWIDTH_MAP = {
    "\uff01": "!",
    "\uff1f": "?",
    "\uff0c": ",",
    "\u3002": ".",
    "\uff1a": ":",
    "\uff1b": ";",
    "\uff08": "(",
    "\uff09": ")",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u3001": ",",
}


# Check whether string contains any CJK characters
def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


# Count CJK characters in string
def _cjk_char_count(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


# Calculate ratio of CJK characters to total non-whitespace characters
def _cjk_ratio(text: str) -> float:
    no_space = "".join(c for c in (text or "") if not c.isspace())
    if not no_space:
        return 0.0
    return len(_CJK_RE.findall(no_space)) / len(no_space)


# Normalize fullwidth punctuation characters
def _normalize_punct(text: str) -> str:
    return normalize_punct(text)


# Check whether region content is trivial or noise
def _is_trivial_region(original_text: str, config) -> bool:
    return not is_meaningful_region(original_text, config)


# Outcome of region triage classification
@dataclass(frozen=True)
class TriageResult:
    reason: str
    effective_text: str


# Strip known watermark patterns and scanlator tags from text
def _strip_watermarks(text: str, config) -> str:
    if not text:
        return ""
    norm = _ESCAPE_SEQ_RE.sub("\n", text)
    patterns = getattr(config, "WATERMARK_REGEX", [])
    fuzzy = getattr(config, "WATERMARK_FUZZY_TOKENS", ())
    kept = []
    for ln in norm.split("\n"):
        s = ln.strip()
        if not s:
            continue
        sl = s.lower()
        is_wm = any(p.search(s) for p in patterns) or any(tok in sl for tok in fuzzy)
        if not is_wm:
            kept.append(ln)
    return "\n".join(kept).strip()


# Count total CJK characters in text
def _cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


_DIALOGUE_SIGNAL_PUNCT = frozenset("?？!！~～…。．⁈⁇‼♥♡")

_CJK_INTERJECTIONS = frozenset(
    "诶欸嗯唔呒啊阿哈呵嘿嘻咦哦噢喔唉哎呃哼呀呐喂嘛啦"
    "咧哇呜咳嘶啧哟唷嚯嗝嗬嘞嚄嘘呸嗤吖呦啐切"
    "额呼嗷噫噗吼咻嗨嘁唏嗞哧咩嗡吭啵"
    "砰啪咚叮嘭哗哐唰嘣哒咔嗖轰嘀"
)


# Check if text contains dialogue punctuation or interjection characters
def _has_dialogue_signal(effective: str) -> bool:
    if any(ch in _DIALOGUE_SIGNAL_PUNCT for ch in effective):
        return True
    cjk_core = "".join(_CJK_RE.findall(effective))
    return cjk_core in _CJK_INTERJECTIONS


# Check if text consists exclusively of punctuation characters
def _is_punct_only(effective: str) -> bool:
    core = "".join(ch for ch in effective if not ch.isspace())
    if not core:
        return False
    return not any(ch.isalnum() for ch in core) and not bool(_CJK_RE.search(core))


# Classify single region text into triage action category
def triage_region(original_text: str, config) -> TriageResult:
    raw = (original_text or "").strip()

    if not raw:
        return TriageResult("empty", "")

    effective = _strip_watermarks(raw, config)

    if not effective:
        return TriageResult("watermark", "")

    cjk_n = _cjk_count(effective)
    if cjk_n == 0:
        if (
            getattr(config, "RENDER_PUNCT_ONLY_REGIONS", True)
            and _is_punct_only(effective)
            and _has_dialogue_signal(effective)
        ):
            return TriageResult("punct_only", effective)
        return TriageResult("no_cjk", effective)

    trivial_max = int(getattr(config, "OLLAMA_TRIVIAL_CJK_CHARS", 1))
    if cjk_n <= trivial_max:
        if getattr(config, "TRANSLATE_SHORT_CJK_DIALOGUE", True) and (
            _has_dialogue_signal(effective)
        ):
            return TriageResult("sfx_passthrough", effective)
        return TriageResult("trivial", effective)

    return TriageResult("translatable", effective)


# Construct skipped region result row
def _skip_row(r: dict, skip_reason: str) -> dict:
    return {
        "region_id": r["region_id"],
        "page_number": r.get("page_number"),
        "original_text": r.get("original_text", ""),
        "effective_text": r.get("effective_text", ""),
        "literal_translation": "",
        "translated": False,
        "skip_reason": skip_reason,
        "glossary_terms_applied": [],
        "glossary_conflict": False,
        "translation_source": None,
    }


# Construct pre-translated or passthrough region result row
def _translated_row(r: dict, text: str, *, source: str, glossary_terms: list) -> dict:
    return {
        "region_id": r["region_id"],
        "page_number": r.get("page_number"),
        "original_text": r.get("original_text", ""),
        "effective_text": r.get("effective_text", ""),
        "literal_translation": text,
        "translated": True,
        "skip_reason": None,
        "glossary_terms_applied": glossary_terms,
        "glossary_conflict": False,
        "translation_source": source,
    }


# Check whether region is eligible for speech bubble routing
def _is_renderable_speech_bubble(region: dict, config) -> bool:
    speech_type = getattr(config, "TYPE_SPEECH", "speech_bubble")
    return (
        region.get("type") == speech_type
        and region.get("render", True) is not False
        and not region.get("is_free_text", False)
    )


# Check whether region is non-dialogue or non-rendered
def _is_explicit_non_speech_region(region: dict, config) -> bool:
    speech_type = getattr(config, "TYPE_SPEECH", "speech_bubble")
    return (
        ("type" in region and region.get("type") != speech_type)
        or region.get("is_free_text") is True
        or region.get("render") is False
    )


# Check whether speech bubble contains short single-character dialogue
def _is_short_speech_bubble(region: dict, effective_text: str, config) -> bool:
    if not getattr(config, "TRANSLATE_SHORT_CJK_DIALOGUE", True):
        return False
    if not getattr(config, "TRANSLATE_SHORT_SPEECH_BUBBLES", True):
        return False
    if not _is_renderable_speech_bubble(region, config):
        return False

    limit = int(getattr(config, "SHORT_SPEECH_BUBBLE_CJK_CHARS", 1))
    if _cjk_count(effective_text) > limit:
        return False

    return True


# Partition regions by triage classification outcome
def _partition_by_triage(
    regions: list[dict], config, locked_index: dict[str, str]
) -> tuple[list[dict], list[dict], list[dict]]:
    translatable: list[dict] = []
    hard_skipped: list[dict] = []
    trivial_fallback: list[dict] = []

    for r in regions:
        raw = r.get("original_text", "")
        res = triage_region(raw, config)
        r["effective_text"] = res.effective_text

        if (
            _is_short_speech_bubble(r, res.effective_text, config)
            and res.reason == "trivial"
        ):
            res = TriageResult("speech_bubble_passthrough", res.effective_text)
        elif res.reason == "sfx_passthrough" and _is_explicit_non_speech_region(
            r, config
        ):
            res = TriageResult("trivial", res.effective_text)

        if res.reason == "empty":
            hard_skipped.append(_skip_row(r, "no_usable_text"))
        elif res.reason == "punct_only":
            hard_skipped.append(
                _translated_row(
                    r,
                    _normalize_punct(res.effective_text),
                    source="punct_passthrough",
                    glossary_terms=[],
                )
            )
        elif res.reason == "watermark":
            hard_skipped.append(_skip_row(r, "watermark"))
        elif res.reason == "no_cjk":
            if getattr(config, "TRANSLATE_NON_CJK_DIALOGUE", False):
                trivial_fallback.append(r)
            else:
                hard_skipped.append(_skip_row(r, "no_cjk"))
        elif res.reason == "sfx_passthrough":
            hard_skipped.append(
                _translated_row(
                    r,
                    res.effective_text,
                    source="sfx_passthrough",
                    glossary_terms=[],
                )
            )
        elif res.reason == "speech_bubble_passthrough":
            hard_skipped.append(
                _translated_row(
                    r,
                    res.effective_text,
                    source="speech_bubble_passthrough",
                    glossary_terms=[],
                )
            )
        elif res.reason == "trivial":
            hit = locked_index.get(res.effective_text)
            if hit:
                hard_skipped.append(
                    _translated_row(
                        r,
                        hit,
                        source="glossary",
                        glossary_terms=[res.effective_text],
                    )
                )
            elif getattr(config, "TRANSLATE_TRIVIAL_WITH_LLM", False):
                trivial_fallback.append(r)
            else:
                hard_skipped.append(_skip_row(r, "trivial_manual_review"))
        else:
            translatable.append(r)

    return translatable, hard_skipped, trivial_fallback
