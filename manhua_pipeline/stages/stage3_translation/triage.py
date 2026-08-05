import re
from dataclasses import dataclass

from manhua_pipeline.translation_validation import (
    is_meaningful_region,
    normalize_punct,
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

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


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _cjk_char_count(text: str) -> int:
    return sum(1 for c in (text or "") if _contains_cjk(c))


def _cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _contains_cjk(c)) / len(chars)


def _normalize_punct(text: str) -> str:
    return normalize_punct(text)


def _is_trivial_region(original_text: str, config) -> bool:
    return not is_meaningful_region(original_text, config)


@dataclass(frozen=True)
class TriageResult:
    reason: str  # empty | watermark | no_cjk | trivial | translatable
    effective_text: str


def _strip_watermarks(text: str, config) -> str:
    if not text:
        return ""
    norm = re.sub(r"\\(?:r\\n|n|r)", "\n", text, flags=re.IGNORECASE)
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


def _cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


_DIALOGUE_SIGNAL_PUNCT = frozenset("?？!！~～…。．⁈⁇‼♥♡")

_CJK_INTERJECTIONS = frozenset(
    "诶欸嗯唔呒啊阿哈呵嘿嘻咦哦噢喔唉哎呃哼呀呐喂嘛啦"
    "咧哇呜咳嘶啧哟唷嚯嗝嗬嘞嚄"
    "嘘呸嗤吖呦啐切"
)


def _has_dialogue_signal(effective: str) -> bool:
    if any(ch in _DIALOGUE_SIGNAL_PUNCT for ch in effective):
        return True
    cjk_core = "".join(ch for ch in effective if _contains_cjk(ch))
    return cjk_core in _CJK_INTERJECTIONS


def _is_punct_only(effective: str) -> bool:
    core = "".join(ch for ch in effective if not ch.isspace())
    if not core:
        return False
    return not any(ch.isalnum() or _contains_cjk(ch) for ch in core)


def triage_region(original_text: str, config) -> TriageResult:
    raw = (original_text or "").strip()

    if not raw:
        return TriageResult("empty", "")

    effective = _strip_watermarks(raw, config)

    if not effective:
        return TriageResult("watermark", "")

    if _cjk_count(effective) == 0:
        if (
            getattr(config, "RENDER_PUNCT_ONLY_REGIONS", True)
            and _is_punct_only(effective)
            and _has_dialogue_signal(effective)
        ):
            return TriageResult("punct_only", effective)
        return TriageResult("no_cjk", effective)

    trivial_max = int(getattr(config, "OLLAMA_TRIVIAL_CJK_CHARS", 1))
    if _cjk_count(effective) <= trivial_max:
        if getattr(config, "TRANSLATE_SHORT_CJK_DIALOGUE", True) and (
            _has_dialogue_signal(effective)
        ):
            return TriageResult("translatable", effective)
        return TriageResult("trivial", effective)

    return TriageResult("translatable", effective)


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
