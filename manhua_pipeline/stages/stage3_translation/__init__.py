# Stage 3: Literal translation package and exported symbol bindings.

import sys
from types import ModuleType

from manhua_pipeline.stages.stage3_translation import (
    backends,
    bundle,
    engine,
    glossary,
    triage,
)
from manhua_pipeline.stages.stage3_translation.backends import (
    ManualBackend,
    McpBackend,
    OllamaBackend,
    TranslatorBackend,
    _get_backend,
)
from manhua_pipeline.stages.stage3_translation.bundle import (
    _PROMPT_INSTRUCTIONS,
    _build_bundle,
    _build_contextual_retry_prompt,
    _validate_response,
    build_translation_bundle,
    write_translation_response,
)
from manhua_pipeline.stages.stage3_translation.engine import (
    TranslationWriteContext,
    _handle_unresolved_region,
    _load_raw_overrides,
    _partition_regions_translation,
    _write_output,
    run_translation,
)
from manhua_pipeline.stages.stage3_translation.glossary import (
    _build_locked_index,
    _enforce_glossary,
    _locked_terms,
    _resolve_locked_glossary,
)
from manhua_pipeline.stages.stage3_translation.triage import (
    _CJK_INTERJECTIONS,
    _CJK_RE,
    _DIALOGUE_SIGNAL_PUNCT,
    _FULLWIDTH_MAP,
    TriageResult,
    _cjk_char_count,
    _cjk_count,
    _cjk_ratio,
    _contains_cjk,
    _has_dialogue_signal,
    _is_punct_only,
    _is_trivial_region,
    _normalize_punct,
    _partition_by_triage,
    _skip_row,
    _strip_watermarks,
    _translated_row,
    triage_region,
)

_SUBMODULE_MAP = {
    "_CJK_RE": triage,
    "_FULLWIDTH_MAP": triage,
    "_DIALOGUE_SIGNAL_PUNCT": triage,
    "_CJK_INTERJECTIONS": triage,
    "_contains_cjk": triage,
    "_cjk_char_count": triage,
    "_cjk_ratio": triage,
    "_normalize_punct": triage,
    "_is_trivial_region": triage,
    "TriageResult": triage,
    "_strip_watermarks": triage,
    "_cjk_count": triage,
    "_has_dialogue_signal": triage,
    "_is_punct_only": triage,
    "triage_region": triage,
    "_skip_row": triage,
    "_translated_row": triage,
    "_partition_by_triage": triage,
    "TranslatorBackend": backends,
    "ManualBackend": backends,
    "McpBackend": backends,
    "OllamaBackend": backends,
    "_get_backend": backends,
    "_locked_terms": glossary,
    "_build_locked_index": glossary,
    "_resolve_locked_glossary": glossary,
    "_enforce_glossary": glossary,
    "_PROMPT_INSTRUCTIONS": bundle,
    "_build_bundle": bundle,
    "_validate_response": bundle,
    "_build_contextual_retry_prompt": bundle,
    "build_translation_bundle": bundle,
    "write_translation_response": bundle,
    "_partition_regions_translation": engine,
    "_handle_unresolved_region": engine,
    "_load_raw_overrides": engine,
    "TranslationWriteContext": engine,
    "_write_output": engine,
    "run_translation": engine,
}


# Dynamic module proxy synchronizing attribute updates into submodules
class _Stage3Module(ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _SUBMODULE_MAP:
            setattr(_SUBMODULE_MAP[name], name, value)


sys.modules[__name__].__class__ = _Stage3Module

__all__ = [
    "_CJK_RE",
    "_FULLWIDTH_MAP",
    "_contains_cjk",
    "_cjk_char_count",
    "_cjk_ratio",
    "_normalize_punct",
    "_is_trivial_region",
    "TriageResult",
    "_strip_watermarks",
    "_cjk_count",
    "_DIALOGUE_SIGNAL_PUNCT",
    "_CJK_INTERJECTIONS",
    "_has_dialogue_signal",
    "_is_punct_only",
    "triage_region",
    "_skip_row",
    "_translated_row",
    "_partition_by_triage",
    "TranslatorBackend",
    "ManualBackend",
    "McpBackend",
    "OllamaBackend",
    "_get_backend",
    "_locked_terms",
    "_build_locked_index",
    "_resolve_locked_glossary",
    "_enforce_glossary",
    "_PROMPT_INSTRUCTIONS",
    "_build_bundle",
    "_validate_response",
    "_build_contextual_retry_prompt",
    "build_translation_bundle",
    "write_translation_response",
    "_partition_regions_translation",
    "_handle_unresolved_region",
    "_load_raw_overrides",
    "TranslationWriteContext",
    "_write_output",
    "run_translation",
]
