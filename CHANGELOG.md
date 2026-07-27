# Changelog

All notable changes to the Manhua Pipeline project will be documented in this file.

## [Unreleased]

### Fixed
- **Stage 1 - Detection Deduplication IoS**:
  - Added IoS (Intersection over Smaller) containment metric (`_ios`) alongside IoU in `_dedupe_raw_boxes` to catch offset and size mismatch duplicate detections (`DEDUP_IOS_THRESHOLD = 0.85`).

## [0.6.0] - 2026-07-26

### Added
- **Stage 3 - Unified Pre-Translation Triage**:
  - Added deterministic classifier (`triage_region`, `_strip_watermarks`, `_partition_by_triage`) running before batching to categorize regions as `empty`, `watermark`, `no_cjk`, `trivial`, or `translatable`.
  - Added `WATERMARK_FUZZY_TOKENS` in `config.py` for OCR-tolerant substring watermark matching (`ww.`, `最新免费`, `包子漫`, `巴子`, `colamanga`, etc.) to handle OCR manglings.
  - Hard-skips pure watermarks and empty regions before sending to LLM backends, ensuring honest completion ratio gate evaluation.
  - Trivial single-CJK regions check locked glossary first; on miss, fall back to non-blocking deferred handling.
  - `_build_bundle` now sends watermark-stripped `effective_text` to translation backends.
- **Stage 1 - Detection Deduplication**:
  - Added IoU-based deduplication (`_dedupe_raw_boxes`) for same-class same-page bounding boxes before reading-order sorting and ID assignment (`DEDUP_IOU_THRESHOLD = 0.90`).

### Fixed
- **Stage 5 - Black Box on Dark / Low-Coverage Bubbles**:
  - Scoped `_dominant_fill_color` sampling to a padded ring around the text box instead of the whole crop, preventing dark panel backgrounds from misidentifying white bubble fill.
  - Raised `BUBBLE_FILL_MIN_COVERAGE` to `0.40` and added a dark-fill safety guard that falls back to white when fill is dark at low coverage.
- **Stage 5 - Credits Page Stretch**:
  - Ensured credits page renders at native template aspect ratio without vertical stretching (`CREDITS_MATCH_PAGE_SIZE = False`).

## [0.5.0] - 2026-07-25

### Added
- **Stage 5 — Two-Pass Render (erase-all, then draw-all)**:
  - Introduced `RenderPlan` dataclass to carry all per-region draw state computed during Pass 1.
  - Added `_plan_and_erase_region()` (Pass 1): selects geometry, erases source text, computes font fit. Returns a `RenderPlan`. No drawing.
  - Added `_draw_region_plan()` (Pass 2): draws stored placements only. No pixel destruction.
  - Added `_finalize_plan()`: emits events and builds the `render.json` result row, called once after all draws.
  - Rewrote `_render_single_page()` to loop erase-all → draw-all → finalize, guaranteeing no erase can clobber an already-drawn glyph regardless of bubble overlap or detection order.
  - Retired `_render_region()` (now raises `NotImplementedError`).
- **Stage 5 — Dominant Fill Color Panel Erase**:
  - Added `_dominant_fill_color()`: samples a coarse grid over the bubble crop (excluding the text-glyph area) and returns the most common quantized color, averaged from actual pixels in the winning bucket. Near-pure white/black snapped to exact values.
  - Added `BUBBLE_FILL_MODE = "dominant"` config key; falls back to white-threshold mode when unset.
  - Added `_color_close_mask()` for Chebyshev-distance fill masking.
- **Stage 5 — Sibling-Region Layout Clip**:
  - Added `_shrink_box_away_from_siblings()`: pulls expanded layout boxes back from adjacent region text boxes so English text from different bubbles can never overlap.
  - `grow_layout_box()` now accepts a `siblings` list and applies the clip after interior expansion.
  - Shared-parent guard: when two regions share the same `parent_bubble`, interior expansion is suppressed for both to avoid cross-bubble layout drift.

### Fixed
- **Stacked bubble text clobbering** (known issue, now resolved): in figure-8 / stacked bubble pairs, a later region's erase step was painting white over an earlier region's already-drawn glyphs. The two-pass refactor eliminates this entirely - all erases complete before any draw, for any number of overlapping bubbles.
- **Off-white bubble fill**: `_dominant_fill_color()` now averages actual pixels in the winning quantization bucket instead of using the bucket midpoint, and snaps results ≥ 248 to pure white. Standard white bubbles now erase byte-identically to the previous output.
- **Case-insensitive `\n`/`\r\n` escape normalization**: `normalize_escaped_newlines()` now handles uppercase `\N`, `\R`, and `\R\N` escape sequences emitted by some LLMs.

## [0.4.0] - 2026-07-23

### Added
- **Human-directed Trailing Page Skip (`--skip-last N`)**:
  - Added `--skip-last` flag to CLI (`import`, `run-all`, `batch`) and a spinner control to the Guided GUI Runner.
  - Automatically flags the last N usable pages as `skip=True` (`skip_reason="trailing_skip"`) at import time so promo, ad, and credit pages are ignored across all downstream stages.
- **Timestamped File Logging**:
  - Added optional timestamped file logging (`run_YYYYMMDD_HHMMSS.log`) inside `logging_setup.py` when `log_dir` is configured.
  - Added `--fresh` log folder wiping in `pipeline.py` and `stage0_import.py` to remove stale logs before initializing new stage execution.

### Changed
- **Stitching Engine Hardening**:
  - Extended multi-page split stitching chains up to a maximum cap of 4 pages (`STITCH_MAX_CHAIN = 4`).
  - Relaxed split chain text probe requirement: a link is accepted if either the bottom half of the upper page or the top half of the lower page contains usable text.
- **Bubble Background Erasing**:
  - Updated render stage bubble erasing logic (`stage5_render.py`): pastes white background directly over the bubble mask when `use_mask` is `True`.
- **Glossary & Watermark Safety**:
  - Prevented region IDs (e.g. `P001_R001`) and non-CJK strings from being auto-seeded into `glossary.json`.
  - Defaulted auto-seeded glossary terms to `locked: false` so hints act as advisory hints rather than strict rejection locks.
  - Hardened translation and paraphrase prompt instructions against promotional watermarks (`包子漫畫`, `漫画屋`).

### Fixed
- Fixed encoding handling for file logging and pipeline JSON outputs on Windows environments.
- **Guided GUI Runner (`pipeline_gui.py`)**: Preserved the `Skip last` spinner value when choosing chapter inputs and kept the `0 · Import` stage button enabled for direct re-imports with `--fresh` or updated `--skip-last N` parameters.
