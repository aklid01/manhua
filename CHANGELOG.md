# Changelog

All notable changes to the Manhua Pipeline project will be documented in this file.

## [Unreleased] - 2026-07-23

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
