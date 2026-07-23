<div align="center">

# Beyond the Horizon

### Translate. Refine. Render.

**A local-first, modular pipeline for translating manhua - on your terms.**

Chinese → English scanlation from raw CBZ to finished pages, with every stage
rerunnable, every backend replaceable, and a human in the loop wherever you want one.

<br />

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Local First](https://img.shields.io/badge/local--first-yes-success)
![MCP Ready](https://img.shields.io/badge/MCP-ready-8A2BE2)

</div>

---

## Philosophy

The pipeline adapts to you - not the other way around.

Manhua Pipeline is built on the idea that a translation workflow should be **yours**:
your hardware, your models, your review process. It runs locally by default, breaks the
work into independent stages you can rerun at will, and never locks you into a vendor or
an API key.

- **Local-first** - runs on your machine; nothing leaves it unless you choose.
- **Modular** - seven independent stages, each with a single responsibility.
- **Vendor-neutral** - swap translation and refinement backends freely.
- **Human-in-the-loop** - review, override, or hand off at any stage.
- **Fully rerunnable** - every stage is idempotent and resumable.
- **Replaceable components** - no stage assumes what came before it, beyond its input file.

> [!NOTE]
> **Your workflow. Your models. Your choice.**
> No mandatory AI credits. No mandatory API keys. Just a pipeline that
> does one job well and gets out of your way.

---

## Why Manhua Pipeline?

This is not "another OCR tool" or "another translation script." It is a **workflow**.

Most tools give you one monolithic button: image in, translation out, no visibility, no
recovery when a step goes wrong. Manhua Pipeline is the opposite - a chain of small,
inspectable stages that each write a plain JSON artifact you can read, edit, and rerun.

| What matters             | How the pipeline delivers it                                                   |
| ------------------------ | ------------------------------------------------------------------------------ |
| **Modular stages**       | Import → Detect → OCR → Translate → Paraphrase → Render → QA, each standalone. |
| **Rerunnable**           | Re-run any stage without redoing the ones before it.                           |
| **Human review**         | Every stage emits editable JSON; overrides are first-class.                    |
| **Backend independence** | Translate/refine with a local model, a cloud agent, or your own hands.         |
| **Reproducible**         | A manifest tracks exactly where each chapter is; resume anytime.               |

If you care about _how_ your translation is produced - not just that it happened - this is
built for you.

---

## Feature Highlights

- **Local-first** - RT-DETR / YOLOv8 text & bubble detection + PaddleOCR / Transformers (PP-OCRv6) + local LLMs, all on your box.
- **Modular** - seven decoupled stages, plain-JSON handoffs.
- **Vendor-neutral** - `manual`, `mcp`, and `ollama` backends for translate & refine.
- **Human-in-the-loop** - per-region overrides, glossary locking, manual handoff.
- **Guided GUI** - visual step-by-step runner with automatic handoff & resume tracking.
- **MCP Ready** - drive translation/refinement from any MCP client (e.g. Antigravity).
- **Fully rerunnable** - idempotent stages, manifest-driven resume, batch processing.

---

## Pipeline Overview

```
        ┌───────────┐
        │  Import    │  CBZ / ZIP / folder → ordered pages + manifest
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ Detection  │  RT-DETR / YOLOv8 speech-bubble + narration boxes
        └─────┬─────┘
              ▼
        ┌───────────┐
        │    OCR     │  PaddleOCR / Transformers (PP-OCRv6 zh) + confidence retry
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ Translation│  literal zh → en  (manual │ mcp │ ollama)
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ Paraphrase │  natural spoken en  (manual │ mcp │ ollama)
        └─────┬─────┘
              ▼
        ┌───────────┐
        │  Rendering │  typeset onto pages + credits page
        └─────┬─────┘
              ▼
        ┌───────────┐
        │     QA     │  warnings, overflow & conflict report
        └─────┬─────┘
              ▼
        ┌───────────┐
        │  Package   │  zip │ cbz │ tar │ pdf   (optional)
        └───────────┘
```

Every arrow is a JSON file on disk. Stop anywhere, inspect it, edit it, rerun from there.

---

## Recommended Workflow

> [!TIP]
> **The sweet spot: local translation, agent-assisted refinement.**
>
> | Stage           | Backend                                                            | Why                                                                                                                                                           |
> | --------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | **Translation** | `ollama` - a strong Chinese→English model (e.g. `qwen2.5:3b-instruct`) | Literal translation is a low-creativity task a small local model handles well - free and fully automated.                                                     |
> | **Refinement**  | `mcp` via Antigravity - Gemini 3.1 Pro / 3.5 Flash                 | Turning literal English into natural dialogue needs real language skill; a capable model shines here, and Antigravity's free tier keeps credit use near zero. |
>
> This split gives you excellent quality while minimizing AI-credit spend - the local model
> does the bulk work, the strong model does only the part that needs judgment.

---

## Workflow Selection

Choose based on your hardware and goals - the pipeline is designed to flex.

| Your situation             | Suggested setup                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **Powerful / capable GPU** | `ollama` for both translate and refine - fully local, fully automated.             |
| **No GPU**                 | `manual` handoff - the pipeline writes prompt bundles; you paste into any chatbot. |
| **Best quality**           | `ollama` translate + `mcp` refine (see Recommended Workflow).                      |
| **Advanced users**         | Replace any backend - implement the small `request()` contract and register it.    |

> [!NOTE]
> A backend is just "given a bundle of regions, return a `{region_id: text}` map." That's
> the entire contract. Local model, remote API, MCP tool, or a human with a text editor -
> the pipeline doesn't care.

---

## Quick Start

> [!IMPORTANT]
> **Prerequisites:** Python 3.10+, and (optionally) [Ollama](https://ollama.com) if you
> want local translation/refinement.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Tell the pipeline where your series live (one time)
python pipeline.py --set-output-dir "/path/to/your/series"

# 3. (Optional) pull a local translation model
ollama pull qwen2.5:3b-instruct

# 4. Run a chapter end-to-end via CLI
python pipeline.py run-all --input "/path/to/chapter_001.cbz"

# ...or launch the Guided GUI Runner
python pipeline_gui.py
```

That's it. The first run downloads the detection model from Hugging Face and caches it;
every run after is offline for detection.

> [!TIP]
> If a stage needs a human (a `manual`/`mcp` handoff), the pipeline stops cleanly and tells
> you exactly what to do. Resume by re-running the same command or clicking **Resume ▶** in the GUI.

---

## Installation

```bash
git clone https://github.com/aklid01/manhua.git
cd manhua
pip install -r requirements.txt
```

**Core dependencies** (`requirements.txt`):

| Package                      | Role                                              |
| ---------------------------- | ------------------------------------------------- |
| `transformers`               | RT-DETR text & bubble detector + HF OCR model     |
| `ultralytics`                | YOLOv8 speech-bubble detection (fallback)        |
| `paddleocr` / `paddlepaddle` | Chinese OCR engine                                |
| `pillow`                     | Image handling & rendering                        |
| `numpy`                      | Array ops for OCR/detection                       |
| `fastmcp`                    | MCP server for agent-driven backends              |

> [!NOTE]
> The detection model (`ogkalu/comic-text-and-bubble-detector` for RT-DETR or `ogkalu/comic-speech-bubble-detector-yolov8m` for YOLOv8) downloads
> automatically on first detection run and is cached by Hugging Face thereafter.

---

## Guided GUI Runner

In addition to the command-line interface, Manhua Pipeline includes a guided graphical runner:

```bash
python pipeline_gui.py
```

- **Series & Input Pickers**: Set your series folder and choose a chapter input (folder or CBZ/ZIP archive).
- **Skip Trailing Pages**: Set a spinner to skip promo, ad, or credit pages at the end of a chapter during import.
- **Sequential Execution**: Stages unlock step-by-step as each preceding stage completes.
- **Handoff & Resume Tracking**: If a `manual` or `mcp` backend requires external input, the stage button dynamically updates to **Resume ▶ <Stage>** and the log displays step-by-step handoff instructions.
- **Idempotent Continuation**: Saving the response file and clicking **Resume ▶** ingests the response, advances the chapter manifest, and unlocks the next stage.
- **Log Filtering**: Output pane filters log messages to surface warnings, re-runs, overrides, and verdicts without cluttering the screen.
- **Packaging**: Select archive formats (`cbz`, `zip`, `tar`, `pdf`) and package chapters directly from the interface.

---

## CLI Usage

Every stage is its own command, and can be run in isolation. `run-all` chains them;
`batch` runs a whole folder of chapters.

### Per-stage

```bash
python pipeline.py import   --input chapter_001.cbz     # CBZ / ZIP / folder → pages
python pipeline.py detect   --chapter chapter_001       # bubble detection
python pipeline.py ocr      --chapter chapter_001       # Chinese OCR
python pipeline.py translate  --chapter chapter_001     # literal zh → en
python pipeline.py paraphrase --chapter chapter_001     # natural english
python pipeline.py render   --chapter chapter_001       # typeset pages
python pipeline.py qa       --chapter chapter_001       # quality report
```

### End-to-end

```bash
# One chapter, all stages, package the result
python pipeline.py run-all --input chapter_001.cbz --package cbz,pdf

# Resume from a specific stage
python pipeline.py run-all --chapter chapter_001 --from-stage paraphrase
```

### Batch a folder of chapters

```bash
python pipeline.py batch --input "/downloads/MySeries" --package cbz
```

> [!TIP]
> `batch` skips already-completed chapters, resumes pending ones, continues past errors,
> and writes a per-run log plus per-chapter error logs. Point it at a folder and walk away.

### Package an existing chapter

```bash
python pipeline.py package --chapter chapter_001 --package zip,cbz,tar,pdf
```

### Common flags

| Flag               | Applies to                | Purpose                                                                |
| ------------------ | ------------------------- | ---------------------------------------------------------------------- |
| `--input`          | import / run-all / batch  | Source CBZ/ZIP/folder (import, run-all) or folder of chapters (batch). |
| `--chapter`        | most stages               | Target an existing chapter by name.                                    |
| `--workspace`      | most stages               | Target a chapter by workspace path (relative or absolute).             |
| `--fresh`          | import / run-all          | Wipe prior stage outputs before importing.                             |
| `--skip-last`      | import / run-all / batch  | Skip the last N pages (promo/ad/credits pages at chapter end).         |
| `--from-stage`     | run-all                   | Resume from a given stage.                                             |
| `--package`        | run-all / batch / package | Comma-separated formats: `zip,cbz,tar,pdf`.                            |
| `--title-romanized`| import / run-all          | Set romanized series/chapter title metadata.                           |
| `--title-en`       | import / run-all          | Set English series/chapter title metadata.                             |
| `--source`         | import / run-all          | Set source URL/provenance metadata.                                    |
| `--no-resume`      | batch                     | Skip any existing chapter folder instead of resuming pending ones.     |
| `--clear-delay`    | batch                     | Seconds a completed chapter's output stays on screen before clearing.  |
| `--set-output-dir` | (top level)               | Persist your series base directory and exit.                           |

---

## Workspace Layout

Everything for a series lives under one base directory. Each chapter is a self-contained
folder; each stage owns a subfolder and one JSON artifact.

```
your-series/
├── glossary.json                  # series-wide locked terms
└── chapter_001/
    ├── manifest.json              # source of truth: stage, page list, status
    ├── overrides.json             # per-region manual overrides (optional)
    ├── pages/                     # normalized page images
    ├── stage1_detection/          # detection.json (+ overlays)
    ├── stage2_ocr/                # ocr.json
    ├── stage3_translation/        # translation.json (+ prompt/response bundles)
    ├── stage4_paraphrase/         # paraphrase.json
    ├── stage5_render/rendered/    # finished pages: 001.png, 002.png, …, zzz_credits.png
    ├── stage6_qa/                 # qa report
    ├── stage7_package/            # archives: chapter_001.cbz, .pdf, …
    └── logs/                      # per-chapter logs
```

> [!NOTE]
> The `manifest.json` `current_stage` field is the pipeline's memory. Resume, batch skip,
> and status all read from it - so a chapter always knows exactly where it left off.

---

## Pipeline Stages

| #   | Stage           | Input              | Output                     | Engine                        |
| --- | --------------- | ------------------ | -------------------------- | ----------------------------- |
| 0   | **Import**      | CBZ / ZIP / folder | `manifest.json` + `pages/` | Pillow                        |
| 1   | **Detection**   | pages              | `detection.json`           | RT-DETR (default) / YOLOv8    |
| 2   | **OCR**         | pages + boxes      | `ocr.json`                 | PaddleOCR / Transformers (zh) |
| 3   | **Translation** | `ocr.json`         | `translation.json`         | pluggable backend             |
| 4   | **Paraphrase**  | `translation.json` | `paraphrase.json`          | pluggable backend             |
| 5   | **Rendering**   | pages + paraphrase | `rendered/*.png`           | Pillow                        |
| 6   | **QA**          | all artifacts      | quality report             | rule-based                    |
| 7   | **Package**     | rendered pages     | `zip / cbz / tar / pdf`    | stdlib + Pillow               |

**Notable stage behaviors**

- **Detection** defaults to RT-DETR (`ogkalu/comic-text-and-bubble-detector`) with YOLOv8 as fallback, skipping text-free regions (SFX/watermark/credits) during downstream OCR.
- **OCR** retries low-confidence regions with escalating image preprocessing, keeping the best result - deterministic OCR only improves when the _input_ changes.
- **Translation / Paraphrase** each pick a backend independently (`manual`, `mcp`, `ollama`) and enforce a locked glossary, flagging conflicts rather than silently rewriting. Single-character CJK noise/SFX is filtered from blocking chapter completion and glossary auto-seeding.
- **Rendering** typesets each bubble with adaptive WCAG text coloring (white text on dark background balloons, black text on light backgrounds) and appends a randomly chosen **credits page** (outside the QA/manifest page count).
- **Split-bubble stitching** (detection sub-step) merges a bubble sliced across two pages into one using container bubble geometry (RT-DETR) and a relaxed text probe for cut-off edge regions.

---

## Translation Backends

Both Translation and Paraphrase share the same three-backend model. Set them independently
in `config.py`:

```python
TRANSLATOR_BACKEND = "ollama"   # "manual" | "mcp" | "ollama"
PARAPHRASE_BACKEND = "mcp"      # "manual" | "mcp" | "ollama"
```

| Backend      | How it works                                                                      | Best for                               |
| ------------ | --------------------------------------------------------------------------------- | -------------------------------------- |
| **`manual`** | Writes a prompt bundle to disk; you paste it into any chatbot and save the reply. | No GPU, no setup, full control.        |
| **`mcp`**    | Exposes the bundle as MCP tools an agent (e.g. Antigravity) calls.                | High-quality, agent-driven refinement. |
| **`ollama`** | Calls a local Ollama server inline - no handoff, fully automated.                 | Local, free, hands-off runs.           |

**Ollama configuration** (`config.py`):

```python
OLLAMA_TRANSLATE_MODEL = "qwen2.5:3b-instruct"   # CJK-capable; fits a 6 GB GPU
OLLAMA_HOST            = "http://localhost:11434"
OLLAMA_BATCH_SIZE      = 8
# Paraphrase uses its own OLLAMA_PARA_* knobs (higher temperature for natural phrasing)
```

> [!WARNING]
> Small local models are excellent at _literal translation_ but weaker at _creative
> paraphrasing_ - the latter needs headroom a 3B model on a 6 GB GPU doesn't have. If
> paraphrase quality matters, use `mcp` for refinement (see Recommended Workflow).

---

## MCP Integration

The pipeline ships an MCP server so any MCP-capable client can drive the `mcp` backends -
translating and refining without leaving your editor.

```bash
python manhua_pipeline/adapters/mcp_server.py
```

**Exposed tools:** `list_pending`, `get_translation_bundle`, `submit_translation`,
`get_paraphrase_bundle`, `submit_paraphrase`, `get_glossary`.

**Antigravity setup** - add to `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "manhua-pipeline": {
      "command": "/path/to/.venv/bin/python",
      "args": ["manhua_pipeline/adapters/mcp_server.py"],
      "cwd": "/path/to/manhua-pipeline"
    }
  }
}
```

Then, in your agent: _"List pending chapters, then translate/paraphrase the pending one."_
The agent calls the tools; the pipeline writes the results and advances.

> [!TIP]
> Set your series directory first (`python pipeline.py --set-output-dir …`) - the MCP
> server reads it from `settings.json`.

---

## Manual Workflow

No GPU and no agent? The `manual` backend turns any chatbot into your translator.

1. Set `TRANSLATOR_BACKEND = "manual"` (and/or `PARAPHRASE_BACKEND = "manual"`).
2. Run the stage - it writes a prompt bundle and stops:
   `stage3_translation/translation_prompt.json`.
3. Paste that JSON into your chatbot of choice.
4. Save its JSON reply as `translation_response.json` in the same folder.
5. Re-run the stage - it ingests the reply and continues.

The same pattern applies to paraphrase. Nothing is automated away that you didn't ask to
automate.

---

## Advanced Configuration

All tunables live in `config.py`. Highlights:

| Area                    | Keys                                                                      |
| ----------------------- | ------------------------------------------------------------------------- |
| **Backends**            | `TRANSLATOR_BACKEND`, `PARAPHRASE_BACKEND`                                |
| **Ollama (translate)**  | `OLLAMA_*` - host, model, batch size, retries, completion gate (`OLLAMA_MAX_MISSING`, `OLLAMA_TRIVIAL_CJK_CHARS`) |
| **Ollama (paraphrase)** | `OLLAMA_PARA_*` - same knobs, tuned for natural phrasing                  |
| **OCR**                 | `OCR_ENGINE`, `OCR_VERSION`, `OCR_CONFIDENCE_THRESHOLD`, `OCR_USE_GPU`, `OCR_RETRY_ENABLED`, `OCR_RETRY_MAX` |
| **Detection**           | `DETECTOR_BACKEND`, `DETECTION_MODEL`, `RTDETR_REPO`, `RTDETR_CONF`, `RTDETR_SKIP_TEXT_FREE`, `DETECTOR_USE_GPU` |
| **Rendering**           | `FONT_PATH`, `FONT_MAX_PT`, `FONT_MIN_PT`, `LINE_SPACING`, adaptive text fill on dark bubbles |
| **Credits**             | `CREDITS_TEMPLATES`, `CREDITS_DIR` (+ `credits` block in `settings.json`) |
| **Stitching**           | `STITCH_ENABLED`, `STITCH_EDGE_EPS`, `STITCH_MIN_X_OVERLAP`, `STITCH_TEXT_MIN_CONF` |
| **Packaging**           | `VALID_PACKAGE_FORMATS`, `PACKAGE_IMAGE_EXTS`                             |
| **QA**                  | `SUCCESS_MAX`, `REVIEW_MAX` warning thresholds                            |

Per-chapter **overrides** (`overrides.json`) let you hand-set any region's final text, and
the series **glossary** (`glossary.json`) locks names/terms across every chapter.

---

## Troubleshooting

> [!WARNING]
> **Detection model won't download.** The first `detect` run fetches the RT-DETR / YOLOv8 weights from
> Hugging Face. If you're offline or rate-limited, pre-cache the model or set
> `HF_HUB_OFFLINE=1` once it's cached.

| Symptom                                    | Likely cause                                   | Fix                                                                                                           |
| ------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Untranslated Chinese renders as boxes (□□) | A backend left text untranslated               | The CJK guard flags these as `needs_translation`; check the translate completion gate / try a stronger model. |
| Paraphrase "ruined" the meaning            | Small local model over-creative on refinement  | Use `mcp` for paraphrase; keep local for translation.                                                         |
| Ollama errors immediately                  | `ollama serve` not running or model not pulled | Start Ollama and `ollama pull <model>`.                                                                       |
| A stage keeps "waiting"                    | `manual`/`mcp` handoff pending                 | Provide the response file / run the MCP tool, then re-run or click **Resume ▶**.                              |
| Batch stops early                          | (it shouldn't)                                 | Batch continues past errors - check `logs/` for the per-chapter error log.                                    |

---

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
Copyright © 2026 Ishan Dev Shakya.

You may use, modify, and distribute this software under the terms of the AGPL-3.0. Network
use counts as distribution - see [`LICENSE`](LICENSE) for the full text.

---

<div align="center">

## Beyond the Horizon

**Translate. Refine. Render.**

Built for readers. Built for translators. Built for the open-source community.

</div>
