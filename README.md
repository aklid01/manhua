# Manhua Translation Pipeline

A modular, local, document-processing pipeline that converts original Chinese manhua chapters into natural, readable US English pages for personal reading.

The project is structured as a step-by-step document-processing system. Each stage has one clear responsibility, reads inputs, writes outputs, and is independently rerunnable.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Requirements](#requirements)
- [Installation (Windows)](#installation-windows)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Workspace Layout](#workspace-layout)
- [Glossary (Series-Level)](#glossary-series-level)
- [Overrides (Human-in-the-Loop)](#overrides-human-in-the-loop)
- [Translation & Paraphrase Backends](#translation--paraphrase-backends)
- [MCP Setup (VS Code, Cursor, Claude Desktop)](#mcp-setup-vs-code-cursor-claude-desktop)
- [Using the MCP Workflow](#using-the-mcp-workflow)
- [QA & Status](#qa--status)
- [Troubleshooting](#troubleshooting)

## Overview
The pipeline ingests a chapter (CBZ or a folder of images), detects speech bubbles, OCRs the Chinese text, translates it faithfully, paraphrases it into natural spoken US English, renders the English back onto the pages, and reports a quality verdict. Chapters are isolated per series, and a shared series glossary keeps names and terms consistent across chapters.

Translation and Paraphrasing run through pluggable backends. By default, they utilize the **MCP** (Model Context Protocol) workflow, letting a coding assistant (such as VS Code, Cursor, or Claude Desktop) read the pending text blocks and submit translations directly.

## Features
- Seven modular stages: **Import -> Detection -> OCR -> Translation -> Paraphrase -> Rendering -> QA**.
- **Per-series / per-chapter isolation** with a shared, consistency-preserving glossary.
- **Text-gated rendering**: original artwork is never erased unless there is trustworthy translated text (protects watermarks and split bubbles).
- **Bubble-aware text removal** + an **overflow ladder** (rewrap -> resize -> warn) using the bundled Comic Neue Bold font.
- **Human-friendly console progress logging** at every stage.
- **Pluggable AI backends** (`manual` | `mcp`); **MCP is the default**.
- **Human-in-the-loop overrides** via a per-chapter `overrides.json` file.
- **Defensive QA** with categorized warnings and a **SUCCESS / REVIEW / FAILED** verdict.

## How It Works (Pipeline)

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 0 Import | `import` | CBZ / folder of images | `pages/`, `manifest.json` |
| 1 Detection | `detect` | `pages/` | `stage1_detection/detection.json` (+ `overlays/`) |
| 2 OCR | `ocr` | `pages/` + `detection.json` | `stage2_ocr/ocr.json` |
| 3 Translation | `translate` | `ocr.json` + glossary | `stage3_translation/translation.json` |
| 4 Paraphrase | `paraphrase` | `translation.json` + glossary | `stage4_paraphrase/paraphrase.json` |
| 5 Rendering | `render` | detection + ocr + paraphrase | `stage5_render/rendered/<original names>` + `render.json` |
| 6 QA | `qa` | all artifacts | `stage6_qa/qa.json` (+ `overrides.json` stub) |

**Core rules:** original files are never modified; region IDs (`P{page:03d}_R{idx:03d}`) are stable across all stages; each stage is idempotent and rerunnable.

## Requirements
- **Windows**
- **Python 3.11+**
- A GPU is optional; OCR runs on CPU by default.
- Python packages (see `requirements.txt`): `ultralytics`, `paddleocr`, `paddlepaddle`, `pillow`, `numpy`, `huggingface_hub`, `fastmcp` (`>=3.4,<4`). `scipy` is optional (speeds up bubble masking; a pure-Python fallback is used if absent).
- An **MCP-capable coding assistant**: VS Code (GitHub Copilot agent mode), Cursor, or Claude Desktop.
- The detection model weights download automatically from Hugging Face on first run (`ogkalu/comic-speech-bubble-detector-yolov8m`).
- **Font:** `assets/fonts/ComicNeue-Bold.ttf` (OFL) must be present.

## Installation (Windows)
```bash
git clone <repository_url>
cd manhua_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Place `ComicNeue-Bold.ttf` in `assets/fonts/` (download from comicneue.com).

## Quick Start
First set the **series output folder** (the base folder that holds all chapters of one series), then run the stages.

```bash
# 1. Set the series base folder once (persisted to settings.json)
python pipeline.py --set-output-dir "<series_base_directory>"

# 2. Import a chapter (creates <series_base_directory>/<chapter_name>/)
python pipeline.py import --input "<path_to_chapter_input>"

# 3. Detect + OCR
python pipeline.py detect --chapter "<chapter_name>"
python pipeline.py ocr --chapter "<chapter_name>"

# 4. Translate + Paraphrase via MCP (see MCP Setup below) — then:
python pipeline.py translate --chapter "<chapter_name>"
python pipeline.py paraphrase --chapter "<chapter_name>"

# 5. Render + QA
python pipeline.py render --chapter "<chapter_name>"
python pipeline.py qa --chapter "<chapter_name>"
```

You can also run everything with resume support:
```bash
python pipeline.py run-all --input "<path_to_chapter_input>"
```
`run-all` stops cleanly at the Translation / Paraphrase **MCP handoff**. After the assistant submits the response via MCP, **re-run `run-all`** to resume — it is manifest-driven and continues from where it stopped.

## CLI Reference

| Command | Purpose | Key options |
|---|---|---|
| `import` | Normalize input into ordered pages + manifest | `--input` (required), `--chapter`, `--title-en`, `--title-romanized`, `--source`, `--fresh` |
| `detect` | Detect speech bubbles / narration | `--chapter` |
| `ocr` | OCR detected regions | `--chapter` |
| `translate` | Faithful literal translation (AI) | `--chapter` |
| `paraphrase` | Natural spoken US English (AI) | `--chapter` |
| `render` | Erase original + draw English | `--chapter` |
| `qa` | Quality checks + verdict | `--chapter` |
| `run-all` | Run every stage in order (resumable) | `--input`, `--from-stage`, `--chapter`, `--fresh` |

**Global options:** `--output-dir <path>` (override the series base for this run), `--set-output-dir <path>` (persist the base and exit).

**Notes:**
- `--chapter <name>` selects the chapter folder under the series base dir. If omitted and no valid manifest is found, the CLI **lists available chapters** and exits.
- `--fresh` (on `import`/`run-all`) wipes prior stage outputs + prompts/overrides for that chapter before re-import.

## Workspace Layout
```
<series_directory>/               # ONE series base folder
  glossary.json                   # series-level, shared across chapters
  <chapter_name>/                 # one folder per chapter
    manifest.json                 # chapter state + page map
    pages/                        # normalized 001.png, 002.png ...
    stage1_detection/
      detection.json
      overlays/                   # debug overlays
    stage2_ocr/ocr.json
    stage3_translation/
      translation.json
      translation_prompt.json
      translation_response.json
    stage4_paraphrase/
      paraphrase.json
      paraphrase_prompt.json
      paraphrase_response.json
    stage5_render/
      rendered/                   # final pages, ORIGINAL CBZ filenames
      render.json
    stage6_qa/qa.json
    overrides.json                # per-chapter overrides stub
```

## Glossary (Series-Level)
`glossary.json` lives at the **series base dir** and is shared across all chapters so names and terms stay consistent. Locked terms (`locked: true`) must be honored by Translation and Paraphrase; glossary conflicts are flagged but never block execution. Edit `glossary.json` directly to seed term rules.

```json
{
  "version": "v1",
  "terms": [
    {
      "term_id": "yu_lili",
      "source_term": "于丽丽",
      "target_term": "Yu Lili",
      "category": "person_name",
      "locked": true
    }
  ]
}
```

## Overrides (Human-in-the-Loop)
After QA, an `overrides.json` stub is written **per chapter** listing regions that could use attention. Fill any `region_id` with the correct English; non-empty values become **authoritative** for that region and are used **verbatim** by Translation and Paraphrase. Then re-run `translate -> paraphrase -> render -> qa`.

- Empty values are ignored; the `_comment` key is ignored.
- An existing `overrides.json` is **never overwritten**.

## Translation & Paraphrase Backends
The AI backend is selected in `config.py`:

- **`mcp` (DEFAULT):** a coding assistant reads the pending bundle and writes the response via MCP tools.
- **`manual`:** the pipeline writes a prompt bundle; you paste it into any assistant and save the JSON reply as the response file, then re-run.

All backends read and write the **same JSON structures**, so switching backends never breaks downstream stages. The translation/paraphrase prompt directives are defined in a single place in the code for consistency.

## MCP Setup (VS Code, Cursor, Claude Desktop)
The MCP server is `manhua_pipeline/adapters/mcp_server.py`. It communicates over **stdio** and logs to **stderr** (stdout is reserved for the protocol). It exposes:
- **Tools:** `list_pending`, `get_translation_bundle`, `submit_translation`, `get_paraphrase_bundle`, `submit_paraphrase`, `get_glossary`
- **Resource:** `series://chapters`
- **Prompts:** `translate_chapter`, `paraphrase_chapter` (formatted prompt templates)

> In all configurations, point `command` at the absolute path of your venv's Python interpreter (`.venv/Scripts/python.exe` on Windows).

### VS Code (GitHub Copilot — agent mode)
Requires VS Code 1.99+ and the GitHub Copilot extension. Create `.vscode/mcp.json` in the project root.

> **Important:** VS Code uses the top-level key **`servers`** (not `mcpServers`) and each server needs **`"type": "stdio"`**.

```json
{
  "servers": {
    "manhua": {
      "type": "stdio",
      "command": "<absolute_path_to_project_venv_python>",
      "args": ["<absolute_path_to_project>/manhua_pipeline/adapters/mcp_server.py"],
      "cwd": "<absolute_path_to_project>"
    }
  }
}
```
Reload VS Code. Verify with **Command Palette -> "MCP: List Servers"**. Open **Copilot Chat**, switch to **Agent mode**, and click **Configure Tools** to enable the `manhua` tools.

### Cursor
Create `.cursor/mcp.json` (project-level) or `~/.cursor/mcp.json` (global). Cursor uses the **`mcpServers`** key.

```json
{
  "mcpServers": {
    "manhua": {
      "command": "<absolute_path_to_project_venv_python>",
      "args": ["<absolute_path_to_project>/manhua_pipeline/adapters/mcp_server.py"]
    }
  }
}
```
Restart Cursor. Enable the server in **Settings -> MCP**. Use it from Cursor's **Agent / Composer** chat.

### Claude Desktop
Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows). Claude Desktop uses **`mcpServers`**.

```json
{
  "mcpServers": {
    "manhua": {
      "command": "<absolute_path_to_project_venv_python>",
      "args": ["<absolute_path_to_project>/manhua_pipeline/adapters/mcp_server.py"]
    }
  }
}
```
**Fully restart** Claude Desktop. The `manhua` tools appear under the tools (hammer) icon.

> **If the server fails to start**, run it directly in a terminal to check logs:
> ```bash
> .venv\Scripts\python.exe manhua_pipeline\adapters\mcp_server.py
> ```

## Using the MCP Workflow
1. Import + detect + ocr a chapter (these are automatic).
2. Run `python pipeline.py translate --chapter <name>` — this writes `translation_prompt.json` and awaits (MCP handoff).
3. In your assistant (**agent mode**), ask: *"List pending manhua chapters, then translate the pending chapter and submit it."* The assistant calls `get_translation_bundle` and `submit_translation`. For best fidelity, invoke the **`translate_chapter` prompt**.
4. Re-run `python pipeline.py translate --chapter <name>` to ingest the response and advance.
5. Repeat for paraphrase (`get_paraphrase_bundle` / `submit_paraphrase`, or the **`paraphrase_chapter` prompt**).
6. Run `render` + `qa`.

> **Tip:** model quality matters — select a strong model in your assistant's model picker.

## QA & Status
QA checks for pipeline warnings and reports a status verdict:

| Status | Condition |
|---|---|
| **SUCCESS** | 0-2 warnings |
| **REVIEW** | 3-10 warnings |
| **FAILED** | >10 warnings **or** any critical failure |

Benign no-text regions (watermarks, split halves) are recorded as **info** and do not count toward the status verdict. `qa.json` lists categorized `checks`, `warnings`, and a `needs_attention` list that feeds `overrides.json`.

## Troubleshooting
- **"Manifest not found" / wrong folder:** pass `--chapter <name>`; the CLI lists available chapters under the series base dir.
- **Font error:** place `ComicNeue-Bold.ttf` in `assets/fonts/` (or set `FONT_MISSING_HARD_ERROR=False` in `config.py` to fall back to a default font).
- **MCP server won't start:** run it directly to see logs; ensure `fastmcp` is installed in the venv and the config points at the correct absolute path.
- **VS Code tools missing:** confirm `.vscode/mcp.json` uses `"servers"` + `"type": "stdio"`, reload VS Code, use Copilot **Agent mode**, run **"MCP: List Servers"**.
- **Clear Chinese bubbles not translating:** OCR upscales crops 2x; extremely stylized / vertical text may still be missed — fill them via `overrides.json`.
- **Re-run a chapter from scratch:** use `--fresh` on `import`.
