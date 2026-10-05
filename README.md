# Manhua Translation Pipeline

Manhua Translation Pipeline is a local-first Python application for converting
Chinese manhua pages into translated English pages. It accepts CBZ, ZIP, or
folders of images and processes them through detection, OCR, translation,
editing, rendering, quality assurance, and optional packaging.

The project is intended for two groups:

- Readers and translators who want a guided workflow with the option to review
  or correct text manually.
- Developers who want an inspectable, rerunnable pipeline with replaceable
  detection, OCR, translation, and paraphrasing components.

The pipeline writes intermediate results as JSON files inside each chapter
workspace. You can inspect or edit those files, rerun one stage, or resume from
the last completed stage.

## What the pipeline does

For each chapter, the application performs these steps:

1. **Import** - normalizes a CBZ, ZIP, or image folder into ordered PNG pages.
2. **Detection** - finds speech bubbles and narration regions.
3. **OCR** - extracts Chinese text and records confidence scores.
4. **Translation** - produces a literal Chinese-to-English translation.
5. **Paraphrase** - optionally turns the literal translation into natural
   dialogue while preserving the meaning and glossary terms.
6. **Rendering** - removes the original text and typesets English onto the
   page.
7. **QA** - reports missing text, untranslated regions, overflow, glossary
   conflicts, and rendering failures.
8. **Package** - optionally creates ZIP or CBZ output.

Packaging is an optional terminal operation. It does not change the chapter's
pipeline stage in `manifest.json`.

### Supported input and output

Supported inputs:

- CBZ archives
- ZIP archives
- Folders containing JPG, JPEG, PNG, or WEBP images

PDF import is not currently implemented.

Supported package formats:

`zip` and `cbz`. CBZ is the preferred comic-reader/upload format; ZIP is
provided for generic archive compatibility. PDF and TAR are not produced by
Stage 7.

## How to choose a workflow

The translation and paraphrase stages can use different backends. Configure
them independently in `config.py`.

| Backend         | Description                                                                  | Suitable when                                                           |
| --------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `manual`        | Writes a JSON prompt bundle. You provide and save the response.              | You want full control or do not have a local model.                     |
| `mcp`           | Exposes pending work through the included MCP server.                        | You use an MCP-capable agent or client.                                 |
| `ollama`        | Sends requests to a local Ollama server.                                     | You want an automated local workflow.                                   |
| `openai_compat` | Routes requests to OpenAI-compatible HTTP APIs with provider fallback chains. | You want automated cloud or self-hosted API models with health probing. |

The current defaults are:

```python
TRANSLATOR_BACKEND = "ollama"
PARAPHRASE_BACKEND = "mcp"
```

This combination uses Ollama for literal translation and an MCP client for
language refinement. For a fully local workflow, set both backends to
`ollama`. For a workflow without a GPU or agent integration, set one or both to
`manual`. For cloud or local API endpoints with automatic failover, set one or
both to `openai_compat`.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/aklid01/manhua.git
cd manhua
python -m venv .venv
```

Activate the virtual environment, then install the core dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The core dependencies include the detection models, PaddleOCR, image
processing, Transformers support, and the MCP server. Detection and OCR models
may download from their model providers on first use and are then cached
locally.

### Optional GUI and touch-up editor

The guided runner and the standalone touch-up editor require PyQt6 and the
additional packages in `requirements-touchup.txt`:

```bash
pip install -r requirements-touchup.txt
```

The touch-up editor can open a page or chapter, select regions, run OCR and
translation, erase source text, position English text, and save the edited
page.

## Quick start

### 1. Choose the series output directory

The pipeline stores all chapter workspaces below one series directory. Set it
once with:

```bash
python pipeline.py --set-output-dir "/path/to/series"
```

This saves the setting in `settings.json`. You can override it for one run with
`--output-dir`.

### 2. Configure translation

For local translation and paraphrasing, make sure Ollama is installed and
running, then pull the configured model:

```bash
ollama pull translategemma:4b
```

Set the backends in `config.py`:

```python
TRANSLATOR_BACKEND = "ollama"
PARAPHRASE_BACKEND = "ollama"
```

Alternatively, keep `PARAPHRASE_BACKEND = "mcp"` or use `manual` for either
stage. The pipeline pauses when external input is required and can be resumed
after the response is supplied.

### 3. Process a chapter

```bash
python pipeline.py run-all --input "/path/to/chapter_001.cbz" --package cbz
```

You can also run the guided GUI:

```bash
python pipeline_gui.py
```

The GUI lets you choose the series directory and chapter input, run stages in
order, resume pending handoffs, view logs, package output, and open a rendered page in the touch-up editor.

## Command-line usage

`pipeline.py` provides a command for every processing stage. Each command
operates on a chapter workspace and can be rerun independently.

### Run individual stages

```bash
python pipeline.py import      --input chapter_001.cbz
python pipeline.py detect      --chapter chapter_001
python pipeline.py ocr         --chapter chapter_001
python pipeline.py translate   --chapter chapter_001
python pipeline.py paraphrase  --chapter chapter_001
python pipeline.py render      --chapter chapter_001
python pipeline.py qa          --chapter chapter_001
```

The import command also accepts a folder of images. When starting from import,
you can add metadata such as `--title-en`, `--title-romanized`, and `--source`.

### Run or resume the complete pipeline

```bash
# Start from import
python pipeline.py run-all --input chapter_001.cbz

# Resume an existing chapter at a later stage
python pipeline.py run-all --chapter chapter_001 --from-stage paraphrase

# Import while skipping trailing promotional or credit pages
python pipeline.py run-all --input chapter_001.cbz --skip-last 2
```

Use `--fresh` with `import` or `run-all` when you intentionally want to remove
the existing stage artifacts for that chapter and start again.

### Process a folder of chapters

```bash
python pipeline.py batch --input "/path/to/chapter-files" --package cbz
```

Batch processing accepts `.cbz` and `.zip` files, resumes incomplete chapter
folders by default, skips completed chapters, and continues after an error.
Use `--no-resume` to skip any chapter folder that already exists.

### Package rendered pages

```bash
python pipeline.py package --chapter chapter_001 --package zip,cbz
```

The package command reads images from `stage5_render/rendered/` and writes the
requested files to `stage7_package/`.

### Resolve manual overrides

QA creates an `overrides.json` file when a region needs attention. You can edit
that file directly or use the interactive resolver:

```bash
python pipeline.py resolve-overrides --chapter chapter_001
```

After changing an override, rerun `translate`, `paraphrase`, `render`, and
`qa` as appropriate.

### Translation recovery and contextual retry

The local Ollama and OpenAI-compatible translators are best-effort. When either
cannot produce a valid translation for a meaningful region, the original CJK text
is handed to Stage 4, where the stronger paraphrase backend performs a combined
translate-and-paraphrase recovery. This is the normal path when
`PARAPHRASE_BACKEND = "mcp"` or when using a capable LLM paraphrase backend.

Stage 3's extra one-region contextual retry is disabled by default because it
duplicates work before the stronger Stage 4 backend runs. Enable
`OLLAMA_CONTEXTUAL_RETRY_ENABLED = True` only when a local contextual retry is
preferred or Stage 4 recovery is unavailable. Single-character CJK/SFX
passthrough remains a separate route and is not affected.

### Common options

| Option                  | Commands                                    | Purpose                                                                              |
| ----------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--input PATH`          | `import`, `run-all`, `batch`                | Input archive, image folder, or batch folder.                                        |
| `--chapter NAME`        | Most stages, `package`, `resolve-overrides` | Chapter folder name below the series directory.                                      |
| `--workspace PATH`      | Most stages                                 | Use a specific relative or absolute workspace path.                                  |
| `--output-dir PATH`     | Top level                                   | Override the configured series directory for one run.                                |
| `--set-output-dir PATH` | Top level                                   | Persist the series directory and exit.                                               |
| `--fresh`               | `import`, `run-all`, `batch`                | Remove prior stage artifacts before importing.                                       |
| `--skip-last N`         | `import`, `run-all`, `batch`                | Mark the final N pages as skipped.                                                   |
| `--from-stage NAME`     | `run-all`                                   | Resume from `import`, `detect`, `ocr`, `translate`, `paraphrase`, `render`, or `qa`. |
| `--package FORMATS`     | `run-all`, `batch`, `package`               | Comma-separated `zip` and/or `cbz`.                                                  |
| `--no-resume`           | `batch`                                     | Skip existing chapter folders instead of resuming them.                              |

Run `python pipeline.py --help` or `python pipeline.py <command> --help` for
the complete parser help.

## Chapter workspace

Each chapter is self-contained. A typical workspace looks like this:

```text
your-series/
|-- glossary.json
`-- chapter_001/
    |-- manifest.json
    |-- overrides.json
    |-- pages/
    |-- stage1_detection/
    |   `-- detection.json
    |-- stage2_ocr/
    |   |-- ocr.json
    |   `-- ocr_review.json       # only when review items exist
    |-- stage3_translation/
    |   |-- translation.json
    |   |-- translation_prompt.json
    |   `-- translation_response.json
    |-- stage4_paraphrase/
    |   |-- paraphrase.json
    |   |-- paraphrase_prompt.json
    |   `-- paraphrase_response.json
    |-- stage5_render/
    |   |-- render.json
    |   `-- rendered/
    |-- stage6_qa/
    |   `-- qa.json
    |-- stage7_package/
    `-- logs/
```

Not every optional file is present in every run. `manifest.json` records the chapter metadata, page list, current stage, completed stages, and QA status. The JSON artifacts are the handoff points between stages and are intended to be readable for debugging or manual review.

The series-level `glossary.json` stores locked names and terms. Chapter-level
`overrides.json` stores region-specific corrections.

## Configuration

Most settings are in `config.py`. The most commonly changed values are:

| Area          | Settings                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------- |
| Translation   | `TRANSLATOR_BACKEND`, `OLLAMA_TRANSLATE_MODEL`, `OLLAMA_HOST`                                   |
| Paraphrasing  | `PARAPHRASE_BACKEND`, `OLLAMA_PARA_MODEL`, `OLLAMA_PARA_HOST`                                   |
| OpenAI-compat | `OPENAI_COMPAT_TIMEOUT`, `OPENAI_COMPAT_MAX_RETRIES`, `OPENAI_COMPAT_BATCH_SIZE`, `OPENAI_COMPAT_CONCURRENCY`, `OPENAI_COMPAT_STREAM`, `OPENAI_COMPAT_PROBE_ENABLED`, `OPENAI_COMPAT_TRANSLATE_TEMPERATURE`, `OPENAI_COMPAT_PARA_TEMPERATURE` |
| Import        | `IMPORT_SLICE_TARGET_HEIGHT`, `IMPORT_SLICE_SEAM_TOLERANCE`, `IMPORT_SLICE_SEAM_BAND`, `IMPORT_SLICE_CONTENT_THRESHOLD`, `IMPORT_MAX_WIDTH` |
| OCR           | `OCR_ENGINE`, `OCR_VERSION`, `OCR_CONFIDENCE_THRESHOLD`, `OCR_USE_GPU`                          |
| Detection     | `RTDETR_REPO`, `RTDETR_CONF`, `DETECTOR_USE_GPU`                                                |
| YOLO-seg      | `YOLO_SEG_ENABLED`, `YOLO_SEG_MODEL`, `YOLO_SEG_CONF`, `YOLO_SEG_DEVICE`, `YOLO_SEG_PAD_FRAC`   |
| Rendering     | `FONT_PATH`, `FONT_MAX_PT`, `FONT_MIN_PT`, `LINE_SPACING`                                       |
| QA            | `SUCCESS_MAX`, `REVIEW_MAX`                                                                     |
| Packaging     | `VALID_PACKAGE_FORMATS`, `PACKAGE_IMAGE_EXTS`                                                   |

The bundled font is `assets/fonts/ComicNeue-Bold.ttf`. Set `FONT_PATH` to a
different font if required. GPU use is enabled by default for detection and OCR
but falls back to CPU when a compatible GPU runtime is unavailable.

### Detection and OCR details

The detector backend is RT-DETR only. The earlier YOLOv8 alternate backend (`DETECTOR_BACKEND = "yolov8"`, `DETECTION_MODEL`, `DETECTION_CONF`) was removed, detection settings are now `RTDETR_REPO` and `RTDETR_CONF`. Detection can skip text-free regions such as watermarks, sound effects, and credits.

OCR uses PaddleOCR with the configured `OCR_ENGINE` and Chinese language model. The current configuration prefers the Transformers engine and falls back to the Paddle engine if initialization fails. Low-confidence reads can be retried with progressively stronger preprocessing. Suspicious empty or punctuation-only dialogue regions are written to `stage2_ocr/ocr_review.json` for optional manual correction.

### Optional YOLO-seg bubble mask refinement

For precise speech bubble boundary masks during erasing and rendering, the pipeline includes an optional two-stage refinement pass:

1. RT-DETR detects speech bubbles and narration bounding boxes.
2. If `YOLO_SEG_ENABLED = True`, RT-DETR is unloaded from GPU memory, and a custom YOLO instance segmentation model runs on padded crops of each bubble region.
3. The resulting binary mask is base64-encoded into `stage1_detection/detection.json`.
4. Stage 5 (Render) uses the YOLO mask to cleanly erase bubble interiors without bleeding into background art or character linework.

To enable YOLO-seg refinement:

```python
# config.py
YOLO_SEG_ENABLED = True
YOLO_SEG_MODEL = "models/best.pt"  # path relative to project root
YOLO_SEG_CONF = 0.4
YOLO_SEG_DEVICE = "0"  # "0" for GPU, "cpu" for CPU
```

## Translation and paraphrase handoffs

### Manual backend

For either stage:

1. Set the corresponding backend to `manual` in `config.py`.
2. Run the stage. It writes a prompt JSON file and waits.
3. Submit a JSON object mapping each `region_id` to its response text.
4. Save the response using the filename requested by the stage.
5. Rerun the same command to ingest the response and continue.

For example, translation uses:

```text
stage3_translation/translation_prompt.json
stage3_translation/translation_response.json
```

Paraphrase uses the equivalent files in `stage4_paraphrase/`.

### Ollama backend

Ollama is expected at `http://localhost:11434` by default. The translation and
paraphrase stages have separate model, batch size, timeout, retry, and
temperature settings. Translation uses a lower temperature by default;
paraphrasing is configured for more natural phrasing.

### MCP backend

The repository includes an MCP server:

```bash
python manhua_pipeline/adapters/mcp_server.py
```

The server exposes these tools:

`list_pending`, `get_translation_bundle`, `submit_translation`,
`get_paraphrase_bundle`, `submit_paraphrase`, and `get_glossary`.

An MCP client can launch the server with a configuration similar to:

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

The MCP server reads the configured series directory from `settings.json`, so
set it with `--set-output-dir` before starting the server.

### OpenAI-compatible backend

The `openai_compat` backend allows Stage 3 (translation) and Stage 4 (paraphrasing) to call OpenAI-compatible HTTP chat completion APIs (such as OpenRouter, DeepSeek, vLLM, or local proxies) with automatic provider fallback, parallel health check probing, streaming SSE support, and zero-CJK prompt hardening.

Configure the backends in `config.py`:

```python
TRANSLATOR_BACKEND = "openai_compat"
PARAPHRASE_BACKEND = "openai_compat"
```

Provider fallback chains and settings are configured in `settings.json` under `translate_fallback` and `paraphrase_fallback`:

```json
{
  "translate_fallback": {
    "enabled": true,
    "max_retries_per_model": 5,
    "batch_size": 20,
    "concurrency": 3,
    "timeout": 120,
    "providers": [
      {
        "name": "primary-provider",
        "base_url": "https://api.example.com/v1",
        "model": "model-name",
        "api_key": "YOUR_API_KEY",
        "batch_size": 20,
        "concurrency": 2,
        "stream": true
      }
    ],
    "disabled_providers": []
  },
  "paraphrase_fallback": {
    "enabled": true,
    "max_retries_per_model": 5,
    "batch_size": 20,
    "concurrency": 2,
    "timeout": 120,
    "providers": [ ... ]
  }
}
```

Key features:
- **Provider failover**: Automatically fails over to the next configured provider if an endpoint encounters fatal errors or exhausts retry limits.
- **Parallel health probing**: Checks provider connectivity in parallel before execution and demotes slow or unresponsive endpoints.
- **Hardened prompts**: Enforces strict JSON output mapping with zero leftover CJK characters and verbatim glossary application.
- **Streaming and connection pooling**: Supports streaming responses and connection pooling via `requests` with standard library `urllib` fallback.

## Optional touch-up editor

The touch-up editor is useful for corrections after automated rendering or for
pages that need more manual control:

```bash
python -m touchup.app
```

To open a particular image immediately:

```bash
python -m touchup.app --image "/path/to/page.png"
```

The editor supports page or chapter navigation, region selection, OCR,
translation assistance, source-text erasing, text placement, and saving the
baked page. It is separate from the core CLI pipeline and requires the optional
GUI dependencies.

## Quality assurance and review

The QA stage writes `stage6_qa/qa.json` with a status of:

- `SUCCESS` for up to two warnings
- `REVIEW` for three to ten warnings
- `FAILED` for more than ten warnings or critical failures

The report can include low OCR confidence, missing translations, missing
paraphrases, text overflow, possible split bubbles, glossary conflicts, and
missing artifacts. QA reports issues; it does not silently modify the artwork or
translation.

## Development

Install the test and lint tools available in the project environment, then run:

```bash
pytest
ruff check .
ruff format --check .
```

The stage implementation is under `manhua_pipeline/stages/`. Shared workspace, schema, logging, and backend code is under `manhua_pipeline/`. The top-level `pipeline.py` contains the CLI orchestration, while `pipeline_gui.py` contains the guided PyQt6 runner.

Each stage is loaded on demand from the stage registry in `pipeline.py`. A new translation or paraphrase backend can implement the existing backend request contract and be registered without changing the other stages.

## Troubleshooting

### Ollama fails immediately

Confirm that the Ollama server is running and that the configured model has been
downloaded:

```bash
ollama pull translategemma:4b
```

If Ollama is not available, switch the affected backend to `manual` or `mcp`.

### A stage is waiting for input

This is expected when using `manual` or `mcp`. Provide the response file or submit the pending bundle through the MCP client, then rerun the same stage or use **Resume** in the GUI.

### Detection or OCR model downloads fail

The first run may need access to the model provider. Check network access and disk space, or pre-cache the models before running offline. Once the models are cached, the pipeline can use the local cache.

### Text is missing or appears as boxes

Inspect `stage2_ocr/ocr.json`, the translation artifact, and the QA report. A low-confidence OCR region may need a manual correction. An untranslated CJK region usually indicates an incomplete backend response or a completion-gate failure.

### Rendered text overflows a bubble

Review the `overflow` entries in `qa.json`. Adjust `FONT_MAX_PT`, `FONT_MIN_PT`,
`LINE_SPACING`, or the relevant manual override, then rerun rendering and QA.

## License

This project is licensed under the GNU Affero General Public License v3.0
(AGPL-3.0). See [`LICENSE`](LICENSE) for the full license text.

Copyright (c) 2026 Ishan Dev Shakya.
