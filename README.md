# Manhua Translation Pipeline

A personal, modular document-processing pipeline that converts original Chinese
manhua chapters into natural US English. **JSON is the source of truth** and AI
is only one stage — this is a document-processing system, not "an AI translator".

## Core philosophy
- JSON is the source of truth; images are temporary inputs/outputs.
- One responsibility per stage.
- Every stage is independently rerunnable.
- Original files are never modified.
- The pipeline is LLM-agnostic (manual handoff in v0; MCP / Ollama planned).

## Pipeline
```
Import -> Detection -> OCR -> Translation -> Paraphrase -> Rendering -> QA
```

## Setup (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running
Each stage is its own command (maximally rerunnable):
```bat
python pipeline.py import   --workspace workspace
python pipeline.py detect   --workspace workspace
python pipeline.py ocr      --workspace workspace
python pipeline.py translate --workspace workspace
python pipeline.py paraphrase --workspace workspace
python pipeline.py render   --workspace workspace
python pipeline.py qa       --workspace workspace
```

Or run everything, with resume support:
```bat
python pipeline.py run-all --workspace workspace
python pipeline.py run-all --workspace workspace --from-stage ocr
```

## Workspace layout
```
workspace/
  pages/                # normalized page images
  stage1_detection/     # detection.json
  stage2_ocr/           # ocr.json
  stage3_translation/   # translation.json
  stage4_paraphrase/    # paraphrase.json
  stage5_render/        # rendered pages
  stage6_qa/            # qa.json
  glossary.json
  manifest.json
  logs/
```

## Translation / Paraphrase (v0 = manual JSON handoff)
In v0 the pipeline emits a JSON prompt; you paste it into your coding assistant
and paste the result back. Planned adapters: **MCP** (assistant reads/writes JSON
directly, no copy-paste) and **Ollama** (fully automated). All three read/write the
same JSON, so the architecture never changes.

## Note on input filenames
Source pages often look like `00000000_00010000.jpg`
(`{volume}_{chapter}{page}`). Import must **sort by the full numeric filename**
and remap to sequential page numbers (001, 002, ...). Do not assume clean names.

## Baseline (v0) scope
Import (folder of images) -> Detect bubbles+narration (YOLO) -> OCR horizontal
(PaddleOCR) -> manual translate/paraphrase via JSON -> Render (mask-fill + font)
-> QA. Deferred: name_label/scene_text detection, vertical-text OCR, OCR
correction, MCP/Ollama automation, inpainting.
