# Unit tests for Stage 6 Quality Assurance validation, warning categorization, and review flags.

import json

import config


# Helper to configure mock workspace with OCR, translation, paraphrase, and render stage outputs
def _setup(ws, ocr=None, tr=None, para=None, render=None, manifest_extra=None):
    ws.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chapter_id": "t",
        "total_pages": 1,
        "pages": [{"page_number": 1, "filename": "001.png", "skip": False}],
        "current_stage": "qa",
        "completed_stages": [
            "import",
            "detect",
            "ocr",
            "translate",
            "paraphrase",
            "render",
        ],
        "warning_count": 0,
        "status": "in_progress",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (ws / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _w(sub, name, data):
        (ws / sub).mkdir(parents=True, exist_ok=True)
        (ws / sub / name).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    _w("stage2_ocr", "ocr.json", ocr or {"results": []})
    _w("stage3_translation", "translation.json", tr or {"results": []})
    _w("stage4_paraphrase", "paraphrase.json", para or {"results": []})
    _w("stage5_render", "render.json", render or {"results": [], "outputs": []})


# Verify flawless chapter passes QA with SUCCESS verdict and zero warnings
def test_qa_clean_success(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ],
            "outputs": [{"page_number": 1, "output_file": "001_render.png"}],
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "SUCCESS"
    assert qa["total_warnings"] == 0
    m = json.loads((ws / "manifest.json").read_text())
    assert m["status"] == "SUCCESS"


# Verify chapters with multiple unrendered regions trigger REVIEW status and overrides template
def test_qa_review_and_overrides_stub(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    ocr_results = []
    for i in range(1, 6):
        ocr_results.append(
            {
                "region_id": f"P001_R00{i}",
                "page_number": 1,
                "has_usable_text": False,
                "needs_correction": True,
                "edge_touching": False,
                "original_text": "",
            }
        )
    _setup(
        ws,
        ocr={"results": ocr_results},
        render={
            "results": [
                {
                    "region_id": f"P001_R00{i}",
                    "rendered": False,
                    "action": "left_original_no_text",
                    "overflow": False,
                }
                for i in range(1, 6)
            ],
            "outputs": [{"page_number": 1, "output_file": "001_render.png"}],
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "REVIEW"
    assert len(qa["needs_attention"]) >= 5
    stub = json.loads((ws / "overrides.json").read_text(encoding="utf-8"))
    assert "_comment" in stub
    assert "P001_R001" in stub
    assert stub["P001_R001"] == ""


# Verify existing manual overrides are preserved when QA runs
def test_qa_preserves_existing_overrides(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": False,
                    "needs_correction": True,
                    "edge_touching": False,
                    "original_text": "",
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "rendered": False,
                    "action": "left_original_no_text",
                    "overflow": False,
                }
            ],
            "outputs": [],
        },
    )
    (ws / "overrides.json").write_text(
        json.dumps({"P001_R001": "My manual text"}), encoding="utf-8"
    )
    run_qa(str(ws), config)
    stub = json.loads((ws / "overrides.json").read_text(encoding="utf-8"))
    assert stub["P001_R001"] == "My manual text"


# Verify missing stage artifact triggers FAILED status
def test_qa_missing_artifact_failed(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={"results": []},
        tr={"results": []},
        para={"results": []},
        render={"results": [], "outputs": []},
    )
    (ws / "stage5_render" / "render.json").unlink()
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "FAILED"
    assert any(w["category"] == "missing_artifact" for w in qa["warnings"])


# Verify QA checks dictionary accurately tallies individual warning categories
def test_qa_category_counts(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": True,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": True,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "rendered": True,
                    "action": "drew",
                    "overflow": True,
                }
            ],
            "outputs": [{"page_number": 1, "output_file": "001_render.png"}],
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["checks"].get("overflow", 0) == 1
    assert qa["checks"].get("glossary_conflict", 0) == 1


# Verify QA correctly evaluates rendered pages without false failures when page list is absent
def test_qa_render_page_fallback_no_false_failure(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    render = {
        "results": [
            {
                "region_id": "P001_R001",
                "page_number": 1,
                "rendered": True,
                "action": "drew",
                "overflow": False,
            }
        ]
    }
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render=render,
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "SUCCESS"
    assert qa["checks"]["rendering_failure"] == 0


# Verify successfully translated/rendered mid-confidence regions are not added to overrides stub
def test_qa_midconf_not_in_overrides(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": True,
                    "ocr_confidence": 0.65,
                    "edge_touching": False,
                    "original_text": "经理",
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ]
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["checks"]["low_ocr_confidence"] == 1
    assert all(item["region_id"] != "P001_R001" for item in qa["needs_attention"])
    ov = ws / "overrides.json"
    if ov.exists():
        stub = json.loads(ov.read_text(encoding="utf-8"))
        assert "P001_R001" not in stub


# Verify missing render artifact is reported as a single critical error without duplicate warnings
def test_qa_missing_render_single_critical(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={"results": []},
        para={"results": []},
        render={"results": []},
    )
    (ws / "stage5_render" / "render.json").unlink()
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "FAILED"
    assert qa["checks"]["missing_artifact"] >= 1
    assert qa["checks"]["rendering_failure"] == 0


# Verify missing paraphrase region is treated as informational and does not trigger failure
def test_qa_missing_paraphrase_is_info(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": False,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ]
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["checks"].get("missing_paraphrase", 0) == 1
    assert qa["status"] == "SUCCESS"
    assert qa["total_warnings"] == 0


# Verify clean QA run marks manifest current_stage as complete
def test_qa_marks_complete(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ]
        },
    )
    run_qa(str(ws), config)
    m = json.loads((ws / "manifest.json").read_text())
    assert m["current_stage"] == "complete"
    assert "qa" in m["completed_stages"]


# Verify flagged OCR review regions are listed as informational without failing verdict
def test_qa_ocr_review_informational(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P022_R002",
                    "page_number": 1,
                    "original_text": "？",
                    "has_usable_text": True,
                    "ocr_confidence": 0.95,
                    "needs_correction": False,
                    "edge_touching": False,
                    "ocr_review": True,
                    "ocr_review_reason": "punct_only",
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P022_R002",
                    "translated": True,
                    "translation_source": "punct_passthrough",
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P022_R002",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P022_R002",
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ],
            "outputs": [{"page_number": 1, "output_file": "001_render.png"}],
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))

    assert qa["status"] == "SUCCESS"
    assert qa["total_warnings"] == 0

    ocr_rev = [w for w in qa["warnings"] if w["category"] == "ocr_review"]
    assert len(ocr_rev) == 1
    assert ocr_rev[0]["severity"] == "info"
    assert ocr_rev[0]["region_id"] == "P022_R002"

    assert qa["summary"]["ocr_review_count"] == 1
    assert qa["summary"]["ocr_review_regions"] == ["P022_R002"]
    assert qa["summary"]["ocr_review_file"].endswith("ocr_review.json")


# Verify QA summary reflects zero OCR review count when no regions are flagged
def test_qa_no_ocr_review_when_none_flagged(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    _setup(
        ws,
        ocr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "page_number": 1,
                    "has_usable_text": True,
                    "needs_correction": False,
                    "edge_touching": False,
                    "ocr_review": False,
                }
            ]
        },
        tr={
            "results": [
                {
                    "region_id": "P001_R001",
                    "translated": True,
                    "glossary_conflict": False,
                }
            ]
        },
        para={
            "results": [
                {
                    "region_id": "P001_R001",
                    "paraphrased": True,
                    "glossary_conflict": False,
                }
            ]
        },
        render={
            "results": [
                {
                    "region_id": "P001_R001",
                    "rendered": True,
                    "action": "drew",
                    "overflow": False,
                }
            ],
            "outputs": [{"page_number": 1, "output_file": "001_render.png"}],
        },
    )
    run_qa(str(ws), config)
    qa = json.loads((ws / "stage6_qa" / "qa.json").read_text(encoding="utf-8"))
    assert qa["summary"]["ocr_review_count"] == 0
    assert qa["summary"]["ocr_review_file"] is None
    assert not [w for w in qa["warnings"] if w["category"] == "ocr_review"]
