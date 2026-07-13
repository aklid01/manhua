import json

import config


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


def test_qa_render_page_fallback_no_false_failure(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    # render.json has NO top-level pages/outputs list — only per-region results with page_number+rendered
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
    assert qa["status"] == "SUCCESS"  # NOT falsely FAILED
    assert qa["checks"]["rendering_failure"] == 0


def test_qa_midconf_not_in_overrides(tmp_path):
    from manhua_pipeline.stages.stage6_qa import run_qa

    ws = tmp_path / "workspace"
    # needs_correction True (0.3-0.7 band) BUT it translated, paraphrased, and rendered fine
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
    # low_ocr_confidence still counted as a warning...
    assert qa["checks"]["low_ocr_confidence"] == 1
    # ...but the region is NOT in needs_attention (it rendered fine)
    assert all(item["region_id"] != "P001_R001" for item in qa["needs_attention"])
    # overrides stub therefore not created for it (or created without this id)
    ov = ws / "overrides.json"
    if ov.exists():
        stub = json.loads(ov.read_text(encoding="utf-8"))
        assert "P001_R001" not in stub


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
    assert qa["checks"]["rendering_failure"] == 0  # not double-counted


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
    assert qa["checks"].get("missing_paraphrase", 0) == 1  # still visible in checks
    assert qa["status"] == "SUCCESS"  # but not counted toward verdict
    assert qa["total_warnings"] == 0


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
