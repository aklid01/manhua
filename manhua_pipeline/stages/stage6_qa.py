# Stage 6: Quality assurance validation, warning aggregation, and report generation.

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from manhua_pipeline.io.workspace import load_manifest, save_manifest
from manhua_pipeline.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

_STAGE_INDEX = 6
_TOTAL_STAGES = 7
_STAGE_NAME = "QA"


# Load stage JSON artifact defensively, recording critical warning if missing
def _load_artifact(path: Path, name: str, warnings_list: list) -> tuple[dict, bool]:
    if not path.exists():
        warnings_list.append(
            {
                "region_id": None,
                "page_number": None,
                "category": "missing_artifact",
                "severity": "critical",
                "message": f"Expected artifact {name} is missing.",
            }
        )
        return {}, True
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data, False
    except Exception as exc:
        warnings_list.append(
            {
                "region_id": None,
                "page_number": None,
                "category": "missing_artifact",
                "severity": "critical",
                "message": f"Failed to load artifact {name}: {exc}",
            }
        )
        return {}, True


# Check for page rendering failures and record critical warnings
def _check_rendering_failures(
    manifest: dict, render_data: dict, warnings_list: list
) -> bool:
    render_pages = (
        render_data.get("pages")
        or render_data.get("outputs")
        or render_data.get("rendered_pages")
        or []
    )
    rendered_page_numbers = {
        p["page_number"] for p in render_pages if "page_number" in p
    }

    if not rendered_page_numbers:
        rendered_page_numbers = {
            r["page_number"]
            for r in render_data.get("results", [])
            if r.get("page_number") is not None and r.get("rendered")
        }

    has_critical = False
    for page in manifest.get("pages", []):
        if not page.get("skip"):
            p_num = page["page_number"]
            if p_num not in rendered_page_numbers:
                warnings_list.append(
                    {
                        "region_id": None,
                        "page_number": p_num,
                        "category": "rendering_failure",
                        "severity": "critical",
                        "message": f"Page {p_num} failed to render (missing from render.json pages list).",
                    }
                )
                has_critical = True
    return has_critical


# Analyze single region across all pipeline artifacts for QA checks
def _analyze_single_region(
    rid, ocr_map, tr_map, para_map, render_map
) -> tuple[list[dict], list[str]]:
    warnings = []
    attention_reasons = []

    ocr_r = ocr_map.get(rid, {})
    tr_r = tr_map.get(rid, {})
    para_r = para_map.get(rid, {})
    render_r = render_map.get(rid, {})

    page_num = (
        ocr_r.get("page_number")
        or tr_r.get("page_number")
        or para_r.get("page_number")
        or render_r.get("page_number")
    )

    ocr_note = (ocr_r.get("note") or "").lower()
    benign_no_text = not ocr_r.get("has_usable_text") and (
        ocr_r.get("watermark_filtered")
        or "watermark" in ocr_note
        or "split" in ocr_note
        or ocr_r.get("edge_touching")
    )
    low_sev = "info" if benign_no_text else "warning"

    if rid in ocr_map and (
        ocr_r.get("needs_correction") or not ocr_r.get("has_usable_text")
    ):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "low_ocr_confidence",
                "severity": low_sev,
                "message": f"Low OCR confidence ({ocr_r.get('ocr_confidence', 0.0):.2f}) or no usable text.",
            }
        )

    if ocr_r.get("ocr_review"):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "ocr_review",
                "severity": "info",
                "message": (
                    "Dialogue bubble OCR'd to "
                    f"{ocr_r.get('ocr_review_reason') or 'no words'}; "
                    "queued for manual OCR correction (see ocr_review.json)."
                ),
            }
        )

    if ocr_r.get("has_usable_text") and (
        rid not in tr_map or not tr_r.get("translated")
    ):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "missing_translation",
                "severity": "warning",
                "message": "Usable text region was not translated.",
            }
        )
        attention_reasons.append("missing_translation")

    if tr_r.get("translated") and (
        rid not in para_map or not para_r.get("paraphrased")
    ):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "missing_paraphrase",
                "severity": "info",
                "message": "Translated text region was not paraphrased (literal fallback used).",
            }
        )

    if render_r.get("overflow"):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "overflow",
                "severity": "warning",
                "message": f"Typography overflow at minimum font size ({render_r.get('font_size_pt')}pt).",
            }
        )

    if render_r.get("action") == "left_original_no_text":
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "left_original_no_text",
                "severity": low_sev,
                "message": "Kept original Chinese artwork because no usable text was found.",
            }
        )
        if not benign_no_text:
            attention_reasons.append("left_original_no_text")

    if ocr_r.get("edge_touching") and not ocr_r.get("has_usable_text"):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "edge_touching_split",
                "severity": "warning",
                "message": f"Possible split bubble touching page edge ({ocr_r.get('edge', 'none')}) with no usable text.",
            }
        )
        attention_reasons.append("edge_touching_split")

    if tr_r.get("glossary_conflict") or para_r.get("glossary_conflict"):
        warnings.append(
            {
                "region_id": rid,
                "page_number": page_num,
                "category": "glossary_conflict",
                "severity": "warning",
                "message": "Locked glossary term was not honored.",
            }
        )

    return warnings, attention_reasons


# Iterate over regions and compile QA warnings and needs_attention lists
def _process_region_analysis(
    sorted_region_ids: list[str],
    ocr_map: dict,
    tr_map: dict,
    para_map: dict,
    render_map: dict,
) -> tuple[list[dict], list[dict], dict]:
    warnings_list = []
    needs_attention = []
    category_counts = {
        "low_ocr_confidence": 0,
        "ocr_review": 0,
        "missing_translation": 0,
        "missing_paraphrase": 0,
        "overflow": 0,
        "left_original_no_text": 0,
        "edge_touching_split": 0,
        "glossary_conflict": 0,
        "rendering_failure": 0,
        "missing_artifact": 0,
    }
    for rid in sorted_region_ids:
        r_warnings, r_reasons = _analyze_single_region(
            rid, ocr_map, tr_map, para_map, render_map
        )
        warnings_list.extend(r_warnings)
        for w in r_warnings:
            category_counts[w["category"]] += 1

        if r_reasons:
            ocr_r = ocr_map.get(rid, {})
            needs_attention.append(
                {
                    "region_id": rid,
                    "page_number": ocr_r.get("page_number") or 0,
                    "original_text": ocr_r.get("original_text") or "",
                    "reason": ", ".join(r_reasons),
                }
            )
    return warnings_list, needs_attention, category_counts


# Compute chapter QA status verdict (SUCCESS, REVIEW, or FAILED)
def _compute_status(has_critical: bool, total_warnings: int, config) -> str:
    if has_critical:
        return "FAILED"
    if total_warnings <= getattr(config, "QA_SUCCESS_MAX", 2):
        return "SUCCESS"
    if total_warnings <= getattr(config, "QA_REVIEW_MAX", 10):
        return "REVIEW"
    return "FAILED"


# Write overrides.json stub with unhandled regions if not already present
def _write_overrides(ws: Path, config, needs_attention: list) -> None:
    overrides_path = ws / getattr(config, "OVERRIDES_NAME", "overrides.json")
    if not overrides_path.exists():
        stub = {
            "_comment": "Fill any region_id with the correct English. Non-empty values override the pipeline for that region. Then re-run: translate -> paraphrase -> render -> qa."
        }
        for item in needs_attention:
            stub[item["region_id"]] = ""
        try:
            with overrides_path.open("w", encoding="utf-8") as fh:
                json.dump(stub, fh, ensure_ascii=False, indent=2)
            logger.info(
                "[%d/%d %s] Wrote overrides.json stub (%d regions need attention)",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                len(needs_attention),
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to write overrides.json stub: %s", _STAGE_NAME, exc
            )
    else:
        logger.info(
            "[%d/%d %s] Existing overrides.json preserved",
            _STAGE_INDEX,
            _TOTAL_STAGES,
            _STAGE_NAME,
        )


# Execute Stage 6 quality assurance verification and write qa.json report
def run_qa(workspace: str, config) -> Path:
    t0 = time.monotonic()
    ws = Path(workspace)
    logger.info(
        "[%d/%d %s] Series: %s | Chapter: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        ws.parent.as_posix(),
        ws.name,
    )
    log_stage(logger, _STAGE_INDEX, _TOTAL_STAGES, _STAGE_NAME, "starting")
    from manhua_pipeline.events import QA_ISSUE_DETECTED, STAGE_COMPLETED, STAGE_STARTED
    from manhua_pipeline.io.event_emitter import emit_stage_event

    emit_stage_event(ws, config, _STAGE_NAME.lower(), STAGE_STARTED, logger=logger)

    manifest = load_manifest(workspace, config)
    if not manifest:
        raise ValueError("Manifest not found. Run import first.")

    logger.info(
        "[%d/%d %s] Reading artifacts: ocr, translation, paraphrase, render",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
    )

    ocr_path = ws / config.STAGE_FOLDERS["ocr"] / "ocr.json"
    tr_path = ws / config.STAGE_FOLDERS["translation"] / "translation.json"
    para_path = ws / config.STAGE_FOLDERS["paraphrase"] / "paraphrase.json"
    render_path = ws / config.STAGE_FOLDERS["render"] / "render.json"

    warnings_list = []

    ocr_data, ocr_crit = _load_artifact(ocr_path, "ocr.json", warnings_list)
    tr_data, tr_crit = _load_artifact(tr_path, "translation.json", warnings_list)
    para_data, para_crit = _load_artifact(para_path, "paraphrase.json", warnings_list)
    render_data, render_crit = _load_artifact(render_path, "render.json", warnings_list)

    has_critical = ocr_crit or tr_crit or para_crit or render_crit

    ocr_results = ocr_data.get("results", [])
    tr_results = tr_data.get("results", [])
    para_results = para_data.get("results", [])
    render_results = render_data.get("results", [])

    ocr_map = {r["region_id"]: r for r in ocr_results if "region_id" in r}
    tr_map = {r["region_id"]: r for r in tr_results if "region_id" in r}
    para_map = {r["region_id"]: r for r in para_results if "region_id" in r}
    render_map = {r["region_id"]: r for r in render_results if "region_id" in r}

    all_region_ids = (
        ocr_map.keys() | tr_map.keys() | para_map.keys() | render_map.keys()
    )

    if not render_crit:
        page_crit = _check_rendering_failures(manifest, render_data, warnings_list)
        has_critical = has_critical or page_crit

    sorted_region_ids = sorted(all_region_ids)
    r_warnings, needs_attention, category_counts = _process_region_analysis(
        sorted_region_ids, ocr_map, tr_map, para_map, render_map
    )
    warnings_list.extend(r_warnings)

    for w in warnings_list:
        if w["category"] in ("rendering_failure", "missing_artifact"):
            category_counts[w["category"]] += 1

    for cat, count in category_counts.items():
        if count > 0 or cat in (
            "low_ocr_confidence",
            "overflow",
            "left_original_no_text",
            "glossary_conflict",
        ):
            logger.info(
                "[%d/%d %s] %s: %d",
                _STAGE_INDEX,
                _TOTAL_STAGES,
                _STAGE_NAME,
                cat,
                count,
            )

    total_warnings = sum(1 for w in warnings_list if w["severity"] == "warning")
    status = _compute_status(has_critical, total_warnings, config)

    _write_overrides(ws, config, needs_attention)

    logger.info(
        "[%d/%d %s] Total warnings: %d -> status: %s",
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        total_warnings,
        status,
    )

    total_pages = len(manifest.get("pages", []))
    summary_drawn = summary_left = summary_overflow = 0
    for r in render_results:
        if r.get("rendered"):
            summary_drawn += 1
        else:
            summary_left += 1
        if r.get("overflow"):
            summary_overflow += 1
    ocr_review_regions = [
        r.get("region_id") for r in ocr_results if r.get("ocr_review")
    ]

    qa_dir = ws / config.STAGE_FOLDERS["qa"]
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_json_path = qa_dir / "qa.json"

    now = datetime.now(timezone.utc).isoformat()
    qa_report = {
        "chapter_id": manifest.get("chapter_id", "unknown"),
        "stage": "qa",
        "generated_at": now,
        "status": status,
        "total_warnings": total_warnings,
        "thresholds": {
            "success_max": getattr(config, "QA_SUCCESS_MAX", 2),
            "review_max": getattr(config, "QA_REVIEW_MAX", 10),
        },
        "checks": category_counts,
        "warnings": warnings_list,
        "needs_attention": needs_attention,
        "summary": {
            "pages": total_pages,
            "regions": len(sorted_region_ids),
            "drawn": summary_drawn,
            "left": summary_left,
            "overflow": summary_overflow,
            "ocr_review_count": len(ocr_review_regions),
            "ocr_review_regions": ocr_review_regions,
            "ocr_review_file": (
                f"{config.STAGE_FOLDERS['ocr']}/ocr_review.json"
                if ocr_review_regions
                else None
            ),
        },
    }

    with qa_json_path.open("w", encoding="utf-8") as fh:
        json.dump(qa_report, fh, ensure_ascii=False, indent=2)

    manifest["status"] = status
    manifest["warning_count"] = total_warnings
    completed = manifest.get("completed_stages", [])
    if "qa" not in completed:
        completed.append("qa")
    manifest["completed_stages"] = completed
    manifest["current_stage"] = "complete"
    manifest["updated_at"] = now
    save_manifest(workspace, config, manifest)

    from manhua_pipeline.events import PipelineEvent
    from manhua_pipeline.io.event_emitter import get_event_emitter

    emitter = get_event_emitter(ws, config, logger=logger)
    for w in warnings_list:
        emitter.emit(
            PipelineEvent(
                event=QA_ISSUE_DETECTED,
                run_id=getattr(config, "_run_id", "unknown"),
                stage=_STAGE_NAME.lower(),
                chapter=ws.name,
                region_id=w.get("region_id"),
                page=w.get("page_number"),
                reason=w.get("category"),
                metadata={"message": w.get("message"), "severity": w.get("severity")},
            )
        )

    elapsed = time.monotonic() - t0
    log_stage(
        logger,
        _STAGE_INDEX,
        _TOTAL_STAGES,
        _STAGE_NAME,
        f"done: {total_pages} pages, {len(sorted_region_ids)} regions, "
        f"{summary_drawn} drawn, {summary_left} left, {summary_overflow} overflow -> qa.json (elapsed {elapsed:.1f}s)",
    )
    emit_stage_event(
        ws,
        config,
        _STAGE_NAME.lower(),
        STAGE_COMPLETED,
        logger=logger,
        metadata={"total_warnings": total_warnings, "qa_status": status},
    )

    return qa_json_path
