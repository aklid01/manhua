# Unit tests for JSON response parsing, auto-repair, and partial recovery.

import pytest

from manhua_pipeline.io.parse_result import (
    _harvest_partial_regions,
    _repair_json_string,
    parse_json_result,
)


# Verify clean JSON object parses with status 'success'
def test_parse_json_clean_success():
    raw = '{"P001_R001": "Hello world", "P001_R002": "Second line"}'
    res = parse_json_result(raw)
    assert res.status == "success"
    assert res.value == {"P001_R001": "Hello world", "P001_R002": "Second line"}


# Verify markdown fences are stripped correctly
def test_parse_json_markdown_fences():
    raw = '```json\n{"P001_R001": "Hello world"}\n```'
    res = parse_json_result(raw)
    assert res.status == "success"
    assert res.value == {"P001_R001": "Hello world"}


# Verify trailing commas before closing braces are auto-repaired
def test_parse_json_trailing_comma_repair():
    raw = '{"P001_R001": "Hello world", "P001_R002": "Second line", }'
    res = parse_json_result(raw)
    assert res.status == "repaired"
    assert res.value == {"P001_R001": "Hello world", "P001_R002": "Second line"}


# Verify unquoted region ID keys are auto-repaired
def test_parse_json_unquoted_keys_repair():
    raw = '{P001_R001: "Hello world", P001_R002: "Second line"}'
    res = parse_json_result(raw)
    assert res.status == "repaired"
    assert res.value == {"P001_R001": "Hello world", "P001_R002": "Second line"}


# Verify missing closing brace is auto-repaired
def test_parse_json_missing_closing_brace():
    raw = '{"P001_R001": "Hello world", "P001_R002": "Second line"'
    res = parse_json_result(raw)
    assert res.status == "repaired"
    assert res.value == {"P001_R001": "Hello world", "P001_R002": "Second line"}


# Verify partial key-values are harvested when syntax breaks mid-document
def test_parse_json_partial_harvesting():
    # Second item has broken syntax (missing quotes or corrupted), but first and third are valid
    raw = '{"P001_R001": "First line", "P001_R002": corrupted syntax without quotes, "P001_R003": "Third line"}'
    res = parse_json_result(raw)
    assert res.status == "partial_recovered"
    assert res.value == {"P001_R001": "First line", "P001_R003": "Third line"}


# Verify blank responses return status 'blank'
def test_parse_json_blank():
    assert parse_json_result("").status == "blank"
    assert parse_json_result("   \n\t  ").status == "blank"


# Verify non-object (e.g. JSON list or int) returns status 'non_object'
def test_parse_json_non_object():
    res = parse_json_result('["P001_R001", "P001_R002"]')
    assert res.status == "non_object"
    assert res.value == {}
