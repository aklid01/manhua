# Unit tests for split-bubble cross-page stitching geometry and guards.

import config
from manhua_pipeline.stages import stage1_detection as s1


# Helper to construct test detection box dictionaries
def _box(x, y, w, h, typ=None):
    return {
        "type": typ or config.TYPE_SPEECH,
        "read_region": {"x": x, "y": y, "w": w, "h": h},
    }


# Helper to construct mock page descriptor lists
def _pages(*nums, h=600):
    return [
        {"page_number": n, "height": h, "skip": False, "filename": f"{n}.png"}
        for n in nums
    ]


# Verify horizontal overlap fraction calculation for overlapping and disjoint boxes
def test_x_overlap_positive_and_zero():
    assert (
        s1._x_overlap_frac(
            {"x": 10, "y": 0, "w": 100, "h": 20}, {"x": 30, "y": 0, "w": 100, "h": 20}
        )
        > 0.5
    )
    assert (
        s1._x_overlap_frac(
            {"x": 0, "y": 0, "w": 50, "h": 20}, {"x": 200, "y": 0, "w": 50, "h": 20}
        )
        == 0.0
    )


# Verify top and bottom edge proximity detection against page borders
def test_edge_touch():
    assert s1._box_touches_bottom({"x": 0, "y": 585, "w": 50, "h": 20}, 600, 6)
    assert s1._box_touches_top({"x": 0, "y": 2, "w": 50, "h": 20}, 6)
    assert not s1._box_touches_top({"x": 0, "y": 50, "w": 50, "h": 20}, 6)


# Verify adjacent split bubbles touching page seams are identified as stitch pairs
def test_find_pair_positive():
    det = {1: [_box(20, 585, 100, 15)], 2: [_box(25, 0, 100, 15)]}
    pairs = s1._find_split_chains(det, _pages(1, 2), config)
    assert len(pairs) == 1 and pairs[0][0] == 1 and pairs[0][1] == 2


# Verify disjoint horizontal positions prevent stitching across pages
def test_find_pair_no_x_overlap():
    det = {1: [_box(0, 585, 40, 15)], 2: [_box(300, 0, 40, 15)]}
    assert s1._find_split_chains(det, _pages(1, 2), config) == []


# Verify bubbles not flush to page boundary are not stitched
def test_find_pair_not_edge_flush():
    det = {1: [_box(20, 400, 100, 15)], 2: [_box(20, 0, 100, 15)]}
    assert s1._find_split_chains(det, _pages(1, 2), config) == []


# Verify stitching chains are processed strictly pairwise
def test_chain_guard_pairwise_only():
    det = {
        1: [_box(10, 585, 100, 15)],
        2: [_box(10, 0, 100, 15), _box(10, 585, 100, 15)],
        3: [_box(10, 0, 100, 15)],
    }
    pairs = s1._find_split_chains(det, _pages(1, 2, 3), config)
    assert len(pairs) == 1 and (pairs[0][0], pairs[0][1]) == (1, 2)


# Verify non-speech regions (e.g. narration) are excluded from bubble stitching
def test_non_speech_boxes_ignored():
    det = {
        1: [_box(20, 585, 100, 15, typ=getattr(config, "TYPE_NARRATION", "narration"))],
        2: [_box(20, 0, 100, 15, typ=getattr(config, "TYPE_NARRATION", "narration"))],
    }
    assert s1._find_split_chains(det, _pages(1, 2), config) == []


# Verify stitching geometry uses parent bubble bounds when available over text box
def test_stitch_box_prefers_parent_bubble():
    box_with_pb = {
        "read_region": {"x": 30, "y": 500, "w": 40, "h": 20},
        "parent_bubble": {"x": 20, "y": 480, "w": 60, "h": 120},
    }
    box_no_pb = {
        "read_region": {"x": 30, "y": 500, "w": 40, "h": 20},
    }

    assert s1._stitch_box(box_with_pb) == {"x": 20, "y": 480, "w": 60, "h": 120}
    assert s1._stitch_box(box_no_pb) == {"x": 30, "y": 500, "w": 40, "h": 20}


# Verify parent bubble edge touch detects split pairs even if inner text box is inset
def test_find_pair_rtdetr_parent_bubble_edge_touch():
    box_a = {
        "type": config.TYPE_SPEECH,
        "read_region": {"x": 30, "y": 500, "w": 40, "h": 50},
        "parent_bubble": {"x": 20, "y": 500, "w": 80, "h": 100},
    }
    box_b = {
        "type": config.TYPE_SPEECH,
        "read_region": {"x": 30, "y": 50, "w": 40, "h": 50},
        "parent_bubble": {"x": 20, "y": 0, "w": 80, "h": 100},
    }
    det = {1: [box_a], 2: [box_b]}
    pairs = s1._find_split_chains(det, _pages(1, 2, h=600), config)
    assert len(pairs) == 1 and pairs[0][0] == 1 and pairs[0][1] == 2
