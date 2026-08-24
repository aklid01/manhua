from PIL import Image, ImageDraw
import config
from manhua_pipeline.stages.stage5_render import planner
from manhua_pipeline.stages.stage5_render.types import Box


def test_uncontained_cjk_sfx_on_art_preserved():
    # Flaming / dark artwork with NO speech bubble
    page = Image.new("RGB", (300, 300), (220, 80, 20))
    draw = ImageDraw.Draw(page)

    region = {
        "region_id": "P001_R001",
        "page_number": 1,
        "type": "speech_bubble",
        "render": True,
        "has_usable_text": True,
        "original_text": "轰",
        "final_text": "BOOM!",
        "bbox": {"x": 50, "y": 50, "w": 200, "h": 200},
        "read_region": {"x": 50, "y": 50, "w": 200, "h": 200},
        "parent_bubble": {"x": 50, "y": 50, "w": 200, "h": 200},
        "bubble_mask": None,
    }

    plan = planner._plan_and_erase_region(
        region=region,
        page_img=page,
        draw=draw,
        page_num=1,
        config=config,
    )

    # Should preserve raw art untouched
    assert plan.rendered is False
    assert plan.action == "left_original_sfx_art"
    # Page background color must be 100% untouched
    assert page.getpixel((100, 100)) == (220, 80, 20)


def test_in_bubble_sfx_is_rendered():
    # Valid white speech bubble on canvas
    page = Image.new("RGB", (300, 300), (50, 50, 50))
    draw = ImageDraw.Draw(page)
    draw.ellipse((50, 50, 250, 250), fill=(255, 255, 255))

    bubble_box = Box(50, 50, 200, 200)

    # A valid bubble mask
    bubble_mask = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(bubble_mask).ellipse((0, 0, 199, 199), fill=255)

    region = {
        "region_id": "P001_R002",
        "page_number": 1,
        "type": "speech_bubble",
        "render": True,
        "has_usable_text": True,
        "original_text": "轰",
        "final_text": "BOOM!",
        "bbox": {"x": 80, "y": 80, "w": 140, "h": 140},
        "read_region": {"x": 80, "y": 80, "w": 140, "h": 140},
        "parent_bubble": {"x": 50, "y": 50, "w": 200, "h": 200},
        "bubble_mask": {
            "valid": True,
            "confidence": 0.95,
        },
    }

    plan = planner._plan_and_erase_region(
        region=region,
        page_img=page,
        draw=draw,
        page_num=1,
        config=config,
    )

    # In-bubble exclamation should be rendered
    assert plan.rendered is True
    assert plan.action == "drew"
    assert plan.placements is not None


def test_non_sfx_dialogue_without_bubble_uses_fallback():
    # Regular spoken dialogue where detector missed the bubble boundary (dark background art)
    page = Image.new("RGB", (300, 300), (30, 30, 30))
    draw = ImageDraw.Draw(page)
    # Tiny white spot under text (less than 40% coverage)
    draw.rectangle((120, 120, 150, 150), fill=(255, 255, 255))

    region = {
        "region_id": "P001_R003",
        "page_number": 1,
        "type": "speech_bubble",
        "render": True,
        "has_usable_text": True,
        "original_text": "快走！我们必须马上离开这里！",
        "final_text": "Hurry! We must leave this place right now!",
        "bbox": {"x": 50, "y": 50, "w": 200, "h": 200},
        "read_region": {"x": 50, "y": 50, "w": 200, "h": 200},
        "parent_bubble": {"x": 50, "y": 50, "w": 200, "h": 200},
        "bubble_mask": None,
    }

    plan = planner._plan_and_erase_region(
        region=region,
        page_img=page,
        draw=draw,
        page_num=1,
        config=config,
    )

    # Spoken dialogue must be rendered using smooth fallback
    assert plan.rendered is True
    assert plan.action == "drew"
    assert plan.bubble_mask_source == "text_rect_fallback"
