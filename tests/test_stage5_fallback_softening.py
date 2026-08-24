from PIL import Image, ImageDraw

import config
from manhua_pipeline.stages.stage5_render import inpainting
from manhua_pipeline.stages.stage5_render.types import Box


def test_solid_rect_mask_generates_smooth_oval():
    bubble_box = Box(0, 0, 200, 100)
    erase_box = Box(30, 20, 140, 60)
    text_box = Box(40, 25, 120, 50)

    mask = inpainting._solid_rect_mask(
        bubble_box,
        erase_box,
        shape="oval",
        text_box=text_box,
        feather_px=0,
    )

    # Core text center should be solid white (255)
    assert mask.getpixel((100, 50)) == 255
    # Text box corners should be solid white (255)
    assert mask.getpixel((40, 25)) == 255
    assert mask.getpixel((159, 74)) == 255
    # Extreme corners of the bubble outside the oval should be black (0)
    assert mask.getpixel((5, 5)) == 0
    assert mask.getpixel((195, 95)) == 0


def test_solid_rect_mask_respects_full_bubble_oval():
    bubble_box = Box(0, 0, 300, 200)
    erase_box = Box(0, 0, 300, 200)

    mask = inpainting._solid_rect_mask(
        bubble_box,
        erase_box,
        shape="oval",
        feather_px=0,
    )

    # Center is solid white
    assert mask.getpixel((150, 100)) == 255
    # Bubble extreme corners are unmasked
    assert mask.getpixel((2, 2)) == 0
    assert mask.getpixel((298, 198)) == 0


def test_fallback_erase_ensures_100_percent_text_core_coverage():
    page = Image.new("RGB", (300, 200), (50, 50, 50))
    draw = ImageDraw.Draw(page)

    bubble_box = Box(20, 20, 260, 160)
    text_box = Box(60, 50, 180, 100)
    erase_box = Box(50, 40, 200, 120)

    # Draw dark Chinese text inside text box
    draw.rectangle((80, 60, 220, 130), fill=(0, 0, 0))

    # Trigger fallback with empty bubble mask
    empty_bubble_mask = Image.new("L", (260, 160), 0)

    inpainting.erase_original_text(
        page_img=page,
        draw=draw,
        text_box=text_box,
        bubble_box=bubble_box,
        erase_box=erase_box,
        bubble_mask=empty_bubble_mask,
        config=config,
        fill_color=(255, 255, 255),
    )

    # The entire text box area should be completely erased (no black pixels remaining)
    text_crop = page.crop(text_box.as_xyxy())
    for pixel in text_crop.getdata():
        assert pixel == (255, 255, 255)


def test_fallback_erase_feathering_applied():
    bubble_box = Box(0, 0, 200, 100)
    erase_box = Box(30, 20, 140, 60)
    text_box = Box(40, 25, 120, 50)

    mask = inpainting._solid_rect_mask(
        bubble_box,
        erase_box,
        shape="oval",
        text_box=text_box,
        feather_px=4,
    )

    # Center is 255
    assert mask.getpixel((100, 50)) == 255
    # Pixels near the boundary should have smooth intermediate alpha (0 < alpha < 255)
    # Check that there are smooth transition pixels
    hist = mask.histogram()
    intermediate_pixels = sum(hist[1:254])
    assert intermediate_pixels > 0
