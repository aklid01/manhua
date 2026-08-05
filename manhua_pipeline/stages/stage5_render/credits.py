import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from manhua_pipeline.io.settings import get_credits
from manhua_pipeline.logging_setup import get_logger

logger = get_logger(__name__)
_STAGE_NAME = "Render"


def _fit_font(draw, text, font_path, max_pt, max_w):
    pt = max_pt
    while pt >= 10:
        f = ImageFont.truetype(font_path, pt)
        if draw.textlength(text, font=f) <= max_w:
            return f
        pt -= 1
    return ImageFont.truetype(font_path, 10)


def _sample_bg(im, cx, cy, W, H, align):
    offs = (0.02, 0.10, 0.18) if align == "left" else (-0.12, -0.06, 0.06, 0.12)
    pts = []
    for dx in offs:
        x, y = int((cx + dx) * W), int(cy * H)
        if 0 <= x < W and 0 <= y < H:
            pts.append(im.getpixel((x, y)))
    if not pts:
        return (12, 14, 22)
    return tuple(sorted(c[i] for c in pts)[len(pts) // 2] for i in range(3))


def _render_credits_page(render_dir: Path, config, page_size) -> Path | None:
    try:
        credits = get_credits()
        templates = getattr(config, "CREDITS_TEMPLATES", {})
        if not templates:
            return None
        root = Path(__file__).resolve().parent.parent.parent.parent
        cdir = root / getattr(config, "CREDITS_DIR", "assets/credits")
        available = [(n, s) for n, s in templates.items() if (cdir / n).exists()]
        if not available:
            logger.warning("[%s] No credit templates found in %s.", _STAGE_NAME, cdir)
            return None

        name, slots = random.choice(available)
        im = Image.open(cdir / name).convert("RGB")
        if page_size and getattr(config, "CREDITS_MATCH_PAGE_SIZE", True):
            im = im.resize(page_size, Image.Resampling.LANCZOS)
        W, H = im.size
        draw = ImageDraw.Draw(im)
        font_path = str(
            root / getattr(config, "FONT_PATH", "assets/fonts/ComicNeue-Bold.ttf")
        )
        fill = getattr(config, "CREDITS_TEXT_FILL", (240, 236, 225))

        for field, cx, cy, align, max_pt, pw, ph in slots:
            val = credits.get(field, "")
            if not val:
                continue
            pwx, phx = int(pw * W), int(ph * H)
            x0 = int(cx * W) - (pwx // 2 if align == "center" else int(0.005 * W))
            y0 = int(cy * H) - phx // 2
            draw.rectangle(
                [x0, y0, x0 + pwx, y0 + phx], fill=_sample_bg(im, cx, cy, W, H, align)
            )
            f = _fit_font(draw, val, font_path, max_pt, pwx - 10)
            tw = draw.textlength(val, font=f)
            tx = int(cx * W) - tw / 2 if align == "center" else x0 + 6
            ty = int(cy * H) - f.size / 2
            draw.text((tx, ty), val, font=f, fill=fill)

        out = render_dir / "zzz_credits.png"
        im.save(out)
        logger.info("[%s] Credits page (%s) -> %s", _STAGE_NAME, name, out.name)
        return out
    except Exception as exc:
        logger.warning("[%s] Credits page skipped (%s).", _STAGE_NAME, exc)
        return None
