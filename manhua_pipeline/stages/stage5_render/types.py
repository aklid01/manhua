from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class RenderPlan:
    """Everything Pass 2 needs to DRAW a region, computed during Pass 1."""

    region_id: str
    page_number: int
    rendered: bool
    action: str

    placements: list[dict] = field(default_factory=list)
    font: object | None = None
    text_fill: tuple[int, int, int] = (0, 0, 0)

    font_size_pt: int | None = None
    lines: int = 0
    overflow: bool = False
    register: str = "neutral"
    style_hint: str = "round"
    glossary_conflict: bool = False
    effective_max_pt: int | None = None
    page_width: int = 0
    page_height: int = 0
    readable_min_pt: int | None = None
    abs_min_pt: int | None = None
    text_box: dict = field(default_factory=dict)
    layout_box: dict = field(default_factory=dict)
    erase_box: dict = field(default_factory=dict)
    bubble_box: dict = field(default_factory=dict)
    erase_strategy: str = ""
    layout_expanded: bool = False
    layout_growth_factor: float = 1.0
    text_bg: tuple[int, int, int] = (255, 255, 255)

    bubble_mask_bbox: list | None = None
    bubble_mask_pixels: int = 0
    bubble_mask_area: int = 0
    bubble_mask_coverage: float = 0.0

    debug_paths: dict = field(default_factory=dict)

    emit_too_small: bool = False
    emit_low_overlap: bool = False
    low_overlap_retention: float = 0.0
