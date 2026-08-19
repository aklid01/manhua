import argparse
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from .cache import TouchupCache
    from .canvas import DraggableTextItem, TouchupCanvas
    from .inpainter import inpaint_text_regions
    from .ocr_helper import extract_text_and_boxes
    from .segmenter import SAMSegmenter
    from .translator import translate_text
except (ImportError, ValueError):
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from touchup.cache import TouchupCache
    from touchup.canvas import DraggableTextItem, TouchupCanvas
    from touchup.inpainter import inpaint_text_regions
    from touchup.ocr_helper import extract_text_and_boxes
    from touchup.segmenter import SAMSegmenter
    from touchup.translator import translate_text


# Convert PIL Image to QPixmap
def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    im = pil_image.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qimg = QImage(data, im.width, im.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class TouchupApp(QMainWindow):
    # Main Touch-Up Editor Window

    MIXED_STYLE_LABEL = "- Mixed -"
    _MIXED = object()

    def __init__(self, initial_image: str | None = None):
        super().__init__()
        self.setWindowTitle("Manhua Touch-Up Editor (SAM + OCR + Typesetter)")
        self.resize(1050, 680)

        self.current_image_path: Path | None = None
        self.chapter_pages: list[Path] = []
        self.current_page_idx: int = -1

        self.original_pil_img: Image.Image | None = None
        self.working_pil_img: Image.Image | None = None

        self.segmenter = SAMSegmenter()
        self.cache: TouchupCache | None = None

        self.current_selected_rect: tuple[int, int, int, int] | None = None
        self.current_sam_mask = None
        self.current_text_boxes: list[tuple[int, int, int, int]] = []
        self.active_text_item: DraggableTextItem | None = None
        self._style_controls: list[QWidget] = []
        self._style_syncing = False
        self._undo_stack: list[dict] = []
        self._undo_restoring = False
        self._undo_coalesce_key: str | None = None
        self._undo_coalesce_generation = 0
        self._max_undo_steps = 30

        self._init_fonts()
        self._build_ui()

        if initial_image:
            self.load_image(Path(initial_image))

    def _init_fonts(self):
        # Discover bundled project fonts and system fonts
        self.available_fonts = ["Arial", "Comic Sans MS", "Impact", "Tahoma", "Verdana"]
        font_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
        if font_dir.exists():
            for f in font_dir.glob("*.ttf"):
                QFontDatabase.addApplicationFont(str(f))
                self.available_fonts.insert(0, f.stem)

        # Remove duplicates while keeping order
        seen = set()
        self.available_fonts = [
            x for x in self.available_fonts if not (x in seen or seen.add(x))
        ]

    # UI Setup
    def _build_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_img_act = QAction("Open &Image…", self)
        open_img_act.setShortcut(QKeySequence.StandardKey.Open)
        open_img_act.triggered.connect(self._on_open_image_dialog)
        file_menu.addAction(open_img_act)

        open_folder_act = QAction("Open Chapter &Folder…", self)
        open_folder_act.triggered.connect(self._on_open_folder_dialog)
        file_menu.addAction(open_folder_act)

        save_act = QAction("&Save & Bake Page", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self.save_and_bake_page)
        file_menu.addAction(save_act)

        self.undo_act = QAction("&Undo Touch-Up Edit", self)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_act.setEnabled(False)
        self.undo_act.triggered.connect(self.undo_last_edit)
        file_menu.addAction(self.undo_act)

        # Top Navigation Bar
        top_bar = QWidget()
        top_bar.setFixedHeight(44)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 4, 10, 4)

        self.btn_open = QPushButton("Open Image…")
        self.btn_open.clicked.connect(self._on_open_image_dialog)
        top_layout.addWidget(self.btn_open)

        self.btn_prev = QPushButton("◀ Prev Page")
        self.btn_prev.clicked.connect(self._prev_page)
        self.btn_prev.setEnabled(False)
        top_layout.addWidget(self.btn_prev)

        self.lbl_page_info = QLabel("No image loaded")
        self.lbl_page_info.setStyleSheet("font-weight: bold; padding: 0 10px;")
        top_layout.addWidget(self.lbl_page_info)

        self.btn_next = QPushButton("Next Page ▶")
        self.btn_next.clicked.connect(self._next_page)
        self.btn_next.setEnabled(False)
        top_layout.addWidget(self.btn_next)

        top_layout.addSpacing(20)

        self.btn_mode_select = QPushButton("⬚ 1. Box + OCR (B)")
        self.btn_mode_select.setCheckable(True)
        self.btn_mode_select.setChecked(True)
        self.btn_mode_select.clicked.connect(lambda: self._set_mode("select"))
        top_layout.addWidget(self.btn_mode_select)

        self.btn_mode_edit = QPushButton("✋ 2. Move & Typeset (V)")
        self.btn_mode_edit.setCheckable(True)
        self.btn_mode_edit.clicked.connect(lambda: self._set_mode("edit"))
        top_layout.addWidget(self.btn_mode_edit)

        # Shortcuts
        self.shortcut_b = QShortcut(QKeySequence("B"), self)
        self.shortcut_b.activated.connect(lambda: self._set_mode("select"))
        self.shortcut_v = QShortcut(QKeySequence("V"), self)
        self.shortcut_v.activated.connect(lambda: self._set_mode("edit"))
        self.shortcut_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.shortcut_escape.activated.connect(self._clear_active_selection)

        top_layout.addStretch()

        self.btn_bake = QPushButton("💾 Save & Bake Page")
        self.btn_bake.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 6px 14px;"
        )
        self.btn_bake.clicked.connect(self.save_and_bake_page)
        top_layout.addWidget(self.btn_bake)

        # Main Splitter: Canvas on Left, Controls on Right
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Canvas
        self.canvas = TouchupCanvas(self)
        self.canvas.region_selected.connect(self._on_region_selected)
        self.canvas.item_selected.connect(self._on_item_selected)
        self.canvas.item_text_edited.connect(self._on_item_text_edited_from_canvas)
        self.canvas.item_change_started.connect(self._on_canvas_item_change_started)
        self.canvas.delete_requested.connect(self._delete_items)
        main_splitter.addWidget(self.canvas)

        # Right Sidebar Controls
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # Panel 1: OCR & Translation
        ocr_group = QGroupBox("1. OCR & Translation")
        ocr_layout = QVBoxLayout(ocr_group)
        ocr_layout.setContentsMargins(6, 6, 6, 6)
        ocr_layout.setSpacing(4)

        self.lbl_region_info = QLabel("Selected: None")
        self.lbl_region_info.setStyleSheet("color: #666;")
        ocr_layout.addWidget(self.lbl_region_info)

        ocr_layout.addWidget(QLabel("Original Chinese Text:"))
        self.txt_chinese = QTextEdit()
        self.txt_chinese.setMaximumHeight(55)
        self.txt_chinese.setPlaceholderText("OCR text will appear here…")
        ocr_layout.addWidget(self.txt_chinese)

        trans_btn_row = QHBoxLayout()
        self.btn_translate = QPushButton("🔄 Auto-Translate (LLM)")
        self.btn_translate.clicked.connect(self._run_auto_translate)
        trans_btn_row.addWidget(self.btn_translate)
        ocr_layout.addLayout(trans_btn_row)

        ocr_layout.addWidget(QLabel("English Translation:"))
        self.txt_english = QTextEdit()
        self.txt_english.setMaximumHeight(60)
        self.txt_english.setPlaceholderText("Enter or verify translated text…")
        self.txt_english.textChanged.connect(self._on_english_text_changed)
        self.txt_english.cursorPositionChanged.connect(self._on_english_cursor_changed)
        ocr_layout.addWidget(self.txt_english)

        sidebar_layout.addWidget(ocr_group)

        # Panel 2: Inpainting & Clean Background
        inpaint_group = QGroupBox("2. Erase Original Text")
        inpaint_layout = QVBoxLayout(inpaint_group)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Erase Dilation:"))
        self.spn_dilation = QSpinBox()
        self.spn_dilation.setRange(0, 30)
        self.spn_dilation.setValue(5)
        pad_row.addWidget(self.spn_dilation)
        inpaint_layout.addLayout(pad_row)

        self.btn_inpaint = QPushButton("🧹 Erase Original Text")
        self.btn_inpaint.clicked.connect(self._run_inpaint)
        inpaint_layout.addWidget(self.btn_inpaint)

        sidebar_layout.addWidget(inpaint_group)

        # Panel 3: Typesetting & Style
        style_group = QGroupBox("3. Typesetting & Styling")
        style_layout = QVBoxLayout(style_group)

        # Target Line Selector
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.cb_target_line = QComboBox()
        self.cb_target_line.addItem("All Lines")
        self.cb_target_line.currentIndexChanged.connect(self._on_target_line_changed)
        target_row.addWidget(self.cb_target_line, stretch=1)
        style_layout.addLayout(target_row)

        # Font & Size
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        self.cb_font = QComboBox()
        self.cb_font.addItems(self.available_fonts)
        self.cb_font.currentTextChanged.connect(self._on_font_changed)
        font_row.addWidget(self.cb_font, stretch=1)

        font_row.addWidget(QLabel("Size:"))
        self.spn_font_size = QSpinBox()
        self.spn_font_size.setRange(8, 120)
        self.spn_font_size.setValue(20)
        self.spn_font_size.valueChanged.connect(self._on_font_size_changed)
        font_row.addWidget(self.spn_font_size)
        style_layout.addLayout(font_row)

        # Quick Color Presets in 2x2 Grid
        color_grid = QGridLayout()
        color_grid.setContentsMargins(0, 0, 0, 0)
        color_grid.setSpacing(4)

        self.btn_preset_gold = QPushButton("🟡 Gold")
        self.btn_preset_gold.setStyleSheet(
            "background-color: #332b00; color: #ffd700; font-weight: bold; padding: 4px;"
        )
        self.btn_preset_gold.clicked.connect(
            lambda: self._apply_preset_color(QColor(255, 215, 0))
        )
        color_grid.addWidget(self.btn_preset_gold, 0, 0)

        self.btn_preset_white = QPushButton("⚪ White")
        self.btn_preset_white.setStyleSheet(
            "background-color: #222; color: #ffffff; font-weight: bold; padding: 4px;"
        )
        self.btn_preset_white.clicked.connect(
            lambda: self._apply_preset_color(QColor(255, 255, 255))
        )
        color_grid.addWidget(self.btn_preset_white, 0, 1)

        self.btn_preset_cyan = QPushButton("🔵 Cyan")
        self.btn_preset_cyan.setStyleSheet(
            "background-color: #002b33; color: #00e5ff; font-weight: bold; padding: 4px;"
        )
        self.btn_preset_cyan.clicked.connect(
            lambda: self._apply_preset_color(QColor(0, 229, 255))
        )
        color_grid.addWidget(self.btn_preset_cyan, 1, 0)

        self.btn_text_color = QPushButton("Custom…")
        self._text_color = QColor(255, 255, 255)
        self.btn_text_color.setStyleSheet("padding: 4px;")
        self.btn_text_color.clicked.connect(self._pick_text_color)
        color_grid.addWidget(self.btn_text_color, 1, 1)
        style_layout.addLayout(color_grid)

        # Stroke Row
        stroke_row = QHBoxLayout()
        stroke_row.addWidget(QLabel("Stroke:"))
        self.spn_stroke_width = QSpinBox()
        self.spn_stroke_width.setRange(0, 15)
        self.spn_stroke_width.setValue(2)
        self.spn_stroke_width.valueChanged.connect(self._on_stroke_width_changed)
        stroke_row.addWidget(self.spn_stroke_width)

        self.btn_stroke_color = QPushButton("Stroke Color")
        self._stroke_color = QColor(0, 0, 0)
        self.btn_stroke_color.setStyleSheet(
            f"background-color: {self._stroke_color.name()}; color: white;"
        )
        self.btn_stroke_color.clicked.connect(self._pick_stroke_color)
        stroke_row.addWidget(self.btn_stroke_color, stretch=1)
        style_layout.addLayout(stroke_row)

        # Alignment & Direction Row
        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("Align:"))
        self.cb_align = QComboBox()
        self.cb_align.addItems(["Left", "Center", "Right"])
        self.cb_align.currentTextChanged.connect(self._on_align_changed)
        align_row.addWidget(self.cb_align, stretch=1)

        align_row.addWidget(QLabel("Dir:"))
        self.cb_direction = QComboBox()
        self.cb_direction.addItems(["Horizontal", "Vertical"])
        self.cb_direction.currentTextChanged.connect(self._on_direction_changed)
        align_row.addWidget(self.cb_direction, stretch=1)
        style_layout.addLayout(align_row)

        # Action row
        action_row = QHBoxLayout()
        self.btn_delete_box = QPushButton("🗑 Delete Box")
        self.btn_delete_box.clicked.connect(self._delete_active_text_item)
        action_row.addWidget(self.btn_delete_box)

        style_layout.addLayout(action_row)
        sidebar_layout.addWidget(style_group)

        self._style_controls = [
            self.cb_target_line,
            self.cb_font,
            self.spn_font_size,
            self.btn_preset_gold,
            self.btn_preset_white,
            self.btn_preset_cyan,
            self.btn_text_color,
            self.spn_stroke_width,
            self.btn_stroke_color,
            self.cb_align,
            self.cb_direction,
            self.btn_delete_box,
        ]
        self._set_style_controls_enabled(False)

        sidebar_layout.addStretch()

        # Wrap sidebar in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(240)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_splitter.addWidget(scroll)

        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([750, 300])

        # Central layout
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(top_bar, 0)
        central_layout.addWidget(main_splitter, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready. Open an image to begin.")

    # Mode Handling
    def _set_mode(self, mode: str):
        if mode == "select":
            self.btn_mode_select.setChecked(True)
            self.btn_mode_edit.setChecked(False)
            self.btn_mode_select.setStyleSheet(
                "background-color: #0288d1; color: white; font-weight: bold; padding: 5px 12px;"
            )
            self.btn_mode_edit.setStyleSheet("padding: 5px 12px;")
            self.canvas.set_mode("select")
            self.statusBar().showMessage(
                "Mode: ⬚ Box + OCR - Drag a region around text."
            )
        else:
            self.btn_mode_select.setChecked(False)
            self.btn_mode_edit.setChecked(True)
            self.btn_mode_edit.setStyleSheet(
                "background-color: #0288d1; color: white; font-weight: bold; padding: 5px 12px;"
            )
            self.btn_mode_select.setStyleSheet("padding: 5px 12px;")
            self.canvas.set_mode("edit")
            self.statusBar().showMessage(
                "Mode: ✋ Move & Typeset - Click box to format, drag background to pan."
            )

    # File Navigation
    def _on_open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Comic Page Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.load_image(Path(file_path))

    def _on_open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Chapter Folder")
        if folder:
            p = Path(folder)
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            pages = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
            if pages:
                self.chapter_pages = pages
                self.load_image(pages[0])
            else:
                QMessageBox.information(
                    self, "No Images", "No image files found in the folder."
                )

    def load_image(self, img_path: Path):
        self.current_image_path = img_path
        self.cache = TouchupCache(img_path.parent)

        if not self.chapter_pages or img_path not in self.chapter_pages:
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            self.chapter_pages = sorted(
                [f for f in img_path.parent.iterdir() if f.suffix.lower() in exts]
            )

        if img_path in self.chapter_pages:
            self.current_page_idx = self.chapter_pages.index(img_path)
            self.lbl_page_info.setText(
                f"{img_path.name}  ({self.current_page_idx + 1}/{len(self.chapter_pages)})"
            )
            self.btn_prev.setEnabled(self.current_page_idx > 0)
            self.btn_next.setEnabled(
                self.current_page_idx < len(self.chapter_pages) - 1
            )
        else:
            self.lbl_page_info.setText(img_path.name)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)

        self.original_pil_img = Image.open(img_path).convert("RGB")
        self.working_pil_img = self.original_pil_img.copy()
        self._undo_stack.clear()
        self._undo_coalesce_key = None
        self._undo_coalesce_generation = 0
        self.undo_act.setEnabled(False)

        self.canvas.load_image(pil_to_qpixmap(self.working_pil_img))
        self.active_text_item = None
        self.statusBar().showMessage(
            f"Loaded {img_path.name} ({self.original_pil_img.width}x{self.original_pil_img.height})"
        )

        # Restore cached edits if any
        self._load_cached_edits()

    def _load_cached_edits(self):
        if not self.cache or not self.current_image_path:
            return
        edits = self._dedupe_cached_edits(
            self.cache.get_edits(self.current_image_path.name)
        )
        for ed in edits:
            self._add_text_item_from_render(
                ed.get("render", {}), translation=ed.get("translation"), select=True
            )
            self._sync_style_controls(reset_target=True)

    def _dedupe_cached_edits(self, edits: list[dict]) -> list[dict]:
        # Keep latest cache record for repeated translations
        latest_by_text = {}
        order = []
        for ed in edits:
            rend = ed.get("render", {})
            key = (ed.get("translation") or rend.get("text") or "").strip()
            if not key:
                key = ed.get("edit_id") or str(id(ed))
            if key not in latest_by_text:
                order.append(key)
            latest_by_text[key] = ed
        return [latest_by_text[key] for key in order]

    def _add_text_item_from_render(
        self, rend: dict, translation: str | None = None, select: bool = False
    ) -> DraggableTextItem:
        rect = QRectF(
            rend.get("x", 0),
            rend.get("y", 0),
            rend.get("w", 200),
            rend.get("h", 100),
        )
        item = self.canvas.add_text_item(
            translation if translation is not None else rend.get("text", ""),
            rect,
            select=select,
        )
        global_style = self._render_style_to_kwargs(rend)
        if any(value is not None for value in global_style.values()):
            item.set_line_style(-1, **global_style)

        lines_data = rend.get("lines", [])
        if lines_data:
            for idx, ld in enumerate(lines_data):
                c = self._qcolor_from_value(ld.get("color"))
                sc_line = self._qcolor_from_value(ld.get("stroke_color"))
                al_str = ld.get("align", "left")
                al_flag = (
                    Qt.AlignmentFlag.AlignCenter
                    if al_str == "center"
                    else Qt.AlignmentFlag.AlignRight
                    if al_str == "right"
                    else Qt.AlignmentFlag.AlignLeft
                )
                item.set_line_style(
                    idx,
                    family=ld.get("font_family"),
                    size=ld.get("font_size"),
                    bold=ld.get("bold"),
                    italic=ld.get("italic"),
                    color=c,
                    stroke_color=sc_line,
                    stroke_width=ld.get("stroke_width"),
                    align=al_flag,
                    direction=ld.get("direction"),
                )
        return item

    def _prev_page(self):
        if self.current_page_idx > 0:
            self.load_image(self.chapter_pages[self.current_page_idx - 1])

    def _next_page(self):
        if self.current_page_idx < len(self.chapter_pages) - 1:
            self.load_image(self.chapter_pages[self.current_page_idx + 1])

    # Undo & State Snapshots
    def _text_items(self) -> list[DraggableTextItem]:
        return self.canvas.text_items()

    def _snapshot_state(self) -> dict:
        items = self._text_items()
        active_idx = (
            items.index(self.active_text_item)
            if self.active_text_item in items
            else None
        )
        return {
            "working_pil_img": self.working_pil_img.copy()
            if self.working_pil_img
            else None,
            "items": [item.get_state() for item in items],
            "active_idx": active_idx,
            "current_selected_rect": self.current_selected_rect,
            "current_text_boxes": list(self.current_text_boxes),
            "txt_chinese": self.txt_chinese.toPlainText(),
            "txt_english": self.txt_english.toPlainText(),
        }

    def _push_undo(self, label: str = "", coalesce_key: str | None = None):
        if self._undo_restoring or not self.working_pil_img:
            return
        if coalesce_key and self._undo_coalesce_key == coalesce_key:
            return

        self._undo_stack.append(self._snapshot_state())
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)
        self._undo_coalesce_key = coalesce_key
        if coalesce_key:
            self._undo_coalesce_generation += 1
            generation = self._undo_coalesce_generation
            QTimer.singleShot(
                800,
                lambda key=coalesce_key, gen=generation: self._clear_undo_coalesce(
                    key, gen
                ),
            )
        self.undo_act.setEnabled(True)

    def _clear_undo_coalesce(self, key: str, generation: int):
        if (
            self._undo_coalesce_key == key
            and self._undo_coalesce_generation == generation
        ):
            self._undo_coalesce_key = None

    def undo_last_edit(self):
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self._undo_coalesce_key = None
        self._restore_snapshot(snapshot)
        self.undo_act.setEnabled(bool(self._undo_stack))
        self.statusBar().showMessage("Undid last touch-up edit.")

    def _restore_snapshot(self, snapshot: dict):
        self._undo_restoring = True
        try:
            self.working_pil_img = (
                snapshot["working_pil_img"].copy()
                if snapshot.get("working_pil_img")
                else None
            )
            if self.working_pil_img:
                self.canvas.pixmap_item.setPixmap(pil_to_qpixmap(self.working_pil_img))

            self.canvas.clear_text_items()
            restored_items = []
            for state in snapshot.get("items", []):
                restored_items.append(
                    self._add_text_item_from_render(state, select=False)
                )

            self.current_selected_rect = snapshot.get("current_selected_rect")
            self.current_text_boxes = list(snapshot.get("current_text_boxes", []))

            self.txt_chinese.blockSignals(True)
            self.txt_chinese.setText(snapshot.get("txt_chinese", ""))
            self.txt_chinese.blockSignals(False)

            self.txt_english.blockSignals(True)
            self.txt_english.setText(snapshot.get("txt_english", ""))
            self.txt_english.blockSignals(False)

            active_idx = snapshot.get("active_idx")
            if isinstance(active_idx, int) and 0 <= active_idx < len(restored_items):
                restored_items[active_idx].setSelected(True)
                self._on_item_selected(restored_items[active_idx])
            else:
                self._on_item_selected(None)
        finally:
            self._undo_restoring = False

    def _clear_active_selection(self):
        self.canvas.deselect_all()

    def _on_canvas_item_change_started(self, item):
        if isinstance(item, DraggableTextItem):
            self._push_undo("Edit text box", coalesce_key=f"item:{id(item)}:drag")

    # Region Selection & Processing
    def _on_region_selected(self, rect: QRectF):
        if not self.working_pil_img:
            return

        x1, y1 = int(round(rect.left())), int(round(rect.top()))
        x2, y2 = int(round(rect.right())), int(round(rect.bottom()))
        self.current_selected_rect = (x1, y1, x2, y2)
        self.lbl_region_info.setText(
            f"Selected: ({x1}, {y1}) → ({x2}, {y2}) [w={x2 - x1}, h={y2 - y1}]"
        )

        self.statusBar().showMessage("Running SAM 2.1 segmentation and OCR…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # 1. SAM Segmentation
            self.current_sam_mask = self.segmenter.segment_box(
                self.working_pil_img, (x1, y1, x2, y2)
            )

            # 2. OCR Text & Bounding Boxes
            full_text, line_items = extract_text_and_boxes(
                self.working_pil_img, (x1, y1, x2, y2)
            )
            self.current_text_boxes = [item["box"] for item in line_items]

            self.txt_chinese.setText(full_text)

            # 3. Auto Translation
            translated = translate_text(
                full_text,
                series_dir=self.current_image_path.parent
                if self.current_image_path
                else None,
            )
            final_text = translated if translated else full_text

            self.txt_english.blockSignals(True)
            self.txt_english.setText(final_text)
            self.txt_english.blockSignals(False)

            # 4. Automatically erase original text & create interactive text box
            self._push_undo("Add translated region")
            self._run_inpaint(record_undo=False)

            # Create text item on canvas
            text_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = self.canvas.add_text_item(final_text, text_rect)
            self.active_text_item = item
            self._sync_style_controls(reset_target=True)

            self._set_mode("edit")
            self.statusBar().showMessage(
                "Region processed. Cleaned background & placed text box."
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _run_auto_translate(self):
        chinese = self.txt_chinese.toPlainText()
        if not chinese.strip():
            return
        self.statusBar().showMessage("Translating text…")
        translated = translate_text(
            chinese,
            series_dir=self.current_image_path.parent
            if self.current_image_path
            else None,
        )
        if translated:
            self._push_undo("Auto-translate text")
            self.txt_english.setText(translated)
            if self.active_text_item:
                self.active_text_item.set_text(translated)
        self.statusBar().showMessage("Translation updated.")

    def _run_inpaint(self, record_undo: bool = True):
        if not self.working_pil_img or not self.current_text_boxes:
            return

        if record_undo:
            self._push_undo("Erase original text")
        pad = self.spn_dilation.value()
        inpainted_img, mask = inpaint_text_regions(
            self.working_pil_img,
            self.current_text_boxes,
            sam_mask=self.current_sam_mask,
            padding_px=pad,
        )
        self.working_pil_img = inpainted_img
        self.canvas.pixmap_item.setPixmap(pil_to_qpixmap(self.working_pil_img))

    # Typesetting Interactions
    def _set_style_controls_enabled(self, enabled: bool):
        for widget in self._style_controls:
            widget.setEnabled(enabled)

    def _populate_target_lines(self, reset_to_all: bool = False):
        if not self.active_text_item:
            self.cb_target_line.blockSignals(True)
            self.cb_target_line.clear()
            self.cb_target_line.addItem("All Lines")
            self.cb_target_line.blockSignals(False)
            return

        curr_idx = 0 if reset_to_all else self.cb_target_line.currentIndex()
        self.cb_target_line.blockSignals(True)
        self.cb_target_line.clear()
        self.cb_target_line.addItem("All Lines")
        for i, ld in enumerate(self.active_text_item._lines_data):
            preview = ld.get("text", "").strip()[:20]
            if not preview:
                preview = "(empty)"
            self.cb_target_line.addItem(f"Line {i + 1}: {preview}")
        if 0 <= curr_idx < self.cb_target_line.count():
            self.cb_target_line.setCurrentIndex(curr_idx)
        else:
            self.cb_target_line.setCurrentIndex(0)
        self.cb_target_line.blockSignals(False)

    def _on_target_line_changed(self):
        if not self.active_text_item:
            return
        self._sync_style_controls(reset_target=False)

    def _on_english_cursor_changed(self):
        if not self.active_text_item:
            return
        line_num = self.txt_english.textCursor().blockNumber()
        target_idx = line_num + 1
        if 0 <= target_idx < self.cb_target_line.count():
            self.cb_target_line.setCurrentIndex(target_idx)

    def _on_item_selected(self, item):
        if not isinstance(item, DraggableTextItem):
            self.active_text_item = None
            self._set_style_controls_enabled(False)
            self._populate_target_lines(reset_to_all=True)
            return

        self.active_text_item = item
        state = item.get_state()

        # Block signals so UI sync does not overwrite item values
        self.txt_english.blockSignals(True)
        self.txt_english.setText(state["text"])
        self.txt_english.blockSignals(False)

        self._sync_style_controls(reset_target=True)

        try:
            item.changed.disconnect(self._on_active_item_changed)
        except Exception:
            pass
        item.changed.connect(self._on_active_item_changed)

    def _on_item_text_edited_from_canvas(self, item, new_text):
        if item == self.active_text_item:
            self.txt_english.blockSignals(True)
            self.txt_english.setText(new_text)
            self.txt_english.blockSignals(False)
            self._populate_target_lines()
            self._sync_style_controls(reset_target=False)

    def _on_active_item_changed(self):
        if not self.active_text_item:
            return
        sender = self.sender()
        if sender is not None and sender is not self.active_text_item:
            return
        self._sync_style_controls(reset_target=False)

    def _on_english_text_changed(self):
        if self.active_text_item:
            self._push_undo(
                "Edit text", coalesce_key=f"text:{id(self.active_text_item)}"
            )
            self.active_text_item.set_text(self.txt_english.toPlainText())
            self._populate_target_lines()
            self._sync_style_controls(reset_target=False)

    def _style_target_index(self) -> int:
        return self.cb_target_line.currentIndex() - 1

    def _apply_style_change(self, **kwargs):
        if not self.active_text_item or self._style_syncing:
            return
        style_key = ",".join(sorted(kwargs))
        self._push_undo(
            "Style text", coalesce_key=f"style:{id(self.active_text_item)}:{style_key}"
        )
        self.active_text_item.set_line_style(self._style_target_index(), **kwargs)

    def _on_font_changed(self, family: str):
        if family == self.MIXED_STYLE_LABEL:
            return
        self._apply_style_change(family=family)

    def _on_font_size_changed(self, size: int):
        self._apply_style_change(size=size)

    def _on_stroke_width_changed(self, width: int):
        self._apply_style_change(stroke_width=width)

    def _on_align_changed(self, align: str):
        if align == self.MIXED_STYLE_LABEL:
            return
        self._apply_style_change(align=self._align_from_name(align))

    def _on_direction_changed(self, direction: str):
        if direction == self.MIXED_STYLE_LABEL:
            return
        self._apply_style_change(direction=direction.lower())

    def _apply_preset_color(self, color: QColor):
        self._text_color = color
        self._set_color_button(self.btn_text_color, color, "Custom…")
        self._apply_style_change(color=color)

    def _pick_text_color(self):
        color = QColorDialog.getColor(self._text_color, self, "Pick Text Color")
        if color.isValid():
            self._apply_preset_color(color)

    def _pick_stroke_color(self):
        color = QColorDialog.getColor(self._stroke_color, self, "Pick Stroke Color")
        if color.isValid():
            self._stroke_color = color
            self._set_color_button(self.btn_stroke_color, color, "Stroke Color")
            self._apply_style_change(stroke_color=color)

    def _delete_active_text_item(self):
        if self.active_text_item:
            self._delete_items([self.active_text_item])

    def _delete_items(self, items: list):
        text_items = [item for item in items if isinstance(item, DraggableTextItem)]
        if not text_items:
            return
        self._push_undo("Delete text box")
        for item in text_items:
            self.canvas.scene.removeItem(item)
        self.active_text_item = None
        self._set_style_controls_enabled(False)
        self._populate_target_lines(reset_to_all=True)
        self.statusBar().showMessage("Text box deleted.")

    # Style Sync Helpers
    def _sync_style_controls(self, reset_target: bool = False):
        if not self.active_text_item:
            self._set_style_controls_enabled(False)
            return

        self._style_syncing = True
        try:
            self._set_style_controls_enabled(True)
            self._populate_target_lines(reset_to_all=reset_target)
            style = self._effective_style_for_target(self.active_text_item)

            self._set_combo_value(self.cb_font, style["font_family"])
            self._set_spin_value(
                self.spn_font_size,
                style["font_size"],
                self._fallback_style_value("font_size"),
            )
            self._set_color_value(
                "text",
                style["color"],
                self._fallback_style_value("color"),
            )
            self._set_spin_value(
                self.spn_stroke_width,
                style["stroke_width"],
                self._fallback_style_value("stroke_width"),
            )
            self._set_color_value(
                "stroke",
                style["stroke_color"],
                self._fallback_style_value("stroke_color"),
            )

            align = style["align"]
            self._set_combo_value(
                self.cb_align,
                self._MIXED if align is self._MIXED else self._align_to_name(align),
            )

            direction = style["direction"]
            self._set_combo_value(
                self.cb_direction,
                self._MIXED if direction is self._MIXED else direction.capitalize(),
            )
        finally:
            self._style_syncing = False

    def _effective_style_for_target(self, item: DraggableTextItem) -> dict:
        props = (
            "font_family",
            "font_size",
            "color",
            "stroke_color",
            "stroke_width",
            "align",
            "direction",
        )
        line_idx = self._style_target_index()
        if 0 <= line_idx < len(item._lines_data):
            line = item._lines_data[line_idx]
            return {prop: self._line_style_value(item, line, prop) for prop in props}

        if not item._lines_data:
            return {prop: self._global_style_value(item, prop) for prop in props}

        style = {}
        for prop in props:
            values = [
                self._line_style_value(item, line, prop) for line in item._lines_data
            ]
            first = values[0]
            style[prop] = (
                first
                if all(self._style_values_equal(first, value) for value in values[1:])
                else self._MIXED
            )
        return style

    def _fallback_style_value(self, prop: str):
        if not self.active_text_item:
            return None
        line_idx = self._style_target_index()
        if 0 <= line_idx < len(self.active_text_item._lines_data):
            return self._line_style_value(
                self.active_text_item,
                self.active_text_item._lines_data[line_idx],
                prop,
            )
        if self.active_text_item._lines_data:
            return self._line_style_value(
                self.active_text_item,
                self.active_text_item._lines_data[0],
                prop,
            )
        return self._global_style_value(self.active_text_item, prop)

    def _global_style_value(self, item: DraggableTextItem, prop: str):
        mapping = {
            "font_family": "_font_family",
            "font_size": "_font_size",
            "color": "_text_color",
            "stroke_color": "_stroke_color",
            "stroke_width": "_stroke_width",
            "align": "_align",
            "direction": "_direction",
        }
        return getattr(item, mapping[prop])

    def _line_style_value(self, item: DraggableTextItem, line: dict, prop: str):
        return line.get(prop, self._global_style_value(item, prop))

    def _style_values_equal(self, left, right) -> bool:
        if isinstance(left, QColor) and isinstance(right, QColor):
            return (
                left.red(),
                left.green(),
                left.blue(),
            ) == (right.red(), right.green(), right.blue())
        return left == right

    def _set_combo_value(self, combo: QComboBox, value):
        combo.blockSignals(True)
        try:
            if value is self._MIXED:
                if combo.findText(self.MIXED_STYLE_LABEL) < 0:
                    combo.insertItem(0, self.MIXED_STYLE_LABEL)
                combo.setCurrentText(self.MIXED_STYLE_LABEL)
                combo.setToolTip("Mixed values. Pick a value to apply it.")
                return

            combo.setToolTip("")
            text = str(value)
            idx = combo.findText(text)
            if idx < 0:
                combo.addItem(text)
            combo.setCurrentText(text)
        finally:
            combo.blockSignals(False)

    def _set_spin_value(self, spin: QSpinBox, value, fallback):
        spin.blockSignals(True)
        try:
            if value is self._MIXED:
                spin.setValue(int(fallback) if fallback is not None else spin.minimum())
                spin.setToolTip("Mixed values. Change this to apply one value.")
            else:
                spin.setValue(int(value))
                spin.setToolTip("")
        finally:
            spin.blockSignals(False)

    def _set_color_value(self, which: str, value, fallback):
        if value is self._MIXED:
            color = fallback if isinstance(fallback, QColor) else QColor(255, 255, 255)
            if which == "text":
                self._text_color = color
                self._set_mixed_color_button(self.btn_text_color)
            else:
                self._stroke_color = color
                self._set_mixed_color_button(self.btn_stroke_color)
            return

        color = value if isinstance(value, QColor) else QColor(255, 255, 255)
        if which == "text":
            self._text_color = color
            self._set_color_button(self.btn_text_color, color, "Custom…")
        else:
            self._stroke_color = color
            self._set_color_button(self.btn_stroke_color, color, "Stroke Color")

    def _set_color_button(self, button: QPushButton, color: QColor, text: str):
        button.setText(text)
        button.setToolTip("")
        button.setStyleSheet(f"background-color: {color.name()};")

    def _set_mixed_color_button(self, button: QPushButton):
        button.setText("Mixed…")
        button.setToolTip("Mixed values. Pick a color to apply one value.")
        button.setStyleSheet("background-color: #555; color: white;")

    def _qcolor_from_value(self, value, fallback: QColor | None = None):
        if isinstance(value, QColor):
            return value
        if isinstance(value, list) and len(value) == 3:
            return QColor(value[0], value[1], value[2])
        return fallback

    def _align_from_name(self, value: str | None):
        value = (value or "Left").lower()
        if value == "center":
            return Qt.AlignmentFlag.AlignCenter
        if value == "right":
            return Qt.AlignmentFlag.AlignRight
        return Qt.AlignmentFlag.AlignLeft

    def _align_to_name(self, align) -> str:
        if align == Qt.AlignmentFlag.AlignCenter:
            return "Center"
        if align == Qt.AlignmentFlag.AlignRight:
            return "Right"
        return "Left"

    def _render_style_to_kwargs(self, rend: dict) -> dict:
        return {
            "family": rend.get("font_family"),
            "size": rend.get("font_size"),
            "bold": rend.get("bold"),
            "italic": rend.get("italic"),
            "color": self._qcolor_from_value(rend.get("text_color")),
            "stroke_color": self._qcolor_from_value(rend.get("stroke_color")),
            "stroke_width": rend.get("stroke_width"),
            "align": self._align_from_name(rend.get("align"))
            if rend.get("align") is not None
            else None,
            "direction": rend.get("direction"),
        }

    # Save & Bake
    def save_and_bake_page(self):
        if not self.working_pil_img or not self.current_image_path:
            QMessageBox.warning(self, "No Image", "No image is currently loaded.")
            return

        final_img = self.working_pil_img.copy()

        # Render all DraggableTextItems onto PIL Image using high-quality QPainter
        qimg = QImage(final_img.width, final_img.height, QImage.Format.Format_RGBA8888)
        qimg.fill(Qt.GlobalColor.transparent)

        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Draw base inpainted image
        painter.drawPixmap(0, 0, pil_to_qpixmap(self.working_pil_img))

        cached_edits = []
        for item in self._text_items():
            state = item.get_state()
            cached_edits.append(
                {
                    "source_box": self.current_selected_rect,
                    "original_text": self.txt_chinese.toPlainText(),
                    "translation": state["text"],
                    "render": state,
                }
            )
            # Paint item in scene coordinates without selection borders or handles
            was_selected = item.isSelected()
            item.setSelected(False)
            painter.save()
            painter.setTransform(item.sceneTransform(), True)
            item.paint(painter, None)
            painter.restore()
            item.setSelected(was_selected)

        painter.end()

        # Convert back to PIL and save
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        import numpy as np

        arr = np.array(ptr).reshape((qimg.height(), qimg.width(), 4))
        final_pil = Image.fromarray(arr, "RGBA").convert("RGB")

        # Save to disk
        final_pil.save(self.current_image_path, quality=95)

        # Save cache
        if self.cache:
            self.cache.replace_page_edits(self.current_image_path.name, cached_edits)

        self.statusBar().showMessage(
            f"Successfully saved and baked {self.current_image_path.name}!"
        )
        QMessageBox.information(
            self,
            "Saved",
            f"Page baked and saved successfully to:\n{self.current_image_path.name}",
        )


def main():
    parser = argparse.ArgumentParser(description="Manhua Touch-Up Editor")
    parser.add_argument("--image", type=str, help="Path to image file to open")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TouchupApp(initial_image=args.image)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
