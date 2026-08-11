import argparse
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRectF, Qt
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


def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """Converts a PIL Image to QPixmap."""
    im = pil_image.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qimg = QImage(data, im.width, im.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class TouchupApp(QMainWindow):
    """Main Touch-Up Editor Window."""

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

        self._init_fonts()
        self._build_ui()

        if initial_image:
            self.load_image(Path(initial_image))

    def _init_fonts(self):
        """Discovers bundled project fonts and system fonts."""
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

    # ---------------- UI Setup ----------------
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
        self.cb_font.currentTextChanged.connect(self._update_active_style)
        font_row.addWidget(self.cb_font, stretch=1)

        font_row.addWidget(QLabel("Size:"))
        self.spn_font_size = QSpinBox()
        self.spn_font_size.setRange(8, 120)
        self.spn_font_size.setValue(20)
        self.spn_font_size.valueChanged.connect(self._update_active_style)
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
        self.spn_stroke_width.valueChanged.connect(self._update_active_style)
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
        self.cb_align.currentTextChanged.connect(self._update_active_style)
        align_row.addWidget(self.cb_align, stretch=1)

        align_row.addWidget(QLabel("Dir:"))
        self.cb_direction = QComboBox()
        self.cb_direction.addItems(["Horizontal", "Vertical"])
        self.cb_direction.currentTextChanged.connect(self._update_active_style)
        align_row.addWidget(self.cb_direction, stretch=1)
        style_layout.addLayout(align_row)

        # Action row
        action_row = QHBoxLayout()
        self.btn_delete_box = QPushButton("🗑 Delete Box")
        self.btn_delete_box.clicked.connect(self._delete_active_text_item)
        action_row.addWidget(self.btn_delete_box)

        style_layout.addLayout(action_row)
        sidebar_layout.addWidget(style_group)

        sidebar_layout.addStretch()

        # Wrap sidebar in scroll area (resizable, no horizontal clipping)
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

    # ---------------- Mode Handling ----------------
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

    # ---------------- File Navigation ----------------
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
        edits = self.cache.get_edits(self.current_image_path.name)
        for ed in edits:
            rend = ed.get("render", {})
            rect = QRectF(
                rend.get("x", 0),
                rend.get("y", 0),
                rend.get("w", 200),
                rend.get("h", 100),
            )
            item = self.canvas.add_text_item(ed.get("translation", ""), rect)
            lines_data = rend.get("lines", [])
            if lines_data:
                for idx, ld in enumerate(lines_data):
                    c = ld.get("color", [255, 255, 255])
                    sc_line = ld.get("stroke_color")
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
                        color=QColor(c[0], c[1], c[2])
                        if isinstance(c, list) and len(c) == 3
                        else None,
                        stroke_color=QColor(sc_line[0], sc_line[1], sc_line[2])
                        if isinstance(sc_line, list) and len(sc_line) == 3
                        else None,
                        stroke_width=ld.get("stroke_width"),
                        align=al_flag,
                        direction=ld.get("direction"),
                    )
            sc = rend.get("stroke_color", [0, 0, 0])
            item._stroke_color = QColor(sc[0], sc[1], sc[2])
            item._stroke_width = rend.get("stroke_width", 2)
            align_val = rend.get("align", "left")
            item.set_alignment(
                Qt.AlignmentFlag.AlignCenter
                if align_val == "center"
                else Qt.AlignmentFlag.AlignLeft
                if align_val == "left"
                else Qt.AlignmentFlag.AlignRight
            )
            item.set_direction(rend.get("direction", "horizontal"))

    def _prev_page(self):
        if self.current_page_idx > 0:
            self.load_image(self.chapter_pages[self.current_page_idx - 1])

    def _next_page(self):
        if self.current_page_idx < len(self.chapter_pages) - 1:
            self.load_image(self.chapter_pages[self.current_page_idx + 1])

    # ---------------- Region Selection & Processing ----------------
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
            self._run_inpaint()

            # Create text item on canvas
            text_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = self.canvas.add_text_item(final_text, text_rect)
            self.active_text_item = item
            self._update_active_style()

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
            self.txt_english.setText(translated)
            if self.active_text_item:
                self.active_text_item.set_text(translated)
        self.statusBar().showMessage("Translation updated.")

    def _run_inpaint(self):
        if not self.working_pil_img or not self.current_text_boxes:
            return

        pad = self.spn_dilation.value()
        inpainted_img, mask = inpaint_text_regions(
            self.working_pil_img,
            self.current_text_boxes,
            sam_mask=self.current_sam_mask,
            padding_px=pad,
        )
        self.working_pil_img = inpainted_img
        self.canvas.pixmap_item.setPixmap(pil_to_qpixmap(self.working_pil_img))

    # ---------------- Typesetting Interactions ----------------
    def _populate_target_lines(self):
        if not self.active_text_item:
            self.cb_target_line.blockSignals(True)
            self.cb_target_line.clear()
            self.cb_target_line.addItem("All Lines")
            self.cb_target_line.blockSignals(False)
            return

        curr_idx = self.cb_target_line.currentIndex()
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
        idx = self.cb_target_line.currentIndex() - 1
        if 0 <= idx < len(self.active_text_item._lines_data):
            ld = self.active_text_item._lines_data[idx]
            self.cb_font.blockSignals(True)
            f_idx = self.cb_font.findText(
                ld.get("font_family", self.active_text_item._font_family)
            )
            if f_idx >= 0:
                self.cb_font.setCurrentIndex(f_idx)
            self.cb_font.blockSignals(False)

            self.spn_font_size.blockSignals(True)
            self.spn_font_size.setValue(
                ld.get("font_size", self.active_text_item._font_size)
            )
            self.spn_font_size.blockSignals(False)

            self._text_color = ld.get("color", self.active_text_item._text_color)
            self.btn_text_color.setStyleSheet(
                f"background-color: {self._text_color.name()};"
            )

            self.spn_stroke_width.blockSignals(True)
            self.spn_stroke_width.setValue(
                ld.get("stroke_width", self.active_text_item._stroke_width)
            )
            self.spn_stroke_width.blockSignals(False)

            self._stroke_color = ld.get(
                "stroke_color", self.active_text_item._stroke_color
            )
            self.btn_stroke_color.setStyleSheet(
                f"background-color: {self._stroke_color.name()}; color: white;"
            )

            self.cb_align.blockSignals(True)
            al = ld.get("align", self.active_text_item._align)
            align_name = (
                "Center"
                if al == Qt.AlignmentFlag.AlignCenter
                else "Right"
                if al == Qt.AlignmentFlag.AlignRight
                else "Left"
            )
            al_idx = self.cb_align.findText(align_name)
            if al_idx >= 0:
                self.cb_align.setCurrentIndex(al_idx)
            self.cb_align.blockSignals(False)

            self.cb_direction.blockSignals(True)
            dir_name = ld.get(
                "direction", self.active_text_item._direction
            ).capitalize()
            dir_idx = self.cb_direction.findText(dir_name)
            if dir_idx >= 0:
                self.cb_direction.setCurrentIndex(dir_idx)
            self.cb_direction.blockSignals(False)
        else:
            self.cb_font.blockSignals(True)
            f_idx = self.cb_font.findText(self.active_text_item._font_family)
            if f_idx >= 0:
                self.cb_font.setCurrentIndex(f_idx)
            self.cb_font.blockSignals(False)

            self.spn_font_size.blockSignals(True)
            self.spn_font_size.setValue(self.active_text_item._font_size)
            self.spn_font_size.blockSignals(False)

            self._text_color = self.active_text_item._text_color
            self.btn_text_color.setStyleSheet(
                f"background-color: {self._text_color.name()};"
            )

            self.spn_stroke_width.blockSignals(True)
            self.spn_stroke_width.setValue(self.active_text_item._stroke_width)
            self.spn_stroke_width.blockSignals(False)

            self._stroke_color = self.active_text_item._stroke_color
            self.btn_stroke_color.setStyleSheet(
                f"background-color: {self._stroke_color.name()}; color: white;"
            )

            self.cb_align.blockSignals(True)
            al = self.active_text_item._align
            align_name = (
                "Center"
                if al == Qt.AlignmentFlag.AlignCenter
                else "Right"
                if al == Qt.AlignmentFlag.AlignRight
                else "Left"
            )
            al_idx = self.cb_align.findText(align_name)
            if al_idx >= 0:
                self.cb_align.setCurrentIndex(al_idx)
            self.cb_align.blockSignals(False)

            self.cb_direction.blockSignals(True)
            dir_name = self.active_text_item._direction.capitalize()
            dir_idx = self.cb_direction.findText(dir_name)
            if dir_idx >= 0:
                self.cb_direction.setCurrentIndex(dir_idx)
            self.cb_direction.blockSignals(False)

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
            self._populate_target_lines()
            return

        self.active_text_item = item
        state = item.get_state()

        # Block signals so UI sync does not overwrite item values
        self.txt_english.blockSignals(True)
        self.txt_english.setText(state["text"])
        self.txt_english.blockSignals(False)

        self._populate_target_lines()

        self.cb_font.blockSignals(True)
        idx = self.cb_font.findText(state["font_family"])
        if idx >= 0:
            self.cb_font.setCurrentIndex(idx)
        self.cb_font.blockSignals(False)

        self.spn_font_size.blockSignals(True)
        self.spn_font_size.setValue(state["font_size"])
        self.spn_font_size.blockSignals(False)

        self.spn_stroke_width.blockSignals(True)
        self.spn_stroke_width.setValue(state["stroke_width"])
        self.spn_stroke_width.blockSignals(False)

        self.cb_align.blockSignals(True)
        align_name = state.get("align", "left").capitalize()
        align_idx = self.cb_align.findText(align_name)
        if align_idx >= 0:
            self.cb_align.setCurrentIndex(align_idx)
        self.cb_align.blockSignals(False)

        self.cb_direction.blockSignals(True)
        dir_name = state.get("direction", "horizontal").capitalize()
        dir_idx = self.cb_direction.findText(dir_name)
        if dir_idx >= 0:
            self.cb_direction.setCurrentIndex(dir_idx)
        self.cb_direction.blockSignals(False)

        tc = state["text_color"]
        self._text_color = QColor(tc[0], tc[1], tc[2])
        self.btn_text_color.setStyleSheet(
            f"background-color: {self._text_color.name()};"
        )

        sc = state["stroke_color"]
        self._stroke_color = QColor(sc[0], sc[1], sc[2])
        self.btn_stroke_color.setStyleSheet(
            f"background-color: {self._stroke_color.name()};"
        )

        # Hook dynamic resize change
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

    def _on_active_item_changed(self):
        if not self.active_text_item:
            return
        idx = self.cb_target_line.currentIndex() - 1
        if 0 <= idx < len(self.active_text_item._lines_data):
            sz = self.active_text_item._lines_data[idx].get(
                "font_size", self.active_text_item._font_size
            )
        else:
            sz = self.active_text_item._font_size
        self.spn_font_size.blockSignals(True)
        self.spn_font_size.setValue(sz)
        self.spn_font_size.blockSignals(False)

    def _on_english_text_changed(self):
        if self.active_text_item:
            self.active_text_item.set_text(self.txt_english.toPlainText())
            self._populate_target_lines()

    def _apply_preset_color(self, color: QColor):
        self._text_color = color
        self.btn_text_color.setStyleSheet(f"background-color: {color.name()};")
        self._update_active_style()

    def _update_active_style(self):
        if not self.active_text_item:
            return
        target_idx = self.cb_target_line.currentIndex() - 1
        align_str = self.cb_align.currentText()
        align_flag = (
            Qt.AlignmentFlag.AlignCenter
            if align_str == "Center"
            else Qt.AlignmentFlag.AlignRight
            if align_str == "Right"
            else Qt.AlignmentFlag.AlignLeft
        )
        self.active_text_item.set_line_style(
            target_idx,
            family=self.cb_font.currentText(),
            size=self.spn_font_size.value(),
            color=self._text_color,
            stroke_color=self._stroke_color,
            stroke_width=self.spn_stroke_width.value(),
            align=align_flag,
            direction=self.cb_direction.currentText().lower(),
        )

    def _pick_text_color(self):
        color = QColorDialog.getColor(self._text_color, self, "Pick Text Color")
        if color.isValid():
            self._apply_preset_color(color)

    def _pick_stroke_color(self):
        color = QColorDialog.getColor(self._stroke_color, self, "Pick Stroke Color")
        if color.isValid():
            self._stroke_color = color
            self.btn_stroke_color.setStyleSheet(f"background-color: {color.name()};")
            self._update_active_style()

    def _delete_active_text_item(self):
        if self.active_text_item:
            self.canvas.scene.removeItem(self.active_text_item)
            self.active_text_item = None
            self.statusBar().showMessage("Text box deleted.")

    # ---------------- Save & Bake ----------------
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
        for item in self.canvas.scene.items():
            if isinstance(item, DraggableTextItem):
                state = item.get_state()
                cached_edits.append(
                    {
                        "source_box": self.current_selected_rect,
                        "original_text": self.txt_chinese.toPlainText(),
                        "translation": state["text"],
                        "render": state,
                    }
                )
                # Paint item without selection borders
                item.setSelected(False)
                item.paint(painter, None)

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
            for ed in cached_edits:
                self.cache.save_edit(self.current_image_path.name, ed)

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
