import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"
SETTINGS = ROOT / "settings.json"
CONFIG_FILE = ROOT / "config.py"
TOUCHUP_APP = ROOT / "touchup" / "app.py"

STAGES = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
STAGE_LABELS = {
    "import": "0 · Import",
    "detect": "1 · Detect",
    "ocr": "2 · OCR",
    "translate": "3 · Translate",
    "paraphrase": "4 · Paraphrase",
    "render": "5 · Render",
    "qa": "6 · QA",
}
STAGE_DESCRIPTIONS = {
    "import": "Extract pages, normalize sizes, generate manifest",
    "detect": "Detect speech bubbles & text regions (RT-DETR / YOLO)",
    "ocr": "Recognize Chinese text via PaddleOCR",
    "translate": "Translate dialogue to literal English (Ollama/MCP)",
    "paraphrase": "Rewrite into natural, punchy comic dialogue",
    "render": "Bubble inpainting & English typesetting",
    "qa": "Validate text coverage, sizes, & formatting",
}
PACKAGE_FORMATS = ["cbz", "zip", "tar", "pdf"]

HANDOFF_HINTS = {
    "translate": (
        "stage3_translation/translation_prompt.json",
        "stage3_translation/translation_response.json",
    ),
    "paraphrase": (
        "stage4_paraphrase/paraphrase_prompt.json",
        "stage4_paraphrase/paraphrase_response.json",
    ),
}

SHOW_KEYWORDS = (
    "warning",
    "error",
    "failed",
    "review",
    "success",
    "done:",
    "held",
    "not advancing",
    "re-run",
    "rerun",
    "handoff",
    "override",
    "glossary",
    "not translated",
    "missing",
    "giving up",
    "completion",
    "conflict",
    "skip",
    "credits",
    "stitch",
)

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #141518;
    color: #e5e7eb;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 12px;
}

QGroupBox {
    background-color: #1c1e24;
    border: 1px solid #2d3139;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
    color: #93c5fd;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 1px 6px;
    background-color: #1c1e24;
    border-radius: 3px;
    color: #93c5fd;
}

QPushButton {
    background-color: #2a2e39;
    color: #f3f4f6;
    border: 1px solid #3e4451;
    border-radius: 5px;
    padding: 4px 10px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #353b49;
    border-color: #60a5fa;
}

QPushButton:pressed {
    background-color: #1f232b;
}

QPushButton:disabled {
    background-color: #181a20;
    color: #525866;
    border-color: #242830;
}

QPushButton#primaryBtn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
    font-weight: 600;
}

QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}

QPushButton#primaryBtn:disabled {
    background-color: #1e3a8a;
    color: #64748b;
    border-color: #1e293b;
}

QPushButton#accentBtn {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
    font-weight: 600;
}

QPushButton#accentBtn:hover {
    background-color: #047857;
}

QPushButton#resumeBtn {
    background-color: #d97706;
    color: #ffffff;
    border: 1px solid #f59e0b;
    font-weight: 600;
}

QPushButton#resumeBtn:hover {
    background-color: #b45309;
}

QLineEdit, QSpinBox, QComboBox {
    background-color: #1c1e24;
    color: #f3f4f6;
    border: 1px solid #374151;
    border-radius: 5px;
    padding: 3px 8px;
    min-height: 18px;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #3b82f6;
    background-color: #20232a;
}

QComboBox::drop-down {
    border: none;
    padding-right: 6px;
}

QTreeWidget, QTableWidget, QTextEdit {
    background-color: #181a1f;
    color: #e5e7eb;
    border: 1px solid #2d3139;
    border-radius: 5px;
    gridline-color: #262a33;
}

QHeaderView::section {
    background-color: #20242c;
    color: #9ca3af;
    padding: 4px 6px;
    border: 1px solid #2d3139;
    font-weight: 600;
}

QTreeWidget::item:selected, QTableWidget::item:selected {
    background-color: #1e3a5f;
    color: #93c5fd;
}

QTreeWidget::item:hover {
    background-color: #232731;
}

QProgressBar {
    background-color: #1c1e24;
    border: 1px solid #2d3139;
    border-radius: 5px;
    text-align: center;
    color: #f9fafb;
    font-weight: 600;
    max-height: 18px;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 4px;
}

QScrollBar:vertical {
    background: #141518;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #374151;
    min-height: 16px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4b5563;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QSplitter::handle {
    background-color: #262932;
}

QSplitter::handle:hover {
    background-color: #3b82f6;
}

QTabWidget::pane {
    border: 1px solid #2d3139;
    background-color: #181a1f;
    border-radius: 5px;
}

QTabBar::tab {
    background: #1f232b;
    color: #9ca3af;
    padding: 5px 12px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #282e3a;
    color: #60a5fa;
    font-weight: 600;
}
"""


def _safe_read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _read_settings() -> dict:
    return _safe_read_json(SETTINGS)


def _write_setting(key: str, value) -> None:
    s = _read_settings()
    s[key] = value
    try:
        SETTINGS.write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _chapter_name(input_path: str) -> str:
    p = Path(input_path)
    if p.suffix.lower() in {".cbz", ".zip"}:
        return p.stem
    return p.name


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class ProcessRunner(QThread):
    line_emitted = pyqtSignal(str)
    finished_emitted = pyqtSignal(int)

    def __init__(self, cmd: list[str]):
        super().__init__()
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self._killed = False

    def run(self):
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            for line in self.proc.stdout:
                if self._killed:
                    break
                self.line_emitted.emit(line.rstrip("\n"))
            self.proc.wait()
            rc = self.proc.returncode if not self._killed else -1
            self.finished_emitted.emit(rc)
        except Exception as exc:
            self.line_emitted.emit(f"[GUI] Error launching process: {exc}")
            self.finished_emitted.emit(1)

    def kill(self):
        self._killed = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Configuration & Settings")
        self.resize(560, 440)
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Paths & General
        t1 = QWidget()
        l1 = QVBoxLayout(t1)

        gb_general = QGroupBox("Directories & Defaults")
        g1 = QGridLayout(gb_general)

        g1.addWidget(QLabel("Input Folder (Raw Scans/CBZ):"), 0, 0)
        self.txt_input = QLineEdit()
        g1.addWidget(self.txt_input, 0, 1)
        btn_br_in = QPushButton("Browse…")
        btn_br_in.clicked.connect(self._browse_input)
        g1.addWidget(btn_br_in, 0, 2)

        g1.addWidget(QLabel("Output Series Folder (Workspace):"), 1, 0)
        self.txt_output = QLineEdit()
        g1.addWidget(self.txt_output, 1, 1)
        btn_br_out = QPushButton("Browse…")
        btn_br_out.clicked.connect(self._browse_output)
        g1.addWidget(btn_br_out, 1, 2)

        g1.addWidget(QLabel("Default Package Format:"), 2, 0)
        self.cmb_pkg = QComboBox()
        self.cmb_pkg.addItems(PACKAGE_FORMATS)
        g1.addWidget(self.cmb_pkg, 2, 1)

        l1.addWidget(gb_general)
        l1.addStretch()
        tabs.addTab(t1, "Paths & General")

        # Tab 2: Translation & Paraphrase (AI)
        t2 = QWidget()
        l2 = QVBoxLayout(t2)

        gb_ollama = QGroupBox("Ollama / Local AI Settings")
        g2 = QGridLayout(gb_ollama)

        g2.addWidget(QLabel("Ollama Host URL:"), 0, 0)
        self.txt_ollama_host = QLineEdit("http://localhost:11434")
        g2.addWidget(self.txt_ollama_host, 0, 1)

        g2.addWidget(QLabel("Translate Model:"), 1, 0)
        self.txt_trans_model = QLineEdit("qwen2.5:3b-instruct")
        g2.addWidget(self.txt_trans_model, 1, 1)

        g2.addWidget(QLabel("Paraphrase Model:"), 2, 0)
        self.txt_para_model = QLineEdit("qwen2.5:3b-instruct")
        g2.addWidget(self.txt_para_model, 2, 1)

        g2.addWidget(QLabel("Paraphrase Tone:"), 3, 0)
        self.txt_para_tone = QLineEdit(
            "preserve crude/rude register; casual US English"
        )
        g2.addWidget(self.txt_para_tone, 3, 1)

        l2.addWidget(gb_ollama)
        l2.addStretch()
        tabs.addTab(t2, "AI & Ollama")

        # Tab 3: OCR & Detection
        t3 = QWidget()
        l3 = QVBoxLayout(t3)

        gb_models = QGroupBox("Detection & OCR Engines")
        g3 = QGridLayout(gb_models)

        g3.addWidget(QLabel("Detector Backend:"), 0, 0)
        self.cmb_detector = QComboBox()
        self.cmb_detector.addItems(["rtdetr", "yolov8"])
        g3.addWidget(self.cmb_detector, 0, 1)

        g3.addWidget(QLabel("Detector GPU:"), 1, 0)
        self.chk_det_gpu = QCheckBox("Enable GPU for Detection")
        self.chk_det_gpu.setChecked(True)
        g3.addWidget(self.chk_det_gpu, 1, 1)

        g3.addWidget(QLabel("OCR GPU:"), 2, 0)
        self.chk_ocr_gpu = QCheckBox("Enable GPU for PaddleOCR")
        self.chk_ocr_gpu.setChecked(True)
        g3.addWidget(self.chk_ocr_gpu, 2, 1)

        l3.addWidget(gb_models)
        l3.addStretch()
        tabs.addTab(t3, "Detection & OCR")

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save_values)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Input Folder (Raw Scans/CBZ)"
        )
        if d:
            self.txt_input.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Series Folder (Workspace)"
        )
        if d:
            self.txt_output.setText(d)

    def _load_values(self):
        settings = _read_settings()
        self.txt_input.setText(settings.get("input_dir", ""))
        self.txt_output.setText(settings.get("output_dir", ""))
        self.cmb_pkg.setCurrentText(settings.get("gui_default_package", "cbz"))
        self.txt_ollama_host.setText(
            settings.get("ollama_host", "http://localhost:11434")
        )
        self.txt_trans_model.setText(
            settings.get("ollama_translate_model", "qwen2.5:3b-instruct")
        )
        self.txt_para_model.setText(
            settings.get("ollama_para_model", "qwen2.5:3b-instruct")
        )
        self.txt_para_tone.setText(
            settings.get(
                "paraphrase_tone",
                "preserve crude/rude register; casual US English",
            )
        )
        self.cmb_detector.setCurrentText(settings.get("detector_backend", "rtdetr"))
        self.chk_det_gpu.setChecked(settings.get("detector_use_gpu", True))
        self.chk_ocr_gpu.setChecked(settings.get("ocr_use_gpu", True))

    def _save_values(self):
        _write_setting("input_dir", self.txt_input.text().strip())
        _write_setting("output_dir", self.txt_output.text().strip())
        _write_setting("gui_default_package", self.cmb_pkg.currentText())
        _write_setting("ollama_host", self.txt_ollama_host.text().strip())
        _write_setting("ollama_translate_model", self.txt_trans_model.text().strip())
        _write_setting("ollama_para_model", self.txt_para_model.text().strip())
        _write_setting("paraphrase_tone", self.txt_para_tone.text().strip())
        _write_setting("detector_backend", self.cmb_detector.currentText())
        _write_setting("detector_use_gpu", self.chk_det_gpu.isChecked())
        _write_setting("ocr_use_gpu", self.chk_ocr_gpu.isChecked())
        self.accept()


class SmoothCanvasView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setStyleSheet("background-color: #121316; border: none;")

        self._rect_items: list[QGraphicsRectItem] = []

    def set_image(self, pixmap: QPixmap):
        self.clear_overlays()
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def add_overlay_box(
        self,
        rect: QRectF,
        color: QColor = QColor(255, 60, 60, 200),
        width: int = 2,
    ):
        box_item = QGraphicsRectItem(rect)
        pen = QPen(color, width)
        box_item.setPen(pen)
        self.scene.addItem(box_item)
        self._rect_items.append(box_item)

    def clear_overlays(self):
        for item in self._rect_items:
            self.scene.removeItem(item)
        self._rect_items.clear()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(factor, factor)


class PipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manhua Translation Pipeline - Studio Runner")
        self.resize(1180, 680)
        self.setMinimumSize(960, 520)

        self.active_runner: ProcessRunner | None = None
        self.is_running = False

        self.current_input_dir: Path | None = None
        self.current_output_dir: Path | None = None
        self.current_chapter: str | None = None
        self.current_input_path: str | None = None
        self.chapter_pages: list[str] = []
        self.current_page_idx: int = 0

        self.chapter_regions: dict = {}

        self.stage_buttons: dict[str, QPushButton] = {}
        self.stage_status_labels: dict[str, QLabel] = {}

        self._build_ui()
        self._load_paths_from_settings()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---------------- Top Bar (Input & Output Paths) ----------------
        top_box = QFrame()
        top_box.setStyleSheet(
            "background-color: #1c1e24; border: 1px solid #2d3139; border-radius: 6px; padding: 2px;"
        )
        top_grid = QGridLayout(top_box)
        top_grid.setContentsMargins(6, 4, 6, 4)
        top_grid.setHorizontalSpacing(8)
        top_grid.setVerticalSpacing(4)

        # Row 0: Input Directory
        lbl_in = QLabel("Input (Raw Scans/CBZ):")
        lbl_in.setStyleSheet("font-weight: 600; color: #93c5fd;")
        top_grid.addWidget(lbl_in, 0, 0)

        self.txt_input_header = QLineEdit()
        self.txt_input_header.setPlaceholderText(
            "Select source folder containing raw chapter folders or .cbz/.zip archives..."
        )
        self.txt_input_header.setReadOnly(True)
        top_grid.addWidget(self.txt_input_header, 0, 1)

        btn_browse_in = QPushButton("Browse…")
        btn_browse_in.clicked.connect(self._on_browse_input)
        top_grid.addWidget(btn_browse_in, 0, 2)

        btn_settings = QPushButton("⚙ Settings")
        btn_settings.clicked.connect(self._open_settings)
        top_grid.addWidget(btn_settings, 0, 3)

        # Row 1: Output Directory
        lbl_out = QLabel("Output (Workspace/Series):")
        lbl_out.setStyleSheet("font-weight: 600; color: #93c5fd;")
        top_grid.addWidget(lbl_out, 1, 0)

        self.txt_output_header = QLineEdit()
        self.txt_output_header.setPlaceholderText(
            "Select output series folder where translated chapters & packages will be stored..."
        )
        self.txt_output_header.setReadOnly(True)
        top_grid.addWidget(self.txt_output_header, 1, 1)

        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._on_browse_output)
        top_grid.addWidget(btn_browse_out, 1, 2)

        main_layout.addWidget(top_box)

        # ---------------- Main Vertical Splitter ----------------
        v_splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper Horizontal Splitter (3 Panels)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel 1: Chapters & Batch
        p1 = self._build_chapters_panel()
        h_splitter.addWidget(p1)

        # Panel 2: Active Chapter & Stages
        p2 = self._build_stages_panel()
        h_splitter.addWidget(p2)

        # Panel 3: Live Preview & Region Inspector
        p3 = self._build_preview_panel()
        h_splitter.addWidget(p3)

        h_splitter.setStretchFactor(0, 2)
        h_splitter.setStretchFactor(1, 3)
        h_splitter.setStretchFactor(2, 4)

        v_splitter.addWidget(h_splitter)

        # Panel 4 (Bottom): Console & Execution Progress
        p4 = self._build_console_panel()
        v_splitter.addWidget(p4)

        v_splitter.setStretchFactor(0, 8)
        v_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(v_splitter, 1)

    # ---------------- UI Sub-Panels ----------------
    def _build_chapters_panel(self) -> QWidget:
        panel = QGroupBox("Chapters & Batch Queue")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_add_folder = QPushButton("📁 Add Folder…")
        btn_add_folder.clicked.connect(self._on_pick_folder)
        btn_row.addWidget(btn_add_folder)

        btn_add_cbz = QPushButton("📦 Add CBZ/ZIP…")
        btn_add_cbz.clicked.connect(self._on_pick_archive)
        btn_row.addWidget(btn_add_cbz)
        layout.addLayout(btn_row)

        self.tree_chapters = QTreeWidget()
        self.tree_chapters.setHeaderLabels(["Chapter", "Status", "Pages"])
        self.tree_chapters.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.tree_chapters.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree_chapters.header().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree_chapters.itemClicked.connect(self._on_chapter_item_clicked)
        layout.addWidget(self.tree_chapters, 1)

        opts_group = QFrame()
        opts_group.setStyleSheet(
            "background-color: #181a1f; border-radius: 5px; padding: 2px;"
        )
        opts_layout = QVBoxLayout(opts_group)
        opts_layout.setContentsMargins(4, 4, 4, 4)
        opts_layout.setSpacing(2)

        self.chk_fresh = QCheckBox("Fresh run (wipe prior chapter outputs)")
        opts_layout.addWidget(self.chk_fresh)

        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("Skip last:"))
        self.spin_skip_last = QSpinBox()
        self.spin_skip_last.setRange(0, 20)
        self.spin_skip_last.setValue(0)
        self.spin_skip_last.setFixedWidth(48)
        skip_row.addWidget(self.spin_skip_last)
        skip_row.addWidget(QLabel("pages (promo/credits)"))
        skip_row.addStretch()
        opts_layout.addLayout(skip_row)

        layout.addWidget(opts_group)

        action_row = QHBoxLayout()
        self.btn_run_selected = QPushButton("▶ Run Chapter")
        self.btn_run_selected.setObjectName("primaryBtn")
        self.btn_run_selected.clicked.connect(self._run_selected_chapter_auto)
        action_row.addWidget(self.btn_run_selected)

        self.btn_run_all = QPushButton("▶ Run All (Batch)")
        self.btn_run_all.setObjectName("accentBtn")
        self.btn_run_all.clicked.connect(self._run_batch_all)
        action_row.addWidget(self.btn_run_all)
        layout.addLayout(action_row)

        return panel

    def _build_stages_panel(self) -> QWidget:
        panel = QGroupBox("Active Chapter & Pipeline Stages")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)

        # Chapter info box
        info_box = QFrame()
        info_box.setStyleSheet(
            "background-color: #181a1f; border: 1px solid #282c34; border-radius: 5px; padding: 4px;"
        )
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(2)

        self.lbl_active_chapter = QLabel("No chapter selected")
        self.lbl_active_chapter.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #f3f4f6;"
        )
        info_layout.addWidget(self.lbl_active_chapter)

        self.lbl_active_details = QLabel(
            "Select or import a chapter to begin processing."
        )
        self.lbl_active_details.setStyleSheet("color: #9ca3af; font-size: 11px;")
        info_layout.addWidget(self.lbl_active_details)

        layout.addWidget(info_box)

        # Stages scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 2, 0, 2)
        scroll_layout.setSpacing(4)

        for name in STAGES:
            stage_card = QFrame()
            stage_card.setStyleSheet(
                "background-color: #181a1f; border: 1px solid #262932; border-radius: 5px; padding: 4px;"
            )
            card_layout = QHBoxLayout(stage_card)
            card_layout.setContentsMargins(6, 4, 6, 4)

            info_sub = QVBoxLayout()
            info_sub.setSpacing(1)
            title = QLabel(STAGE_LABELS[name])
            title.setStyleSheet("font-weight: 600; color: #e5e7eb;")
            info_sub.addWidget(title)

            desc = QLabel(STAGE_DESCRIPTIONS[name])
            desc.setStyleSheet("color: #6b7280; font-size: 10px;")
            info_sub.addWidget(desc)
            card_layout.addLayout(info_sub, 1)

            st_lbl = QLabel("○ Pending")
            st_lbl.setStyleSheet("color: #9ca3af; font-weight: 500;")
            st_lbl.setFixedWidth(74)
            card_layout.addWidget(st_lbl)
            self.stage_status_labels[name] = st_lbl

            btn = QPushButton("Run")
            btn.setFixedWidth(74)
            btn.clicked.connect(lambda _, n=name: self._run_single_stage(n))
            btn.setEnabled(False)
            card_layout.addWidget(btn)
            self.stage_buttons[name] = btn

            scroll_layout.addWidget(stage_card)

        # Stage 7 (Package)
        pkg_card = QFrame()
        pkg_card.setStyleSheet(
            "background-color: #181a1f; border: 1px solid #262932; border-radius: 5px; padding: 4px;"
        )
        pkg_layout = QHBoxLayout(pkg_card)
        pkg_layout.setContentsMargins(6, 4, 6, 4)

        pkg_info = QVBoxLayout()
        pkg_info.setSpacing(1)
        pkg_title = QLabel("7 · Package")
        pkg_title.setStyleSheet("font-weight: 600; color: #e5e7eb;")
        pkg_info.addWidget(pkg_title)
        pkg_desc = QLabel("Pack rendered pages into archive")
        pkg_desc.setStyleSheet("color: #6b7280; font-size: 10px;")
        pkg_info.addWidget(pkg_desc)
        pkg_layout.addLayout(pkg_info, 1)

        self.cmb_pkg_format = QComboBox()
        self.cmb_pkg_format.addItems(PACKAGE_FORMATS)
        self.cmb_pkg_format.setFixedWidth(58)
        pkg_layout.addWidget(self.cmb_pkg_format)

        self.btn_package = QPushButton("Package")
        self.btn_package.setFixedWidth(74)
        self.btn_package.clicked.connect(self._run_package_stage)
        self.btn_package.setEnabled(False)
        pkg_layout.addWidget(self.btn_package)

        scroll_layout.addWidget(pkg_card)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Stage Action Bar
        action_bar = QHBoxLayout()
        self.btn_open_touchup = QPushButton("🎨 Open Chapter in Touch-Up")
        self.btn_open_touchup.clicked.connect(self._launch_touchup_chapter)
        action_bar.addWidget(self.btn_open_touchup)

        layout.addLayout(action_bar)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QGroupBox("Live Preview & Region Inspector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)

        # Top Bar: Mode selector & page navigation
        nav_bar = QHBoxLayout()
        nav_bar.addWidget(QLabel("Mode:"))

        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(
            [
                "Rendered Page (Stage 5)",
                "Original Page (Stage 0)",
                "Detection Overlays (Stage 1)",
                "Side-by-Side Split",
            ]
        )
        self.cmb_preview_mode.currentIndexChanged.connect(self._update_preview_image)
        nav_bar.addWidget(self.cmb_preview_mode, 1)

        self.btn_prev_page = QPushButton("◀")
        self.btn_prev_page.setFixedWidth(28)
        self.btn_prev_page.clicked.connect(self._prev_page)
        nav_bar.addWidget(self.btn_prev_page)

        self.lbl_page_counter = QLabel("Page 0 / 0")
        self.lbl_page_counter.setStyleSheet("font-weight: 600; color: #93c5fd;")
        nav_bar.addWidget(self.lbl_page_counter)

        self.btn_next_page = QPushButton("▶")
        self.btn_next_page.setFixedWidth(28)
        self.btn_next_page.clicked.connect(self._next_page)
        nav_bar.addWidget(self.btn_next_page)

        layout.addLayout(nav_bar)

        # Image Canvas Splitter (for side-by-side or single view)
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.canvas_left = SmoothCanvasView()
        self.preview_splitter.addWidget(self.canvas_left)

        self.canvas_right = SmoothCanvasView()
        self.preview_splitter.addWidget(self.canvas_right)
        self.canvas_right.setVisible(False)

        layout.addWidget(self.preview_splitter, 3)

        # Page slider
        self.page_slider = QSlider(Qt.Orientation.Horizontal)
        self.page_slider.setRange(0, 0)
        self.page_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.page_slider)

        # Regions Table / Inspector
        self.tbl_regions = QTableWidget(0, 4)
        self.tbl_regions.setHorizontalHeaderLabels(
            ["Region", "Original (OCR)", "Translation", "Paraphrase"]
        )
        self.tbl_regions.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tbl_regions.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.tbl_regions.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.tbl_regions.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.tbl_regions.setMaximumHeight(130)
        layout.addWidget(self.tbl_regions, 1)

        # Quick edit button
        self.btn_touchup_page = QPushButton("🎨 Edit Active Page in Touch-Up")
        self.btn_touchup_page.clicked.connect(self._launch_touchup_page)
        layout.addWidget(self.btn_touchup_page)

        return panel

    def _build_console_panel(self) -> QWidget:
        panel = QGroupBox("Console & Event Stream")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)

        # Toolbar
        bar = QHBoxLayout()
        self.chk_filter_important = QCheckBox(
            "Filter meaningful only (warnings, overrides, verdicts)"
        )
        self.chk_filter_important.setChecked(True)
        bar.addWidget(self.chk_filter_important)

        bar.addStretch()

        btn_clear = QPushButton("Clear Console")
        btn_clear.clicked.connect(self._clear_console)
        bar.addWidget(btn_clear)

        layout.addLayout(bar)

        # Output text
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setFont(QFont("Consolas", 9))
        self.txt_console.setStyleSheet(
            "background-color: #0f1012; color: #d1d5db; border: 1px solid #23262d; border-radius: 5px; padding: 4px;"
        )
        layout.addWidget(self.txt_console, 1)

        # Status & Progress Footer
        footer = QHBoxLayout()
        self.lbl_status = QLabel("● Idle")
        self.lbl_status.setStyleSheet("font-weight: 600; color: #10b981;")
        footer.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        footer.addWidget(self.progress_bar, 1)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_running_process)
        footer.addWidget(self.btn_stop)

        layout.addLayout(footer)
        return panel

    # ---------------- Paths & Settings Loading ----------------
    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._load_paths_from_settings()

    def _on_browse_input(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Input Folder (Raw Scans/CBZ)"
        )
        if d:
            _write_setting("input_dir", d)
            self._load_paths_from_settings()

    def _on_browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Series Folder (Workspace)"
        )
        if d:
            _write_setting("output_dir", d)
            self._load_paths_from_settings()

    def _load_paths_from_settings(self):
        settings = _read_settings()
        in_path = settings.get("input_dir", "")
        out_path = settings.get("output_dir", "")

        self.txt_input_header.setText(in_path)
        self.txt_output_header.setText(out_path)

        self.current_input_dir = (
            Path(in_path) if in_path and Path(in_path).exists() else None
        )
        self.current_output_dir = (
            Path(out_path) if out_path and Path(out_path).exists() else None
        )

        self._scan_chapters()

    def _collect_from_dir(
        self, folder: Path | None, seen: set[str], entries: list[tuple[str, str]]
    ):
        if not folder or not folder.exists():
            return
        for child in sorted(folder.iterdir()):
            if child.name.startswith("."):
                continue
            is_valid = child.is_dir() or child.suffix.lower() in {".cbz", ".zip"}
            if not is_valid:
                continue
            ch_name = _chapter_name(str(child))
            if ch_name not in seen:
                seen.add(ch_name)
                entries.append((ch_name, str(child)))

    def _get_chapter_status(self, ch_name: str) -> tuple[str, str]:
        if not self.current_output_dir:
            return "○ Raw", "-"
        manifest_file = self.current_output_dir / ch_name / "manifest.json"
        if not manifest_file.exists():
            return "○ Raw", "-"
        try:
            data = _safe_read_json(manifest_file)
            curr = data.get("current_stage", "pending")
            page_count = str(data.get("total_pages", len(data.get("pages", []))))
            if curr in {"complete", "package"}:
                return "✓ Complete", page_count
            if curr == "qa":
                return "⚠ QA Review", page_count
            return f"● {curr.capitalize()}", page_count
        except Exception:
            return "○ Raw", "-"

    def _scan_chapters(self):
        self.tree_chapters.clear()
        seen_names: set[str] = set()
        chapter_entries: list[tuple[str, str]] = []

        self._collect_from_dir(self.current_input_dir, seen_names, chapter_entries)
        self._collect_from_dir(self.current_output_dir, seen_names, chapter_entries)

        for ch_name, src_path in chapter_entries:
            status, page_count = self._get_chapter_status(ch_name)
            item = QTreeWidgetItem([ch_name, status, page_count])
            item.setData(0, Qt.ItemDataRole.UserRole, src_path)
            self.tree_chapters.addTopLevelItem(item)

    # ---------------- Pickers & Chapter Selection ----------------
    def _on_pick_folder(self):
        if self.is_running:
            QMessageBox.information(
                self, "Busy", "A pipeline process is still running."
            )
            return
        d = QFileDialog.getExistingDirectory(self, "Select Chapter Folder (Images)")
        if d:
            self._select_input_path(d)

    def _on_pick_archive(self):
        if self.is_running:
            QMessageBox.information(
                self, "Busy", "A pipeline process is still running."
            )
            return
        f, _ = QFileDialog.getOpenFileName(
            self, "Select CBZ/ZIP Archive", "", "Comic Archives (*.cbz *.zip)"
        )
        if f:
            self._select_input_path(f)

    def _on_chapter_item_clicked(self, item: QTreeWidgetItem, column: int):
        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if path_str:
            self._select_input_path(path_str, auto_import=False)

    def _select_input_path(self, path_str: str, auto_import: bool = False):
        if not self.current_output_dir:
            QMessageBox.warning(
                self,
                "Output Series Folder",
                "Please select an Output Series Folder (Workspace) first.",
            )
            return

        self.current_input_path = path_str
        self.current_chapter = _chapter_name(path_str)

        self.lbl_active_chapter.setText(f"Chapter: {self.current_chapter}")
        self.lbl_active_details.setText(f"Source: {Path(path_str).name}")

        self._refresh_stage_states()
        self._load_chapter_pages_for_preview()

        if auto_import:
            self._run_single_stage("import")

    def _get_active_chapter_dir(self) -> Path | None:
        if not self.current_output_dir or not self.current_chapter:
            return None
        return self.current_output_dir / self.current_chapter

    def _manifest_current_stage(self) -> str | None:
        ch_dir = self._get_active_chapter_dir()
        if not ch_dir:
            return None
        mpath = ch_dir / "manifest.json"
        if mpath.exists():
            try:
                return json.loads(mpath.read_text(encoding="utf-8")).get(
                    "current_stage"
                )
            except Exception:
                return None
        return None

    def _refresh_stage_states(self):
        curr_stage = self._manifest_current_stage()

        # Reset all stage buttons
        for name in STAGES:
            self.stage_buttons[name].setEnabled(False)
            self.stage_buttons[name].setText("Run")
            self.stage_buttons[name].setObjectName("")
            self.stage_status_labels[name].setText("○ Pending")
            self.stage_status_labels[name].setStyleSheet("color: #9ca3af;")

        self.btn_package.setEnabled(False)

        # Import is always runnable for selected input
        self.stage_buttons["import"].setEnabled(not self.is_running)

        if not curr_stage:
            return

        # Unlock completed / current stages
        stage_order = STAGES
        if curr_stage in stage_order:
            curr_idx = stage_order.index(curr_stage)
            for i in range(curr_idx):
                sname = stage_order[i]
                self.stage_buttons[sname].setEnabled(not self.is_running)
                self.stage_buttons[sname].setText("Re-run")
                self.stage_status_labels[sname].setText("✓ Done")
                self.stage_status_labels[sname].setStyleSheet(
                    "color: #10b981; font-weight: 600;"
                )

            # Active stage
            act_name = stage_order[curr_idx]
            self.stage_buttons[act_name].setEnabled(not self.is_running)
            self.stage_buttons[act_name].setObjectName("primaryBtn")

            # Check if handoff pending
            if act_name in HANDOFF_HINTS and self._is_handoff_pending(act_name):
                self.stage_buttons[act_name].setText("Resume ▶")
                self.stage_buttons[act_name].setObjectName("resumeBtn")
                self.stage_status_labels[act_name].setText("Waiting MCP")
                self.stage_status_labels[act_name].setStyleSheet(
                    "color: #f59e0b; font-weight: 600;"
                )
            else:
                self.stage_buttons[act_name].setText("Run")
                self.stage_status_labels[act_name].setText("● Ready")
                self.stage_status_labels[act_name].setStyleSheet(
                    "color: #60a5fa; font-weight: 600;"
                )

            if curr_stage == "qa" or curr_idx >= stage_order.index("render"):
                self.btn_package.setEnabled(not self.is_running)
        elif curr_stage in {"complete", "package"}:
            for sname in stage_order:
                self.stage_buttons[sname].setEnabled(not self.is_running)
                self.stage_buttons[sname].setText("Re-run")
                self.stage_status_labels[sname].setText("✓ Done")
                self.stage_status_labels[sname].setStyleSheet(
                    "color: #10b981; font-weight: 600;"
                )
            self.btn_package.setEnabled(not self.is_running)

        self._reapply_styles()

    def _is_handoff_pending(self, stage_name: str) -> bool:
        ch_dir = self._get_active_chapter_dir()
        if not ch_dir or stage_name not in HANDOFF_HINTS:
            return False
        prompt_rel, resp_rel = HANDOFF_HINTS[stage_name]
        p_path = ch_dir / prompt_rel
        r_path = ch_dir / resp_rel
        return p_path.exists() and not r_path.exists()

    def _reapply_styles(self):
        for btn in self.stage_buttons.values():
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ---------------- Process Execution Plumbing ----------------
    def _run_cmd(self, cmd: list[str], on_done=None):
        if self.is_running:
            return
        self.is_running = True
        self.btn_stop.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.lbl_status.setText("● Processing...")
        self.lbl_status.setStyleSheet("color: #60a5fa; font-weight: 600;")
        self._refresh_stage_states()

        self._log_sys("$ " + " ".join(Path(c).name if c == cmd[1] else c for c in cmd))

        self.active_runner = ProcessRunner(cmd)
        self.active_runner.line_emitted.connect(self._handle_process_line)
        self.active_runner.finished_emitted.connect(
            lambda rc: self._handle_process_finished(rc, on_done)
        )
        self.active_runner.start()

    def _handle_process_line(self, line: str):
        m = re.search(r"Page\s+(\d+)/(\d+)", line, re.IGNORECASE)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            self.progress_bar.setRange(0, tot)
            self.progress_bar.setValue(cur)

        if not self.chk_filter_important.isChecked():
            self._append_console(line)
            return

        low = line.lower()
        if any(k in low for k in SHOW_KEYWORDS):
            tag = (
                "err"
                if any(x in low for x in ["error", "failed", "giving up"])
                else "warn"
                if any(x in low for x in ["warning", "held", "missing", "conflict"])
                else "ok"
                if any(x in low for x in ["success", "done:"])
                else "sys"
            )
            self._append_console(line, tag)

    def _handle_process_finished(self, rc: int, on_done=None):
        self.is_running = False
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if rc == 0 else 0)

        if rc == 0:
            self.lbl_status.setText("● Complete")
            self.lbl_status.setStyleSheet("color: #10b981; font-weight: 600;")
        else:
            self.lbl_status.setText("● Failed / Interrupted")
            self.lbl_status.setStyleSheet("color: #ef4444; font-weight: 600;")

        self._scan_chapters()
        self._refresh_stage_states()
        self._load_chapter_pages_for_preview()

        if on_done:
            on_done(rc)

    def _stop_running_process(self):
        if self.active_runner and self.is_running:
            self.active_runner.kill()
            self._log_sys("Process cancelled by user.", "err")

    # ---------------- Stage Actions ----------------
    def _run_single_stage(self, name: str):
        if not self.current_chapter:
            return

        cmd = [sys.executable, str(PIPELINE)]
        if name == "import":
            if not self.current_input_path:
                return
            cmd += ["import", "--input", self.current_input_path]
            if self.chk_fresh.isChecked():
                cmd += ["--fresh"]
            if self.spin_skip_last.value() > 0:
                cmd += ["--skip-last", str(self.spin_skip_last.value())]
        else:
            cmd += [name, "--chapter", self.current_chapter]

        self._run_cmd(
            cmd,
            on_done=lambda rc: self._log_sys(
                f"{STAGE_LABELS.get(name, name)} finished (code {rc}).",
                "ok" if rc == 0 else "err",
            ),
        )

    def _run_package_stage(self):
        if not self.current_chapter:
            return
        fmt = self.cmb_pkg_format.currentText()
        _write_setting("gui_default_package", fmt)
        cmd = [
            sys.executable,
            str(PIPELINE),
            "package",
            "--chapter",
            self.current_chapter,
            "--package",
            fmt,
        ]
        self._run_cmd(
            cmd,
            on_done=lambda rc: self._log_sys(
                f"Package finished ({fmt}).", "ok" if rc == 0 else "err"
            ),
        )

    def _run_selected_chapter_auto(self):
        if not self.current_chapter or not self.current_input_path:
            QMessageBox.warning(
                self, "Select Chapter", "Please select a chapter first."
            )
            return

        cmd = [
            sys.executable,
            str(PIPELINE),
            self.current_chapter,
            "--all",
            "--input",
            self.current_input_path,
        ]
        if self.chk_fresh.isChecked():
            cmd += ["--fresh"]
        if self.spin_skip_last.value() > 0:
            cmd += ["--skip-last", str(self.spin_skip_last.value())]

        self._run_cmd(
            cmd,
            on_done=lambda rc: self._log_sys(
                f"Chapter {self.current_chapter} pipeline auto-run complete.",
                "ok" if rc == 0 else "err",
            ),
        )

    def _run_batch_all(self):
        target_dir = self.current_input_dir or self.current_output_dir
        if not target_dir:
            QMessageBox.warning(
                self,
                "No Directory",
                "Please select an Input Folder or Output Series Folder first.",
            )
            return

        if (
            QMessageBox.question(
                self,
                "Run Batch",
                f"Run batch processing for all chapters in {target_dir.name}?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        cmd = [
            sys.executable,
            str(PIPELINE),
            "batch",
            str(target_dir),
        ]
        if self.chk_fresh.isChecked():
            cmd += ["--fresh"]
        if self.spin_skip_last.value() > 0:
            cmd += ["--skip-last", str(self.spin_skip_last.value())]

        self._run_cmd(
            cmd,
            on_done=lambda rc: self._log_sys(
                "Batch run finished.", "ok" if rc == 0 else "err"
            ),
        )

    # ---------------- Touch-Up Integration ----------------
    def _launch_touchup_chapter(self):
        if not self.current_chapter:
            return
        ch_dir = self._get_active_chapter_dir()
        if not ch_dir:
            return
        target_dir = ch_dir / "pages"
        first_img = None
        if target_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                imgs = sorted(target_dir.glob(f"*{ext}"))
                if imgs:
                    first_img = imgs[0]
                    break
        self._spawn_touchup(first_img)

    def _launch_touchup_page(self):
        if not self.chapter_pages or self.current_page_idx >= len(self.chapter_pages):
            return
        img_name = self.chapter_pages[self.current_page_idx]
        ch_dir = self._get_active_chapter_dir()
        if not ch_dir:
            return
        page_path = ch_dir / "pages" / img_name
        self._spawn_touchup(page_path if page_path.exists() else None)

    def _spawn_touchup(self, img_path: Path | None):
        cmd = [sys.executable, str(TOUCHUP_APP)]
        if img_path and img_path.exists():
            cmd += ["--image", str(img_path)]
        try:
            subprocess.Popen(cmd, cwd=str(ROOT))
            self._log_sys(f"Launched Touch-Up Editor ({img_path or 'Empty'}).", "sys")
        except Exception as exc:
            self._log_sys(f"Failed to launch Touch-Up: {exc}", "err")

    # ---------------- Preview & Inspector ----------------
    def _load_chapter_pages_for_preview(self):
        ch_dir = self._get_active_chapter_dir()
        if not ch_dir:
            self.chapter_pages = []
            self.current_page_idx = 0
            self._update_preview_image()
            return

        pages_dir = ch_dir / "pages"
        if pages_dir.exists():
            exts = {".png", ".jpg", ".jpeg", ".webp"}
            self.chapter_pages = sorted(
                [f.name for f in pages_dir.iterdir() if f.suffix.lower() in exts]
            )
        else:
            self.chapter_pages = []

        self._load_regions_data(ch_dir)

        tot = len(self.chapter_pages)
        self.page_slider.setRange(1 if tot > 0 else 0, tot)
        self.current_page_idx = 0
        if tot > 0:
            self.page_slider.setValue(1)
        self._update_preview_image()

    def _load_regions_data(self, ch_dir: Path):
        self.chapter_regions.clear()
        det_data = _safe_read_json(ch_dir / "stage1_detection" / "detection.json")
        ocr_data = _safe_read_json(ch_dir / "stage2_ocr" / "ocr_results.json")
        trans_map = _safe_read_json(
            ch_dir / "stage3_translation" / "translation_response.json"
        )
        para_map = _safe_read_json(
            ch_dir / "stage4_paraphrase" / "paraphrase_response.json"
        )

        det_map = {
            r["region_id"]: r
            for r in det_data.get("regions", [])
            if isinstance(r, dict) and "region_id" in r
        }
        ocr_map = {
            r["region_id"]: r.get("original_text", "")
            for r in ocr_data.get("results", [])
            if isinstance(r, dict) and "region_id" in r
        }

        for rid, rdata in det_map.items():
            pnum = rdata.get("page_number", 1)
            self.chapter_regions.setdefault(pnum, []).append(
                {
                    "region_id": rid,
                    "bbox": rdata.get("bbox", {}),
                    "ocr": ocr_map.get(rid, ""),
                    "trans": trans_map.get(rid, ""),
                    "para": para_map.get(rid, ""),
                }
            )

    def _on_slider_changed(self, val: int):
        if val > 0 and val <= len(self.chapter_pages):
            self.current_page_idx = val - 1
            self._update_preview_image()

    def _prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.page_slider.setValue(self.current_page_idx + 1)
            self._update_preview_image()

    def _next_page(self):
        if self.current_page_idx < len(self.chapter_pages) - 1:
            self.current_page_idx += 1
            self.page_slider.setValue(self.current_page_idx + 1)
            self._update_preview_image()

    def _update_preview_image(self):
        ch_dir = self._get_active_chapter_dir()
        tot = len(self.chapter_pages)

        if not ch_dir or tot == 0 or self.current_page_idx >= tot:
            self.lbl_page_counter.setText("Page 0 / 0")
            self.tbl_regions.setRowCount(0)
            return

        page_name = self.chapter_pages[self.current_page_idx]
        self.lbl_page_counter.setText(
            f"Page {self.current_page_idx + 1} / {tot}  ({page_name})"
        )

        mode = self.cmb_preview_mode.currentText()
        is_split = "Side-by-Side" in mode
        self.canvas_right.setVisible(is_split)

        raw_path = ch_dir / "pages" / page_name
        render_path = ch_dir / "stage5_render" / "pages" / page_name
        overlay_path = ch_dir / "stage1_detection" / "overlays" / page_name

        if is_split:
            img_left = raw_path if raw_path.exists() else None
            img_right = render_path if render_path.exists() else raw_path
            self._display_on_canvas(self.canvas_left, img_left)
            self._display_on_canvas(self.canvas_right, img_right)
        else:
            target = (
                render_path
                if "Rendered" in mode and render_path.exists()
                else overlay_path
                if "Detection" in mode and overlay_path.exists()
                else raw_path
            )
            self._display_on_canvas(self.canvas_left, target)

        self._populate_region_table(self.current_page_idx + 1)

    def _display_on_canvas(self, canvas: SmoothCanvasView, img_path: Path | None):
        if img_path and img_path.exists():
            try:
                pil_img = Image.open(img_path)
                canvas.set_image(pil_to_qpixmap(pil_img))
            except Exception:
                pass

    def _populate_region_table(self, page_num: int):
        self.tbl_regions.setRowCount(0)
        regs = self.chapter_regions.get(page_num, [])
        self.tbl_regions.setRowCount(len(regs))
        for row, r in enumerate(regs):
            self.tbl_regions.setItem(row, 0, QTableWidgetItem(r["region_id"]))
            self.tbl_regions.setItem(row, 1, QTableWidgetItem(r["ocr"]))
            self.tbl_regions.setItem(row, 2, QTableWidgetItem(r["trans"]))
            self.tbl_regions.setItem(row, 3, QTableWidgetItem(r["para"]))

    # ---------------- Console Helpers ----------------
    def _append_console(self, text: str, tag: str | None = None):
        color = (
            "#ef4444"
            if tag == "err"
            else "#f59e0b"
            if tag == "warn"
            else "#10b981"
            if tag == "ok"
            else "#60a5fa"
            if tag == "sys"
            else "#e5e7eb"
        )
        html = f"<span style='color:{color}; font-family:Consolas;'>{text}</span><br>"
        self.txt_console.insertHtml(html)
        self.txt_console.ensureCursorVisible()

    def _log_sys(self, msg: str, tag: str = "sys"):
        self._append_console(f"» {msg}", tag)

    def _clear_console(self):
        self.txt_console.clear()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = PipelineGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
