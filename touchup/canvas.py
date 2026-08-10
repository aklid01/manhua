from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QTextEdit,
)


class SelectionOverlay(QGraphicsRectItem):
    """Temporary rubberband box during region selection."""

    def __init__(self):
        super().__init__()
        self.setPen(QPen(QColor(0, 162, 255, 220), 2, Qt.PenStyle.DashLine))
        self.setBrush(QBrush(QColor(0, 162, 255, 40)))
        self.setZValue(100)


class DraggableTextItem(QGraphicsObject):
    """Interactive, draggable, and resizable text box with complete per-line styling."""

    changed = pyqtSignal()

    HANDLE_NONE = 0
    HANDLE_TL = 1
    HANDLE_TR = 2
    HANDLE_BL = 3
    HANDLE_BR = 4
    HANDLE_T = 5
    HANDLE_B = 6
    HANDLE_L = 7
    HANDLE_R = 8

    def __init__(self, text: str, rect: QRectF, parent=None):
        super().__init__(parent)
        self._rect = rect
        self._font_family = "Segoe UI"
        self._font_size = 20
        self._bold = True
        self._italic = False
        self._text_color = QColor(255, 255, 255)
        self._stroke_color = QColor(0, 0, 0)
        self._stroke_width = 2
        self._align = Qt.AlignmentFlag.AlignLeft
        self._direction = "horizontal"

        # Per-line style data: list of dicts
        self._lines_data: list[dict] = []
        self.set_text(text)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._active_handle = self.HANDLE_NONE
        self._handle_size = 10.0
        self._initial_rect = QRectF()
        self._initial_pos = QPointF()
        self._initial_font_size = 20
        self._mouse_press_pos = QPointF()

        self.setZValue(50)

    # ---------------- Bounding Rect & Shape ----------------
    def boundingRect(self) -> QRectF:
        margin = max(self._stroke_width, self._handle_size) + 8
        return self._rect.adjusted(-margin, -margin, margin, margin)

    # ---------------- Text & Per-Line Properties ----------------
    @property
    def raw_text(self) -> str:
        return "\n".join(line["text"] for line in self._lines_data)

    def set_text(self, text: str):
        paragraphs = text.split("\n")
        new_lines = []
        for i, p in enumerate(paragraphs):
            if i < len(self._lines_data):
                old = self._lines_data[i]
                new_lines.append(
                    {
                        "text": p,
                        "font_family": old.get("font_family", self._font_family),
                        "font_size": old.get("font_size", self._font_size),
                        "bold": old.get("bold", self._bold),
                        "italic": old.get("italic", self._italic),
                        "color": old.get("color", self._text_color),
                        "stroke_color": old.get("stroke_color", self._stroke_color),
                        "stroke_width": old.get("stroke_width", self._stroke_width),
                        "align": old.get("align", self._align),
                        "direction": old.get("direction", self._direction),
                    }
                )
            else:
                new_lines.append(
                    {
                        "text": p,
                        "font_family": self._font_family,
                        "font_size": self._font_size,
                        "bold": self._bold,
                        "italic": self._italic,
                        "color": self._text_color,
                        "stroke_color": self._stroke_color,
                        "stroke_width": self._stroke_width,
                        "align": self._align,
                        "direction": self._direction,
                    }
                )
        self._lines_data = new_lines
        self._auto_expand_height()
        self.prepareGeometryChange()
        self.update()
        self.changed.emit()

    def _auto_expand_height(self):
        """Ensures the bounding box expands to fit the text when font size increases."""
        wrapped = self._compute_wrapped_lines()
        needed_h = sum(fm.lineSpacing() for _, _, _, fm in wrapped) + 16.0
        if needed_h > self._rect.height():
            self.prepareGeometryChange()
            self._rect = QRectF(
                self._rect.x(), self._rect.y(), self._rect.width(), needed_h
            )
            self.update()

    def _apply_style_dict(self, target: dict, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                if k == "font_size":
                    target[k] = max(6, v)
                elif k == "stroke_width":
                    target[k] = max(0, v)
                else:
                    target[k] = v

    def _apply_global_style_kw(self, **kw):
        mapping = {
            "font_family": "_font_family",
            "bold": "_bold",
            "italic": "_italic",
            "color": "_text_color",
            "stroke_color": "_stroke_color",
            "align": "_align",
            "direction": "_direction",
        }
        for k, attr in mapping.items():
            if kw.get(k) is not None:
                setattr(self, attr, kw[k])
        if kw.get("font_size") is not None:
            self._font_size = max(6, kw["font_size"])
        if kw.get("stroke_width") is not None:
            self._stroke_width = max(0, kw["stroke_width"])

    def set_line_style(
        self,
        line_idx: int,
        family: str | None = None,
        size: int | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
        color: QColor | None = None,
        stroke_color: QColor | None = None,
        stroke_width: int | None = None,
        align: Qt.AlignmentFlag | None = None,
        direction: str | None = None,
    ):
        """Applies style to a specific line index, or to all lines if line_idx == -1."""
        kw = {
            "font_family": family,
            "font_size": size,
            "bold": bold,
            "italic": italic,
            "color": color,
            "stroke_color": stroke_color,
            "stroke_width": stroke_width,
            "align": align,
            "direction": direction,
        }
        target_indices = (
            range(len(self._lines_data))
            if line_idx < 0
            else [line_idx]
            if 0 <= line_idx < len(self._lines_data)
            else []
        )

        for idx in target_indices:
            self._apply_style_dict(self._lines_data[idx], **kw)

        if line_idx < 0:
            self._apply_global_style_kw(**kw)

        self._auto_expand_height()
        self.prepareGeometryChange()
        self.update()
        self.changed.emit()

    def set_colors(self, text_color: QColor, stroke_color: QColor, stroke_width: int):
        self._text_color = text_color
        self._stroke_color = stroke_color
        self._stroke_width = max(0, stroke_width)
        for ld in self._lines_data:
            ld["color"] = text_color
            ld["stroke_color"] = stroke_color
            ld["stroke_width"] = max(0, stroke_width)
        self.update()
        self.changed.emit()

    def set_alignment(self, align: Qt.AlignmentFlag):
        self._align = align
        for ld in self._lines_data:
            ld["align"] = align
        self.update()
        self.changed.emit()

    def set_direction(self, direction: str):
        self._direction = direction
        for ld in self._lines_data:
            ld["direction"] = direction
        self.prepareGeometryChange()
        self.update()
        self.changed.emit()

    def get_state(self) -> dict:
        pos = self.pos()
        serialized_lines = []
        for ld in self._lines_data:
            c = ld.get("color", self._text_color)
            sc = ld.get("stroke_color", self._stroke_color)
            al = ld.get("align", self._align)
            serialized_lines.append(
                {
                    "text": ld.get("text", ""),
                    "font_family": ld.get("font_family", self._font_family),
                    "font_size": ld.get("font_size", self._font_size),
                    "bold": ld.get("bold", self._bold),
                    "italic": ld.get("italic", self._italic),
                    "color": [c.red(), c.green(), c.blue()],
                    "stroke_color": [sc.red(), sc.green(), sc.blue()],
                    "stroke_width": ld.get("stroke_width", self._stroke_width),
                    "direction": ld.get("direction", self._direction),
                    "align": "center"
                    if al == Qt.AlignmentFlag.AlignCenter
                    else "right"
                    if al == Qt.AlignmentFlag.AlignRight
                    else "left",
                }
            )

        return {
            "text": self.raw_text,
            "lines": serialized_lines,
            "x": int(round(pos.x() + self._rect.x())),
            "y": int(round(pos.y() + self._rect.y())),
            "w": int(round(self._rect.width())),
            "h": int(round(self._rect.height())),
            "rotation": int(round(self.rotation())),
            "font_family": self._font_family,
            "font_size": self._font_size,
            "bold": self._bold,
            "italic": self._italic,
            "text_color": [
                self._text_color.red(),
                self._text_color.green(),
                self._text_color.blue(),
            ],
            "stroke_color": [
                self._stroke_color.red(),
                self._stroke_color.green(),
                self._stroke_color.blue(),
            ],
            "stroke_width": self._stroke_width,
            "direction": self._direction,
            "align": "center"
            if self._align == Qt.AlignmentFlag.AlignCenter
            else "right"
            if self._align == Qt.AlignmentFlag.AlignRight
            else "left",
        }

    # ---------------- Handles & Interaction ----------------
    def _handle_rects(self) -> dict[int, QRectF]:
        hs = self._handle_size
        r = self._rect
        cx = r.center().x()
        cy = r.center().y()
        return {
            self.HANDLE_TL: QRectF(r.left() - hs / 2, r.top() - hs / 2, hs, hs),
            self.HANDLE_TR: QRectF(r.right() - hs / 2, r.top() - hs / 2, hs, hs),
            self.HANDLE_BL: QRectF(r.left() - hs / 2, r.bottom() - hs / 2, hs, hs),
            self.HANDLE_BR: QRectF(r.right() - hs / 2, r.bottom() - hs / 2, hs, hs),
            self.HANDLE_T: QRectF(cx - hs / 2, r.top() - hs / 2, hs, hs),
            self.HANDLE_B: QRectF(cx - hs / 2, r.bottom() - hs / 2, hs, hs),
            self.HANDLE_L: QRectF(r.left() - hs / 2, cy - hs / 2, hs, hs),
            self.HANDLE_R: QRectF(r.right() - hs / 2, cy - hs / 2, hs, hs),
        }

    def _hit_test_handle(self, pt: QPointF) -> int:
        for hid, hrect in self._handle_rects().items():
            if hrect.contains(pt):
                return hid
        return self.HANDLE_NONE

    def hoverMoveEvent(self, event):
        if not self.isSelected():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            super().hoverMoveEvent(event)
            return

        hid = self._hit_test_handle(event.pos())
        if hid in (self.HANDLE_TL, self.HANDLE_BR):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif hid in (self.HANDLE_TR, self.HANDLE_BL):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif hid in (self.HANDLE_T, self.HANDLE_B):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif hid in (self.HANDLE_L, self.HANDLE_R):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            hid = self._hit_test_handle(event.pos())
            if hid != self.HANDLE_NONE:
                self._active_handle = hid
                self._initial_rect = QRectF(self._rect)
                self._initial_pos = self.pos()
                self._initial_font_size = self._font_size
                self._mouse_press_pos = event.pos()
                event.accept()
                return

        super().mousePressEvent(event)

    def _calc_handle_geometry(self, delta: QPointF) -> QRectF:
        r = QRectF(self._initial_rect)
        min_w, min_h = 30.0, 20.0
        h = self._active_handle

        if h == self.HANDLE_BR:
            r.setWidth(max(min_w, r.width() + delta.x()))
            r.setHeight(max(min_h, r.height() + delta.y()))
        elif h == self.HANDLE_BL:
            new_w = max(min_w, r.width() - delta.x())
            r.setLeft(r.right() - new_w)
            r.setHeight(max(min_h, r.height() + delta.y()))
        elif h == self.HANDLE_TR:
            new_h = max(min_h, r.height() - delta.y())
            r.setWidth(max(min_w, r.width() + delta.x()))
            r.setTop(r.bottom() - new_h)
        elif h == self.HANDLE_TL:
            new_w = max(min_w, r.width() - delta.x())
            new_h = max(min_h, r.height() - delta.y())
            r.setLeft(r.right() - new_w)
            r.setTop(r.bottom() - new_h)
        elif h == self.HANDLE_R:
            r.setWidth(max(min_w, r.width() + delta.x()))
        elif h == self.HANDLE_L:
            r.setLeft(r.right() - max(min_w, r.width() - delta.x()))
        elif h == self.HANDLE_B:
            r.setHeight(max(min_h, r.height() + delta.y()))
        elif h == self.HANDLE_T:
            r.setTop(r.bottom() - max(min_h, r.height() - delta.y()))
        return r

    def mouseMoveEvent(self, event):
        if self._active_handle != self.HANDLE_NONE:
            delta = event.pos() - self._mouse_press_pos
            r = self._calc_handle_geometry(delta)

            corners = (
                self.HANDLE_TL,
                self.HANDLE_TR,
                self.HANDLE_BL,
                self.HANDLE_BR,
            )
            if self._active_handle in corners:
                scale = r.height() / max(1.0, self._initial_rect.height())
                scaled_font = int(round(self._initial_font_size * scale))
                new_size = max(8, min(120, scaled_font))
                ratio = new_size / max(1, self._font_size)
                self._font_size = new_size
                for ld in self._lines_data:
                    ld["font_size"] = max(
                        6, min(120, int(round(ld["font_size"] * ratio)))
                    )

            self.prepareGeometryChange()
            self._rect = r
            self.update()
            self.changed.emit()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._active_handle != self.HANDLE_NONE:
            self._active_handle = self.HANDLE_NONE
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.changed.emit()
        return super().itemChange(change, value)

    def _compute_wrapped_lines(self) -> list[tuple[str, dict, QFont, QFontMetricsF]]:
        """Returns list of (sub_line_text, line_dict, font, font_metrics)."""
        lines = []
        avail_w = max(10.0, self._rect.width())

        for ld in self._lines_data:
            text = ld.get("text", "")
            font = QFont(
                ld.get("font_family", self._font_family),
                ld.get("font_size", self._font_size),
            )
            font.setBold(ld.get("bold", self._bold))
            font.setItalic(ld.get("italic", self._italic))
            fm = QFontMetricsF(font)
            line_dir = ld.get("direction", self._direction)

            if line_dir == "vertical":
                for char in text:
                    lines.append((char, ld, font, fm))
                lines.append(("", ld, font, fm))
                continue

            words = text.split(" ")
            curr_line = ""
            for word in words:
                test = f"{curr_line} {word}".strip()
                if fm.horizontalAdvance(test) <= avail_w or not curr_line:
                    curr_line = test
                else:
                    lines.append((curr_line, ld, font, fm))
                    curr_line = word
            if curr_line:
                lines.append((curr_line, ld, font, fm))
        return lines

    # ---------------- Paint Rendering ----------------
    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        wrapped = self._compute_wrapped_lines()

        # Draw selection boundary and handles if selected
        if self.isSelected():
            painter.setPen(QPen(QColor(0, 150, 255, 220), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._rect)

            painter.setPen(QPen(QColor(0, 100, 220), 1.5))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            for hrect in self._handle_rects().values():
                painter.drawRect(hrect)

        # Compute vertical height
        total_text_h = sum(fm.lineSpacing() for _, _, _, fm in wrapped)
        start_y = self._rect.top() + max(
            0.0, (self._rect.height() - total_text_h) / 2.0
        )
        cursor_y = start_y

        for line_text, ld, font, fm in wrapped:
            if not line_text:
                continue

            color = ld.get("color", self._text_color)
            stroke_col = ld.get("stroke_color", self._stroke_color)
            stroke_w = ld.get("stroke_width", self._stroke_width)
            line_align = ld.get("align", self._align)

            painter.setFont(font)
            text_w = fm.horizontalAdvance(line_text)

            if line_align == Qt.AlignmentFlag.AlignCenter:
                x = self._rect.left() + max(0.0, (self._rect.width() - text_w) / 2.0)
            elif line_align == Qt.AlignmentFlag.AlignRight:
                x = self._rect.right() - text_w
            else:
                x = self._rect.left()

            y = cursor_y + fm.ascent()
            cursor_y += fm.lineSpacing()

            path = QPainterPath()
            path.addText(QPointF(x, y), font, line_text)

            if stroke_w > 0:
                stroker = QPainterPathStroker()
                stroker.setWidth(stroke_w * 2)
                stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                stroke_path = stroker.createStroke(path)
                painter.fillPath(stroke_path, QBrush(stroke_col))

            painter.fillPath(path, QBrush(color))


class TouchupCanvas(QGraphicsView):
    """Canvas view supporting smooth Photos-app panning, zoom, selection, and in-place text editing."""

    region_selected = pyqtSignal(QRectF)
    item_selected = pyqtSignal(object)
    item_text_edited = pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.selection_overlay = SelectionOverlay()
        self.selection_overlay.setVisible(False)
        self.scene.addItem(self.selection_overlay)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._selecting = False
        self._select_start = QPointF()
        self._mode = "select"
        self._panning = False
        self._pan_start = QPointF()

        # In-place text editor widget
        self._inline_editor = QTextEdit(self.viewport())
        self._inline_editor.setVisible(False)
        self._inline_editor.setStyleSheet(
            "background: rgba(25, 25, 25, 240); color: white; border: 2px solid #00a2ff; border-radius: 4px; padding: 4px;"
        )
        self._inline_editor.focusOutEvent = self._on_inline_editor_focus_out
        self._editing_item: DraggableTextItem | None = None

        self.set_mode("select")

    def set_mode(self, mode: str):
        self._mode = mode
        cursor = (
            Qt.CursorShape.CrossCursor
            if mode == "select"
            else Qt.CursorShape.OpenHandCursor
        )
        self.setCursor(cursor)
        self.viewport().setCursor(cursor)

    def load_image(self, pixmap: QPixmap):
        for it in list(self.scene.items()):
            if isinstance(it, DraggableTextItem):
                self.scene.removeItem(it)

        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def add_text_item(self, text: str, rect: QRectF) -> DraggableTextItem:
        item = DraggableTextItem(text, rect)
        self.scene.addItem(item)
        self.clearSelection()
        item.setSelected(True)
        self.item_selected.emit(item)
        return item

    def clearSelection(self):
        for it in self.scene.selectedItems():
            it.setSelected(False)

    # ---------------- Inline Direct Text Editing ----------------
    def mouseDoubleClickEvent(self, event):
        if self._mode == "edit":
            item = self.itemAt(event.pos())
            if isinstance(item, DraggableTextItem):
                self._open_inline_editor(item)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _open_inline_editor(self, item: DraggableTextItem):
        self._editing_item = item
        scene_rect = item.mapToScene(item._rect).boundingRect()
        view_rect = self.mapFromScene(scene_rect).boundingRect()

        self._inline_editor.setGeometry(view_rect.adjusted(-10, -10, 10, 10))
        self._inline_editor.setText(item.raw_text)
        self._inline_editor.selectAll()
        self._inline_editor.setVisible(True)
        self._inline_editor.setFocus()

    def _on_inline_editor_focus_out(self, event):
        if self._editing_item and self._inline_editor.isVisible():
            new_text = self._inline_editor.toPlainText()
            item = self._editing_item
            item.set_text(new_text)
            self._inline_editor.setVisible(False)
            self._editing_item = None
            item.setSelected(True)
            self.item_selected.emit(item)
            self.item_text_edited.emit(item, new_text)
        QTextEdit.focusOutEvent(self._inline_editor, event)

    # ---------------- Zoom & Photos-App Hand Panning ----------------
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        # Middle click, Right click, or Spacebar drag -> Pan immediately
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or event.button() == Qt.MouseButton.RightButton
            or (
                event.button() == Qt.MouseButton.LeftButton
                and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            )
        ):
            self._panning = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self._mode == "select" and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, DraggableTextItem):
                super().mousePressEvent(event)
                self.item_selected.emit(item)
                return

            self._selecting = True
            scene_pos = self.mapToScene(event.pos())
            self._select_start = scene_pos
            self.selection_overlay.setRect(QRectF(scene_pos.x(), scene_pos.y(), 0, 0))
            self.selection_overlay.setVisible(True)
            event.accept()
            return

        if self._mode == "edit" and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item is None or item == self.pixmap_item:
                self._panning = True
                self._pan_start = event.position()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)
        selected = self.scene.selectedItems()
        if selected:
            for it in selected:
                if isinstance(it, DraggableTextItem):
                    self.item_selected.emit(it)
                    break

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
            return

        if self._selecting:
            scene_pos = self.mapToScene(event.pos())
            x1 = min(self._select_start.x(), scene_pos.x())
            y1 = min(self._select_start.y(), scene_pos.y())
            x2 = max(self._select_start.x(), scene_pos.x())
            y2 = max(self._select_start.y(), scene_pos.y())
            self.selection_overlay.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            event.accept()
            return

        if not self._selecting and not self._panning:
            if self._mode == "select":
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            elif self._mode == "edit":
                item = self.itemAt(event.pos())
                if item is None or item == self.pixmap_item:
                    self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            cursor = (
                Qt.CursorShape.CrossCursor
                if self._mode == "select"
                else Qt.CursorShape.OpenHandCursor
            )
            self.setCursor(cursor)
            self.viewport().setCursor(cursor)
            event.accept()
            return

        if self._selecting:
            self._selecting = False
            self.selection_overlay.setVisible(False)
            rect = self.selection_overlay.rect()
            if rect.width() >= 35 and rect.height() >= 20:
                self.region_selected.emit(rect)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected = [
                it
                for it in self.scene.selectedItems()
                if isinstance(it, DraggableTextItem)
            ]
            if selected:
                for it in selected:
                    self.scene.removeItem(it)
                self.item_selected.emit(None)
                event.accept()
                return
        super().keyPressEvent(event)
