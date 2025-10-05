"""PySide6 control panel for Terra layer selection (modern UI).

Provides the same public API as the previous Tkinter variant:
  - class ControlState
  - class TerraControlPanel with methods:
      poll, destroy, consume_changes, current_state,
      set_fps, update_status, set_date_info, set_layer_description

This widget integrates with an external render loop by calling
QApplication.processEvents() inside poll(), so it does not block.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import datetime as dt
import sys
from typing import Deque, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QPixmap
from PIL import Image

from .terra_config import TerraLayerOption


from analysis import (
    AnalysisKind,
    AnalysisRequest,
    GeoBoundingBox,
    FULL_EARTH_BBOX,
    parse_bbox_string,
)


FIRE_LAYER_ID = "MODIS_Terra_Thermal_Anomalies_All"
FLUX_LAYER_ID = "CERES_Terra_TOA_Longwave_Flux_All_Sky_Monthly"
DEFAULT_ROI_TEXT = "24,44,41,52"


@dataclass(frozen=True)
class ControlState:
    layer_id: str
    resolution: int
    offset_days: int
    use_backfill: bool


class TerraControlPanel(QtCore.QObject):
    def __init__(
        self,
        layers: list[TerraLayerOption],
        resolutions: list[int],
        *,
        initial_layer_id: str,
        initial_resolution: int,
        initial_offset: int,
        initial_backfill: bool,
    ) -> None:
        super().__init__()
        # Create or reuse application
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv or [])

        self._layers = layers
        self._resolutions = resolutions
        self._layer_by_label = {layer.label: layer for layer in layers}
        layer_labels = [layer.label for layer in layers] or ["<no layers>"]

        initial_layer_label = next(
            (l.label for l in layers if l.layer_id == initial_layer_id),
            layer_labels[0],
        )
        initial_resolution_val = (
            initial_resolution if initial_resolution in resolutions else (resolutions[0] if resolutions else 0)
        )

        self._closed = False
        self._changed = True
        self._force_refresh = False
        self._updating = False

        # Track per-layer time bounds / mode
        self._min_date: Optional[dt.date] = None
        self._max_date: Optional[dt.date] = None
        self._analysis_requests: Deque[AnalysisRequest] = deque()
        self._analysis_status: QtWidgets.QLabel | None = None
        self._viewer = None  # Посилання на головне вікно
        self._analysis_output: QtWidgets.QPlainTextEdit | None = None
        self._analysis_start: QtWidgets.QDateEdit | None = None
        self._analysis_end: QtWidgets.QDateEdit | None = None
        self._roi_edit: QtWidgets.QLineEdit | None = None
        self._analysis_preview: QtWidgets.QLabel | None = None
        self._analysis_preview_pixmap: QPixmap | None = None

        # UI
        self._win = QtWidgets.QWidget()
        self._win.setWindowTitle("TERRA TOOLS")
        try:
            from pathlib import Path
            icon_path = Path(__file__).resolve().parents[1] / "icon.ico"
            if icon_path.exists():
                self._win.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass
        self._win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        
        # Встановлюємо обмеження розміру вікна для виправлення помилки геометрії
        self._win.setMinimumSize(440, 600)
        self._win.setMaximumSize(800, 1200)  # Обмежуємо максимальний розмір

        layout = QtWidgets.QGridLayout(self._win)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        row = 0
        layout.addWidget(QtWidgets.QLabel("Layer"), row, 0)
        self._layer_combo = QtWidgets.QComboBox()
        self._layer_combo.addItems(layer_labels)
        self._layer_combo.setCurrentText(initial_layer_label)
        layout.addWidget(self._layer_combo, row, 1)

        row += 1
        self._desc_label = QtWidgets.QLabel(self._layer_description(initial_layer_label))
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label, row, 0, 1, 2)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Resolution (px)"), row, 0)
        self._res_combo = QtWidgets.QComboBox()
        self._res_combo.addItems([str(r) for r in resolutions] or ["0"])
        self._res_combo.setCurrentText(str(initial_resolution_val))
        layout.addWidget(self._res_combo, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Date offset (days)"), row, 0)
        self._offset_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._offset_slider.setRange(0, max(0, int(initial_offset)))
        self._offset_slider.setValue(max(0, int(initial_offset)))
        layout.addWidget(self._offset_slider, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Date"), row, 0)
        self._date_edit = QtWidgets.QDateEdit()
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        self._date_edit.setEnabled(False)
        layout.addWidget(self._date_edit, row, 1)

        row += 1
        self._range_label = QtWidgets.QLabel("Date range: ?")
        layout.addWidget(self._range_label, row, 0, 1, 2)

        row += 1
        self._selected_label = QtWidgets.QLabel("Selected date: ?")
        layout.addWidget(self._selected_label, row, 0, 1, 2)

        row += 1
        layout.addWidget(self._hline(), row, 0, 1, 2)

        row += 1
        btn_row = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton("Refresh now")
        btn_row.addWidget(self._refresh_btn, 0)
        btn_row.addStretch(1)
        self._fps_label = QtWidgets.QLabel("FPS: ?")
        btn_row.addWidget(self._fps_label, 0)
        layout.addLayout(btn_row, row, 0, 1, 2)

        row += 1
        self._backfill_check = QtWidgets.QCheckBox("Fill gaps with previous day")
        self._backfill_check.setChecked(bool(initial_backfill))
        layout.addWidget(self._backfill_check, row, 0, 1, 2)

        row += 1
        layout.addWidget(self._hline(), row, 0, 1, 2)

        row += 1
        self._status_label = QtWidgets.QLabel("Status: Idle")
        layout.addWidget(self._status_label, row, 0, 1, 2)

        row += 1
        self._detail_label = QtWidgets.QLabel("No imagery requested yet.")
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label, row, 0, 1, 2)

        row += 1
        self._texture_label = QtWidgets.QLabel("Texture: ?")
        self._texture_label.setWordWrap(True)
        layout.addWidget(self._texture_label, row, 0, 1, 2)

        row += 1
        self._updated_label = QtWidgets.QLabel("Updated: ?")
        layout.addWidget(self._updated_label, row, 0, 1, 2)

        row += 1
        layout.addWidget(self._hline(), row, 0, 1, 2)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Analysis"), row, 0)
        self._analysis_status = QtWidgets.QLabel("No analysis queued.")
        layout.addWidget(self._analysis_status, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Start date"), row, 0)
        self._analysis_start = QtWidgets.QDateEdit()
        self._analysis_start.setCalendarPopup(True)
        self._analysis_start.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self._analysis_start, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("End date"), row, 0)
        self._analysis_end = QtWidgets.QDateEdit()
        self._analysis_end.setCalendarPopup(True)
        self._analysis_end.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self._analysis_end, row, 1)

        today = dt.date.today()
        qtoday = QtCore.QDate(today.year, today.month, today.day)
        self._analysis_start.setDate(qtoday)
        self._analysis_end.setDate(qtoday)

        row += 1
        layout.addWidget(QtWidgets.QLabel("ROI lon/lat"), row, 0)
        self._roi_edit = QtWidgets.QLineEdit(DEFAULT_ROI_TEXT)
        self._roi_edit.setPlaceholderText("lon_min,lat_min,lon_max,lat_max")
        layout.addWidget(self._roi_edit, row, 1)

        row += 1
        analysis_btns = QtWidgets.QHBoxLayout()
        self._fire_btn = QtWidgets.QPushButton("Analyze Fires")
        self._flux_btn = QtWidgets.QPushButton("Analyze Flux")
        analysis_btns.addWidget(self._fire_btn)
        analysis_btns.addWidget(self._flux_btn)
        analysis_btns.addStretch(1)
        layout.addLayout(analysis_btns, row, 0, 1, 2)

        row += 1
        # Кнопки експорту
        export_btns = QtWidgets.QHBoxLayout()
        self._export_globe_btn = QtWidgets.QPushButton("Globe")
        self._export_analysis_btn = QtWidgets.QPushButton("Analysis")
        self._open_results_folder_btn = QtWidgets.QPushButton("Folder")
        export_btns.addWidget(self._export_globe_btn)
        export_btns.addWidget(self._export_analysis_btn)
        export_btns.addWidget(self._open_results_folder_btn)
        export_btns.addStretch(1)
        layout.addLayout(export_btns, row, 0, 1, 2)

        row += 1
        # Інформація про папку збереження
        self._save_location_label = QtWidgets.QLabel("Save location: Current directory")
        self._save_location_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._save_location_label, row, 0, 1, 2)

        row += 1
        # Прогрес-бар для аналізу
        self._analysis_progress = QtWidgets.QProgressBar()
        self._analysis_progress.setVisible(False)  # Прихований за замовчуванням
        self._analysis_progress.setRange(0, 100)
        self._analysis_progress.setValue(0)
        layout.addWidget(self._analysis_progress, row, 0, 1, 2)

        row += 1
        self._analysis_output = QtWidgets.QPlainTextEdit()
        self._analysis_output.setReadOnly(True)
        self._analysis_output.setPlaceholderText("Analysis summary will appear here.")
        self._analysis_output.setFixedHeight(160)
        layout.addWidget(self._analysis_output, row, 0, 1, 2)

        row += 1
        self._analysis_preview = QtWidgets.QLabel("No preview available.")
        self._analysis_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._analysis_preview.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self._analysis_preview.setMinimumHeight(160)
        self._analysis_preview.setWordWrap(True)
        self._analysis_preview.setScaledContents(True)
        layout.addWidget(self._analysis_preview, row, 0, 1, 2)

        # Events
        self._win.destroyed.connect(self._on_destroyed)
        self._layer_combo.currentTextChanged.connect(self._on_layer_change)
        self._res_combo.currentTextChanged.connect(self._on_resolution_change)
        self._offset_slider.valueChanged.connect(self._on_offset_change)
        self._date_edit.dateChanged.connect(self._on_date_changed)
        self._refresh_btn.clicked.connect(self._on_refresh)
        self._fire_btn.clicked.connect(self._on_fire_analysis)
        self._flux_btn.clicked.connect(self._on_flux_analysis)
        self._backfill_check.stateChanged.connect(self._on_backfill_toggle)
        
        # Обробники кнопок експорту
        self._export_globe_btn.clicked.connect(self._on_export_globe)
        self._export_analysis_btn.clicked.connect(self._on_export_analysis)
        self._open_results_folder_btn.clicked.connect(self._on_open_results_folder)

        # Apply a modern style if available
        try:
            QtWidgets.QApplication.setStyle("Fusion")
        except Exception:
            pass
        self._win.setFixedSize(440, 600)  # Встановлюємо фіксований розмір
        self._win.show()

    # ---- Public API ----
    def poll(self) -> bool:
        if self._closed:
            return False
        self._app.processEvents()
        return not self._closed

    def destroy(self) -> None:
        if not self._closed:
            self._closed = True
            self._win.close()

    def consume_changes(self) -> tuple[bool, bool, ControlState]:
        state = self.current_state()
        changed, force = self._changed, self._force_refresh
        self._changed = False
        self._force_refresh = False
        return changed, force, state

    def current_state(self) -> ControlState:
        label = self._layer_combo.currentText()
        layer = self._layer_by_label.get(label)
        resolution = int(self._res_combo.currentText() or 0)
        offset = int(self._offset_slider.value())
        return ControlState(
            layer_id=layer.layer_id if layer is not None else "",
            resolution=resolution,
            offset_days=max(0, offset),
            use_backfill=self._backfill_check.isChecked(),
        )

    def set_fps(self, fps_text: str) -> None:
        self._fps_label.setText(fps_text)

    def update_status(
        self,
        status: str,
        detail: str,
        texture_description: str,
        updated_at: Optional[str],
    ) -> None:
        self._status_label.setText(f"Status: {status}")
        self._detail_label.setText(detail or "")
        self._texture_label.setText(f"Texture: {texture_description}" if texture_description else "Texture: ?")
        self._updated_label.setText(updated_at or "Updated: ?")

    def set_date_info(
        self,
        min_date: Optional[str],
        max_date: Optional[str],
        selected_date: Optional[str],
        time_mode: str,
        max_offset: int,
        offset_value: int,
    ) -> None:
        self._updating = True
        self._time_mode = time_mode
        try:
            self._min_date = dt.date.fromisoformat(min_date) if min_date else None
            self._max_date = dt.date.fromisoformat(max_date) if max_date else None
        except Exception:
            self._min_date = None
            self._max_date = None

        if time_mode == "none" or self._min_date is None or self._max_date is None:
            self._offset_slider.setEnabled(False)
            self._offset_slider.setRange(0, 0)
            self._offset_slider.setValue(0)
            self._range_label.setText("Date range: static layer")
            self._selected_label.setText("Selected date: ?")
            self._date_edit.setEnabled(False)
        else:
            self._offset_slider.setEnabled(True)
            self._offset_slider.setRange(0, max(0, int(max_offset)))
            self._offset_slider.setValue(max(0, min(int(max_offset), int(offset_value))))
            self._range_label.setText(f"Date range: {self._min_date.isoformat()} -> {self._max_date.isoformat()}")
            self._selected_label.setText("Selected date: " + (selected_date or "?"))
            self._date_edit.setEnabled(True)
            # Set bounds and current date on the editor
            qmin = QtCore.QDate(self._min_date.year, self._min_date.month, self._min_date.day)
            qmax = QtCore.QDate(self._max_date.year, self._max_date.month, self._max_date.day)
            self._date_edit.setMinimumDate(qmin)
            self._date_edit.setMaximumDate(qmax)
            sel = dt.date.fromisoformat(selected_date) if selected_date else self._max_date
            if self._time_mode == "monthly" and sel is not None:
                sel = sel.replace(day=1)
            qsel = QtCore.QDate(sel.year, sel.month, sel.day)
            self._date_edit.setDate(qsel)
        if (
            self._analysis_start is not None
            and self._analysis_end is not None
            and self._min_date is not None
            and self._max_date is not None
        ):
            qmin = QtCore.QDate(self._min_date.year, self._min_date.month, self._min_date.day)
            qmax = QtCore.QDate(self._max_date.year, self._max_date.month, self._max_date.day)
            for editor in (self._analysis_start, self._analysis_end):
                editor.setMinimumDate(qmin)
                editor.setMaximumDate(qmax)
            if self._analysis_start.date() < qmin:
                self._analysis_start.setDate(qmin)
            if self._analysis_end.date() > qmax:
                self._analysis_end.setDate(qmax)
        self._updating = False

    def pop_analysis_request(self) -> Optional[AnalysisRequest]:
        if self._analysis_requests:
            return self._analysis_requests.popleft()
        return None

    def set_analysis_status(self, status: str) -> None:
        if self._analysis_status is not None:
            self._analysis_status.setText(status or "")

    def set_analysis_result(self, summary: str) -> None:
        if self._analysis_output is not None:
            self._analysis_output.setPlainText(summary or "No analysis result available.")

    def set_analysis_preview(self, image: Optional[Image.Image]) -> None:
        if self._analysis_preview is None:
            return
        if image is None:
            self._analysis_preview.clear()
            self._analysis_preview.setText("No preview available.")
            self._analysis_preview_pixmap = None
            return
        try:
            from PIL.ImageQt import ImageQt
        except Exception:
            self._analysis_preview.clear()
            self._analysis_preview.setText("Preview unavailable (ImageQt missing).")
            self._analysis_preview_pixmap = None
            return
        qim = ImageQt(image)
        pixmap = QPixmap.fromImage(qim)
        self._analysis_preview.setPixmap(pixmap)
        self._analysis_preview.setText("")
        self._analysis_preview_pixmap = pixmap

    def _enqueue_analysis(self, kind: AnalysisKind, layer_id: str) -> None:
        if self._analysis_start is None or self._analysis_end is None:
            return
        start_q = self._analysis_start.date()
        end_q = self._analysis_end.date()
        start = dt.date(start_q.year(), start_q.month(), start_q.day())
        end = dt.date(end_q.year(), end_q.month(), end_q.day())
        if kind is AnalysisKind.FLUX:
            start = start.replace(day=1)
            end = end.replace(day=1)
        if start > end:
            start, end = end, start
        roi_text = self._roi_edit.text().strip() if self._roi_edit is not None else ""
        roi = parse_bbox_string(roi_text) if roi_text else FULL_EARTH_BBOX
        print(f"🔍 ROI text: '{roi_text}' -> bbox: {roi}")
        if roi is None:
            self.set_analysis_status("Invalid ROI format. Expected lon_min,lat_min,lon_max,lat_max.")
            return
        resolution_text = self._res_combo.currentText() if self._res_combo is not None else "1024"
        try:
            resolution = int(resolution_text or 1024)
        except ValueError:
            resolution = 1024
        resolution = max(256, resolution)
        request = AnalysisRequest(
            kind=kind,
            layer_id=layer_id,
            start_date=start,
            end_date=end,
            bbox=roi,
            resolution=resolution,
        )
        self._analysis_requests.append(request)
        if self._analysis_output is not None:
            self._analysis_output.clear()
        self.set_analysis_preview(None)
        self.set_analysis_status(
            f"Queued {kind.value} analysis {start.isoformat()} -> {end.isoformat()}"
        )

    def _on_fire_analysis(self) -> None:
        self._enqueue_analysis(AnalysisKind.FIRE, FIRE_LAYER_ID)

    def _on_flux_analysis(self) -> None:
        self._enqueue_analysis(AnalysisKind.FLUX, FLUX_LAYER_ID)

    def _on_export_globe(self) -> None:
        """Обробник кнопки експорту глобуса."""
        print("🔵 Кнопка Globe натиснута")
        if hasattr(self, '_viewer') and self._viewer:
            print("✅ Viewer знайдено")
            # Викликаємо той самий метод, що і клавіатурне скорочення G
            if hasattr(self._viewer, '_quick_export_globe'):
                print("✅ Викликаємо _quick_export_globe()")
                self._viewer._quick_export_globe()
            else:
                print("⚠️ _quick_export_globe не знайдено, використовуємо fallback")
                # Fallback до розширеного експорту
                self._export_globe_image()
        else:
            print("❌ Viewer не знайдено")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self._win, "Error", "Viewer not available")

    def _on_export_analysis(self) -> None:
        """Обробник кнопки експорту аналізу."""
        print("🟡 Кнопка Analysis натиснута")
        if hasattr(self, '_viewer') and self._viewer:
            print("✅ Viewer знайдено")
            if hasattr(self._viewer, '_current_analysis_data') and self._viewer._current_analysis_data is not None:
                print("✅ Дані аналізу знайдено")
                # Викликаємо метод з головного файлу, якщо він існує
                if hasattr(self._viewer, '_open_export_dialog'):
                    print("✅ Викликаємо _open_export_dialog()")
                    self._viewer._open_export_dialog()
                else:
                    print("⚠️ _open_export_dialog не знайдено, використовуємо fallback")
                    # Fallback до Qt версії
                    self._export_analysis_data()
            else:
                print("❌ Дані аналізу не знайдено")
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self._win, "No Analysis Data", "No analysis data available to export. Please run an analysis first.")
        else:
            print("❌ Viewer не знайдено")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self._win, "Error", "Viewer not available")

    def _on_open_results_folder(self) -> None:
        """Відкриває папку з результатами."""
        print("🟢 Кнопка Folder натиснута")
        import os
        import subprocess
        import platform
        
        try:
            # Отримуємо поточну робочу папку
            current_dir = os.getcwd()
            print(f"📁 Відкриваємо папку: {current_dir}")
            
            # Відкриваємо папку залежно від операційної системи
            if platform.system() == "Windows":
                os.startfile(current_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", current_dir])
            else:  # Linux
                subprocess.run(["xdg-open", current_dir])
            
            print("✅ Папка відкрита успішно")
                
        except Exception as e:
            print(f"❌ Помилка відкриття папки: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self._win, "Error", f"Could not open results folder: {e}")

    def set_save_location(self, path: str) -> None:
        """Встановлює інформацію про папку збереження."""
        self._save_location_label.setText(f"Save location: {path}")

    def set_analysis_date_info(self, min_date: Optional[str], max_date: Optional[str]) -> None:
        """Встановлює інформацію про доступні дати для аналізу."""
        # Цей метод може бути використаний для відображення діапазону дат
        # Наразі просто ігноруємо, але можна додати відображення в майбутньому
        pass

    def show_analysis_progress(self, message: str = "Analyzing..."):
        """Показує прогрес-бар аналізу."""
        self._analysis_progress.setVisible(True)
        self._analysis_progress.setValue(0)
        self._analysis_progress.setFormat(f"{message} %p%")
        self._app.processEvents()

    def update_analysis_progress(self, value: int, message: str = None):
        """Оновлює прогрес аналізу."""
        self._analysis_progress.setValue(value)
        if message:
            self._analysis_progress.setFormat(f"{message} %p%")
        self._app.processEvents()

    def hide_analysis_progress(self):
        """Приховує прогрес-бар аналізу."""
        self._analysis_progress.setVisible(False)
        self._app.processEvents()

    def _export_globe_image(self) -> None:
        """Експортує фото глобуса з Qt діалогом."""
        try:
            from PySide6.QtWidgets import QFileDialog
            import datetime
            import os
            
            # Відкриваємо діалог вибору папки
            save_dir = QFileDialog.getExistingDirectory(
                self._win,
                "Select folder to save globe image",
                os.getcwd()
            )
            
            if not save_dir:
                return
            
            # Створюємо скріншот через viewer
            if hasattr(self._viewer, '_capture_globe_screenshot'):
                globe_img = self._viewer._capture_globe_screenshot()
                if globe_img is None:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self._win, "Error", "Failed to capture globe screenshot")
                    return
                
                # Генеруємо ім'я файлу з timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"globe_image_{timestamp}.png"
                full_path = os.path.join(save_dir, filename)
                
                # Зберігаємо зображення
                globe_img.save(full_path)
                
                # Оновлюємо інформацію про папку збереження
                self.set_save_location(save_dir)
                
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self._win, "Success", f"Globe image exported to {full_path}")
                
                # Якщо є аналіз, додаємо його як overlay
                if (hasattr(self._viewer, '_current_analysis_data') and 
                    hasattr(self._viewer, '_current_analysis_mask') and
                    self._viewer._current_analysis_data is not None and 
                    self._viewer._current_analysis_mask is not None):
                    self._add_analysis_overlay_to_globe(globe_img, full_path)
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self._win, "Error", "Globe capture functionality not available")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self._win, "Error", f"Error exporting globe image: {e}")

    def _export_analysis_data(self) -> None:
        """Експортує дані аналізу з Qt діалогом."""
        try:
            from PySide6.QtWidgets import QFileDialog
            import datetime
            import os
            
            # Відкриваємо діалог вибору папки
            save_dir = QFileDialog.getExistingDirectory(
                self._win,
                "Select folder to save analysis results",
                os.getcwd()
            )
            
            if not save_dir:
                return
            
            # Генеруємо ім'я файлу з timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.png"
            full_path = os.path.join(save_dir, filename)
            
            # Експортуємо результати через viewer
            if hasattr(self._viewer, 'export_analysis_results'):
                if self._viewer.export_analysis_results(full_path):
                    # Оновлюємо інформацію про папку збереження
                    self.set_save_location(save_dir)
                    
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(self._win, "Success", f"Analysis results exported to {full_path}")
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self._win, "Error", "Failed to export analysis results")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self._win, "Error", "Analysis export functionality not available")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self._win, "Error", f"Error exporting analysis: {e}")

    def _add_analysis_overlay_to_globe(self, globe_img, filename: str) -> None:
        """Додає overlay аналізу до зображення глобуса."""
        try:
            # Отримуємо дані аналізу з головного вікна
            analysis_data = self._viewer._current_analysis_data
            analysis_mask = self._viewer._current_analysis_mask
            
            if analysis_data is None or analysis_mask is None:
                return
            
            # Створюємо heatmap аналізу
            from analysis.analysis_visualization import AnalysisVisualizer
            visualizer = AnalysisVisualizer()
            
            heatmap_img = visualizer.create_heatmap_texture(
                analysis_data,
                analysis_mask,
                width=globe_img.width,
                height=globe_img.height,
                show_legend=True
            )
            
            # Конвертуємо heatmap в RGBA
            if heatmap_img.mode != 'RGBA':
                heatmap_img = heatmap_img.convert('RGBA')
            
            # Створюємо копію зображення глобуса
            result_img = globe_img.copy()
            
            # Додаємо heatmap як overlay з прозорістю
            alpha = 0.7  # Прозорість overlay
            result_img = Image.blend(result_img.convert('RGBA'), heatmap_img, alpha)
            
            # Зберігаємо результат
            base_name = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
            overlay_filename = f"{base_name}_with_analysis.{ext}"
            
            result_img.save(overlay_filename)
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self._win, "Success", f"Globe image with analysis overlay exported to {overlay_filename}")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self._win, "Warning", f"Error adding analysis overlay: {e}")

    def set_layer_description(self, label: str) -> None:
        self._desc_label.setText(self._layer_description(label))

    # ---- Event handlers / internals ----
    def _layer_description(self, label: str) -> str:
        layer = self._layer_by_label.get(label)
        return layer.description if layer is not None else ""

    def _on_destroyed(self, _obj=None) -> None:
        self._closed = True

    def _on_layer_change(self, _txt: str) -> None:
        if self._updating:
            return
        self._desc_label.setText(self._layer_description(self._layer_combo.currentText()))
        self._changed = True

    def _on_resolution_change(self, _txt: str) -> None:
        if self._updating:
            return
        self._changed = True

    def _on_offset_change(self, value: int) -> None:
        if self._updating:
            return
        self._changed = True
        if self._max_date is not None:
            sel = self._max_date - dt.timedelta(days=int(value))
            if self._time_mode == "monthly":
                sel = sel.replace(day=1)
            self._updating = True
            try:
                self._date_edit.setDate(QtCore.QDate(sel.year, sel.month, sel.day))
            finally:
                self._updating = False

    def _on_date_changed(self, qdate: QtCore.QDate) -> None:
        if self._updating or self._max_date is None:
            return
        sel = dt.date(qdate.year(), qdate.month(), qdate.day())
        if self._time_mode == "monthly":
            sel = sel.replace(day=1)
        if self._min_date and sel < self._min_date:
            sel = self._min_date
        if self._max_date and sel > self._max_date:
            sel = self._max_date
        offset = (self._max_date - sel).days
        self._updating = True
        try:
            self._offset_slider.setValue(max(0, int(offset)))
        finally:
            self._updating = False
        self._changed = True

    def _on_refresh(self) -> None:
        self._force_refresh = True
        self._changed = True

    def _on_backfill_toggle(self, _state: int) -> None:
        if self._updating:
            return
        self._force_refresh = True
        self._changed = True

    @staticmethod
    def _hline() -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return f
