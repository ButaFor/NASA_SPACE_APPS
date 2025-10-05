"""Tkinter control panel for Terra layer selection.

Includes an optional calendar-based date picker when ``tkcalendar`` is available.
Falls back to a simple text field if not installed.
"""
from __future__ import annotations

import tkinter as tk
from collections import deque
from dataclasses import dataclass
import datetime as dt
from tkinter import ttk
from typing import Deque, Optional

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
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _HAS_PIL = False

try:  # optional modern theming
    import ttkbootstrap as _tb  # type: ignore
    _HAS_BOOTSTRAP = True
except Exception:  # pragma: no cover
    _tb = None  # type: ignore
    _HAS_BOOTSTRAP = False

try:  # optional pretty date picker
    from tkcalendar import DateEntry as _DateEntry  # type: ignore
    _HAS_DATEENTRY = True
except Exception:  # pragma: no cover - optional
    _DateEntry = None  # type: ignore
    _HAS_DATEENTRY = False


@dataclass(frozen=True)
class ControlState:
    layer_id: str
    resolution: int
    offset_days: int
    use_backfill: bool


class TerraControlPanel:
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
        self._layers = layers
        self._resolutions = resolutions
        self._layer_by_label = {layer.label: layer for layer in layers}
        layer_labels = [layer.label for layer in layers] or ["<no layers>"]

        initial_layer_label = next(
            (layer.label for layer in layers if layer.layer_id == initial_layer_id),
            layer_labels[0],
        )
        initial_resolution_val = initial_resolution if initial_resolution in resolutions else (resolutions[0] if resolutions else 0)

        self._closed = False
        self._changed = True
        self._force_refresh = False
        self._updating = False

        self._root = tk.Tk()
        self._root.title("TERRA TOOLS")
        # Try to apply application icon if available
        try:
            from pathlib import Path
            icon_path = Path(__file__).resolve().parents[1] / "icon.ico"
            if icon_path.exists():
                self._root.iconbitmap(default=str(icon_path))
        except Exception:
            pass
        self._root.resizable(True, True)
        self._root.geometry("400x400")  # Встановлюємо менший початковий розмір
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.columnconfigure(1, weight=1)
        
        # Try a modern bootstrap theme if available
        if _HAS_BOOTSTRAP:
            try:
                _tb.Style(master=self._root, theme="flatly")
            except Exception:
                pass
        # Apply a more modern ttk theme if available
        try:
            style = ttk.Style(self._root)
            for name in ("clam", "vista", "default"):
                if name in style.theme_names():
                    style.theme_use(name)
                    break
        except Exception:
            pass

        padding = {"padx": 4, "pady": 2}  # Зменшуємо відступи для компактності

        # Track active layer date constraints and mode
        self._min_date: dt.date | None = None
        self._max_date: dt.date | None = None
        self._time_mode: str = "none"
        self._analysis_requests: Deque[AnalysisRequest] = deque()
        self._analysis_status: ttk.Label | None = None
        self._analysis_output: tk.Text | None = None
        self._analysis_start_entry: ttk.Entry | None = None
        self._analysis_end_entry: ttk.Entry | None = None
        self._analysis_roi_entry: ttk.Entry | None = None
        self._analysis_preview_label: ttk.Label | None = None
        self._analysis_preview_image: Optional[tk.PhotoImage] = None
        self._analysis_start_var = tk.StringVar()
        self._analysis_end_var = tk.StringVar()
        self._roi_var = tk.StringVar(value=DEFAULT_ROI_TEXT)

        row = 0
        ttk.Label(self._root, text="Layer").grid(row=row, column=0, sticky="w", **padding)
        self._layer_var = tk.StringVar(value=initial_layer_label)
        self._layer_combo = ttk.Combobox(
            self._root,
            textvariable=self._layer_var,
            values=layer_labels,
            state="readonly",
            width=36,
        )
        self._layer_combo.grid(row=row, column=1, sticky="ew", **padding)
        self._layer_combo.bind("<<ComboboxSelected>>", self._on_layer_change)

        row += 1
        self._description_var = tk.StringVar(value=self._layer_description(initial_layer_label))
        ttk.Label(self._root, textvariable=self._description_var, wraplength=360, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        ttk.Label(self._root, text="Resolution (px)").grid(row=row, column=0, sticky="w", **padding)
        res_labels = [str(res) for res in resolutions] or ["0"]
        self._resolution_var = tk.StringVar(value=str(initial_resolution_val))
        self._resolution_combo = ttk.Combobox(
            self._root,
            textvariable=self._resolution_var,
            values=res_labels,
            state="readonly",
        )
        self._resolution_combo.grid(row=row, column=1, sticky="ew", **padding)
        self._resolution_combo.bind("<<ComboboxSelected>>", self._on_resolution_change)

        row += 1
        ttk.Label(self._root, text="Date offset (days)").grid(row=row, column=0, sticky="w", **padding)
        self._offset_var = tk.IntVar(value=max(0, initial_offset))
        self._offset_scale = tk.Scale(
            self._root,
            orient=tk.HORIZONTAL,
            from_=0,
            to=max(0, initial_offset),
            resolution=1,
            variable=self._offset_var,
            command=self._on_offset_change,
        )
        self._offset_scale.grid(row=row, column=1, sticky="ew", **padding)

        row += 1
        # Date picker (calendar) row
        ttk.Label(self._root, text="Date").grid(row=row, column=0, sticky="w", **padding)
        self._date_var = tk.StringVar(value="")
        if _HAS_DATEENTRY:
            self._date_entry = _DateEntry(
                self._root,
                width=18,
                date_pattern="yyyy-mm-dd",
                state="disabled",
                textvariable=self._date_var,
            )
            self._date_entry.grid(row=row, column=1, sticky="ew", **padding)
            self._date_entry.bind("<<DateEntrySelected>>", self._on_date_picker_change)
        else:
            self._date_entry = ttk.Entry(self._root, textvariable=self._date_var, state="disabled")
            self._date_entry.grid(row=row, column=1, sticky="ew", **padding)
            self._date_entry.bind("<Return>", self._on_date_picker_change)

        row += 1
        self._date_range_var = tk.StringVar(value="Date range: —")
        ttk.Label(self._root, textvariable=self._date_range_var).grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        self._date_value_var = tk.StringVar(value="Selected date: —")
        ttk.Label(self._root, textvariable=self._date_value_var).grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        ttk.Separator(self._root, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 4))

        row += 1
        self._refresh_button = ttk.Button(self._root, text="Refresh now", command=self._on_refresh)
        self._refresh_button.grid(row=row, column=0, sticky="w", **padding)
        self._fps_var = tk.StringVar(value="FPS: —")
        ttk.Label(self._root, textvariable=self._fps_var).grid(row=row, column=1, sticky="e", **padding)

        row += 1
        self._backfill_var = tk.BooleanVar(value=initial_backfill)
        self._backfill_check = ttk.Checkbutton(
            self._root,
            text="Fill gaps with previous day",
            variable=self._backfill_var,
            command=self._on_backfill_toggle,
        )
        self._backfill_check.grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        ttk.Separator(self._root, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 4))

        row += 1
        self._status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(self._root, textvariable=self._status_var).grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        self._detail_var = tk.StringVar(value="No imagery requested yet.")
        ttk.Label(self._root, textvariable=self._detail_var, wraplength=360, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        self._texture_var = tk.StringVar(value="Texture: —")
        ttk.Label(self._root, textvariable=self._texture_var, wraplength=360, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        self._updated_var = tk.StringVar(value="Updated: —")
        ttk.Label(self._root, textvariable=self._updated_var).grid(row=row, column=0, columnspan=2, sticky="w", **padding)

        row += 1
        ttk.Separator(self._root, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 4))

        row += 1
        # Створюємо окремий Frame для секції аналізу з прокруткою
        analysis_frame = ttk.LabelFrame(self._root, text="Analysis", padding=5)
        analysis_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **padding)
        
        # Canvas для прокрутки секції аналізу
        self._analysis_canvas = tk.Canvas(analysis_frame, height=200)
        self._analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=self._analysis_canvas.yview)
        self._analysis_scrollable_frame = ttk.Frame(self._analysis_canvas)
        
        self._analysis_scrollable_frame.bind(
            "<Configure>",
            lambda e: self._analysis_canvas.configure(scrollregion=self._analysis_canvas.bbox("all"))
        )
        
        self._analysis_canvas.create_window((0, 0), window=self._analysis_scrollable_frame, anchor="nw")
        self._analysis_canvas.configure(yscrollcommand=self._analysis_scrollbar.set)
        
        # Розміщуємо Canvas та Scrollbar
        self._analysis_canvas.pack(side="left", fill="both", expand=True)
        self._analysis_scrollbar.pack(side="right", fill="y")
        
        # Прив'язуємо прокрутку до колеса миші для секції аналізу
        def _on_analysis_mousewheel(event):
            self._analysis_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self._analysis_canvas.bind_all("<MouseWheel>", _on_analysis_mousewheel)
        
        # Тепер створюємо елементи аналізу в scrollable_frame
        analysis_row = 0
        self._analysis_status = ttk.Label(self._analysis_scrollable_frame, text="No analysis queued.")
        self._analysis_status.grid(row=analysis_row, column=0, columnspan=2, sticky="ew", **padding)

        today_str = dt.date.today().isoformat()
        self._analysis_start_var.set(today_str)
        self._analysis_end_var.set(today_str)

        analysis_row += 1
        ttk.Label(self._analysis_scrollable_frame, text="Start date").grid(row=analysis_row, column=0, sticky="w", **padding)
        self._analysis_start_entry = ttk.Entry(self._analysis_scrollable_frame, textvariable=self._analysis_start_var, width=18)
        self._analysis_start_entry.grid(row=analysis_row, column=1, sticky="ew", **padding)

        analysis_row += 1
        ttk.Label(self._analysis_scrollable_frame, text="End date").grid(row=analysis_row, column=0, sticky="w", **padding)
        self._analysis_end_entry = ttk.Entry(self._analysis_scrollable_frame, textvariable=self._analysis_end_var, width=18)
        self._analysis_end_entry.grid(row=analysis_row, column=1, sticky="ew", **padding)

        analysis_row += 1
        # Додаємо інформацію про доступні дати для аналізу
        self._analysis_date_info_var = tk.StringVar(value="Date range: —")
        ttk.Label(self._analysis_scrollable_frame, textvariable=self._analysis_date_info_var).grid(row=analysis_row, column=0, columnspan=2, sticky="w", **padding)

        analysis_row += 1
        # Кнопки експорту
        export_frame = ttk.Frame(self._analysis_scrollable_frame)
        export_frame.grid(row=analysis_row, column=0, columnspan=2, sticky="ew", **padding)
        
        # Кнопка експорту глобуса
        self._export_globe_btn = ttk.Button(export_frame, text="Globe", command=self._on_export_globe)
        self._export_globe_btn.pack(side="left", padx=(0, 3))
        
        # Кнопка експорту аналізу
        self._export_analysis_btn = ttk.Button(export_frame, text="Analysis", command=self._on_export_analysis)
        self._export_analysis_btn.pack(side="left", padx=(0, 3))
        
        # Кнопка відкриття папки з результатами
        self._open_results_folder_btn = ttk.Button(export_frame, text="Folder", command=self._on_open_results_folder)
        self._open_results_folder_btn.pack(side="left")

        analysis_row += 1
        # Інформація про папку збереження
        self._save_location_var = tk.StringVar(value="Save location: Current directory")
        ttk.Label(self._analysis_scrollable_frame, textvariable=self._save_location_var, font=("Arial", 7)).grid(row=analysis_row, column=0, columnspan=2, sticky="w", **padding)

        analysis_row += 1
        ttk.Label(self._analysis_scrollable_frame, text="ROI lon/lat").grid(row=analysis_row, column=0, sticky="w", **padding)
        self._analysis_roi_entry = ttk.Entry(self._analysis_scrollable_frame, textvariable=self._roi_var, width=28)
        self._analysis_roi_entry.grid(row=analysis_row, column=1, sticky="ew", **padding)

        analysis_row += 1
        btn_frame = ttk.Frame(self._analysis_scrollable_frame)
        btn_frame.grid(row=analysis_row, column=0, columnspan=2, sticky="ew", **padding)
        btn_frame.columnconfigure(2, weight=1)
        self._fire_button = ttk.Button(btn_frame, text="Analyze Fires", command=self._on_fire_analysis)
        self._fire_button.grid(row=0, column=0, padx=(0, 6))
        self._flux_button = ttk.Button(btn_frame, text="Analyze Flux", command=self._on_flux_analysis)
        self._flux_button.grid(row=0, column=1, padx=(0, 6))

        analysis_row += 1
        self._analysis_output = tk.Text(self._analysis_scrollable_frame, height=4, width=48, state="disabled")  # Зменшуємо висоту
        self._analysis_output.grid(row=analysis_row, column=0, columnspan=2, sticky="nsew", padx=8, pady=(0, 4))

        analysis_row += 1
        self._analysis_preview_label = ttk.Label(self._analysis_scrollable_frame, text="No preview available.", anchor="center", relief="groove")
        self._analysis_preview_label.grid(row=analysis_row, column=0, columnspan=2, sticky="nsew", padx=4, pady=(0, 4))

    def poll(self) -> bool:
        if self._closed:
            return False
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True
            return False
        return True

    def destroy(self) -> None:
        if not self._closed:
            self._closed = True
            self._root.destroy()

    def consume_changes(self) -> tuple[bool, bool, ControlState]:
        state = self.current_state()
        changed, force = self._changed, self._force_refresh
        self._changed = False
        self._force_refresh = False
        return changed, force, state

    def current_state(self) -> ControlState:
        layer_label = self._layer_var.get()
        layer = self._layer_by_label.get(layer_label)
        if layer is None and self._layers:
            layer = self._layers[0]
        resolution = int(self._resolution_var.get() or 0)
        offset = max(0, self._offset_var.get())
        return ControlState(
            layer_id=layer.layer_id if layer is not None else "",
            resolution=resolution,
            offset_days=offset,
            use_backfill=self._backfill_var.get(),
        )

    def set_fps(self, fps_text: str) -> None:
        self._fps_var.set(fps_text)

    def update_status(
        self,
        status: str,
        detail: str,
        texture_description: str,
        updated_at: Optional[str],
    ) -> None:
        self._status_var.set(f"Status: {status}")
        self._detail_var.set(detail or "")
        self._texture_var.set(f"Texture: {texture_description}" if texture_description else "Texture: —")
        self._updated_var.set(updated_at or "Updated: —")

    def pop_analysis_request(self) -> Optional[AnalysisRequest]:
        if self._analysis_requests:
            return self._analysis_requests.popleft()
        return None

    def set_analysis_status(self, status: str) -> None:
        if self._analysis_status is not None:
            self._analysis_status.config(text=status or "")

    def set_analysis_result(self, summary: str) -> None:
        self._set_analysis_output_text(summary or "No analysis result available.")

    def set_analysis_preview(self, image: Optional[Image.Image]) -> None:
        label = self._analysis_preview_label
        if label is None:
            return
        if not _HAS_PIL or Image is None or ImageTk is None:
            label.config(text="Preview unavailable (Pillow missing).", image="")
            self._analysis_preview_image = None
            return
        if image is None:
            label.config(text="No preview available.", image="")
            self._analysis_preview_image = None
            return
        preview = image.copy()
        max_dim = 360
        preview.thumbnail((200, 200), Image.BILINEAR)  # Зменшуємо розмір preview
        photo = ImageTk.PhotoImage(preview)
        label.config(image=photo, text="")
        self._analysis_preview_image = photo

    def set_analysis_date_info(self, min_date: Optional[str], max_date: Optional[str]) -> None:
        """Встановлює інформацію про доступні дати для аналізу."""
        if min_date and max_date:
            self._analysis_date_info_var.set(f"Analysis date range: {min_date} to {max_date}")
        else:
            self._analysis_date_info_var.set("Analysis date range: —")

    def set_save_location(self, path: str) -> None:
        """Встановлює інформацію про папку збереження."""
        self._save_location_var.set(f"Save location: {path}")

    def _on_export_globe(self) -> None:
        """Обробник кнопки експорту глобуса."""
        if hasattr(self, '_viewer') and self._viewer:
            self._viewer.export_globe_image()

    def _on_export_analysis(self) -> None:
        """Обробник кнопки експорту аналізу."""
        if hasattr(self, '_viewer') and self._viewer:
            if hasattr(self._viewer, '_current_analysis_data') and self._viewer._current_analysis_data is not None:
                self._viewer._open_export_dialog()
            else:
                import tkinter.messagebox as msgbox
                msgbox.showwarning("No Analysis Data", "No analysis data available to export. Please run an analysis first.")

    def _on_open_results_folder(self) -> None:
        """Відкриває папку з результатами."""
        import os
        import subprocess
        import platform
        
        try:
            # Отримуємо поточну робочу папку
            current_dir = os.getcwd()
            
            # Відкриваємо папку залежно від операційної системи
            if platform.system() == "Windows":
                os.startfile(current_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", current_dir])
            else:  # Linux
                subprocess.run(["xdg-open", current_dir])
                
        except Exception as e:
            import tkinter.messagebox as msgbox
            msgbox.showerror("Error", f"Could not open results folder: {e}")

    def _set_analysis_output_text(self, text: str) -> None:
        if self._analysis_output is None:
            return
        self._analysis_output.configure(state="normal")
        self._analysis_output.delete("1.0", tk.END)
        if text:
            self._analysis_output.insert(tk.END, text)
        self._analysis_output.configure(state="disabled")

    def _enqueue_analysis(self, kind: AnalysisKind, layer_id: str) -> None:
        try:
            start = dt.date.fromisoformat(self._analysis_start_var.get().strip() or dt.date.today().isoformat())
            end = dt.date.fromisoformat(self._analysis_end_var.get().strip() or dt.date.today().isoformat())
        except ValueError:
            self.set_analysis_status("Invalid date format. Use YYYY-MM-DD.")
            return
        if kind is AnalysisKind.FLUX:
            start = start.replace(day=1)
            end = end.replace(day=1)
        if start > end:
            start, end = end, start
        roi_text = (self._roi_var.get().strip() if self._roi_var is not None else "")
        roi = parse_bbox_string(roi_text) if roi_text else FULL_EARTH_BBOX
        if roi is None:
            self.set_analysis_status("Invalid ROI format. Expected lon_min,lat_min,lon_max,lat_max.")
            return
        try:
            resolution = int(self._resolution_var.get() or 1024)
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
        self._set_analysis_output_text("")
        self.set_analysis_preview(None)
        self.set_analysis_status(
            f"Queued {kind.value} analysis {start.isoformat()} -> {end.isoformat()}"
        )

    def _on_fire_analysis(self) -> None:
        self._enqueue_analysis(AnalysisKind.FIRE, FIRE_LAYER_ID)

    def _on_flux_analysis(self) -> None:
        self._enqueue_analysis(AnalysisKind.FLUX, FLUX_LAYER_ID)

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
        # Persist bounds/mode for date conversions
        self._time_mode = time_mode
        try:
            self._min_date = dt.date.fromisoformat(min_date) if min_date else None
            self._max_date = dt.date.fromisoformat(max_date) if max_date else None
        except Exception:
            self._min_date = None
            self._max_date = None
        if time_mode == "none" or min_date is None or max_date is None:
            self._offset_scale.configure(state="disabled", to=0)
            self._offset_var.set(0)
            self._date_range_var.set("Date range: static layer")
            self._date_value_var.set("Selected date: —")
        else:
            self._offset_scale.configure(state="normal", to=max(0, max_offset))
            self._offset_var.set(max(0, min(max_offset, offset_value)))
            self._date_range_var.set(f"Date range: {min_date} -> {max_date}")
            self._date_value_var.set("Selected date: " + (selected_date or "—"))
        # Synchronize calendar/date-entry controls
        try:
            if time_mode == "none" or self._min_date is None or self._max_date is None:
                self._date_entry.configure(state="disabled")
                try:
                    self._calendar_btn.configure(state="disabled")
                except Exception:
                    pass
                self._date_var.set("")
            else:
                d = None
                try:
                    d = dt.date.fromisoformat(selected_date) if selected_date else self._max_date
                except Exception:
                    d = self._max_date
                if self._time_mode == "monthly" and d is not None:
                    d = d.replace(day=1)
                self._date_entry.configure(state="normal")
                try:
                    self._calendar_btn.configure(state="normal")
                except Exception:
                    pass
                try:
                    self._date_entry.configure(mindate=self._min_date, maxdate=self._max_date)
                except Exception:
                    pass
                if hasattr(self._date_entry, 'set_date'):
                    self._date_entry.set_date(d)
                else:
                    self._date_var.set((d or self._max_date).isoformat())
        except Exception:
            pass
        if self._min_date is not None and self._max_date is not None:
            try:
                start = dt.date.fromisoformat(self._analysis_start_var.get())
            except Exception:
                start = self._min_date
            try:
                end = dt.date.fromisoformat(self._analysis_end_var.get())
            except Exception:
                end = self._max_date
            start = min(max(start, self._min_date), self._max_date)
            end = min(max(end, self._min_date), self._max_date)
            self._analysis_start_var.set(start.isoformat())
            self._analysis_end_var.set(end.isoformat())
        self._updating = False

    def set_layer_description(self, label: str) -> None:
        self._description_var.set(self._layer_description(label))

    def _layer_description(self, label: str) -> str:
        layer = self._layer_by_label.get(label)
        return layer.description if layer is not None else ""

    def _on_close(self) -> None:
        self.destroy()

    def _on_layer_change(self, _event: object = None) -> None:
        if self._updating:
            return
        self._description_var.set(self._layer_description(self._layer_var.get()))
        self._changed = True

    def _on_resolution_change(self, _event: object = None) -> None:
        if self._updating:
            return
        self._changed = True

    def _on_offset_change(self, value: str) -> None:
        if self._updating:
            return
        try:
            int(value)
        except ValueError:
            return
        self._changed = True
        # If we have date bounds, mirror the slider position into the date picker
        if self._max_date is not None:
            try:
                offset = max(0, int(value))
                sel = self._max_date - dt.timedelta(days=offset)
                if self._time_mode == "monthly":
                    sel = sel.replace(day=1)
                self._updating = True
                if hasattr(self._date_entry, 'set_date'):
                    self._date_entry.set_date(sel)
                else:
                    self._date_var.set(sel.isoformat())
            finally:
                self._updating = False

    def _on_refresh(self) -> None:
        self._force_refresh = True
        self._changed = True

    def _on_backfill_toggle(self) -> None:
        if self._updating:
            return
        self._changed = True
        # Force immediate refresh so the blend applies
        self._force_refresh = True

    def _on_date_picker_change(self, _event: object = None) -> None:
        if self._updating:
            return
        if self._max_date is None:
            return
        try:
            if hasattr(self._date_entry, 'get_date'):
                d = self._date_entry.get_date()
                sel = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))
            else:
                sel = dt.date.fromisoformat(self._date_var.get())
        except Exception:
            return
        # Clamp and adapt for monthly mode
        if self._time_mode == "monthly":
            sel = sel.replace(day=1)
        if self._min_date and sel < self._min_date:
            sel = self._min_date
        if self._max_date and sel > self._max_date:
            sel = self._max_date
        offset = (self._max_date - sel).days
        self._updating = True
        try:
            self._offset_var.set(max(0, int(offset)))
            self._date_var.set(sel.isoformat())
        finally:
            self._updating = False
        self._changed = True

    def _open_calendar_dialog(self) -> None:
        # Open a persistent calendar popup using tkcalendar.Calendar
        if not _HAS_DATEENTRY or _Calendar is None:
            return
        if self._max_date is None:
            return
        # Determine current selection
        try:
            if hasattr(self._date_entry, 'get_date'):
                d = self._date_entry.get_date()
                current = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))
            else:
                current = dt.date.fromisoformat(self._date_var.get()) if self._date_var.get() else self._max_date
        except Exception:
            current = self._max_date or dt.date.today()
        top = tk.Toplevel(self._root)
        top.title("Select date")
        top.resizable(False, False)
        top.transient(self._root)
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass
        cal = _Calendar(
            top,
            selectmode='day',
            year=current.year,
            month=current.month,
            day=current.day,
            mindate=self._min_date,
            maxdate=self._max_date,
            date_pattern='yyyy-mm-dd',
        )
        cal.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        def on_ok() -> None:
            try:
                d = cal.selection_get()
                if isinstance(d, dt.date):
                    self._date_var.set(d.isoformat())
                    self._on_date_picker_change()
            finally:
                top.destroy()
        def on_cancel() -> None:
            top.destroy()
        ttk.Button(top, text="Cancel", command=on_cancel).grid(row=1, column=0, sticky='e', padx=10, pady=(0,10))
        ttk.Button(top, text="OK", command=on_ok).grid(row=1, column=1, sticky='w', padx=10, pady=(0,10))
        try:
            top.grab_set()
        except Exception:
            pass
        top.focus_set()


