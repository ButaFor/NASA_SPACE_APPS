"""Інтерактивний селектор ROI для Terra Tools.

Цей модуль надає GUI для вибору області аналізу (ROI) з різними типами
та можливостями інтерактивного редагування.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from analysis.common import GeoBoundingBox, FULL_EARTH_BBOX


@dataclass
class ROISelection:
    """Результат вибору ROI."""
    bbox: GeoBoundingBox
    roi_type: str
    points: List[Tuple[float, float]]  # lat, lon
    is_valid: bool


class ROISelector:
    """Інтерактивний селектор ROI."""
    
    def __init__(self, parent: tk.Widget, width: int = 800, height: int = 400):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Змінні стану
        self.roi_type = "rectangle"
        self.points: List[Tuple[float, float]] = []
        self.is_drawing = False
        self.current_point: Optional[Tuple[float, float]] = None
        
        # Callbacks
        self.on_selection_changed: Optional[Callable[[ROISelection], None]] = None
        
        # Створюємо GUI
        self._create_widgets()
        self._setup_bindings()
        
        # Початковий ROI (весь світ)
        self._current_selection = ROISelection(
            bbox=FULL_EARTH_BBOX,
            roi_type="rectangle",
            points=[],
            is_valid=True
        )
    
    def _create_widgets(self) -> None:
        """Створює GUI елементи."""
        # Основний фрейм
        self.main_frame = tk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель керування
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Вибір типу ROI
        tk.Label(self.control_frame, text="ROI Type:").pack(side=tk.LEFT, padx=5)
        self.roi_type_var = tk.StringVar(value="rectangle")
        roi_type_combo = tk.ttk.Combobox(
            self.control_frame,
            textvariable=self.roi_type_var,
            values=["rectangle", "polygon", "circle"],
            state="readonly",
            width=12
        )
        roi_type_combo.pack(side=tk.LEFT, padx=5)
        roi_type_combo.bind("<<ComboboxSelected>>", self._on_roi_type_changed)
        
        # Кнопки керування
        tk.Button(self.control_frame, text="Clear", command=self._clear_selection).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Reset to World", command=self._reset_to_world).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Apply", command=self._apply_selection).pack(side=tk.LEFT, padx=5)
        
        # Інформаційна панель
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = tk.Label(
            self.info_frame,
            text="Click and drag to select ROI. Right-click to finish polygon.",
            font=("Arial", 10)
        )
        self.info_label.pack(side=tk.LEFT)
        
        # Canvas для малювання
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.width,
            height=self.height,
            bg="black",
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Створюємо початкове зображення
        self._create_base_map()
    
    def _create_base_map(self) -> None:
        """Створює базову карту світу."""
        # Створюємо просту карту світу
        img = Image.new("RGB", (self.width, self.height), (20, 30, 50))
        draw = ImageDraw.Draw(img)
        
        # Малюємо сітку координат
        grid_color = (100, 100, 100)
        for lat in range(-90, 91, 30):
            y = int((90.0 - lat) / 180.0 * self.height)
            draw.line([(0, y), (self.width, y)], fill=grid_color, width=1)
        
        for lon in range(-180, 181, 30):
            x = int((lon + 180.0) / 360.0 * self.width)
            draw.line([(x, 0), (x, self.height)], fill=grid_color, width=1)
        
        # Малюємо континенти (спрощено)
        continent_color = (60, 80, 100)
        continents = [
            # Північна Америка
            [(240, 200), (300, 200), (300, 250), (240, 250)],
            # Європа
            [(400, 150), (450, 150), (450, 200), (400, 200)],
            # Азія
            [(450, 150), (550, 150), (550, 200), (450, 200)],
            # Африка
            [(420, 250), (470, 250), (470, 350), (420, 350)],
            # Австралія
            [(500, 350), (550, 350), (550, 400), (500, 400)],
        ]
        
        for continent in continents:
            draw.polygon(continent, fill=continent_color)
        
        # Конвертуємо в PhotoImage
        self.base_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.base_image)
    
    def _setup_bindings(self) -> None:
        """Налаштовує обробники подій."""
        self.canvas.bind("<Button-1>", self._on_mouse_click)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
    
    def _on_roi_type_changed(self, event=None) -> None:
        """Обробляє зміну типу ROI."""
        self.roi_type = self.roi_type_var.get()
        self._clear_selection()
        self._update_info_text()
    
    def _on_mouse_click(self, event) -> None:
        """Обробляє клік миші."""
        x, y = event.x, event.y
        lat, lon = self._pixel_to_geo(x, y)
        
        if self.roi_type == "rectangle":
            if not self.is_drawing:
                # Початок малювання прямокутника
                self.points = [(lat, lon)]
                self.is_drawing = True
                self.current_point = (lat, lon)
        elif self.roi_type == "polygon":
            # Додаємо точку до полігону
            self.points.append((lat, lon))
            self.is_drawing = True
        elif self.roi_type == "circle":
            if not self.is_drawing:
                # Початок малювання кола
                self.points = [(lat, lon)]
                self.is_drawing = True
                self.current_point = (lat, lon)
    
    def _on_mouse_drag(self, event) -> None:
        """Обробляє перетягування миші."""
        if not self.is_drawing:
            return
        
        x, y = event.x, event.y
        lat, lon = self._pixel_to_geo(x, y)
        self.current_point = (lat, lon)
        
        # Перемальовуємо canvas
        self._redraw_canvas()
    
    def _on_mouse_release(self, event) -> None:
        """Обробляє відпускання миші."""
        if not self.is_drawing:
            return
        
        x, y = event.x, event.y
        lat, lon = self._pixel_to_geo(x, y)
        
        if self.roi_type == "rectangle":
            if len(self.points) == 1:
                # Завершуємо прямокутник
                self.points.append((lat, lon))
                self.is_drawing = False
                self._update_selection()
        elif self.roi_type == "circle":
            if len(self.points) == 1:
                # Завершуємо коло
                self.points.append((lat, lon))
                self.is_drawing = False
                self._update_selection()
    
    def _on_right_click(self, event) -> None:
        """Обробляє правий клік миші."""
        if self.roi_type == "polygon" and len(self.points) >= 3:
            # Завершуємо полігон
            self.is_drawing = False
            self._update_selection()
    
    def _on_mouse_move(self, event) -> None:
        """Обробляє рух миші."""
        x, y = event.x, event.y
        lat, lon = self._pixel_to_geo(x, y)
        
        # Оновлюємо інформацію про координати
        self._update_coordinate_info(lat, lon)
    
    def _pixel_to_geo(self, x: int, y: int) -> Tuple[float, float]:
        """Конвертує піксельні координати в географічні."""
        lon = (x / self.width) * 360.0 - 180.0
        lat = 90.0 - (y / self.height) * 180.0
        return lat, lon
    
    def _geo_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Конвертує географічні координати в піксельні."""
        x = int((lon + 180.0) / 360.0 * self.width)
        y = int((90.0 - lat) / 180.0 * self.height)
        return x, y
    
    def _redraw_canvas(self) -> None:
        """Перемальовує canvas."""
        # Очищаємо canvas
        self.canvas.delete("roi")
        
        # Малюємо поточний ROI
        if self.points:
            self._draw_roi()
    
    def _draw_roi(self) -> None:
        """Малює поточний ROI на canvas."""
        if not self.points:
            return
        
        # Конвертуємо координати в пікселі
        pixel_points = [self._geo_to_pixel(lat, lon) for lat, lon in self.points]
        
        if self.roi_type == "rectangle":
            if len(pixel_points) >= 2:
                x1, y1 = pixel_points[0]
                x2, y2 = pixel_points[1]
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="red", width=2, tags="roi"
                )
        elif self.roi_type == "polygon":
            if len(pixel_points) >= 2:
                # Малюємо лінії між точками
                for i in range(len(pixel_points) - 1):
                    x1, y1 = pixel_points[i]
                    x2, y2 = pixel_points[i + 1]
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill="red", width=2, tags="roi"
                    )
                
                # Малюємо точки
                for x, y in pixel_points:
                    self.canvas.create_oval(
                        x-3, y-3, x+3, y+3,
                        fill="red", outline="white", width=1, tags="roi"
                    )
        elif self.roi_type == "circle":
            if len(pixel_points) >= 2:
                x1, y1 = pixel_points[0]
                x2, y2 = pixel_points[1]
                # Розраховуємо радіус
                radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                self.canvas.create_oval(
                    x1 - radius, y1 - radius, x1 + radius, y1 + radius,
                    outline="red", width=2, tags="roi"
                )
    
    def _update_selection(self) -> None:
        """Оновлює поточний вибір ROI."""
        if not self.points:
            return
        
        try:
            if self.roi_type == "rectangle":
                if len(self.points) >= 2:
                    lat1, lon1 = self.points[0]
                    lat2, lon2 = self.points[1]
                    bbox = GeoBoundingBox(
                        min_lon=min(lon1, lon2),
                        min_lat=min(lat1, lat2),
                        max_lon=max(lon1, lon2),
                        max_lat=max(lat1, lat2)
                    )
                    self._current_selection = ROISelection(
                        bbox=bbox,
                        roi_type=self.roi_type,
                        points=self.points,
                        is_valid=True
                    )
            elif self.roi_type == "polygon":
                if len(self.points) >= 3:
                    # Розраховуємо bounding box для полігону
                    lats = [p[0] for p in self.points]
                    lons = [p[1] for p in self.points]
                    bbox = GeoBoundingBox(
                        min_lon=min(lons),
                        min_lat=min(lats),
                        max_lon=max(lons),
                        max_lat=max(lats)
                    )
                    self._current_selection = ROISelection(
                        bbox=bbox,
                        roi_type=self.roi_type,
                        points=self.points,
                        is_valid=True
                    )
            elif self.roi_type == "circle":
                if len(self.points) >= 2:
                    center_lat, center_lon = self.points[0]
                    edge_lat, edge_lon = self.points[1]
                    
                    # Розраховуємо радіус в градусах
                    radius = ((edge_lat - center_lat) ** 2 + (edge_lon - center_lon) ** 2) ** 0.5
                    
                    bbox = GeoBoundingBox(
                        min_lon=center_lon - radius,
                        min_lat=center_lat - radius,
                        max_lon=center_lon + radius,
                        max_lat=center_lat + radius
                    )
                    self._current_selection = ROISelection(
                        bbox=bbox,
                        roi_type=self.roi_type,
                        points=self.points,
                        is_valid=True
                    )
            
            # Викликаємо callback
            if self.on_selection_changed:
                self.on_selection_changed(self._current_selection)
                
        except Exception as e:
            print(f"Error updating selection: {e}")
            self._current_selection = ROISelection(
                bbox=FULL_EARTH_BBOX,
                roi_type=self.roi_type,
                points=[],
                is_valid=False
            )
    
    def _clear_selection(self) -> None:
        """Очищає поточний вибір."""
        self.points = []
        self.is_drawing = False
        self.current_point = None
        self.canvas.delete("roi")
        self._current_selection = ROISelection(
            bbox=FULL_EARTH_BBOX,
            roi_type=self.roi_type,
            points=[],
            is_valid=True
        )
        if self.on_selection_changed:
            self.on_selection_changed(self._current_selection)
    
    def _reset_to_world(self) -> None:
        """Скидає ROI до всього світу."""
        self._clear_selection()
        self._current_selection = ROISelection(
            bbox=FULL_EARTH_BBOX,
            roi_type="rectangle",
            points=[],
            is_valid=True
        )
        if self.on_selection_changed:
            self.on_selection_changed(self._current_selection)
    
    def _apply_selection(self) -> None:
        """Застосовує поточний вибір."""
        if self.on_selection_changed:
            self.on_selection_changed(self._current_selection)
    
    def _update_info_text(self) -> None:
        """Оновлює інформаційний текст."""
        if self.roi_type == "rectangle":
            text = "Click and drag to select rectangular ROI."
        elif self.roi_type == "polygon":
            text = "Click to add points. Right-click to finish polygon."
        elif self.roi_type == "circle":
            text = "Click center, then drag to set radius."
        else:
            text = "Select ROI type and draw on map."
        
        self.info_label.config(text=text)
    
    def _update_coordinate_info(self, lat: float, lon: float) -> None:
        """Оновлює інформацію про координати."""
        # Можна додати відображення поточних координат
        pass
    
    def get_current_selection(self) -> ROISelection:
        """Повертає поточний вибір ROI."""
        return self._current_selection
    
    def set_selection(self, selection: ROISelection) -> None:
        """Встановлює вибір ROI."""
        self._current_selection = selection
        self.roi_type = selection.roi_type
        self.roi_type_var.set(selection.roi_type)
        self.points = selection.points.copy()
        self._redraw_canvas()
        self._update_info_text()


def create_roi_selector_dialog(parent: tk.Widget, initial_bbox: Optional[GeoBoundingBox] = None) -> Optional[ROISelection]:
    """Створює діалог вибору ROI."""
    dialog = tk.Toplevel(parent)
    dialog.title("Select ROI")
    dialog.geometry("900x500")
    dialog.resizable(True, True)
    
    # Центруємо діалог
    dialog.transient(parent)
    dialog.grab_set()
    
    result = [None]
    
    def on_selection_changed(selection: ROISelection) -> None:
        result[0] = selection
    
    # Створюємо селектор
    selector = ROISelector(dialog, width=800, height=400)
    selector.on_selection_changed = on_selection_changed
    
    # Встановлюємо початковий ROI
    if initial_bbox:
        initial_selection = ROISelection(
            bbox=initial_bbox,
            roi_type="rectangle",
            points=[],
            is_valid=True
        )
        selector.set_selection(initial_selection)
    
    # Кнопки
    button_frame = tk.Frame(dialog)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def on_ok():
        result[0] = selector.get_current_selection()
        dialog.destroy()
    
    def on_cancel():
        result[0] = None
        dialog.destroy()
    
    tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
    
    # Очікуємо закриття діалогу
    dialog.wait_window()
    
    return result[0]
