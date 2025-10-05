"""Розширена візуалізація області аналізу (ROI) для Terra Tools.

Цей модуль надає функції для створення інтерактивних візуалізацій області аналізу
на глобусі, включаючи різні типи ROI та їх стилізацію.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from .common import GeoBoundingBox, FULL_EARTH_BBOX


@dataclass
class ROIStyle:
    """Стиль для візуалізації ROI."""
    line_color: Tuple[float, float, float] = (1.0, 0.58, 0.0)  # Помаранчевий
    fill_color: Tuple[float, float, float] = (1.0, 0.58, 0.0)  # Помаранчевий
    line_width: float = 2.0
    fill_alpha: float = 0.2
    line_alpha: float = 0.8
    dash_pattern: Optional[List[float]] = None  # [dash_length, gap_length, ...]
    glow_radius: float = 0.0  # Радіус свічення
    glow_intensity: float = 0.0  # Інтенсивність свічення


@dataclass
class ROIType:
    """Тип ROI."""
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    CIRCLE = "circle"
    CUSTOM = "custom"


class ROIVisualizer:
    """Клас для створення візуалізацій ROI."""
    
    def __init__(self, style: Optional[ROIStyle] = None):
        self.style = style or ROIStyle()
    
    def create_roi_path(
        self, 
        bbox: GeoBoundingBox, 
        roi_type: str = ROIType.RECTANGLE,
        segments: int = 90,
        center: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Створює шлях для візуалізації ROI на основі типу."""
        if roi_type == ROIType.RECTANGLE:
            return self._create_rectangle_path(bbox, segments)
        elif roi_type == ROIType.CIRCLE and center and radius:
            return self._create_circle_path(center, radius, segments)
        elif roi_type == ROIType.POLYGON:
            return self._create_polygon_path(bbox, segments)
        else:
            return self._create_rectangle_path(bbox, segments)
    
    def _create_rectangle_path(
        self, 
        bbox: GeoBoundingBox, 
        segments: int = 90
    ) -> List[Tuple[float, float]]:
        """Створює прямокутний шлях для ROI."""
        segments = max(4, segments)
        lons = np.linspace(bbox.min_lon, bbox.max_lon, segments, dtype=np.float32)
        lats = np.linspace(bbox.min_lat, bbox.max_lat, segments, dtype=np.float32)
        
        path: List[Tuple[float, float]] = []
        
        # Нижня сторона
        for lon in lons:
            path.append((float(bbox.min_lat), float(lon)))
        
        # Права сторона
        for lat in lats[1:]:
            path.append((float(lat), float(bbox.max_lon)))
        
        # Верхня сторона
        for lon in lons[-2::-1]:
            path.append((float(bbox.max_lat), float(lon)))
        
        # Ліва сторона
        for lat in lats[-2:0:-1]:
            path.append((float(lat), float(bbox.min_lon)))
        
        # Замикаємо контур
        if path:
            path.append(path[0])
        
        return path
    
    def _create_circle_path(
        self, 
        center: Tuple[float, float], 
        radius: float, 
        segments: int = 90
    ) -> List[Tuple[float, float]]:
        """Створює круговий шлях для ROI."""
        center_lat, center_lon = center
        segments = max(8, segments)
        
        path: List[Tuple[float, float]] = []
        
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            # Проекція кола на сферу
            lat_offset = radius * math.cos(angle)
            lon_offset = radius * math.sin(angle) / math.cos(math.radians(center_lat))
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            
            # Обмежуємо координати
            lat = max(-90.0, min(90.0, lat))
            lon = max(-180.0, min(180.0, lon))
            
            path.append((float(lat), float(lon)))
        
        return path
    
    def _create_polygon_path(
        self, 
        bbox: GeoBoundingBox, 
        segments: int = 90
    ) -> List[Tuple[float, float]]:
        """Створює полігональний шлях для ROI (зараз просто прямокутник)."""
        return self._create_rectangle_path(bbox, segments)
    
    def create_roi_texture(
        self, 
        width: int, 
        height: int, 
        bbox: GeoBoundingBox,
        roi_type: str = ROIType.RECTANGLE
    ) -> Image.Image:
        """Створює текстуру для ROI."""
        # Створюємо прозоре зображення
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Конвертуємо географічні координати в піксельні
        def geo_to_pixel(lat: float, lon: float) -> Tuple[int, int]:
            x = int((lon + 180.0) / 360.0 * width)
            y = int((90.0 - lat) / 180.0 * height)
            return x, y
        
        # Створюємо шлях для ROI
        path = self.create_roi_path(bbox, roi_type)
        
        if not path:
            return img
        
        # Конвертуємо координати
        pixel_path = [geo_to_pixel(lat, lon) for lat, lon in path]
        
        # Малюємо заливку
        if self.style.fill_alpha > 0:
            fill_color = (*[int(c * 255) for c in self.style.fill_color], 
                         int(self.style.fill_alpha * 255))
            draw.polygon(pixel_path, fill=fill_color)
        
        # Малюємо контур
        if self.style.line_alpha > 0:
            line_color = (*[int(c * 255) for c in self.style.line_color], 
                         int(self.style.line_alpha * 255))
            if self.style.dash_pattern:
                # Простий пунктирний контур
                for i in range(0, len(pixel_path) - 1, 2):
                    if i + 1 < len(pixel_path):
                        draw.line([pixel_path[i], pixel_path[i + 1]], 
                                fill=line_color, width=int(self.style.line_width))
            else:
                draw.polygon(pixel_path, outline=line_color, width=int(self.style.line_width))
        
        return img
    
    def create_roi_overlay_geometry(
        self, 
        bbox: GeoBoundingBox,
        roi_type: str = ROIType.RECTANGLE,
        segments: int = 90
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Створює геометрію для накладання ROI на глобус."""
        path = self.create_roi_path(bbox, roi_type, segments)
        
        if not path:
            return [], []
        
        # Конвертуємо в 3D координати на сфері
        segments_3d: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []
        
        radius = 1.0028  # Трохи вище поверхні глобуса
        
        coords = []
        for lat, lon in path:
            theta = math.radians(90.0 - lat)
            phi = math.radians(lon)
            x = radius * math.sin(theta) * math.sin(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.cos(phi)
            coords.append([x, y, z])
        
        if coords:
            coords_array = np.asarray(coords, dtype=np.float32)
            segments_3d.append(coords_array)
            offsets.append((0, coords_array.shape[0]))
        
        return segments_3d, offsets


def create_analysis_heatmap_texture(
    data: np.ndarray,
    bbox: GeoBoundingBox,
    width: int,
    height: int,
    colormap: str = "hot"
) -> Image.Image:
    """Створює текстуру heatmap для результатів аналізу."""
    # Нормалізуємо дані
    if data.size == 0:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    vmin = float(valid_data.min())
    vmax = float(valid_data.max())
    
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    
    # Нормалізуємо до 0-1
    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Застосовуємо colormap
    if colormap == "hot":
        # Hot colormap: чорний -> червоний -> жовтий -> білий
        r = np.clip(3 * normalized, 0, 1)
        g = np.clip(3 * normalized - 1, 0, 1)
        b = np.clip(3 * normalized - 2, 0, 1)
    elif colormap == "cool":
        # Cool colormap: блакитний -> пурпурний
        r = normalized
        g = 1.0 - normalized
        b = 1.0
    elif colormap == "viridis":
        # Viridis colormap
        r = np.clip(0.267 + 0.004 * normalized, 0, 1)
        g = np.clip(0.004 + 0.99 * normalized, 0, 1)
        b = np.clip(0.329 + 0.67 * normalized, 0, 1)
    else:
        # Default: grayscale
        r = g = b = normalized
    
    # Створюємо RGBA зображення
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = (r * 255).astype(np.uint8)
    rgba[..., 1] = (g * 255).astype(np.uint8)
    rgba[..., 2] = (b * 255).astype(np.uint8)
    rgba[..., 3] = (normalized * 255).astype(np.uint8)  # Alpha на основі інтенсивності
    
    return Image.fromarray(rgba, mode="RGBA")


def create_roi_selection_interface(
    bbox: GeoBoundingBox,
    width: int = 800,
    height: int = 400
) -> Image.Image:
    """Створює інтерфейс для вибору ROI."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Малюємо сітку координат
    grid_color = (100, 100, 100, 100)
    for lat in range(-90, 91, 30):
        y = int((90.0 - lat) / 180.0 * height)
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    for lon in range(-180, 181, 30):
        x = int((lon + 180.0) / 360.0 * width)
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    
    # Малюємо поточний ROI
    roi_color = (255, 150, 0, 200)
    x1 = int((bbox.min_lon + 180.0) / 360.0 * width)
    y1 = int((90.0 - bbox.max_lat) / 180.0 * height)
    x2 = int((bbox.max_lon + 180.0) / 360.0 * width)
    y2 = int((90.0 - bbox.min_lat) / 180.0 * height)
    
    draw.rectangle([x1, y1, x2, y2], outline=roi_color, width=2)
    
    return img
