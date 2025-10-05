"""Розширена візуалізація результатів аналізу для Terra Tools.

Цей модуль надає функції для створення детальних візуалізацій результатів аналізу
з легендою, кольоровою шкалою та інтерактивними елементами.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import GeoBoundingBox, FULL_EARTH_BBOX


@dataclass
class ColorMap:
    """Кольорова схема для візуалізації."""
    name: str
    colors: List[Tuple[float, float, float]]  # RGB values 0-1
    positions: List[float]  # Positions 0-1
    
    @classmethod
    def hot(cls) -> "ColorMap":
        """Hot colormap: чорний -> червоний -> жовтий -> білий."""
        return cls(
            name="hot",
            colors=[
                (0.0, 0.0, 0.0),      # Чорний
                (1.0, 0.0, 0.0),      # Червоний
                (1.0, 1.0, 0.0),      # Жовтий
                (1.0, 1.0, 1.0),      # Білий
            ],
            positions=[0.0, 0.33, 0.66, 1.0]
        )
    
    def apply_colormap(self, data: np.ndarray, width: int, height: int) -> Image.Image:
        """Застосовує colormap до даних та створює зображення."""
        # Нормалізуємо дані до 0-1
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Масштабуємо дані до потрібного розміру
        if normalized_data.shape != (height, width):
            temp_img = Image.fromarray((normalized_data * 255).astype(np.uint8))
            scaled_img = temp_img.resize((width, height), Image.Resampling.LANCZOS)
            normalized_data = np.array(scaled_img) / 255.0
        
        # Створюємо RGBA масив
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Застосовуємо colormap
        for i in range(len(self.colors) - 1):
            pos1 = self.positions[i]
            pos2 = self.positions[i + 1]
            color1 = self.colors[i]
            color2 = self.colors[i + 1]
            
            # Маска для поточного інтервалу
            mask = (normalized_data >= pos1) & (normalized_data < pos2)
            if i == len(self.colors) - 2:  # Останній інтервал включає кінець
                mask = (normalized_data >= pos1) & (normalized_data <= pos2)
            
            if not np.any(mask):
                continue
            
            # Лінійна інтерполяція між кольорами
            t = (normalized_data[mask] - pos1) / (pos2 - pos1)
            t = np.clip(t, 0.0, 1.0)
            
            r = color1[0] + t * (color2[0] - color1[0])
            g = color1[1] + t * (color2[1] - color1[1])
            b = color1[2] + t * (color2[2] - color1[2])
            
            rgba[mask, 0] = (r * 255).astype(np.uint8)
            rgba[mask, 1] = (g * 255).astype(np.uint8)
            rgba[mask, 2] = (b * 255).astype(np.uint8)
            rgba[mask, 3] = (normalized_data[mask] * 255).astype(np.uint8)
        
        return Image.fromarray(rgba)
    
    def get_color(self, t: float) -> Tuple[float, float, float]:
        """Отримує колір для позиції t (0-1)."""
        t = np.clip(t, 0.0, 1.0)
        
        # Знаходимо інтервал
        for i in range(len(self.colors) - 1):
            pos1 = self.positions[i]
            pos2 = self.positions[i + 1]
            
            if pos1 <= t <= pos2:
                # Лінійна інтерполяція
                local_t = (t - pos1) / (pos2 - pos1)
                color1 = self.colors[i]
                color2 = self.colors[i + 1]
                
                r = color1[0] + local_t * (color2[0] - color1[0])
                g = color1[1] + local_t * (color2[1] - color1[1])
                b = color1[2] + local_t * (color2[2] - color1[2])
                
                return (r, g, b)
        
        # Fallback до останнього кольору
        return self.colors[-1]
    
    @classmethod
    def cool(cls) -> "ColorMap":
        """Cool colormap: блакитний -> пурпурний."""
        return cls(
            name="cool",
            colors=[
                (0.0, 1.0, 1.0),      # Блакитний
                (1.0, 0.0, 1.0),      # Пурпурний
            ],
            positions=[0.0, 1.0]
        )
    
    @classmethod
    def viridis(cls) -> "ColorMap":
        """Viridis colormap."""
        return cls(
            name="viridis",
            colors=[
                (0.267, 0.004, 0.329),
                (0.282, 0.140, 0.457),
                (0.253, 0.265, 0.529),
                (0.206, 0.371, 0.557),
                (0.164, 0.471, 0.558),
                (0.128, 0.567, 0.550),
                (0.120, 0.658, 0.530),
                (0.197, 0.740, 0.490),
                (0.346, 0.810, 0.430),
                (0.529, 0.866, 0.346),
                (0.761, 0.906, 0.240),
                (1.000, 0.925, 0.153),
            ],
            positions=[i / 11.0 for i in range(12)]
        )
    
    @classmethod
    def plasma(cls) -> "ColorMap":
        """Plasma colormap."""
        return cls(
            name="plasma",
            colors=[
                (0.050, 0.030, 0.528),
                (0.196, 0.018, 0.585),
                (0.290, 0.000, 0.605),
                (0.378, 0.000, 0.590),
                (0.462, 0.000, 0.540),
                (0.540, 0.000, 0.460),
                (0.610, 0.000, 0.360),
                (0.670, 0.000, 0.240),
                (0.720, 0.000, 0.100),
                (0.760, 0.000, 0.000),
                (0.800, 0.000, 0.000),
                (0.840, 0.000, 0.000),
                (0.880, 0.000, 0.000),
                (0.920, 0.000, 0.000),
                (0.960, 0.000, 0.000),
                (1.000, 0.000, 0.000),
            ],
            positions=[i / 15.0 for i in range(16)]
        )


@dataclass
class LegendConfig:
    """Конфігурація легенди."""
    width: int = 200
    height: int = 20
    font_size: int = 12
    title: str = "Intensity"
    unit: str = ""
    show_values: bool = True
    num_ticks: int = 5
    position: str = "bottom_right"  # "top_left", "top_right", "bottom_left", "bottom_right"


class AnalysisVisualizer:
    """Клас для створення розширених візуалізацій результатів аналізу."""
    
    def __init__(self, colormap: Optional[ColorMap] = None, legend_config: Optional[LegendConfig] = None):
        self.colormap = colormap or ColorMap.hot()
        self.legend_config = legend_config or LegendConfig()
    
    def create_heatmap_texture(self, data: np.ndarray, mask: np.ndarray, 
                              width: int = 1024, height: int = 512, 
                              show_legend: bool = True) -> Image.Image:
        """Створює текстуру heatmap для відображення на глобусі."""
        # Нормалізуємо дані
        if mask is not None and mask.size > 0:
            valid_data = data[mask]
            if valid_data.size > 0:
                data_min, data_max = valid_data.min(), valid_data.max()
            else:
                data_min, data_max = 0, 1
        else:
            data_min, data_max = data.min(), data.max()
        
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Застосовуємо маску
        if mask is not None:
            normalized_data = np.where(mask, normalized_data, 0)
        
        # Створюємо heatmap з правильним розміром для ROI
        heatmap_width = width
        heatmap_height = height
        
        # Використовуємо інтерполяцію для кращої якості при масштабуванні
        try:
            from scipy import ndimage
            use_scipy = True
        except ImportError:
            use_scipy = False
            
        # Масштабуємо дані до потрібного розміру з інтерполяцією
        if normalized_data.shape != (heatmap_height, heatmap_width):
            if use_scipy:
                # Використовуємо scipy для кращої якості інтерполяції
                scale_y = heatmap_height / normalized_data.shape[0]
                scale_x = heatmap_width / normalized_data.shape[1]
                scaled_data = ndimage.zoom(normalized_data, (scale_y, scale_x), order=1)
            else:
                # Fallback до PIL з високою якістю інтерполяції
                temp_img = Image.fromarray((normalized_data * 255).astype(np.uint8))
                scaled_img = temp_img.resize((heatmap_width, heatmap_height), Image.Resampling.LANCZOS)
                scaled_data = np.array(scaled_img) / 255.0
        else:
            scaled_data = normalized_data
            
        # Створюємо heatmap зображення
        heatmap = self.colormap.apply_colormap(scaled_data, heatmap_width, heatmap_height)
        
        if show_legend:
            # Додаємо легенду
            legend_height = 120
            legend_width = 400
            legend_img = self._create_legend_with_values(data_min, data_max, legend_width, legend_height)
            
            # Об'єднуємо heatmap з легендою
            combined_width = heatmap_width + legend_width
            combined_height = max(heatmap_height, legend_height)
            combined = Image.new('RGBA', (combined_width, combined_height), (0, 0, 0, 0))
            combined.paste(heatmap, (0, 0))
            combined.paste(legend_img, (heatmap_width, 0))
            return combined
        
        return heatmap
    
    def create_heatmap_for_roi(self, data: np.ndarray, mask: np.ndarray, 
                              bbox: GeoBoundingBox,
                              width: int = 1024, height: int = 512, 
                              show_legend: bool = True) -> Image.Image:
        """Створює heatmap для ROI, яка займає весь розмір зображення."""
        # Нормалізуємо дані
        if mask is not None and mask.size > 0:
            valid_data = data[mask]
            if valid_data.size > 0:
                data_min, data_max = valid_data.min(), valid_data.max()
            else:
                data_min, data_max = 0, 1
        else:
            data_min, data_max = data.min(), data.max()
        
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Застосовуємо маску
        if mask is not None:
            normalized_data = np.where(mask, normalized_data, 0)
        
        # Створюємо heatmap з правильним розміром для ROI
        heatmap_width = width
        heatmap_height = height
        
        # Масштабуємо дані до потрібного розміру з інтерполяцією
        try:
            from scipy import ndimage
            use_scipy = True
        except ImportError:
            use_scipy = False
            
        # Масштабуємо дані до потрібного розміру з інтерполяцією
        if normalized_data.shape != (heatmap_height, heatmap_width):
            if use_scipy:
                # Використовуємо scipy для кращої якості інтерполяції
                scale_y = heatmap_height / normalized_data.shape[0]
                scale_x = heatmap_width / normalized_data.shape[1]
                scaled_data = ndimage.zoom(normalized_data, (scale_y, scale_x), order=1)
            else:
                # Fallback до PIL з високою якістю інтерполяції
                temp_img = Image.fromarray((normalized_data * 255).astype(np.uint8))
                scaled_img = temp_img.resize((heatmap_width, heatmap_height), Image.Resampling.LANCZOS)
                scaled_data = np.array(scaled_img) / 255.0
        else:
            scaled_data = normalized_data
        
        # Створюємо heatmap зображення без чорного фону
        # Використовуємо білий фон замість прозорого для кращого відображення
        rgba = np.zeros((heatmap_height, heatmap_width, 4), dtype=np.uint8)
        rgba[..., 3] = 255  # Непрозорість для всього зображення
        
        # Застосовуємо colormap
        for i in range(len(self.colormap.colors) - 1):
            pos1 = self.colormap.positions[i]
            pos2 = self.colormap.positions[i + 1]
            color1 = self.colormap.colors[i]
            color2 = self.colormap.colors[i + 1]
            
            # Маска для поточного інтервалу
            mask_interval = (scaled_data >= pos1) & (scaled_data < pos2)
            if i == len(self.colormap.colors) - 2:  # Останній інтервал включає кінець
                mask_interval = (scaled_data >= pos1) & (scaled_data <= pos2)
            
            if not np.any(mask_interval):
                continue
            
            # Лінійна інтерполяція між кольорами
            t = (scaled_data[mask_interval] - pos1) / (pos2 - pos1)
            t = np.clip(t, 0.0, 1.0)
            
            r = color1[0] + t * (color2[0] - color1[0])
            g = color1[1] + t * (color2[1] - color1[1])
            b = color1[2] + t * (color2[2] - color1[2])
            
            rgba[mask_interval, 0] = (r * 255).astype(np.uint8)
            rgba[mask_interval, 1] = (g * 255).astype(np.uint8)
            rgba[mask_interval, 2] = (b * 255).astype(np.uint8)
            rgba[mask_interval, 3] = 255  # Повна непрозорість для ROI
        
        heatmap = Image.fromarray(rgba)
        
        if show_legend:
            # Додаємо легенду
            legend_height = 120
            legend_width = 400
            legend_img = self._create_legend_with_values(data_min, data_max, legend_width, legend_height)
            
            # Об'єднуємо heatmap з легендою
            combined_width = heatmap_width + legend_width
            combined_height = max(heatmap_height, legend_height)
            combined = Image.new('RGBA', (combined_width, combined_height), (255, 255, 255, 255))  # Білий фон
            combined.paste(heatmap, (0, 0))
            combined.paste(legend_img, (heatmap_width, 0))
            return combined
        
        return heatmap
    
    def create_heatmap_for_export(self, data: np.ndarray, mask: np.ndarray, 
                                 bbox: GeoBoundingBox,
                                 width: int = 1024, height: int = 512, 
                                 show_legend: bool = True,
                                 show_grid: bool = True,
                                 analysis_info: dict = None) -> Image.Image:
        """Створює heatmap для експорту з ROI на повний розмір."""
        # Нормалізуємо дані тільки в межах ROI
        if mask is not None and mask.size > 0:
            valid_data = data[mask]
            if valid_data.size > 0:
                data_min, data_max = valid_data.min(), valid_data.max()
            else:
                data_min, data_max = 0, 1
        else:
            data_min, data_max = data.min(), data.max()
        
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Створюємо heatmap зображення з білим фоном для експорту
        rgba = np.full((height, width, 4), 255, dtype=np.uint8)  # Білий фон
        rgba[..., 3] = 255  # Повна непрозорість
        
        # ВИПРАВЛЕННЯ: Розтягуємо ROI на весь розмір зображення
        # Знаходимо координати ROI в оригінальних даних
        roi_coords = np.where(mask)
        if len(roi_coords[0]) > 0:
            # Знаходимо межі ROI
            min_row, max_row = roi_coords[0].min(), roi_coords[0].max()
            min_col, max_col = roi_coords[1].min(), roi_coords[1].max()
            
            # Розтягуємо ROI на весь розмір зображення
            roi_height = max_row - min_row + 1
            roi_width = max_col - min_col + 1
            
            # Витягуємо ROI дані
            roi_data = normalized_data[min_row:max_row+1, min_col:max_col+1]
            
            # Масштабуємо ROI дані на весь розмір зображення
            try:
                from scipy import ndimage
                use_scipy = True
            except ImportError:
                use_scipy = False
            
            if use_scipy:
                # Використовуємо scipy для кращої якості інтерполяції
                scale_y = height / roi_height
                scale_x = width / roi_width
                scaled_roi_data = ndimage.zoom(roi_data, (scale_y, scale_x), order=1)
            else:
                # Fallback до PIL з високою якістю інтерполяції
                temp_img = Image.fromarray((roi_data * 255).astype(np.uint8))
                scaled_img = temp_img.resize((width, height), Image.Resampling.LANCZOS)
                scaled_roi_data = np.array(scaled_img) / 255.0
            
            # Застосовуємо colormap до всього зображення
            for i in range(len(self.colormap.colors) - 1):
                pos1 = self.colormap.positions[i]
                pos2 = self.colormap.positions[i + 1]
                color1 = self.colormap.colors[i]
                color2 = self.colormap.colors[i + 1]
                
                # Маска для поточного інтервалу
                mask_interval = (scaled_roi_data >= pos1) & (scaled_roi_data < pos2)
                if i == len(self.colormap.colors) - 2:  # Останній інтервал включає кінець
                    mask_interval = (scaled_roi_data >= pos1) & (scaled_roi_data <= pos2)
                
                if not np.any(mask_interval):
                    continue
                
                # Лінійна інтерполяція між кольорами
                t = (scaled_roi_data[mask_interval] - pos1) / (pos2 - pos1)
                t = np.clip(t, 0.0, 1.0)
                
                r = color1[0] + t * (color2[0] - color1[0])
                g = color1[1] + t * (color2[1] - color1[1])
                b = color1[2] + t * (color2[2] - color1[2])
                
                rgba[mask_interval, 0] = (r * 255).astype(np.uint8)
                rgba[mask_interval, 1] = (g * 255).astype(np.uint8)
                rgba[mask_interval, 2] = (b * 255).astype(np.uint8)
                rgba[mask_interval, 3] = 255  # Повна непрозорість
        
        heatmap = Image.fromarray(rgba)
        
        # Додаємо сітку координат
        if show_grid:
            heatmap = self._add_coordinate_grid(heatmap, bbox)
        
        # Додаємо інформаційну панель
        if analysis_info:
            heatmap = self._add_analysis_info_panel(heatmap, bbox, data_min, data_max, analysis_info)
        
        if show_legend:
            # Додаємо легенду під heatmap
            legend_height = 120
            legend_width = width  # Легенда на всю ширину heatmap
            legend_img = self._create_legend_with_values(data_min, data_max, legend_width, legend_height)
            
            # Об'єднуємо heatmap з легендою (легенда знизу)
            combined_width = width
            combined_height = height + legend_height + 10  # Додаємо відступ
            combined = Image.new('RGBA', (combined_width, combined_height), (255, 255, 255, 255))  # Білий фон
            combined.paste(heatmap, (0, 0))
            combined.paste(legend_img, (0, height + 10))  # Легенда знизу
            return combined
        
        return heatmap

    def create_heatmap_with_legend(
        self,
        data: np.ndarray,
        bbox: GeoBoundingBox,
        width: int,
        height: int,
        title: str = "Analysis Results"
    ) -> Image.Image:
        """Створює heatmap з легендою."""
        # Створюємо основне зображення
        heatmap = self._create_heatmap(data, bbox, width, height)
        
        # Додаємо легенду
        legend = self._create_legend(data, title)
        
        # Об'єднуємо heatmap та легенду
        return self._combine_heatmap_and_legend(heatmap, legend)
    
    def _create_heatmap(
        self,
        data: np.ndarray,
        bbox: GeoBoundingBox,
        width: int,
        height: int
    ) -> Image.Image:
        """Створює heatmap без легенди."""
        if data.size == 0:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # Нормалізуємо дані
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
        rgba = self._apply_colormap(normalized)
        
        return Image.fromarray(rgba, mode="RGBA")
    
    def _apply_colormap(self, normalized_data: np.ndarray) -> np.ndarray:
        """Застосовує colormap до нормалізованих даних."""
        height, width = normalized_data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Інтерполюємо кольори
        for i in range(len(self.colormap.colors) - 1):
            pos1 = self.colormap.positions[i]
            pos2 = self.colormap.positions[i + 1]
            color1 = self.colormap.colors[i]
            color2 = self.colormap.colors[i + 1]
            
            mask = (normalized_data >= pos1) & (normalized_data < pos2)
            if i == len(self.colormap.colors) - 2:
                mask = (normalized_data >= pos1) & (normalized_data <= pos2)
            
            if not np.any(mask):
                continue
            
            # Лінійна інтерполяція між кольорами
            t = (normalized_data[mask] - pos1) / (pos2 - pos1)
            t = np.clip(t, 0.0, 1.0)
            
            r = color1[0] + t * (color2[0] - color1[0])
            g = color1[1] + t * (color2[1] - color1[1])
            b = color1[2] + t * (color2[2] - color1[2])
            
            rgba[mask, 0] = (r * 255).astype(np.uint8)
            rgba[mask, 1] = (g * 255).astype(np.uint8)
            rgba[mask, 2] = (b * 255).astype(np.uint8)
            rgba[mask, 3] = (normalized_data[mask] * 255).astype(np.uint8)
        
        return rgba
    
    def _create_legend(self, data: np.ndarray, title: str) -> Image.Image:
        """Створює легенду для heatmap."""
        config = self.legend_config
        
        # Розраховуємо розміри легенди
        legend_width = config.width
        legend_height = config.height + 40  # Додаємо місце для тексту
        
        # Створюємо зображення легенди
        legend_img = Image.new("RGBA", (legend_width, legend_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(legend_img)
        
        # Завантажуємо шрифт
        try:
            font = ImageFont.truetype("arial.ttf", config.font_size)
            title_font = ImageFont.truetype("arial.ttf", config.font_size + 2)
        except OSError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Малюємо заголовок
        title_text = f"{title} {config.unit}".strip()
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (legend_width - title_width) // 2
        draw.text((title_x, 5), title_text, font=title_font, fill=(255, 255, 255, 255))
        
        # Малюємо кольорову шкалу
        scale_y = 25
        scale_height = config.height
        
        for x in range(legend_width):
            t = x / legend_width
            color = self._interpolate_color(t)
            draw.line([(x, scale_y), (x, scale_y + scale_height)], fill=color)
        
        # Малюємо рамку
        draw.rectangle([(0, scale_y), (legend_width - 1, scale_y + scale_height)], 
                      outline=(255, 255, 255, 255), width=1)
        
        # Малюємо значення
        if config.show_values and data.size > 0:
            valid_data = data[np.isfinite(data)]
            if valid_data.size > 0:
                vmin = float(valid_data.min())
                vmax = float(valid_data.max())
                
                for i in range(config.num_ticks):
                    t = i / (config.num_ticks - 1)
                    value = vmin + t * (vmax - vmin)
                    x = int(t * (legend_width - 1))
                    
                    # Форматуємо значення
                    if abs(value) < 0.001:
                        value_str = "0"
                    elif abs(value) < 1:
                        value_str = f"{value:.3f}"
                    elif abs(value) < 100:
                        value_str = f"{value:.1f}"
                    else:
                        value_str = f"{value:.0f}"
                    
                    # Малюємо мітку
                    draw.line([(x, scale_y + scale_height), (x, scale_y + scale_height + 5)], 
                            fill=(255, 255, 255, 255), width=1)
                    
                    # Малюємо текст
                    text_bbox = draw.textbbox((0, 0), value_str, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_x = x - text_width // 2
                    text_x = max(0, min(text_x, legend_width - text_width))
                    
                    draw.text((text_x, scale_y + scale_height + 8), value_str, 
                            font=font, fill=(255, 255, 255, 255))
        
        return legend_img

    def _create_legend_with_values(self, vmin: float, vmax: float, width: int, height: int) -> Image.Image:
        """Створює легенду з заданими мінімальними та максимальними значеннями."""
        # Створюємо зображення
        legend_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(legend_img)
        
        # Завантажуємо шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Малюємо кольорову шкалу
        colorbar_width = width - 20
        colorbar_height = 30
        colorbar_x = 10
        colorbar_y = height - colorbar_height - 10
        
        for i in range(colorbar_width):
            t = i / (colorbar_width - 1)
            color = self.colormap.get_color(t)
            color_rgba = tuple(int(c * 255) for c in color) + (255,)
            draw.rectangle([colorbar_x + i, colorbar_y, colorbar_x + i + 1, colorbar_y + colorbar_height], 
                          fill=color_rgba)
        
        # Додаємо рамку
        draw.rectangle([colorbar_x, colorbar_y, colorbar_x + colorbar_width, colorbar_y + colorbar_height], 
                      outline=(0, 0, 0, 255), width=2)
        
        # Додаємо підписи
        # Мінімальне значення
        draw.text((colorbar_x, colorbar_y - 25), f"{vmin:.2f}", fill=(0, 0, 0, 255), font=font)
        # Максимальне значення
        draw.text((colorbar_x + colorbar_width - 60, colorbar_y - 25), f"{vmax:.2f}", fill=(0, 0, 0, 255), font=font)
        
        # Додаємо заголовок
        title_text = "Analysis Intensity"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 5), title_text, fill=(0, 0, 0, 255), font=title_font)
        
        return legend_img
    
    def _add_coordinate_grid(self, image: Image.Image, bbox: GeoBoundingBox) -> Image.Image:
        """Додає сітку широти і довготи до зображення."""
        from PIL import ImageDraw
        import numpy as np
        
        # Створюємо копію зображення
        img_with_grid = image.copy()
        draw = ImageDraw.Draw(img_with_grid)
        
        width, height = img_with_grid.size
        
        # Параметри сітки
        grid_color = (255, 255, 255, 180)  # Білий з прозорістю
        grid_width = 1
        label_color = (255, 255, 255, 255)  # Білий для підписів
        
        # Розраховуємо кроки для сітки
        lon_range = bbox.max_lon - bbox.min_lon
        lat_range = bbox.max_lat - bbox.min_lat
        
        # Автоматично вибираємо крок залежно від розміру ROI
        if lon_range <= 1:
            lon_step = 0.1
        elif lon_range <= 5:
            lon_step = 0.5
        elif lon_range <= 10:
            lon_step = 1.0
        else:
            lon_step = 2.0
            
        if lat_range <= 1:
            lat_step = 0.1
        elif lat_range <= 5:
            lat_step = 0.5
        elif lat_range <= 10:
            lat_step = 1.0
        else:
            lat_step = 2.0
        
        # Малюємо вертикальні лінії (довгота)
        # Використовуємо реальні координати ROI
        lon_start = np.ceil(bbox.min_lon / lon_step) * lon_step
        lon_end = np.floor(bbox.max_lon / lon_step) * lon_step
        
        for lon in np.arange(lon_start, lon_end + lon_step, lon_step):
            if bbox.min_lon <= lon <= bbox.max_lon:
                # Розраховуємо X координату (прямо пропорційно до довготи)
                x = int((lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon) * width)
                
                # Малюємо лінію
                draw.line([(x, 0), (x, height)], fill=grid_color, width=grid_width)
                
                # Додаємо підпис з реальною довготою
                label = f"{lon:.1f}°E" if lon >= 0 else f"{abs(lon):.1f}°W"
                bbox_text = draw.textbbox((0, 0), label)
                text_width = bbox_text[2] - bbox_text[0]
                draw.text((x - text_width//2, 5), label, fill=label_color)
        
        # Малюємо горизонтальні лінії (широта)
        # Використовуємо реальні координати ROI
        lat_start = np.ceil(bbox.min_lat / lat_step) * lat_step
        lat_end = np.floor(bbox.max_lat / lat_step) * lat_step
        
        for lat in np.arange(lat_start, lat_end + lat_step, lat_step):
            if bbox.min_lat <= lat <= bbox.max_lat:
                # Розраховуємо Y координату (інвертуємо, бо Y зверху вниз)
                y = int(height - (lat - bbox.min_lat) / (bbox.max_lat - bbox.min_lat) * height)
                
                # Малюємо лінію
                draw.line([(0, y), (width, y)], fill=grid_color, width=grid_width)
                
                # Додаємо підпис з реальною широтою
                label = f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S"
                bbox_text = draw.textbbox((0, 0), label)
                text_width = bbox_text[2] - bbox_text[0]
                draw.text((5, y - 10), label, fill=label_color)
        
        return img_with_grid
    
    def _add_analysis_info_panel(self, image: Image.Image, bbox: GeoBoundingBox, 
                                data_min: float, data_max: float, analysis_info: dict) -> Image.Image:
        """Додає інформаційну панель з параметрами аналізу."""
        from PIL import ImageDraw
        
        # Створюємо копію зображення
        img_with_info = image.copy()
        draw = ImageDraw.Draw(img_with_info)
        
        width, height = img_with_info.size
        
        # Параметри панелі
        panel_width = 250
        panel_height = 120
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Фон панелі
        panel_bg = (255, 255, 255, 230)  # Білий з прозорістю
        panel_border = (0, 0, 0, 255)    # Чорна рамка
        
        # Малюємо фон панелі
        draw.rectangle([panel_x, panel_y, panel_x + panel_width, panel_y + panel_height], 
                      fill=panel_bg, outline=panel_border, width=2)
        
        # Текст інформації
        info_lines = []
        
        # ROI координати
        info_lines.append(f"ROI: {bbox.min_lon:.2f}° - {bbox.max_lon:.2f}°E")
        info_lines.append(f"      {bbox.min_lat:.2f}° - {bbox.max_lat:.2f}°N")
        
        # Діапазон значень
        info_lines.append(f"Range: {data_min:.3f} - {data_max:.3f}")
        
        # Додаткова інформація з analysis_info
        if 'analysis_type' in analysis_info:
            info_lines.append(f"Type: {analysis_info['analysis_type']}")
        if 'start_date' in analysis_info:
            info_lines.append(f"Start: {analysis_info['start_date']}")
        if 'end_date' in analysis_info:
            info_lines.append(f"End: {analysis_info['end_date']}")
        if 'resolution' in analysis_info:
            info_lines.append(f"Res: {analysis_info['resolution']}")
        
        # Малюємо текст
        y_offset = panel_y + 10
        for line in info_lines:
            draw.text((panel_x + 10, y_offset), line, fill=(0, 0, 0, 255))
            y_offset += 18
        
        return img_with_info
    
    def _interpolate_color(self, t: float) -> Tuple[int, int, int, int]:
        """Інтерполює колір на основі позиції t (0-1)."""
        t = np.clip(t, 0.0, 1.0)
        
        # Знаходимо інтервал
        for i in range(len(self.colormap.colors) - 1):
            pos1 = self.colormap.positions[i]
            pos2 = self.colormap.positions[i + 1]
            
            if pos1 <= t <= pos2:
                # Лінійна інтерполяція
                local_t = (t - pos1) / (pos2 - pos1)
                color1 = self.colormap.colors[i]
                color2 = self.colormap.colors[i + 1]
                
                r = color1[0] + local_t * (color2[0] - color1[0])
                g = color1[1] + local_t * (color2[1] - color1[1])
                b = color1[2] + local_t * (color2[2] - color1[2])
                
                return (int(r * 255), int(g * 255), int(b * 255), 255)
        
        # Fallback
        return (255, 255, 255, 255)
    
    def _combine_heatmap_and_legend(self, heatmap: Image.Image, legend: Image.Image) -> Image.Image:
        """Об'єднує heatmap та легенду в одне зображення."""
        config = self.legend_config
        
        # Розраховуємо розміри результуючого зображення
        heatmap_width, heatmap_height = heatmap.size
        legend_width, legend_height = legend.size
        
        # Визначаємо позицію легенди
        if config.position == "top_left":
            legend_x = 10
            legend_y = 10
        elif config.position == "top_right":
            legend_x = heatmap_width - legend_width - 10
            legend_y = 10
        elif config.position == "bottom_left":
            legend_x = 10
            legend_y = heatmap_height - legend_height - 10
        else:  # bottom_right
            legend_x = heatmap_width - legend_width - 10
            legend_y = heatmap_height - legend_height - 10
        
        # Створюємо результуюче зображення
        result = Image.new("RGBA", (heatmap_width, heatmap_height), (0, 0, 0, 0))
        result.paste(heatmap, (0, 0))
        result.paste(legend, (legend_x, legend_y), legend)
        
        return result
    
    def create_statistics_overlay(
        self,
        data: np.ndarray,
        width: int,
        height: int,
        title: str = "Statistics"
    ) -> Image.Image:
        """Створює накладання зі статистикою."""
        if data.size == 0:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # Розраховуємо статистику
        stats = {
            "min": float(valid_data.min()),
            "max": float(valid_data.max()),
            "mean": float(valid_data.mean()),
            "std": float(valid_data.std()),
            "count": int(valid_data.size),
        }
        
        # Створюємо зображення для статистики
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Малюємо фон
        bg_width = 200
        bg_height = 120
        bg_x = width - bg_width - 10
        bg_y = 10
        
        draw.rectangle([(bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height)], 
                      fill=(0, 0, 0, 180), outline=(255, 255, 255, 255), width=1)
        
        # Малюємо заголовок
        draw.text((bg_x + 10, bg_y + 10), title, font=title_font, fill=(255, 255, 255, 255))
        
        # Малюємо статистику
        y_offset = 35
        for key, value in stats.items():
            if key == "count":
                text = f"{key}: {value:,}"
            elif key in ["min", "max", "mean", "std"]:
                text = f"{key}: {value:.3f}"
            else:
                text = f"{key}: {value}"
            
            draw.text((bg_x + 10, bg_y + y_offset), text, font=font, fill=(255, 255, 255, 255))
            y_offset += 20
        
        return overlay


def create_analysis_export_image(
    heatmap: Image.Image,
    legend: Image.Image,
    statistics: Image.Image,
    title: str = "Analysis Results"
) -> Image.Image:
    """Створює зображення для експорту аналізу."""
    # Розраховуємо розміри
    heatmap_width, heatmap_height = heatmap.size
    legend_width, legend_height = legend.size
    stats_width, stats_height = statistics.size
    
    # Створюємо результуюче зображення
    total_width = max(heatmap_width, legend_width, stats_width)
    total_height = heatmap_height + legend_height + stats_height + 50  # Додаємо відступи
    
    result = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
    
    # Додаємо heatmap
    result.paste(heatmap, (0, 0))
    
    # Додаємо легенду
    legend_x = (total_width - legend_width) // 2
    legend_y = heatmap_height + 10
    result.paste(legend, (legend_x, legend_y), legend)
    
    # Додаємо статистику
    stats_x = (total_width - stats_width) // 2
    stats_y = legend_y + legend_height + 10
    result.paste(statistics, (stats_x, stats_y), statistics)
    
    return result
