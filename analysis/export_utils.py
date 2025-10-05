"""Утиліти для експорту результатів аналізу.

Цей модуль надає функції для збереження результатів аналізу в різних форматах
та з різними опціями візуалізації.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import GeoBoundingBox


@dataclass
class ExportConfig:
    """Конфігурація експорту."""
    format: str = "png"  # png, jpg, tiff, pdf
    quality: int = 95  # для jpg
    dpi: int = 300
    include_legend: bool = True
    include_statistics: bool = True
    include_metadata: bool = True
    colormap: str = "hot"
    width: int = 1024
    height: int = 1024
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)


class AnalysisExporter:
    """Клас для експорту результатів аналізу."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
    
    def export_heatmap(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        output_path: str,
        title: str = "Analysis Results"
    ) -> bool:
        """Експортує heatmap як зображення."""
        try:
            # Створюємо heatmap
            heatmap_img = self._create_heatmap_image(data, mask, bbox, title)
            
            # Зберігаємо зображення
            self._save_image(heatmap_img, output_path)
            
            # Зберігаємо метадані, якщо потрібно
            if self.config.include_metadata:
                self._save_metadata(data, mask, bbox, title, output_path)
            
            return True
            
        except Exception as e:
            print(f"Error exporting heatmap: {e}")
            return False
    
    def export_analysis_report(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        output_path: str,
        title: str = "Analysis Results",
        additional_info: Optional[Dict] = None
    ) -> bool:
        """Експортує повний звіт аналізу."""
        try:
            # Створюємо зображення звіту
            report_img = self._create_report_image(data, mask, bbox, title, additional_info)
            
            # Зберігаємо зображення
            self._save_image(report_img, output_path)
            
            # Зберігаємо дані як JSON
            if self.config.include_metadata:
                json_path = output_path.rsplit('.', 1)[0] + '.json'
                self._save_data_json(data, mask, bbox, title, json_path, additional_info)
            
            return True
            
        except Exception as e:
            print(f"Error exporting analysis report: {e}")
            return False
    
    def export_data_csv(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        output_path: str,
        include_coordinates: bool = True
    ) -> bool:
        """Експортує дані як CSV файл."""
        try:
            # Підготовлюємо дані
            masked_data = np.where(mask, data, np.nan)
            valid_indices = np.where(np.isfinite(masked_data))
            
            if len(valid_indices[0]) == 0:
                print("No valid data to export")
                return False
            
            # Створюємо CSV
            with open(output_path, 'w') as f:
                if include_coordinates:
                    f.write("lat,lon,value\n")
                    
                    # Розраховуємо координати
                    height, width = data.shape
                    lat_step = bbox.height() / height
                    lon_step = bbox.width() / width
                    
                    for i, j in zip(valid_indices[0], valid_indices[1]):
                        lat = bbox.max_lat - (i + 0.5) * lat_step
                        lon = bbox.min_lon + (j + 0.5) * lon_step
                        value = masked_data[i, j]
                        f.write(f"{lat:.6f},{lon:.6f},{value:.6f}\n")
                else:
                    f.write("value\n")
                    for i, j in zip(valid_indices[0], valid_indices[1]):
                        value = masked_data[i, j]
                        f.write(f"{value:.6f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False
    
    def _create_heatmap_image(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        title: str
    ) -> Image.Image:
        """Створює зображення heatmap."""
        # Застосовуємо маску
        masked_data = np.where(mask, data, np.nan)
        
        # Нормалізуємо дані
        valid_data = masked_data[np.isfinite(masked_data)]
        if valid_data.size == 0:
            return Image.new("RGBA", (self.config.width, self.config.height), (0, 0, 0, 0))
        
        vmin = float(valid_data.min())
        vmax = float(valid_data.max())
        
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        
        normalized = (masked_data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Застосовуємо colormap
        rgba = self._apply_colormap(normalized)
        
        # Створюємо зображення
        img = Image.fromarray(rgba, mode="RGBA")
        
        # Масштабуємо до потрібного розміру
        if img.size != (self.config.width, self.config.height):
            img = img.resize((self.config.width, self.config.height), Image.BILINEAR)
        
        return img
    
    def _create_report_image(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        title: str,
        additional_info: Optional[Dict] = None
    ) -> Image.Image:
        """Створює повне зображення звіту."""
        # Розраховуємо розміри
        report_width = self.config.width
        report_height = self.config.height
        
        if self.config.include_legend:
            report_height += 100  # Місце для легенди
        
        if self.config.include_statistics:
            report_height += 200  # Місце для статистики
        
        # Створюємо зображення
        img = Image.new("RGBA", (report_width, report_height), self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        # Завантажуємо шрифти
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            header_font = ImageFont.truetype("arial.ttf", 16)
            text_font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        y_offset = 10
        
        # Заголовок
        draw.text((10, y_offset), title, font=title_font, fill=(255, 255, 255, 255))
        y_offset += 40
        
        # Heatmap
        heatmap_img = self._create_heatmap_image(data, mask, bbox, title)
        img.paste(heatmap_img, (10, y_offset), heatmap_img)
        y_offset += self.config.height + 20
        
        # Легенда
        if self.config.include_legend:
            legend_img = self._create_legend_image(data, mask)
            img.paste(legend_img, (10, y_offset), legend_img)
            y_offset += 100
        
        # Статистика
        if self.config.include_statistics:
            stats_text = self._create_statistics_text(data, mask, bbox, additional_info)
            draw.text((10, y_offset), "Statistics:", font=header_font, fill=(255, 255, 255, 255))
            y_offset += 30
            
            for line in stats_text.split('\n'):
                draw.text((10, y_offset), line, font=text_font, fill=(255, 255, 255, 255))
                y_offset += 20
        
        return img
    
    def _create_legend_image(self, data: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Створює зображення легенди."""
        legend_width = self.config.width - 20
        legend_height = 80
        
        img = Image.new("RGBA", (legend_width, legend_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Розраховуємо діапазон значень
        masked_data = np.where(mask, data, np.nan)
        valid_data = masked_data[np.isfinite(masked_data)]
        
        if valid_data.size == 0:
            return img
        
        vmin = float(valid_data.min())
        vmax = float(valid_data.max())
        
        # Малюємо кольорову шкалу
        for x in range(legend_width):
            t = x / legend_width
            color = self._get_colormap_color(t)
            draw.line([(x, 20), (x, 40)], fill=color, width=1)
        
        # Малюємо рамку
        draw.rectangle([(0, 20), (legend_width - 1, 40)], outline=(255, 255, 255, 255), width=1)
        
        # Малюємо значення
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except OSError:
            font = ImageFont.load_default()
        
        # Мінімальне значення
        draw.text((0, 45), f"{vmin:.3f}", font=font, fill=(255, 255, 255, 255))
        
        # Максимальне значення
        text_width = draw.textlength(f"{vmax:.3f}", font=font)
        draw.text((legend_width - text_width, 45), f"{vmax:.3f}", font=font, fill=(255, 255, 255, 255))
        
        return img
    
    def _create_statistics_text(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        additional_info: Optional[Dict] = None
    ) -> str:
        """Створює текст статистики."""
        masked_data = np.where(mask, data, np.nan)
        valid_data = masked_data[np.isfinite(masked_data)]
        
        if valid_data.size == 0:
            return "No valid data available."
        
        # Базова статистика
        stats = [
            f"Valid pixels: {valid_data.size:,}",
            f"Min value: {float(valid_data.min()):.6f}",
            f"Max value: {float(valid_data.max()):.6f}",
            f"Mean value: {float(valid_data.mean()):.6f}",
            f"Std deviation: {float(valid_data.std()):.6f}",
            f"Median: {float(np.median(valid_data)):.6f}",
        ]
        
        # ROI інформація
        stats.extend([
            f"ROI longitude: {bbox.min_lon:.3f} to {bbox.max_lon:.3f}",
            f"ROI latitude: {bbox.min_lat:.3f} to {bbox.max_lat:.3f}",
            f"ROI area: {bbox.area_deg2():.3f} deg²",
        ])
        
        # Додаткова інформація
        if additional_info:
            for key, value in additional_info.items():
                stats.append(f"{key}: {value}")
        
        return "\n".join(stats)
    
    def _apply_colormap(self, normalized_data: np.ndarray) -> np.ndarray:
        """Застосовує colormap до нормалізованих даних."""
        height, width = normalized_data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        if self.config.colormap == "hot":
            r = np.clip(3 * normalized_data, 0, 1)
            g = np.clip(3 * normalized_data - 1, 0, 1)
            b = np.clip(3 * normalized_data - 2, 0, 1)
        elif self.config.colormap == "cool":
            r = normalized_data
            g = 1.0 - normalized_data
            b = 1.0
        elif self.config.colormap == "viridis":
            r = np.clip(0.267 + 0.004 * normalized_data, 0, 1)
            g = np.clip(0.004 + 0.99 * normalized_data, 0, 1)
            b = np.clip(0.329 + 0.67 * normalized_data, 0, 1)
        else:  # grayscale
            r = g = b = normalized_data
        
        rgba[..., 0] = (r * 255).astype(np.uint8)
        rgba[..., 1] = (g * 255).astype(np.uint8)
        rgba[..., 2] = (b * 255).astype(np.uint8)
        rgba[..., 3] = (normalized_data * 255).astype(np.uint8)
        
        return rgba
    
    def _get_colormap_color(self, t: float) -> Tuple[int, int, int, int]:
        """Отримує колір з colormap для позиції t (0-1)."""
        t = np.clip(t, 0.0, 1.0)
        
        if self.config.colormap == "hot":
            r = np.clip(3 * t, 0, 1)
            g = np.clip(3 * t - 1, 0, 1)
            b = np.clip(3 * t - 2, 0, 1)
        elif self.config.colormap == "cool":
            r = t
            g = 1.0 - t
            b = 1.0
        elif self.config.colormap == "viridis":
            r = np.clip(0.267 + 0.004 * t, 0, 1)
            g = np.clip(0.004 + 0.99 * t, 0, 1)
            b = np.clip(0.329 + 0.67 * t, 0, 1)
        else:  # grayscale
            r = g = b = t
        
        return (int(r * 255), int(g * 255), int(b * 255), 255)
    
    def _save_image(self, img: Image.Image, output_path: str) -> None:
        """Зберігає зображення в потрібному форматі."""
        # Визначаємо формат з розширення файлу
        ext = Path(output_path).suffix.lower()
        
        if ext == '.jpg' or ext == '.jpeg':
            # Конвертуємо в RGB для JPEG
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[-1])
                img = background
            img.save(output_path, 'JPEG', quality=self.config.quality, dpi=(self.config.dpi, self.config.dpi))
        elif ext == '.tiff' or ext == '.tif':
            img.save(output_path, 'TIFF', dpi=(self.config.dpi, self.config.dpi))
        elif ext == '.pdf':
            # Для PDF потрібно конвертувати в RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            img.save(output_path, 'PDF', resolution=self.config.dpi)
        else:  # PNG
            img.save(output_path, 'PNG', dpi=(self.config.dpi, self.config.dpi))
    
    def _save_metadata(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        title: str,
        output_path: str
    ) -> None:
        """Зберігає метадані як JSON файл."""
        metadata_path = output_path.rsplit('.', 1)[0] + '_metadata.json'
        
        masked_data = np.where(mask, data, np.nan)
        valid_data = masked_data[np.isfinite(masked_data)]
        
        metadata = {
            "title": title,
            "export_time": datetime.datetime.now().isoformat(),
            "roi": {
                "min_lon": float(bbox.min_lon),
                "min_lat": float(bbox.min_lat),
                "max_lon": float(bbox.max_lon),
                "max_lat": float(bbox.max_lat),
                "area_deg2": float(bbox.area_deg2())
            },
            "data": {
                "shape": data.shape,
                "valid_pixels": int(valid_data.size),
                "min_value": float(valid_data.min()) if valid_data.size > 0 else None,
                "max_value": float(valid_data.max()) if valid_data.size > 0 else None,
                "mean_value": float(valid_data.mean()) if valid_data.size > 0 else None,
                "std_value": float(valid_data.std()) if valid_data.size > 0 else None
            },
            "export_config": {
                "format": self.config.format,
                "quality": self.config.quality,
                "dpi": self.config.dpi,
                "colormap": self.config.colormap,
                "width": self.config.width,
                "height": self.config.height
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_data_json(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        title: str,
        output_path: str,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Зберігає дані як JSON файл."""
        masked_data = np.where(mask, data, np.nan)
        valid_data = masked_data[np.isfinite(masked_data)]
        
        # Підготовлюємо дані для JSON
        data_dict = {
            "title": title,
            "export_time": datetime.datetime.now().isoformat(),
            "roi": {
                "min_lon": float(bbox.min_lon),
                "min_lat": float(bbox.min_lat),
                "max_lon": float(bbox.max_lon),
                "max_lat": float(bbox.max_lat),
                "area_deg2": float(bbox.area_deg2())
            },
            "data_shape": data.shape,
            "valid_data": valid_data.tolist(),
            "statistics": {
                "count": int(valid_data.size),
                "min": float(valid_data.min()) if valid_data.size > 0 else None,
                "max": float(valid_data.max()) if valid_data.size > 0 else None,
                "mean": float(valid_data.mean()) if valid_data.size > 0 else None,
                "std": float(valid_data.std()) if valid_data.size > 0 else None,
                "median": float(np.median(valid_data)) if valid_data.size > 0 else None
            }
        }
        
        if additional_info:
            data_dict["additional_info"] = additional_info
        
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def _add_analysis_overlay_to_globe(self, globe_img: Image.Image, filename: str, parent) -> None:
        """Додає overlay аналізу до зображення глобуса."""
        try:
            # Отримуємо дані аналізу з головного вікна
            analysis_data = parent._current_analysis_data
            analysis_mask = parent._current_analysis_mask
            
            if analysis_data is None or analysis_mask is None:
                return
            
            # Створюємо heatmap аналізу
            from .analysis_visualization import AnalysisVisualizer
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
            
            self._save_image(result_img, overlay_filename)
            print(f"Globe image with analysis overlay exported to {overlay_filename}")
            
        except Exception as e:
            print(f"Error adding analysis overlay: {e}")


def create_export_dialog(parent, data: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, 
                        bbox: Optional[GeoBoundingBox] = None, title: str = "Analysis Results", 
                        globe_export: bool = False) -> None:
    """Створює діалог експорту."""
    import tkinter as tk
    from tkinter import filedialog, ttk
    
    dialog = tk.Toplevel(parent)
    dialog.title("Export Globe Image" if globe_export else "Export Analysis Results")
    dialog.geometry("500x400")
    dialog.resizable(False, False)
    
    # Центруємо діалог
    dialog.transient(parent)
    dialog.grab_set()
    
    # Конфігурація експорту
    config = ExportConfig()
    
    # Основний фрейм
    main_frame = tk.Frame(dialog)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Формат файлу
    tk.Label(main_frame, text="File Format:").grid(row=0, column=0, sticky="w", pady=5)
    format_var = tk.StringVar(value=config.format)
    if globe_export:
        format_combo = ttk.Combobox(main_frame, textvariable=format_var, values=["png", "jpg", "tiff", "bmp"])
    else:
        format_combo = ttk.Combobox(main_frame, textvariable=format_var, values=["png", "jpg", "tiff", "pdf"])
    format_combo.grid(row=0, column=1, sticky="ew", pady=5)
    
    # Якість (для JPEG)
    tk.Label(main_frame, text="Quality (JPEG):").grid(row=1, column=0, sticky="w", pady=5)
    quality_var = tk.IntVar(value=config.quality)
    quality_scale = tk.Scale(main_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=quality_var)
    quality_scale.grid(row=1, column=1, sticky="ew", pady=5)
    
    # DPI
    tk.Label(main_frame, text="DPI:").grid(row=2, column=0, sticky="w", pady=5)
    dpi_var = tk.IntVar(value=config.dpi)
    dpi_scale = tk.Scale(main_frame, from_=72, to=600, orient=tk.HORIZONTAL, variable=dpi_var)
    dpi_scale.grid(row=2, column=1, sticky="ew", pady=5)
    
    # Розміри
    tk.Label(main_frame, text="Width:").grid(row=3, column=0, sticky="w", pady=5)
    width_var = tk.IntVar(value=config.width)
    width_entry = tk.Entry(main_frame, textvariable=width_var)
    width_entry.grid(row=3, column=1, sticky="ew", pady=5)
    
    tk.Label(main_frame, text="Height:").grid(row=4, column=0, sticky="w", pady=5)
    height_var = tk.IntVar(value=config.height)
    height_entry = tk.Entry(main_frame, textvariable=height_var)
    height_entry.grid(row=4, column=1, sticky="ew", pady=5)
    
    # Colormap
    tk.Label(main_frame, text="Colormap:").grid(row=5, column=0, sticky="w", pady=5)
    colormap_var = tk.StringVar(value=config.colormap)
    colormap_combo = ttk.Combobox(main_frame, textvariable=colormap_var, values=["hot", "cool", "viridis", "plasma"])
    colormap_combo.grid(row=5, column=1, sticky="ew", pady=5)
    
    # Опції
    tk.Label(main_frame, text="Options:").grid(row=6, column=0, sticky="w", pady=5)
    options_frame = tk.Frame(main_frame)
    options_frame.grid(row=6, column=1, sticky="ew", pady=5)
    
    if globe_export:
        include_analysis_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Include Analysis Overlay", variable=include_analysis_var).pack(anchor="w")
        
        include_legend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Include Legend", variable=include_legend_var).pack(anchor="w")
    else:
        include_legend_var = tk.BooleanVar(value=config.include_legend)
        tk.Checkbutton(options_frame, text="Include Legend", variable=include_legend_var).pack(anchor="w")
        
        include_stats_var = tk.BooleanVar(value=config.include_statistics)
        tk.Checkbutton(options_frame, text="Include Statistics", variable=include_stats_var).pack(anchor="w")
        
        include_metadata_var = tk.BooleanVar(value=config.include_metadata)
        tk.Checkbutton(options_frame, text="Include Metadata", variable=include_metadata_var).pack(anchor="w")
    
    # Кнопки
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=7, column=0, columnspan=2, pady=20)
    
    def export_heatmap():
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{format_var.get()}",
            filetypes=[(f"{format_var.get().upper()} files", f"*.{format_var.get()}"), ("All files", "*.*")]
        )
        if filename:
            config.format = format_var.get()
            config.quality = quality_var.get()
            config.dpi = dpi_var.get()
            config.width = width_var.get()
            config.height = height_var.get()
            config.colormap = colormap_var.get()
            config.include_legend = include_legend_var.get()
            if not globe_export:
                config.include_statistics = include_stats_var.get()
                config.include_metadata = include_metadata_var.get()
            
            if globe_export:
                # Для експорту глобуса потрібно зробити скріншот
                # Отримуємо посилання на головне вікно через parent
                try:
                    # Спробуємо знайти метод _capture_globe_screenshot в головному вікні
                    if hasattr(parent, '_capture_globe_screenshot'):
                        globe_img = parent._capture_globe_screenshot(config.width, config.height)
                        if globe_img:
                            # Зберігаємо зображення глобуса
                            self._save_image(globe_img, filename)
                            print(f"Globe image exported to {filename}")
                            
                            # Якщо потрібно додати аналіз
                            if include_analysis_var.get() and hasattr(parent, '_current_analysis_data') and hasattr(parent, '_current_analysis_mask'):
                                if parent._current_analysis_data is not None and parent._current_analysis_mask is not None:
                                    self._add_analysis_overlay_to_globe(globe_img, filename, parent)
                        else:
                            print("Failed to capture globe screenshot")
                    else:
                        print("Globe capture functionality not available")
                except Exception as e:
                    print(f"Error exporting globe: {e}")
            else:
                exporter = AnalysisExporter(config)
                if exporter.export_heatmap(data, mask, bbox, filename, title):
                    print(f"Exported heatmap to {filename}")
                else:
                    print("Failed to export heatmap")
    
    def export_report():
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{format_var.get()}",
            filetypes=[(f"{format_var.get().upper()} files", f"*.{format_var.get()}"), ("All files", "*.*")]
        )
        if filename:
            config.format = format_var.get()
            config.quality = quality_var.get()
            config.dpi = dpi_var.get()
            config.width = width_var.get()
            config.height = height_var.get()
            config.colormap = colormap_var.get()
            config.include_legend = include_legend_var.get()
            config.include_statistics = include_stats_var.get()
            config.include_metadata = include_metadata_var.get()
            
            exporter = AnalysisExporter(config)
            if exporter.export_analysis_report(data, mask, bbox, filename, title):
                print(f"Exported report to {filename}")
            else:
                print("Failed to export report")
    
    def export_csv():
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            exporter = AnalysisExporter(config)
            if exporter.export_data_csv(data, mask, bbox, filename):
                print(f"Exported CSV to {filename}")
            else:
                print("Failed to export CSV")
    
    if globe_export:
        tk.Button(button_frame, text="Export Globe", command=export_heatmap).pack(side=tk.LEFT, padx=5)
    else:
        tk.Button(button_frame, text="Export Heatmap", command=export_heatmap).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export Report", command=export_report).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export CSV", command=export_csv).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    # Налаштовуємо grid weights
    main_frame.columnconfigure(1, weight=1)
