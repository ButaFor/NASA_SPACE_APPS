"""Розширений дисплей результатів аналізу для Terra Tools.

Цей модуль надає детальний дисплей результатів аналізу з графіками,
статистикою та інтерактивними елементами.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from analysis.common import GeoBoundingBox


@dataclass
class AnalysisDisplayConfig:
    """Конфігурація дисплею аналізу."""
    width: int = 800
    height: int = 600
    font_size: int = 12
    title_font_size: int = 16
    show_statistics: bool = True
    show_timeline: bool = True
    show_heatmap: bool = True
    show_legend: bool = True


class AnalysisDisplay:
    """Дисплей результатів аналізу."""
    
    def __init__(self, parent: tk.Widget, config: Optional[AnalysisDisplayConfig] = None):
        self.parent = parent
        self.config = config or AnalysisDisplayConfig()
        
        # Дані аналізу
        self.analysis_data: Optional[np.ndarray] = None
        self.analysis_mask: Optional[np.ndarray] = None
        self.analysis_bbox: Optional[GeoBoundingBox] = None
        self.analysis_title: str = "Analysis Results"
        self.analysis_stats: dict = {}
        
        # Створюємо GUI
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Створює GUI елементи."""
        # Основний фрейм
        self.main_frame = tk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        self.title_label = tk.Label(
            self.main_frame,
            text=self.analysis_title,
            font=("Arial", self.config.title_font_size, "bold")
        )
        self.title_label.pack(pady=10)
        
        # Створюємо notebook для різних вкладок
        self.notebook = tk.ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка з heatmap
        self.heatmap_frame = tk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="Heatmap")
        
        # Canvas для heatmap
        self.heatmap_canvas = tk.Canvas(
            self.heatmap_frame,
            width=self.config.width,
            height=self.config.height // 2,
            bg="black"
        )
        self.heatmap_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка зі статистикою
        self.stats_frame = tk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Текстовий віджет для статистики
        self.stats_text = tk.Text(
            self.stats_frame,
            width=80,
            height=20,
            font=("Courier", self.config.font_size)
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar для статистики
        stats_scrollbar = tk.Scrollbar(self.stats_frame)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        stats_scrollbar.config(command=self.stats_text.yview)
        
        # Вкладка з графіками
        self.graphs_frame = tk.Frame(self.notebook)
        self.notebook.add(self.graphs_frame, text="Graphs")
        
        # Canvas для графіків
        self.graphs_canvas = tk.Canvas(
            self.graphs_frame,
            width=self.config.width,
            height=self.config.height // 2,
            bg="white"
        )
        self.graphs_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель керування
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Кнопки експорту
        tk.Button(self.control_frame, text="Export Image", command=self._export_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Export Data", command=self._export_data).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Refresh", command=self._refresh_display).pack(side=tk.LEFT, padx=5)
    
    def set_analysis_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        bbox: GeoBoundingBox,
        title: str = "Analysis Results",
        stats: Optional[dict] = None
    ) -> None:
        """Встановлює дані аналізу."""
        self.analysis_data = data
        self.analysis_mask = mask
        self.analysis_bbox = bbox
        self.analysis_title = title
        self.analysis_stats = stats or {}
        
        # Оновлюємо відображення
        self._update_display()
    
    def _update_display(self) -> None:
        """Оновлює відображення."""
        if self.analysis_data is None:
            return
        
        # Оновлюємо заголовок
        self.title_label.config(text=self.analysis_title)
        
        # Оновлюємо heatmap
        self._update_heatmap()
        
        # Оновлюємо статистику
        self._update_statistics()
        
        # Оновлюємо графіки
        self._update_graphs()
    
    def _update_heatmap(self) -> None:
        """Оновлює heatmap."""
        if self.analysis_data is None or self.analysis_mask is None:
            return
        
        try:
            # Створюємо heatmap зображення
            masked_data = np.where(self.analysis_mask, self.analysis_data, np.nan)
            
            # Нормалізуємо дані
            valid_data = masked_data[np.isfinite(masked_data)]
            if valid_data.size == 0:
                return
            
            vmin = float(valid_data.min())
            vmax = float(valid_data.max())
            
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            
            normalized = (masked_data - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Застосовуємо colormap
            height, width = normalized.shape
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Hot colormap
            r = np.clip(3 * normalized, 0, 1)
            g = np.clip(3 * normalized - 1, 0, 1)
            b = np.clip(3 * normalized - 2, 0, 1)
            
            rgba[..., 0] = (r * 255).astype(np.uint8)
            rgba[..., 1] = (g * 255).astype(np.uint8)
            rgba[..., 2] = (b * 255).astype(np.uint8)
            rgba[..., 3] = (normalized * 255).astype(np.uint8)
            
            # Створюємо зображення
            img = Image.fromarray(rgba, mode="RGBA")
            
            # Масштабуємо до розміру canvas
            canvas_width = self.heatmap_canvas.winfo_width()
            canvas_height = self.heatmap_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img = img.resize((canvas_width, canvas_height), Image.BILINEAR)
                
                # Конвертуємо в PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Очищаємо canvas та додаємо зображення
                self.heatmap_canvas.delete("all")
                self.heatmap_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                
                # Зберігаємо посилання на зображення
                self.heatmap_canvas.image = photo
                
        except Exception as e:
            print(f"Error updating heatmap: {e}")
    
    def _update_statistics(self) -> None:
        """Оновлює статистику."""
        if self.analysis_data is None or self.analysis_mask is None:
            return
        
        try:
            masked_data = np.where(self.analysis_mask, self.analysis_data, np.nan)
            valid_data = masked_data[np.isfinite(masked_data)]
            
            if valid_data.size == 0:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, "No valid data available.")
                return
            
            # Розраховуємо статистику
            stats = {
                "Count": int(valid_data.size),
                "Min": float(valid_data.min()),
                "Max": float(valid_data.max()),
                "Mean": float(valid_data.mean()),
                "Median": float(np.median(valid_data)),
                "Std": float(valid_data.std()),
                "Q25": float(np.percentile(valid_data, 25)),
                "Q75": float(np.percentile(valid_data, 75)),
            }
            
            # Додаємо користувацьку статистику
            if self.analysis_stats:
                stats.update(self.analysis_stats)
            
            # Форматуємо вивід
            self.stats_text.delete(1.0, tk.END)
            
            # Заголовок
            self.stats_text.insert(tk.END, f"{self.analysis_title}\n")
            self.stats_text.insert(tk.END, "=" * len(self.analysis_title) + "\n\n")
            
            # ROI інформація
            if self.analysis_bbox:
                self.stats_text.insert(tk.END, "ROI Information:\n")
                self.stats_text.insert(tk.END, f"  Longitude: {self.analysis_bbox.min_lon:.3f} to {self.analysis_bbox.max_lon:.3f}\n")
                self.stats_text.insert(tk.END, f"  Latitude: {self.analysis_bbox.min_lat:.3f} to {self.analysis_bbox.max_lat:.3f}\n")
                self.stats_text.insert(tk.END, f"  Area: {self.analysis_bbox.area_deg2():.3f} deg²\n\n")
            
            # Статистика даних
            self.stats_text.insert(tk.END, "Data Statistics:\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    self.stats_text.insert(tk.END, f"  {key}: {value:.6f}\n")
                else:
                    self.stats_text.insert(tk.END, f"  {key}: {value}\n")
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Error calculating statistics: {e}")
    
    def _update_graphs(self) -> None:
        """Оновлює графіки."""
        if self.analysis_data is None or self.analysis_mask is None:
            return
        
        try:
            # Очищаємо canvas
            self.graphs_canvas.delete("all")
            
            masked_data = np.where(self.analysis_mask, self.analysis_data, np.nan)
            valid_data = masked_data[np.isfinite(masked_data)]
            
            if valid_data.size == 0:
                return
            
            # Розраховуємо гістограму
            hist, bins = np.histogram(valid_data, bins=50)
            
            # Масштабуємо до розміру canvas
            canvas_width = self.graphs_canvas.winfo_width()
            canvas_height = self.graphs_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Малюємо гістограму
            max_hist = float(hist.max())
            if max_hist > 0:
                bar_width = canvas_width // len(hist)
                
                for i, count in enumerate(hist):
                    bar_height = int((count / max_hist) * canvas_height * 0.8)
                    x1 = i * bar_width
                    y1 = canvas_height - bar_height
                    x2 = (i + 1) * bar_width
                    y2 = canvas_height
                    
                    # Кольорова схема на основі висоти
                    color_intensity = int((count / max_hist) * 255)
                    color = f"#{color_intensity:02x}0000"
                    
                    self.graphs_canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=color, outline="black", width=1
                    )
            
            # Малюємо осі
            self.graphs_canvas.create_line(0, canvas_height - 1, canvas_width, canvas_height - 1, fill="black", width=2)
            self.graphs_canvas.create_line(0, 0, 0, canvas_height, fill="black", width=2)
            
            # Додаємо підписи
            self.graphs_canvas.create_text(
                canvas_width // 2, canvas_height - 10,
                text="Value", font=("Arial", 10)
            )
            self.graphs_canvas.create_text(
                10, canvas_height // 2,
                text="Frequency", font=("Arial", 10), angle=90
            )
            
        except Exception as e:
            print(f"Error updating graphs: {e}")
    
    def _export_image(self) -> None:
        """Експортує зображення."""
        if self.analysis_data is None:
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filename:
                # Створюємо зображення для експорту
                masked_data = np.where(self.analysis_mask, self.analysis_data, np.nan)
                
                # Нормалізуємо дані
                valid_data = masked_data[np.isfinite(masked_data)]
                if valid_data.size == 0:
                    return
                
                vmin = float(valid_data.min())
                vmax = float(valid_data.max())
                
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1e-6
                
                normalized = (masked_data - vmin) / (vmax - vmin)
                normalized = np.clip(normalized, 0.0, 1.0)
                
                # Застосовуємо colormap
                height, width = normalized.shape
                rgba = np.zeros((height, width, 4), dtype=np.uint8)
                
                r = np.clip(3 * normalized, 0, 1)
                g = np.clip(3 * normalized - 1, 0, 1)
                b = np.clip(3 * normalized - 2, 0, 1)
                
                rgba[..., 0] = (r * 255).astype(np.uint8)
                rgba[..., 1] = (g * 255).astype(np.uint8)
                rgba[..., 2] = (b * 255).astype(np.uint8)
                rgba[..., 3] = (normalized * 255).astype(np.uint8)
                
                # Створюємо зображення
                img = Image.fromarray(rgba, mode="RGBA")
                img.save(filename)
                print(f"Image exported to {filename}")
                
        except Exception as e:
            print(f"Error exporting image: {e}")
    
    def _export_data(self) -> None:
        """Експортує дані."""
        if self.analysis_data is None:
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                masked_data = np.where(self.analysis_mask, self.analysis_data, np.nan)
                valid_data = masked_data[np.isfinite(masked_data)]
                
                with open(filename, 'w') as f:
                    f.write(f"Analysis Results: {self.analysis_title}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    if self.analysis_bbox:
                        f.write(f"ROI: {self.analysis_bbox.min_lon:.3f}, {self.analysis_bbox.min_lat:.3f} -> {self.analysis_bbox.max_lon:.3f}, {self.analysis_bbox.max_lat:.3f}\n\n")
                    
                    f.write("Data values:\n")
                    for value in valid_data:
                        f.write(f"{value:.6f}\n")
                
                print(f"Data exported to {filename}")
                
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def _refresh_display(self) -> None:
        """Оновлює відображення."""
        self._update_display()


def create_analysis_display_dialog(
    parent: tk.Widget,
    data: np.ndarray,
    mask: np.ndarray,
    bbox: GeoBoundingBox,
    title: str = "Analysis Results",
    stats: Optional[dict] = None
) -> None:
    """Створює діалог відображення результатів аналізу."""
    dialog = tk.Toplevel(parent)
    dialog.title("Analysis Results")
    dialog.geometry("900x700")
    dialog.resizable(True, True)
    
    # Центруємо діалог
    dialog.transient(parent)
    dialog.grab_set()
    
    # Створюємо дисплей
    display = AnalysisDisplay(dialog)
    display.set_analysis_data(data, mask, bbox, title, stats)
    
    # Кнопка закриття
    button_frame = tk.Frame(dialog)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    tk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
