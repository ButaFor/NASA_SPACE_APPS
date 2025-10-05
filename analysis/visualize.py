from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .fire_analysis import FireAnalysisResult
from .flux_analysis import FluxAnalysisResult


def _resize_if_needed(image: Image.Image, *, max_dimension: int = 768) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BILINEAR)


def fire_heatmap_image(result: FireAnalysisResult, *, max_dimension: int = 768) -> Optional[Image.Image]:
    if not result.has_heatmap:
        return None
    data = result.density_map.astype(np.float32)
    mask = result.roi_mask.astype(bool)
    valid = data[mask]
    if valid.size == 0:
        return None
    peak = float(valid.max())
    if peak <= 0.0:
        return None
    
    # Нормалізуємо дані
    norm = np.zeros_like(data, dtype=np.float32)
    norm[mask] = data[mask] / peak
    norm = np.clip(norm, 0.0, 1.0)

    # Розраховуємо розмір зображення
    aspect_ratio = result.width / result.height
    if aspect_ratio > 1:
        new_width = max_dimension
        new_height = int(max_dimension / aspect_ratio)
    else:
        new_width = int(max_dimension * aspect_ratio)
        new_height = max_dimension

    # Створюємо зображення з білим фоном
    rgba = np.full((new_height, new_width, 4), 255, dtype=np.uint8)  # Білий фон
    
    # ВИПРАВЛЕННЯ: Розтягуємо ROI на весь розмір зображення
    roi_coords = np.where(mask)
    if len(roi_coords[0]) > 0:
        # Знаходимо межі ROI
        min_row, max_row = roi_coords[0].min(), roi_coords[0].max()
        min_col, max_col = roi_coords[1].min(), roi_coords[1].max()
        
        # Витягуємо ROI дані
        roi_data = norm[min_row:max_row+1, min_col:max_col+1]
        
        # Масштабуємо ROI дані на весь розмір зображення
        from PIL import Image
        temp_img = Image.fromarray((roi_data * 255).astype(np.uint8))
        scaled_img = temp_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        scaled_roi_data = np.array(scaled_img) / 255.0
        
        # Застосовуємо colormap до всього зображення
        r = 255 * scaled_roi_data
        g = 200 * (1.0 - 0.25 * scaled_roi_data)
        b = 32 * (1.0 - scaled_roi_data)
        
        rgba[..., 0] = r.astype(np.uint8)
        rgba[..., 1] = g.astype(np.uint8)
        rgba[..., 2] = b.astype(np.uint8)
        rgba[..., 3] = 255  # Повна непрозорість

    image = Image.fromarray(rgba, mode="RGBA")
    return image


def flux_heatmap_image(result: FluxAnalysisResult, *, max_dimension: int = 768) -> Optional[Image.Image]:
    if not result.has_heatmap:
        return None
    data = result.mean_map.astype(np.float32)
    mask = result.roi_mask.astype(bool)
    valid = data[mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None
    vmin = float(valid.min())
    vmax = float(valid.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax - vmin <= 1e-6:
        vmax = vmin + 1e-6
    
    # Нормалізуємо дані
    norm = np.zeros_like(data, dtype=np.float32)
    norm[mask] = (data[mask] - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # Розраховуємо розмір зображення
    aspect_ratio = result.width / result.height
    if aspect_ratio > 1:
        new_width = max_dimension
        new_height = int(max_dimension / aspect_ratio)
    else:
        new_width = int(max_dimension * aspect_ratio)
        new_height = max_dimension

    # Створюємо зображення з білим фоном
    rgba = np.full((new_height, new_width, 4), 255, dtype=np.uint8)  # Білий фон
    
    # ВИПРАВЛЕННЯ: Розтягуємо ROI на весь розмір зображення
    roi_coords = np.where(mask)
    if len(roi_coords[0]) > 0:
        # Знаходимо межі ROI
        min_row, max_row = roi_coords[0].min(), roi_coords[0].max()
        min_col, max_col = roi_coords[1].min(), roi_coords[1].max()
        
        # Витягуємо ROI дані
        roi_data = norm[min_row:max_row+1, min_col:max_col+1]
        
        # Масштабуємо ROI дані на весь розмір зображення
        from PIL import Image
        temp_img = Image.fromarray((roi_data * 255).astype(np.uint8))
        scaled_img = temp_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        scaled_roi_data = np.array(scaled_img) / 255.0
        
        # Застосовуємо colormap до всього зображення
        # Simple blue -> white -> red diverging colormap
        mid = 0.5
        low_mask = scaled_roi_data <= mid
        high_mask = scaled_roi_data > mid

        # Low values: blue to white
        if np.any(low_mask):
            frac = np.zeros_like(scaled_roi_data)
            frac[low_mask] = scaled_roi_data[low_mask] / mid
            rgba[..., 0][low_mask] = (255 * frac[low_mask]).astype(np.uint8)
            rgba[..., 1][low_mask] = (255 * frac[low_mask]).astype(np.uint8)
            rgba[..., 2][low_mask] = 255

        # High values: white to red
        if np.any(high_mask):
            frac = np.zeros_like(scaled_roi_data)
            frac[high_mask] = (scaled_roi_data[high_mask] - mid) / (1.0 - mid)
            rgba[..., 0][high_mask] = 255
            rgba[..., 1][high_mask] = (255 * (1.0 - frac[high_mask])).astype(np.uint8)
            rgba[..., 2][high_mask] = (255 * (1.0 - frac[high_mask])).astype(np.uint8)

        rgba[..., 3] = 255  # Повна непрозорість

    image = Image.fromarray(rgba, mode="RGBA")
    return image
