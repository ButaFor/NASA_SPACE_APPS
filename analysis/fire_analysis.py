from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from gui.terra_config import TerraLayerOption, TerraRequest, TerraTextureController

from .common import FULL_EARTH_BBOX, GeoBoundingBox, lon_lat_grids, roi_mask
from .interface import AnalysisKind, AnalysisRequest

LOGGER = logging.getLogger(__name__)

_ALPHA_THRESHOLD = 8


@dataclass(frozen=True)
class FireDailyStats:
    date: dt.date
    fire_pixels: int
    coverage_fraction: float
    intensity_index: float
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]


@dataclass(frozen=True)
class FireAnalysisResult:
    request: AnalysisRequest
    stats: List[FireDailyStats]
    total_fire_pixels: int
    peak_day: Optional[dt.date]
    peak_pixels: int
    days_with_fire: int
    valid_days: int
    density_map: np.ndarray
    roi_mask: np.ndarray
    width: int
    height: int

    @property
    def mean_daily_pixels(self) -> float:
        if not self.valid_days:
            return 0.0
        return self.total_fire_pixels / self.valid_days

    @property
    def has_heatmap(self) -> bool:
        return bool(self.density_map.size) and int(self.density_map.max()) > 0


class FireAnalyzer:
    """Detect daily fire hotspots within a bounding box."""

    def __init__(self, *, image_bbox: GeoBoundingBox = FULL_EARTH_BBOX) -> None:
        self._image_bbox = image_bbox

    def analyze(
        self,
        controller: TerraTextureController,
        layer: TerraLayerOption,
        request: AnalysisRequest,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> FireAnalysisResult:
        if request.kind != AnalysisKind.FIRE:
            raise ValueError("FireAnalyzer expects a FIRE analysis request")
        if layer.time_mode != "daily":
            raise ValueError("Fire analysis requires a daily layer")
        normalized = request.normalized()
        total_days = normalized.spans()
        resolution = normalized.resolution
        width = resolution
        height = max(1, resolution // 2)

        roi = normalized.bbox or FULL_EARTH_BBOX
        roi_mask_arr = roi_mask(width, height, self._image_bbox, roi)
        roi_pixels = int(roi_mask_arr.sum())
        if roi_pixels == 0:
            raise ValueError("ROI does not overlap requested imagery")

        lon_grid, lat_grid = lon_lat_grids(width, height, self._image_bbox)

        stats: List[FireDailyStats] = []
        total_fire_pixels = 0
        peak_pixels = 0
        peak_day: Optional[dt.date] = None
        days_with_fire = 0
        valid_days = 0
        density_map = np.zeros((height, width), dtype=np.uint16)

        current = normalized.start_date
        day_index = 0
        while current <= normalized.end_date:
            terra_request = TerraRequest(
                layer_id=layer.layer_id,
                date=current,
                width=width,
                height=height,
                time_mode=layer.time_mode,
                layers=layer.layers,
                snapshot=layer.snapshot_config,
            )
            result = controller.fetch_sync(terra_request, use_cache=True)
            if result is None:
                LOGGER.warning("No imagery returned for %s on %s", layer.layer_id, current.isoformat())
                current += dt.timedelta(days=1)
                day_index += 1
                if progress_callback is not None:
                    progress_callback(day_index, total_days)
                continue

            image = result.image.convert("RGBA")
            rgba = np.array(image, dtype=np.uint8)
            if rgba.shape[0] != height or rgba.shape[1] != width:
                LOGGER.debug(
                    "Resizing imagery from %s to %s for analysis",
                    rgba.shape[:2],
                    (height, width),
                )
                image = image.resize((width, height), Image.BILINEAR)
                rgba = np.array(image, dtype=np.uint8)

            alpha = rgba[:, :, 3] > _ALPHA_THRESHOLD
            roi_valid = roi_mask_arr & alpha
            valid_count = int(roi_valid.sum())
            if valid_count == 0:
                LOGGER.debug("Skipping %s (%s) - no valid pixels in ROI", layer.layer_id, current)
                current += dt.timedelta(days=1)
                day_index += 1
                if progress_callback is not None:
                    progress_callback(day_index, total_days)
                continue

            red = rgba[:, :, 0].astype(np.int16)
            green = rgba[:, :, 1].astype(np.int16)
            blue = rgba[:, :, 2].astype(np.int16)

            fire_mask = (
                roi_valid
                & (red > 160)
                & (red - green > 45)
                & (red - blue > 45)
                & (green < 200)
            )
            fire_pixels = int(fire_mask.sum())
            coverage_fraction = fire_pixels / valid_count if valid_count else 0.0

            centroid_lat: Optional[float] = None
            centroid_lon: Optional[float] = None
            intensity_index = 0.0
            if fire_pixels:
                days_with_fire += 1
                fire_lats = lat_grid[fire_mask]
                fire_lons = lon_grid[fire_mask]
                centroid_lat = float(fire_lats.mean())
                centroid_lon = float(fire_lons.mean())
                red_dom = red[fire_mask]
                green_dom = green[fire_mask]
                blue_dom = blue[fire_mask]
                red_surplus = np.maximum(0, red_dom - np.maximum(green_dom, blue_dom))
                intensity_index = float(np.clip(red_surplus.mean() / 255.0, 0.0, 1.0))
                np.add(density_map, 1, out=density_map, where=fire_mask, casting="unsafe")
            valid_days += 1
            stats.append(
                FireDailyStats(
                    date=current,
                    fire_pixels=fire_pixels,
                    coverage_fraction=coverage_fraction,
                    intensity_index=intensity_index,
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                )
            )
            total_fire_pixels += fire_pixels
            if fire_pixels > peak_pixels:
                peak_pixels = fire_pixels
                peak_day = current

            current += dt.timedelta(days=1)
            day_index += 1
            if progress_callback is not None:
                progress_callback(day_index, total_days)

        if density_map.size:
            density_map[~roi_mask_arr] = 0

        return FireAnalysisResult(
            request=normalized,
            stats=stats,
            total_fire_pixels=total_fire_pixels,
            peak_day=peak_day,
            peak_pixels=peak_pixels,
            days_with_fire=days_with_fire,
            valid_days=valid_days,
            density_map=density_map,
            roi_mask=roi_mask_arr,
            width=width,
            height=height,
        )


def format_fire_analysis(result: FireAnalysisResult) -> str:
    lines: List[str] = []
    rq = result.request
    lines.append(
        f"Fire analysis ({rq.start_date.isoformat()} to {rq.end_date.isoformat()}, {result.valid_days} valid days)"
    )
    bbox = rq.bbox
    lines.append(
        f"ROI lon/lat: [{bbox.min_lon:.2f}, {bbox.min_lat:.2f}] -> [{bbox.max_lon:.2f}, {bbox.max_lat:.2f}]"
    )
    if not result.valid_days:
        lines.append("No valid imagery available for the selected period.")
        return "\n".join(lines)
    if result.days_with_fire == 0:
        lines.append("No active fire pixels detected inside ROI.")
        return "\n".join(lines)

    lines.append(
        f"Active fire days: {result.days_with_fire}/{result.valid_days}; "
        f"mean pixels/day: {result.mean_daily_pixels:.1f}"
    )
    if result.peak_day is not None:
        lines.append(
            f"Peak activity on {result.peak_day.isoformat()}: {result.peak_pixels} pixels"
        )

    top_days = sorted(result.stats, key=lambda s: s.fire_pixels, reverse=True)[:3]
    lines.append("Top days inside ROI:")
    for stat in top_days:
        if stat.centroid_lat is not None and stat.centroid_lon is not None:
            centroid = f" @ ({stat.centroid_lat:.2f} deg, {stat.centroid_lon:.2f} deg)"
        else:
            centroid = ""
        lines.append(
            f"  - {stat.date.isoformat()}: {stat.fire_pixels} px, "
            f"coverage {stat.coverage_fraction * 100:.3f}%{centroid}"
        )
    return "\n".join(lines)
