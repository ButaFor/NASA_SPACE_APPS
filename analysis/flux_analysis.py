from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from gui.terra_config import TerraLayerOption, TerraRequest, TerraTextureController

from .common import FULL_EARTH_BBOX, GeoBoundingBox, roi_mask
from .interface import AnalysisKind, AnalysisRequest

LOGGER = logging.getLogger(__name__)
_ALPHA_THRESHOLD = 8


@dataclass(frozen=True)
class FluxMonthlyStats:
    date: dt.date
    mean_index: float
    median_index: float
    p10_index: float
    p90_index: float
    std_index: float


@dataclass(frozen=True)
class FluxAnalysisResult:
    request: AnalysisRequest
    stats: List[FluxMonthlyStats]
    trend_per_year: float
    valid_months: int
    mean_map: np.ndarray
    roi_mask: np.ndarray
    width: int
    height: int

    @property
    def baseline_mean(self) -> float:
        if not self.stats:
            return 0.0
        return self.stats[0].mean_index

    @property
    def latest_mean(self) -> float:
        if not self.stats:
            return 0.0
        return self.stats[-1].mean_index

    @property
    def has_heatmap(self) -> bool:
        return bool(self.mean_map.size) and np.isfinite(self.mean_map).any()


class FluxAnalyzer:
    """Quantify longwave flux variability across a period."""

    def __init__(self, *, image_bbox: GeoBoundingBox = FULL_EARTH_BBOX) -> None:
        self._image_bbox = image_bbox

    def analyze(
        self,
        controller: TerraTextureController,
        layer: TerraLayerOption,
        request: AnalysisRequest,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> FluxAnalysisResult:
        if request.kind != AnalysisKind.FLUX:
            raise ValueError("FluxAnalyzer expects a FLUX analysis request")
        if layer.time_mode not in {"monthly", "daily"}:
            raise ValueError("Flux analysis expects a monthly-capable layer")
        normalized = request.normalized()
        months = self._collect_months(normalized.start_date, normalized.end_date)
        total_months = len(months)
        resolution = normalized.resolution
        width = resolution
        height = max(1, resolution // 2)

        roi = normalized.bbox or FULL_EARTH_BBOX
        roi_mask_arr = roi_mask(width, height, self._image_bbox, roi)
        roi_pixels = int(roi_mask_arr.sum())
        if roi_pixels == 0:
            raise ValueError("ROI does not overlap requested imagery")

        stats: List[FluxMonthlyStats] = []
        sum_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.uint16)

        month_index = 0
        for current in months:
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
                LOGGER.warning("No flux imagery for %s on %s", layer.layer_id, current.isoformat())
                month_index += 1
                if progress_callback is not None:
                    progress_callback(month_index, total_months)
                continue

            image = result.image.convert("RGBA")
            rgba = np.array(image, dtype=np.uint8)
            if rgba.shape[0] != height or rgba.shape[1] != width:
                image = image.resize((width, height), Image.BILINEAR)
                rgba = np.array(image, dtype=np.uint8)

            alpha = rgba[:, :, 3] > _ALPHA_THRESHOLD
            roi_valid = roi_mask_arr & alpha
            valid_count = int(roi_valid.sum())
            if valid_count == 0:
                LOGGER.debug("Skipping %s (%s) - no valid pixels in ROI", layer.layer_id, current)
                month_index += 1
                if progress_callback is not None:
                    progress_callback(month_index, total_months)
                continue

            rgb = rgba[:, :, :3].astype(np.float32) / 255.0
            luminance = (
                0.2126 * rgb[:, :, 0]
                + 0.7152 * rgb[:, :, 1]
                + 0.0722 * rgb[:, :, 2]
            )
            samples = luminance[roi_valid]
            mean_index = float(samples.mean()) if samples.size else 0.0
            median_index = float(np.median(samples)) if samples.size else 0.0
            p10_index = float(np.percentile(samples, 10)) if samples.size else 0.0
            p90_index = float(np.percentile(samples, 90)) if samples.size else 0.0
            std_index = float(samples.std(ddof=0)) if samples.size else 0.0

            stats.append(
                FluxMonthlyStats(
                    date=current,
                    mean_index=mean_index,
                    median_index=median_index,
                    p10_index=p10_index,
                    p90_index=p90_index,
                    std_index=std_index,
                )
            )

            np.add(sum_map, luminance, out=sum_map, where=roi_valid, casting="unsafe")
            np.add(count_map, 1, out=count_map, where=roi_valid, casting="unsafe")

            month_index += 1
            if progress_callback is not None:
                progress_callback(month_index, total_months)

        valid_months = len(stats)
        trend_per_year = 0.0
        if valid_months >= 2:
            x = np.arange(valid_months, dtype=np.float64)
            y = np.array([entry.mean_index for entry in stats], dtype=np.float64)
            slope, _intercept = np.polyfit(x, y, 1)
            trend_per_year = float(slope * 12.0)

        mean_map = np.zeros_like(sum_map, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(sum_map, np.maximum(count_map, 1), out=mean_map, where=count_map > 0)
        mean_map[~roi_mask_arr] = np.nan

        return FluxAnalysisResult(
            request=normalized,
            stats=stats,
            trend_per_year=trend_per_year,
            valid_months=valid_months,
            mean_map=mean_map,
            roi_mask=roi_mask_arr,
            width=width,
            height=height,
        )

    @staticmethod
    def _collect_months(start: dt.date, end: dt.date) -> List[dt.date]:
        start_month = start.replace(day=1)
        end_month = end.replace(day=1)
        months: List[dt.date] = []
        current = start_month
        while current <= end_month:
            months.append(current)
            year = current.year + (current.month // 12)
            month = (current.month % 12) + 1
            current = dt.date(year, month, 1)
        return months


def format_flux_analysis(result: FluxAnalysisResult) -> str:
    lines: List[str] = []
    rq = result.request
    lines.append(
        f"Flux analysis ({rq.start_date.isoformat()} to {rq.end_date.isoformat()}, {result.valid_months} valid months)"
    )
    bbox = rq.bbox
    lines.append(
        f"ROI lon/lat: [{bbox.min_lon:.2f}, {bbox.min_lat:.2f}] -> [{bbox.max_lon:.2f}, {bbox.max_lat:.2f}]"
    )
    if not result.valid_months:
        lines.append("No valid flux imagery available for the selected period.")
        return "\n".join(lines)

    delta = (result.latest_mean - result.baseline_mean) * 100.0
    lines.append(
        f"Mean index change (latest vs first): {delta:+0.2f} pp"
    )
    lines.append(
        f"Linear trend: {result.trend_per_year * 100.0:+0.2f} pp/year"
    )

    extrema_high = max(result.stats, key=lambda entry: entry.mean_index)
    extrema_low = min(result.stats, key=lambda entry: entry.mean_index)
    lines.append(
        f"Highest mean index {extrema_high.mean_index * 100.0:0.2f} pp on {extrema_high.date.isoformat()}"
    )
    lines.append(
        f"Lowest mean index {extrema_low.mean_index * 100.0:0.2f} pp on {extrema_low.date.isoformat()}"
    )

    recent = result.stats[-3:]
    if recent:
        avg_recent = sum(entry.mean_index for entry in recent) / len(recent)
        lines.append(
            f"Recent 3-month average: {avg_recent * 100.0:0.2f} pp"
        )

    return "\n".join(lines)
