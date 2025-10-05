from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GeoBoundingBox:
    """Geographic bounding box expressed in degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def validate(self) -> None:
        if not (
            math.isfinite(self.min_lon)
            and math.isfinite(self.max_lon)
            and math.isfinite(self.min_lat)
            and math.isfinite(self.max_lat)
        ):
            raise ValueError("Bounding box coordinates must be finite numbers")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon < -180.0 or self.max_lon > 180.0:
            raise ValueError("Longitude bounds must lie within [-180, 180]")
        if self.min_lat < -90.0 or self.max_lat > 90.0:
            raise ValueError("Latitude bounds must lie within [-90, 90]")

    def intersection(self, other: "GeoBoundingBox") -> Optional["GeoBoundingBox"]:
        min_lon = max(self.min_lon, other.min_lon)
        max_lon = min(self.max_lon, other.max_lon)
        min_lat = max(self.min_lat, other.min_lat)
        max_lat = min(self.max_lat, other.max_lat)
        if min_lon >= max_lon or min_lat >= max_lat:
            return None
        return GeoBoundingBox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.min_lon <= lon <= self.max_lon
            and self.min_lat <= lat <= self.max_lat
        )

    def width(self) -> float:
        return self.max_lon - self.min_lon

    def height(self) -> float:
        return self.max_lat - self.min_lat

    def area_deg2(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())


FULL_EARTH_BBOX = GeoBoundingBox(
    min_lon=-180.0,
    min_lat=-90.0,
    max_lon=180.0,
    max_lat=90.0,
)


def parse_bbox_string(text: str) -> Optional[GeoBoundingBox]:
    cleaned = text.strip()
    if not cleaned:
        return None
    parts = cleaned.replace(";", ",").split(",")
    if len(parts) != 4:
        return None
    try:
        values = [float(p.strip()) for p in parts]
    except ValueError:
        return None
    bbox = GeoBoundingBox(
        min_lon=values[0],
        min_lat=values[1],
        max_lon=values[2],
        max_lat=values[3],
    )
    try:
        bbox.validate()
    except ValueError:
        return None
    return bbox


def lon_lat_grids(
    width: int,
    height: int,
    bbox: GeoBoundingBox,
) -> Tuple[np.ndarray, np.ndarray]:
    bbox.validate()
    if width <= 0 or height <= 0:
        raise ValueError("Raster dimensions must be positive")
    lon_step = bbox.width() / width
    lat_step = bbox.height() / height
    lon_centers = bbox.min_lon + (np.arange(width, dtype=np.float32) + 0.5) * lon_step
    lat_centers = bbox.max_lat - (np.arange(height, dtype=np.float32) + 0.5) * lat_step
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    return lon_grid, lat_grid


def roi_mask(
    width: int,
    height: int,
    image_bbox: GeoBoundingBox,
    roi: GeoBoundingBox,
) -> np.ndarray:
    intersection = image_bbox.intersection(roi)
    if intersection is None:
        return np.zeros((height, width), dtype=bool)
    lon_grid, lat_grid = lon_lat_grids(width, height, image_bbox)
    mask = (
        (lon_grid >= intersection.min_lon)
        & (lon_grid <= intersection.max_lon)
        & (lat_grid >= intersection.min_lat)
        & (lat_grid <= intersection.max_lat)
    )
    return mask

