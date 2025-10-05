from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

from .common import FULL_EARTH_BBOX, GeoBoundingBox


class AnalysisKind(str, Enum):
    FIRE = "fire"
    FLUX = "flux"


@dataclass(frozen=True)
class AnalysisRequest:
    kind: AnalysisKind
    layer_id: str
    start_date: dt.date
    end_date: dt.date
    bbox: GeoBoundingBox = FULL_EARTH_BBOX
    resolution: int = 1024

    def normalized(self) -> "AnalysisRequest":
        if self.start_date > self.end_date:
            return AnalysisRequest(
                kind=self.kind,
                layer_id=self.layer_id,
                start_date=self.end_date,
                end_date=self.start_date,
                bbox=self.bbox,
                resolution=self.resolution,
            )
        return self

    def clamp_to(self, *, min_date: dt.date, max_date: dt.date) -> "AnalysisRequest":
        start = max(min_date, self.start_date)
        end = min(max_date, self.end_date)
        if start > end:
            start = end
        return AnalysisRequest(
            kind=self.kind,
            layer_id=self.layer_id,
            start_date=start,
            end_date=end,
            bbox=self.bbox,
            resolution=self.resolution,
        )

    def spans(self) -> int:
        return (self.end_date - self.start_date).days + 1

