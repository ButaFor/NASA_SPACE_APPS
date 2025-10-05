"""Utilities for Terra layer metadata and imagery downloading."""
from __future__ import annotations

import bisect
import datetime as dt
import logging
import io
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError
from requests import exceptions as requests_exceptions

WMS_ENDPOINT = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
WMS_VERSION = "1.1.1"
BBOX_FULL_EARTH = "-180,-90,180,90"
DEFAULT_DATE_RANGE_DAYS = 365
GLOBAL_MIN_DATE = dt.date(2000, 1, 1)
TERRA_RESOLUTIONS = [1024, 2048, 4096]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeDomainInfo:
    min_date: Optional[dt.date]
    max_date: Optional[dt.date]
    period: Optional[str]
    explicit_dates: tuple[dt.date, ...]
    raw_value: str

    def contains(self, value: dt.date) -> bool:
        if self.explicit_dates:
            return value in self.explicit_dates
        if self.min_date is not None and value < self.min_date:
            return False
        if self.max_date is not None and value > self.max_date:
            return False
        if self.period == "P1M":
            return value.day == 1
        return True

    def normalize(self, value: dt.date) -> dt.date:
        if self.explicit_dates:
            if not self.explicit_dates:
                return value
            idx = bisect.bisect_right(self.explicit_dates, value)
            if idx:
                return self.explicit_dates[idx - 1]
            return self.explicit_dates[0]
        normalized = value
        if self.min_date is not None and normalized < self.min_date:
            normalized = self.min_date
        if self.max_date is not None and normalized > self.max_date:
            normalized = self.max_date
        if self.period == "P1M":
            normalized = normalized.replace(day=1)
        return normalized


class WMSCapabilitiesCache:
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._lock = threading.Lock()
        self._domains: Optional[dict[str, TimeDomainInfo]] = None

    def get_time_domain(self, layer_name: str) -> Optional[TimeDomainInfo]:
        domains = self._ensure_domains()
        return domains.get(layer_name)

    def _ensure_domains(self) -> dict[str, TimeDomainInfo]:
        with self._lock:
            if self._domains is not None:
                return self._domains
        domains = self._load_domains()
        with self._lock:
            if self._domains is None:
                self._domains = domains
            return self._domains

    def _load_domains(self) -> dict[str, TimeDomainInfo]:
        params = {"SERVICE": "WMS", "REQUEST": "GetCapabilities", "VERSION": WMS_VERSION}
        try:
            response = requests.get(self._endpoint, params=params, timeout=(5, 30))
            response.raise_for_status()
        except requests_exceptions.RequestException as exc:
            LOGGER.warning("Failed to fetch WMS capabilities: %s", exc)
            return {}
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            LOGGER.warning("Failed to parse WMS capabilities XML: %s", exc)
            return {}
        return self._parse_capabilities(root)

    @staticmethod
    def _parse_capabilities(root: ET.Element) -> dict[str, TimeDomainInfo]:
        result: dict[str, TimeDomainInfo] = {}
        for layer_el in root.findall('.//{*}Layer'):
            name_el = layer_el.find('{*}Name')
            if name_el is None or not (name_el.text and name_el.text.strip()):
                continue
            layer_name = name_el.text.strip()
            time_el = None
            for tag in ('{*}Dimension', '{*}Extent'):
                for candidate in layer_el.findall(tag):
                    attr_name = (candidate.get('name') or '').lower()
                    if attr_name == 'time':
                        time_el = candidate
                        break
                if time_el is not None:
                    break
            if time_el is None:
                continue
            value = (time_el.text or '').strip()
            if not value:
                continue
            info = WMSCapabilitiesCache._parse_time_extent(value)
            if info is not None:
                result[layer_name] = info
        return result

    @staticmethod
    def _parse_time_extent(value: str) -> Optional[TimeDomainInfo]:
        min_date: Optional[dt.date] = None
        max_date: Optional[dt.date] = None
        explicit: set[dt.date] = set()
        period: Optional[str] = None
        for chunk in value.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '/' in chunk:
                parts = chunk.split('/')
                start = WMSCapabilitiesCache._parse_iso_date(parts[0])
                end = WMSCapabilitiesCache._parse_iso_date(parts[1]) if len(parts) > 1 else None
                step = parts[2] if len(parts) > 2 else None
                if start is not None:
                    min_date = start if min_date is None else min(min_date, start)
                if end is not None:
                    max_date = end if max_date is None else max(max_date, end)
                if step:
                    period = step
            else:
                date_val = WMSCapabilitiesCache._parse_iso_date(chunk)
                if date_val is None:
                    continue
                explicit.add(date_val)
                min_date = date_val if min_date is None else min(min_date, date_val)
                max_date = date_val if max_date is None else max(max_date, date_val)
        explicit_dates = tuple(sorted(explicit))
        if min_date is None and max_date is None and not explicit_dates:
            return None
        return TimeDomainInfo(
            min_date=min_date,
            max_date=max_date,
            period=period,
            explicit_dates=explicit_dates,
            raw_value=value,
        )

    @staticmethod
    def _parse_iso_date(value: str) -> Optional[dt.date]:
        text = value.strip()
        if not text:
            return None
        if text.lower() == 'present':
            return dt.date.today()
        cleaned = text
        if cleaned.endswith('Z'):
            cleaned = cleaned[:-1] + '+00:00'
        try:
            if 'T' in cleaned:
                return dt.datetime.fromisoformat(cleaned).date()
            return dt.date.fromisoformat(cleaned)
        except ValueError:
            try:
                return dt.date.fromisoformat(cleaned[:10])
            except ValueError:
                return None


_CAPABILITIES_CACHE = WMSCapabilitiesCache(WMS_ENDPOINT)


def get_time_domain(layer_id: str) -> Optional[TimeDomainInfo]:
    """Return cached WMS time-domain information for the requested layer."""
    return _CAPABILITIES_CACHE.get_time_domain(layer_id)


@dataclass(frozen=True)
class SnapshotConfig:
    endpoint: str
    layer: str
    bbox: str
    colormap: Optional[str] = None
    crs: str = "EPSG:4326"
    wrap: str = "none"
    image_format: str = "image/png"
    request: str = "GetSnapshot"


@dataclass(frozen=True)
class TerraLayerOption:
    layer_id: str
    label: str
    description: str
    time_mode: str  # "daily", "monthly", "none"
    min_date: Optional[dt.date] = None
    max_date: Optional[dt.date] = None
    layer_stack: tuple[str, ...] = ()
    snapshot_config: Optional[SnapshotConfig] = None

    @property
    def layers(self) -> tuple[str, ...]:
        """Return the sequence of WMS layers to request for this option."""
        return self.layer_stack or (self.layer_id,)


@dataclass(frozen=True)
class TerraRequest:
    layer_id: str
    date: Optional[dt.date]
    width: int
    height: int
    time_mode: str
    layers: Optional[tuple[str, ...]] = None
    snapshot: Optional[SnapshotConfig] = None

    def time_param(self) -> Optional[str]:
        if self.time_mode == "none" or self.date is None:
            return None
        if self.time_mode == "monthly":
            return self.date.strftime("%Y-%m-01")
        return self.date.strftime("%Y-%m-%d")

    def resolved_layers(self) -> tuple[str, ...]:
        """Return the stack of layers to send in the WMS request."""
        if self.snapshot is not None:
            return (self.snapshot.layer,)
        return self.layers or (self.layer_id,)

    def is_snapshot(self) -> bool:
        return self.snapshot is not None

    def cache_key(self) -> str:
        date_part = self.time_param() or "static"
        if self.snapshot is not None:
            base_id = self.snapshot.layer
            mode = "snapshot"
        else:
            base_id = "-".join(self.resolved_layers())
            mode = "wms"
        # Add a cache version suffix to invalidate older files saved before
        # transparency/alpha handling changes.
        return f"{mode}_{base_id}_{date_part}_{self.width}x{self.height}_v3"


@dataclass
class TerraDownloadResult:
    params: TerraRequest
    image: Image.Image
    from_cache: bool
    cache_path: Optional[Path]


def _daily_max_date(today: dt.date, lag_days: int = 5) -> dt.date:
    return max(GLOBAL_MIN_DATE, today - dt.timedelta(days=lag_days))


def _monthly_max_date(today: dt.date) -> dt.date:
    first_of_month = today.replace(day=1)
    previous_month_end = first_of_month - dt.timedelta(days=1)
    return previous_month_end.replace(day=1)


def default_layers() -> list[TerraLayerOption]:
    today = dt.date.today()
    daily_max = _daily_max_date(today)
    monthly_max = _monthly_max_date(today)
    return [
        TerraLayerOption(
            layer_id="MODIS_Terra_CorrectedReflectance_TrueColor",
            label="MODIS: True Color",
            description="Daily natural-color imagery.",
            time_mode="daily",
            min_date=GLOBAL_MIN_DATE,
            max_date=daily_max,
        ),
        TerraLayerOption(
            layer_id="MODIS_Terra_CorrectedReflectance_Bands721",
            label="MODIS: Bands 7-2-1",
            description="Infrared false-color composite (7-2-1).",
            time_mode="daily",
            min_date=GLOBAL_MIN_DATE,
            max_date=daily_max,
        ),
        TerraLayerOption(
            layer_id="MODIS_Terra_Thermal_Anomalies_All",
            label="MODIS: Thermal Anomalies",
            description="Active fire and thermal anomaly detections (FIRMS).",
            time_mode="daily",
            # GIBS WMS 'best' for this layer has reliable coverage from ~2018
            # Older dates frequently return ServiceException XML (no data).
            min_date=dt.date(2018, 1, 1),
            max_date=daily_max,
        ),
        TerraLayerOption(
            layer_id="MOPITT_CO_Daily_Surface_Mixing_Ratio_Day",
            label="MOPITT: CO Day",
            description="Surface-level carbon monoxide (daytime overpass).",
            time_mode="daily",
            min_date=dt.date(2017, 8, 18),
            max_date=min(dt.date(2025, 2, 1), daily_max),
            snapshot_config=SnapshotConfig(
                endpoint="https://wvs.earthdata.nasa.gov/api/v1/snapshot",
                layer="MOPITT_CO_Daily_Total_Column_L2",
                bbox="-90,-180,90,180",
                colormap="MOPITT_CO_Daily_Total_Column_L2",
            ),
        ),
        TerraLayerOption(
            layer_id="MISR_AM1_Ellipsoid_Radiance_RGB_BF",
            label="MISR: Radiance RGB",
            description="Multi-angle RGB imagery from MISR instrument.",
            time_mode="daily",
            min_date=dt.date(2020, 4, 9),
            max_date=dt.date(2022, 10, 12),
            snapshot_config=SnapshotConfig(
                endpoint="https://wvs.earthdata.nasa.gov/api/v1/snapshot",
                layer="MISR_AM1_Ellipsoid_Radiance_RGB_AA",
                bbox="-90,-180,90,180",
            ),
        ),
        TerraLayerOption(
            layer_id="CERES_Terra_TOA_Longwave_Flux_All_Sky_Monthly",
            label="CERES: Longwave Flux",
            description="Monthly TOA longwave flux (all sky).",
            time_mode="monthly",
            min_date=dt.date(2000, 3, 1),
            max_date=dt.date(2018, 10, 31),
            snapshot_config=SnapshotConfig(
                endpoint="https://wvs.earthdata.nasa.gov/api/v1/snapshot",
                layer="CERES_EBAF_TOA_CRE_Net_Flux_Monthly",
                bbox="-90,-180,90,180",
                colormap="CERES_EBAF_TOA_CRE_Net_Flux_Monthly",
                image_format="image/jpeg",
            ),
        ),
        TerraLayerOption(
            layer_id="ASTER_GDEM_Color_Shaded_Relief",
            label="ASTER: GDEM Relief",
            description="Static color shaded relief from ASTER GDEM.",
            time_mode="none",
            min_date=GLOBAL_MIN_DATE,
            max_date=today,
        ),
    ]



class TerraTextureController:
    """Handles Terra imagery downloads on a worker thread with local caching."""

    def __init__(self, *, cache_root: Optional[Path] = None) -> None:
        fallback_root = Path(__file__).resolve().parent / "cache" / "terra"
        root = cache_root if cache_root is not None else fallback_root
        self.cache_root = root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terra-fetch")
        self._future: Optional[Future[TerraDownloadResult]] = None
        self._active_params: Optional[TerraRequest] = None
        self._pending_params: Optional[TerraRequest] = None
        self._pending_force: bool = False
        self._current_params: Optional[TerraRequest] = None
        self.status: str = "Idle"
        self.status_detail: str = "No imagery requested yet."

    def request(self, params: TerraRequest, *, force: bool = False) -> None:
        if not force and (
            params == self._current_params
            or (self._future is not None and self._active_params == params)
            or params == self._pending_params
        ):
            return
        if self._future is not None:
            self._pending_params = params
            self._pending_force = force
            self.status = "Queued"
            self.status_detail = "A new download request is queued."
            return
        self._start_download(params, force)

    def poll(self) -> Optional[TerraDownloadResult]:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "Error"
            self.status_detail = str(exc)
            self._current_params = None
            return None
        origin = "cache" if result.from_cache else "network"
        time_info = result.params.time_param() or "static layer"
        self.status = "Ready"
        layers_desc = ", ".join(result.params.resolved_layers())
        self.status_detail = f"{layers_desc} ({time_info}) [{origin}]"
        self._current_params = result.params
        if self._pending_params is not None:
            params = self._pending_params
            force = self._pending_force
            self._pending_params = None
            self._pending_force = False
            self._start_download(params, force)
        return result

    def fetch_sync(self, params: TerraRequest, *, use_cache: bool = True) -> Optional[TerraDownloadResult]:
        try:
            return self._download_image(params, use_cache)
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.warning(
                "Backfill fetch failed for %s on %s: %s",
                params.layer_id,
                params.time_param(),
                exc,
            )
            return None

    def shutdown(self) -> None:
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._future = None
        self._pending_params = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_download(self, params: TerraRequest, force: bool) -> None:
        use_cache = not force
        self._active_params = params
        time_info = params.time_param() or "static layer"
        layers_desc = ", ".join(params.resolved_layers())
        self.status = "Loading"
        self.status_detail = f"{layers_desc} ({time_info})"
        self._future = self._executor.submit(self._download_image, params, use_cache)

    def _download_image(self, params: TerraRequest, use_cache: bool) -> TerraDownloadResult:
        cache_path = self._cache_path(params)
        if use_cache and cache_path.exists():
            with Image.open(cache_path) as img:
                # Preserve alpha channel when available (many layers use transparency)
                image = img.convert("RGBA")
            return TerraDownloadResult(params=params, image=image, from_cache=True, cache_path=cache_path)

        if params.is_snapshot():
            image, raw_bytes = self._download_snapshot_image(params)
        else:
            image, raw_bytes = self._download_wms_image(params)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw_bytes)
        return TerraDownloadResult(params=params, image=image, from_cache=False, cache_path=cache_path)

    def _download_wms_image(self, params: TerraRequest) -> tuple[Image.Image, bytes]:
        query = self._build_query(params)

        def _do_request() -> requests.Response:
            resp = requests.get(WMS_ENDPOINT, params=query, timeout=(5, 60))
            resp.raise_for_status()
            return resp

        try:
            response = _do_request()
        except requests_exceptions.RequestException as exc:
            raise RuntimeError(f"GIBS request failed: {exc}") from exc

        try:
            image = self._response_to_image(response, service_name="GIBS")
        except RuntimeError:
            # Retry once in case of transient truncation
            response = _do_request()
            image = self._response_to_image(response, service_name="GIBS")
        return image, response.content

    def _download_snapshot_image(self, params: TerraRequest) -> tuple[Image.Image, bytes]:
        snapshot = params.snapshot
        if snapshot is None:
            raise RuntimeError("Snapshot configuration missing for request")
        if params.date is None:
            raise RuntimeError("Snapshot layers require an explicit date")
        time_value = params.date.isoformat()
        if params.time_mode != "none":
            time_value = f"{time_value}T00:00:00Z"
        query = {
            "REQUEST": snapshot.request,
            "TIME": time_value,
            "BBOX": snapshot.bbox,
            "CRS": snapshot.crs,
            "LAYERS": snapshot.layer,
            "WRAP": snapshot.wrap,
            "FORMAT": snapshot.image_format,
            "WIDTH": str(params.width),
            "HEIGHT": str(params.height),
        }
        if snapshot.colormap:
            query["colormaps"] = snapshot.colormap
        query["ts"] = str(int(time.time() * 1000))
        try:
            response = requests.get(snapshot.endpoint, params=query, timeout=(5, 60))
            response.raise_for_status()
        except requests_exceptions.RequestException as exc:
            raise RuntimeError(f"Snapshot request failed: {exc}") from exc
        image = self._response_to_image(response, service_name="WVS snapshot")
        return image, response.content

    @staticmethod
    def _response_to_image(resp: requests.Response, *, service_name: str) -> Image.Image:
        ct = (resp.headers.get("Content-Type") or "").lower()
        data = resp.content or b""
        if ("image" not in ct) or len(data) < 32:
            preview = data[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"{service_name} error content-type '{ct}' or short body: {preview}")
        try:
            with Image.open(io.BytesIO(data)) as img:
                return img.convert("RGBA")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"{service_name} returned an unreadable image") from exc

    def _cache_path(self, params: TerraRequest) -> Path:
        layers_part = "_".join(params.resolved_layers()).replace('/', '_')
        if params.is_snapshot():
            layers_part = f"snapshot_{layers_part}"
        layer_dir = self.cache_root / layers_part
        return layer_dir / f"{params.cache_key()}.png"

    @staticmethod
    def _build_query(params: TerraRequest) -> dict[str, str]:
        query = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": WMS_VERSION,
            "SRS": "EPSG:4326",
            "LAYERS": ",".join(params.resolved_layers()),
            "BBOX": BBOX_FULL_EARTH,
            "WIDTH": str(params.width),
            "HEIGHT": str(params.height),
            "FORMAT": "image/png",
            "STYLES": "",
            # Keep transparency from the server so overlays/voids are correct
            "TRANSPARENT": "TRUE",
        }
        time_value = params.time_param()
        if time_value is not None:
            query["TIME"] = time_value
        return query

