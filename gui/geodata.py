"""Helpers for loading coastline and Ukraine outlines for the GPU viewer.

Adapted from the legacy LEO Ops project (V0.5-b/base_layers.py + geo_loader.py)
so we keep the same data selection and simplification pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import requests

LOGGER = logging.getLogger(__name__)

Coordinate = Tuple[float, float]  # (lat, lon)
CoordinatePath = List[Coordinate]


NE_DOWNLOADS = {
    'ne_50m_admin_0_countries.json': 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson',
    'ne_50m_admin_0_countries.geojson': 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson',
    'ne_50m_admin_0_map_subunits.json': 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_map_subunits.geojson',
    'ne_50m_admin_0_map_subunits.geojson': 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_map_subunits.geojson',
}


@dataclass
class BaseLayers:
    continents: List[CoordinatePath]
    ukraine: List[CoordinatePath]


def load_base_layers(root_hint: Path | None = None) -> BaseLayers:
    """Load simplified coastline and Ukraine (with Crimea) outlines."""
    roots = _candidate_roots(root_hint)
    continents = _load_coastlines(roots, pattern="ne_50m_coastline.json", simplify_tol=0.05)
    ukraine = _load_ukraine(roots, simplify_tol=0.02)
    return BaseLayers(continents=continents, ukraine=ukraine)


def _candidate_roots(root_hint: Path | None) -> List[Path]:
    project_root = Path(__file__).resolve().parents[1]
    hints = [
        root_hint,
        project_root / "data",
        project_root / "V0.5-b" / "data",
        project_root / "natural-earth-geojson-master",
        project_root,
    ]
    unique: List[Path] = []
    seen: set[Path] = set()
    for hint in hints:
        if not hint:
            continue
        try:
            resolved = Path(hint).resolve()
        except FileNotFoundError:
            continue
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    # widen search to subdirectories that typically hold geojson dumps
    expanded: List[Path] = []
    for root in unique:
        expanded.append(root)
        if root.is_dir():
            for extra in (root / "50m", root / "physical", root / "cultural"):
                if extra.exists() and extra not in expanded:
                    expanded.append(extra)
    return expanded


def _load_coastlines(roots: Sequence[Path], pattern: str, simplify_tol: float) -> List[CoordinatePath]:
    files = list(_first_match_all(roots, pattern))
    if not files:
        alt_pattern = pattern.replace('.json', '.geojson')
        files = list(_first_match_all(roots, alt_pattern))
    if not files:
        LOGGER.warning("Coastline files matching %s not found", pattern)
        return []
    paths: List[CoordinatePath] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.exception("Failed to parse %s: %s", file, exc)
            continue
        for feature in data.get("features", []):
            geom = feature.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if not coords:
                continue
            if gtype == "LineString":
                for segment in _split_antimeridian(coords):
                    path = [_normalize_point(pt) for pt in segment]
                    if simplify_tol > 0:
                        path = _rdp(path, simplify_tol)
                    paths.append(path)
            elif gtype == "MultiLineString":
                for ls in coords:
                    for segment in _split_antimeridian(ls):
                        path = [_normalize_point(pt) for pt in segment]
                        if simplify_tol > 0:
                            path = _rdp(path, simplify_tol)
                        paths.append(path)
    LOGGER.debug("Loaded %d coastline polylines", len(paths))
    return paths


def _first_geojson_match(roots: Sequence[Path], names: tuple[str, ...]) -> Path | None:
    for name in names:
        match = _first_match(roots, name)
        if match:
            return match
    stem = names[0]
    alt = stem + '.geojson' if not stem.endswith('.geojson') else stem[:-8] + '.json'
    return _first_match(roots, alt)



def _ensure_natural_earth_resource(project_root: Path, names: tuple[str, ...]) -> Path | None:
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    candidates: list[str] = []
    for name in names:
        base = Path(name).name
        candidates.append(base)
        if base.endswith('.json'):
            candidates.append(base.replace('.json', '.geojson'))
        elif base.endswith('.geojson'):
            candidates.append(base.replace('.geojson', '.json'))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        url = NE_DOWNLOADS.get(candidate)
        if not url:
            continue
        target = data_dir / candidate
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Failed to download %s: %s", candidate, exc)
            continue
        target.write_bytes(response.content)
        LOGGER.info("Downloaded %s to %s", candidate, target)
        return target
    return None



def _load_ukraine(roots: Sequence[Path], simplify_tol: float) -> List[CoordinatePath]:
    project_root = Path(__file__).resolve().parents[1]
    names = ("ne_50m_admin_0_countries.json", "ne_50m_admin_0_countries.geojson")
    country_file = _first_geojson_match(roots, names)
    if not country_file:
        downloaded = _ensure_natural_earth_resource(project_root, names)
        if downloaded is not None:
            country_file = downloaded
            roots = (downloaded.parent,) + tuple(roots)
    if not country_file:
        LOGGER.warning("Ukraine country file missing")
        return []
    polygons = _extract_polygons(country_file, match_key="ADMIN", match_value="Ukraine", simplify_tol=simplify_tol)
    if _has_crimea(polygons):
        return polygons
    LOGGER.info("Crimea not bundled with Ukraine in %s, loading supplemental geometry", country_file.name)
    crimea_names = ("crimea.json", "crimea.geojson")
    local_crimea = _first_geojson_match(roots, crimea_names) or _ensure_natural_earth_resource(project_root, crimea_names)
    if local_crimea:
        polygons.extend(_load_geojson_polygons(local_crimea, simplify_tol))
        if _has_crimea(polygons):
            return polygons
    subunit_names = ("ne_50m_admin_0_map_subunits.json", "ne_50m_admin_0_map_subunits.geojson")
    subunit_file = _first_geojson_match(roots, subunit_names)
    if not subunit_file:
        subunit_file = _ensure_natural_earth_resource(project_root, subunit_names)
    if subunit_file:
        polygons.extend(
            _extract_polygons(subunit_file, match_key="NAME", match_value="Crimea", simplify_tol=simplify_tol)
        )
    if not _has_crimea(polygons):
        LOGGER.warning("Falling back to hard-coded Crimea outline")
        polygons.append(_fallback_crimea())
    return polygons

    country_file = _first_match(roots, "ne_50m_admin_0_countries.json")
    if not country_file:
        LOGGER.warning("Ukraine country file missing")
        return []
    polygons = _extract_polygons(country_file, match_key="ADMIN", match_value="Ukraine", simplify_tol=simplify_tol)
    if _has_crimea(polygons):
        return polygons
    LOGGER.info("Crimea not bundled with Ukraine in %s, loading supplemental geometry", country_file.name)
    local_crimea = _first_match(roots, "crimea.json")
    if local_crimea:
        polygons.extend(_load_geojson_polygons(local_crimea, simplify_tol))
        if _has_crimea(polygons):
            return polygons
    subunit_file = _first_match(roots, "ne_50m_admin_0_map_subunits.json")
    if subunit_file:
        polygons.extend(
            _extract_polygons(subunit_file, match_key="NAME", match_value="Crimea", simplify_tol=simplify_tol)
        )
    if not _has_crimea(polygons):
        LOGGER.warning("Falling back to hard-coded Crimea outline")
        polygons.append(_fallback_crimea())
    return polygons


def _first_match(roots: Sequence[Path], pattern: str) -> Path | None:
    return next(_first_match_all(roots, pattern), None)


def _first_match_all(roots: Sequence[Path], pattern: str) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for match in root.rglob(pattern):
                resolved = match.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved
        except Exception:
            continue


def _extract_polygons(path: Path, match_key: str, match_value: str, simplify_tol: float) -> List[CoordinatePath]:
    results: List[CoordinatePath] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOGGER.exception("Failed to parse %s: %s", path, exc)
        return results
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        if props.get(match_key) != match_value:
            continue
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "Polygon":
            results.extend(_prepare_rings(coords, simplify_tol))
        elif gtype == "MultiPolygon":
            for polygon in coords:
                results.extend(_prepare_rings(polygon, simplify_tol))
    return results


def _load_geojson_polygons(path: Path, simplify_tol: float) -> List[CoordinatePath]:
    results: List[CoordinatePath] = []
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except Exception as exc:  # pragma: no cover - parse fallback
        LOGGER.debug("Strict parse failed for %s: %s", path, exc)
        trimmed = raw_text.strip().rstrip(",")
        data = json.loads(trimmed)
    data_type = data.get("type")
    if data_type == "FeatureCollection":
        features = data.get("features") or []
    elif data_type == "Feature":
        features = [data]
    else:
        LOGGER.warning("Unsupported GeoJSON type %s in %s", data_type, path.name)
        return results
    for feature in features:
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "Polygon":
            results.extend(_prepare_rings(coords, simplify_tol))
        elif gtype == "MultiPolygon":
            for polygon in coords:
                results.extend(_prepare_rings(polygon, simplify_tol))
    return results


def _prepare_rings(rings: Iterable[Sequence[Sequence[float]]], simplify_tol: float) -> List[CoordinatePath]:
    prepared: List[CoordinatePath] = []
    for ring in rings:
        if len(ring) < 3:
            continue
        path = [_normalize_point(pt) for pt in ring]
        if simplify_tol > 0:
            path = _rdp(path, simplify_tol, closed=True)
        if path and path[0] != path[-1]:
            path.append(path[0])
        prepared.append(path)
    return prepared


def _normalize_point(point: Sequence[float]) -> Coordinate:
    lon, lat = float(point[0]), float(point[1])
    if abs(lat) > 90 and abs(lon) <= 90:
        lat, lon = lon, lat
    lon = ((lon + 180.0) % 360.0) - 180.0
    return (lat, lon)


def _split_antimeridian(points: Sequence[Sequence[float]]) -> List[List[Sequence[float]]]:
    if len(points) < 2:
        return [list(points)]
    segments: List[List[Sequence[float]]] = []
    current: List[Sequence[float]] = [points[0]]
    prev_lon = points[0][0]
    for pt in points[1:]:
        lon = pt[0]
        if abs(lon - prev_lon) > 180:
            segments.append(current)
            current = [pt]
        else:
            current.append(pt)
        prev_lon = lon
    if current:
        segments.append(current)
    return segments


def _rdp(points: CoordinatePath, epsilon: float, closed: bool = False) -> CoordinatePath:
    if len(points) < 3 or epsilon <= 0:
        return points
    if closed and points[0] == points[-1]:
        inner = _rdp(points[:-1], epsilon, closed=False)
        return inner + [inner[0]]
    first, last = points[0], points[-1]
    index = -1
    max_dist = 0.0
    for i in range(1, len(points) - 1):
        dist = _point_segment_distance(points[i], first, last)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon and index != -1:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [first, last]


def _point_segment_distance(point: Coordinate, a: Coordinate, b: Coordinate) -> float:
    x0, y0 = point[1], point[0]
    x1, y1 = a[1], a[0]
    x2, y2 = b[1], b[0]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2) ** 0.5


def _has_crimea(polygons: Sequence[CoordinatePath]) -> bool:
    for path in polygons:
        lats = [lat for lat, _ in path]
        lons = [lon for _, lon in path]
        if not lats or not lons:
            continue
        if min(lats) < 46.0 and 30.0 < min(lons) and max(lons) < 37.5:
            return True
    return False


def _fallback_crimea() -> CoordinatePath:
    coords = [
        (45.3396, 32.5120),
        (44.8083, 33.6578),
        (44.3870, 34.5449),
        (44.1032, 34.9629),
        (45.0371, 36.5691),
        (45.7733, 35.2959),
        (46.2200, 33.8821),
        (46.0980, 32.3730),
        (45.6509, 33.0703),
        (45.3396, 32.5120),
    ]
    return coords

