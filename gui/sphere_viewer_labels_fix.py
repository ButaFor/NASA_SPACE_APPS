"""GPU-accelerated globe viewer prototype.

Stage 1 delivers a standalone window with a shaded sphere that exposes
meridians and parallels. The window is built on glfw + moderngl so that
rendering happens on the GPU and rotations stay responsive for the next
project stages.

Run with:
    python -m gui.sphere_viewer_labels_fix

Dependencies:
    pip install -r gui/requirements.txt
"""
from __future__ import annotations

import datetime as dt
import math
import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from analysis import (
    AnalysisKind,
    AnalysisRequest,
    FireAnalyzer,
    FluxAnalyzer,
    GeoBoundingBox,
    FULL_EARTH_BBOX,
    fire_heatmap_image,
    flux_heatmap_image,
    format_fire_analysis,
    format_flux_analysis,
)
from analysis.roi_visualization import (
    ROIVisualizer,
    ROIStyle,
    ROIType,
    create_analysis_heatmap_texture,
    create_roi_selection_interface,
)
from analysis.analysis_visualization import (
    AnalysisVisualizer,
    ColorMap,
    LegendConfig,
    create_analysis_export_image,
)
from gui.roi_selector import create_roi_selector_dialog
from gui.analysis_display import create_analysis_display_dialog
from analysis.export_utils import create_export_dialog
from gui.geodata import BaseLayers, load_base_layers
from gui.terra_config import (
    DEFAULT_DATE_RANGE_DAYS,
    GLOBAL_MIN_DATE,
    TERRA_RESOLUTIONS,
    TerraLayerOption,
    TerraRequest,
    TerraTextureController,
    TimeDomainInfo,
    default_layers,
    get_time_domain,
)
try:
    from gui.qt_controls import ControlState, TerraControlPanel  # Modern PySide6 UI
except Exception:  # pragma: no cover - fallback to Tk controls if PySide6 unavailable
    from gui.terra_controls import ControlState, TerraControlPanel


def _hex_to_rgb(value: str) -> Tuple[float, float, float]:
    value = value.lstrip('#')
    if len(value) != 6:
        raise ValueError(f"Expected hex color RRGGBB, got {value!r}")
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return (r, g, b)


THEME_BACKGROUND = _hex_to_rgb('#001b2e')
THEME_SPHERE = _hex_to_rgb('#0d2b45')
THEME_GRID = _hex_to_rgb('#1f8ef1')
THEME_COASTLINE = _hex_to_rgb('#0c4f70')
THEME_UKRAINE = _hex_to_rgb('#ffd43b')
THEME_MARKER_FILL = _hex_to_rgb('#ffffff')
THEME_MARKER_OUTLINE = _hex_to_rgb('#f9c80e')
# Off-white label color to match UI reference


@dataclass
class LabelConfig:
    surface_normal_offset: float = 0.0006
    right_offset: float = 0.025
    up_offset: float = -0.015
    min_screen_px: float = 16.0
    max_screen_px: float = 64.0
    pixel_scale: float = 0.0003


THEME_LABEL_FOREGROUND = (240.0 / 255.0, 244.0 / 255.0, 255.0 / 255.0, 1.0)


TERRA_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "terra"
TERRA_LAYER_OPTIONS: List[TerraLayerOption] = default_layers()

MARKER_SURFACE_RADIUS = 1.001
MARKER_DOT_ELEVATION = 0.0012
MARKER_DOT_RADIUS = 0.007
MARKER_DOT_SEGMENTS = 48
MARKER_LABEL_NORMAL_OFFSET = 0.045
MARKER_LABEL_VERTICAL_OFFSET = 0.0025
# Adjust these values to control text size and appearance
LABEL_PIXEL_PADDING = 8       # Padding around text in pixels (smaller = more compact labels)

MARKER_OUTLINE_SCALE = 1.45   # Outline ring slightly larger than dot

# Label placement relative to the point on the sphere surface - adjust these to change text position

# Camera distance limits for zoom
CAMERA_DISTANCE_MIN = 1.25
CAMERA_DISTANCE_MAX = 8.0
CAMERA_ZOOM_STEP = 0.3

# Minimum on-screen height of a label in pixels

# Debug: visualize SDF alpha instead of text to diagnose geometry/placement
LABEL_DEBUG_SDF = False

BACKFILL_VOID_THRESHOLD = 80

KYIV_COORDS = (50.4501, 30.5234)
KYIV_LABEL = "KYIV"

LOGGER = logging.getLogger(__name__)

@dataclass
class ViewerConfig:
    width: int = 1280
    height: int = 720
    title: str = "TERRA TOOLS"
    lat_segments: int = 64
    lon_segments: int = 128
    grid_lat_step: int = 30
    grid_lon_step: int = 30
    geodata_root: Path | None = None
    label_config: LabelConfig = field(default_factory=LabelConfig)
    enable_backfill: bool = False
    backfill_max_lookups: int = 1
    mirror_texture_v: bool = False
    # Convert near-black pixels to transparent to expose base sphere
    transparentize_black: bool = True


@dataclass
class MarkerData:
    normal: np.ndarray
    label_anchor: np.ndarray
    line_vertices: np.ndarray
    triangle_vertices: np.ndarray


@dataclass
class PointLabel:
    label: str
    lat: float
    lon: float
    normal: np.ndarray
    east: np.ndarray
    north: np.ndarray
    label_anchor: np.ndarray
    line_vbo: moderngl.Buffer | None
    line_vao: moderngl.VertexArray | None
    line_vertex_count: int
    tri_vbo: moderngl.Buffer | None
    tri_vao: moderngl.VertexArray | None
    tri_vertex_count: int
    label_texture: moderngl.Texture | None
    label_vbo: moderngl.Buffer | None
    label_vao: moderngl.VertexArray | None
    label_size_px: np.ndarray
    label_world_size: np.ndarray

    def release(self) -> None:
        for vao in (self.line_vao, self.tri_vao, self.label_vao):
            if vao is not None:
                vao.release()
        for vbo in (self.line_vbo, self.tri_vbo, self.label_vbo):
            if vbo is not None:
                vbo.release()
        if self.label_texture is not None:
            self.label_texture.release()


class SphereViewer:
    def __init__(self, config: ViewerConfig | None = None) -> None:
        self.cfg = config or ViewerConfig()
        self._ensure_glfw()
        self.window = self._create_window()

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if hasattr(moderngl, "MULTISAMPLE"):
            self.ctx.enable(moderngl.MULTISAMPLE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        try:
            self.ctx.line_width = 1.5
        except AttributeError:
            pass

        self._geodata: BaseLayers | None = None
        self._coast_vbo = None
        self._coast_vao = None
        self._coast_offsets: List[Tuple[int, int]] = []
        self._ukraine_vbo = None
        self._ukraine_vao = None
        self._ukraine_offsets: List[Tuple[int, int]] = []
        self._points: list[PointLabel] = []

        self._sphere_vbo = None
        self._normal_vbo = None
        self._uv_vbo = None
        self._sphere_vao = None
        self._grid_vbo = None
        self._grid_vao = None
        self._sphere_texture: moderngl.Texture | None = None

        self._label_cfg = self.cfg.label_config
        self._use_backfill = bool(self.cfg.enable_backfill)
        self._backfill_max_steps = max(1, int(self.cfg.backfill_max_lookups))
        self._terra_controller = TerraTextureController(cache_root=TERRA_CACHE_DIR)
        self._analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terra-analysis")
        self._analysis_future: Future[tuple[AnalysisRequest, str, Optional[Image.Image]]] | None = None
        self._analysis_pending: Deque[AnalysisRequest] = deque()
        self._analysis_active_request: AnalysisRequest | None = None
        self._fire_analyzer = FireAnalyzer()
        self._flux_analyzer = FluxAnalyzer()
        self._roi_overlay_vbo = None
        self._roi_overlay_vao = None
        self._roi_overlay_offsets: List[Tuple[int, int]] = []
        self._roi_bbox: GeoBoundingBox | None = None
        self._roi_visualizer = ROIVisualizer()
        self._roi_type = ROIType.RECTANGLE
        self._roi_style = ROIStyle()
        self._analysis_heatmap_texture: moderngl.Texture | None = None
        self._analysis_heatmap_vao = None
        self._analysis_heatmap_vbo = None
        self._analysis_visualizer = AnalysisVisualizer()
        self._current_analysis_data: np.ndarray | None = None
        self._current_analysis_mask: np.ndarray | None = None
        self._show_analysis_legend = True
        self._analysis_colormap = ColorMap.hot()
        self._terra_layers = list(TERRA_LAYER_OPTIONS)
        self._terra_resolutions = list(TERRA_RESOLUTIONS)
        self._terra_resolution_index = min(1, len(self._terra_resolutions) - 1)
        self._terra_date_offset_days = 0
        self._terra_date_range_days = DEFAULT_DATE_RANGE_DAYS
        self._selected_layer_index = 0
        self._terra_last_request: TerraRequest | None = None
        self._terra_last_update: dt.datetime | None = None
        self._terra_texture_info: str = "Default placeholder texture"

        initial_layer_id = self._terra_layers[self._selected_layer_index].layer_id if self._terra_layers else ""
        initial_resolution = self._terra_resolutions[self._terra_resolution_index] if self._terra_resolutions else 1024
        self._controls = TerraControlPanel(
            layers=self._terra_layers,
            resolutions=self._terra_resolutions,
            initial_layer_id=initial_layer_id,
            initial_resolution=initial_resolution,
            initial_offset=self._terra_date_offset_days,
            initial_backfill=self._use_backfill,
        )
        
        # Передаємо посилання на viewer в контроли
        self._controls._viewer = self
        
        # Встановлюємо початкову папку збереження
        import os
        if hasattr(self._controls, 'set_save_location'):
            self._controls.set_save_location(os.getcwd())

        self._init_callbacks()
        self._compile_programs()
        self._load_geometry()
        self._request_current_terra_texture(force=True)

        self._camera_position = np.array([0.0, 0.0, 3.2], dtype=np.float32)
        self._camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.rotation_pitch = 0.0
        self.rotation_yaw = 0.0
        
        # Анімація глобуса
        self._animation_active = False
        self._animation_start_time = 0.0
        self._animation_speed = 2 * math.pi / 10.0  # Один оберт за 10 секунд (радіан/сек)
        self._drag_active = False
        self._last_cursor: Tuple[float, float] | None = None

        framebuffer_size = glfw.get_framebuffer_size(self.window)
        self._refresh_viewport(*framebuffer_size)

    @staticmethod
    def _ensure_glfw() -> None:
        if not glfw.init():
            raise RuntimeError("Unable to initialise GLFW")

    def _create_window(self) -> glfw._GLFWwindow:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(self.cfg.width, self.cfg.height, self.cfg.title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        # Try to set a custom icon for the GLFW window
        try:
            self._apply_window_icon(window)
        except Exception:
            pass
        return window

    def _apply_window_icon(self, window: glfw._GLFWwindow) -> None:
        icon_path = Path(__file__).resolve().parents[1] / "icon.ico"
        if not icon_path.exists():
            return
        try:
            img = Image.open(icon_path).convert("RGBA")
            w, h = img.size
            pixels = img.tobytes()
            image = glfw.Image(width=w, height=h, pixels=pixels)
            glfw.set_window_icon(window, [image])
        except Exception:
            return

    def _init_callbacks(self) -> None:
        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key_press)


    def _compile_programs(self) -> None:
        self.sphere_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_uv;
                uniform mat4 mvp;
                uniform mat3 normal_matrix;
                out vec3 v_normal;
                out vec2 v_uv;
                void main() {
                    v_normal = normalize(normal_matrix * in_normal);
                    v_uv = in_uv;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_normal;
                in vec2 v_uv;
                out vec4 fragColor;
                uniform vec3 base_color;
                uniform vec3 light_dir;
                uniform bool use_texture;
                uniform sampler2D surface_tex;
                void main() {
                    float light = max(dot(normalize(v_normal), normalize(light_dir)), 0.0);
                    float ambient = 0.35;
                    float diffuse = 0.65 * light;
                    vec3 surface = base_color;
                    if (use_texture) {
                        vec4 texel = texture(surface_tex, v_uv);
                        surface = mix(base_color, texel.rgb, texel.a);
                    }
                    fragColor = vec4(surface * (ambient + diffuse), 1.0);
                }
            """,
        )
        self.sphere_prog["surface_tex"].value = 0
        self.sphere_prog["use_texture"].value = False
        self.grid_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                uniform vec3 line_color;
                void main() {
                    fragColor = vec4(line_color, 1.0);
                }
            """,
        )
        self.marker_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                uniform vec3 fill_color;
                void main() {
                    fragColor = vec4(fill_color, 1.0);
                }
            """,
        )
        self.label_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec2 in_uv;
                uniform mat4 mvp;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                uniform sampler2D tex;
                uniform vec4 label_color;
                out vec4 fragColor;
                void main() {
                    vec4 texColor = texture(tex, v_uv);
                    fragColor = vec4(label_color.rgb, texColor.a * label_color.a);
                }
            """,
        )
        self.label_prog["tex"].value = 0
        self.label_prog["label_color"].value = THEME_LABEL_FOREGROUND

    def _load_geometry(self) -> None:
        sphere_positions, sphere_normals, sphere_uvs = self._build_sphere_mesh(
            self.cfg.lat_segments, self.cfg.lon_segments
        )
        grid_positions = self._build_grid(self.cfg.grid_lat_step, self.cfg.grid_lon_step)

        self._sphere_vbo = self.ctx.buffer(sphere_positions.tobytes())
        self._normal_vbo = self.ctx.buffer(sphere_normals.tobytes())
        self._uv_vbo = self.ctx.buffer(sphere_uvs.tobytes())
        self._grid_vbo = self.ctx.buffer(grid_positions.tobytes())

        self._sphere_vao = self.ctx.vertex_array(
            self.sphere_prog,
            [
                (self._sphere_vbo, "3f", "in_position"),
                (self._normal_vbo, "3f", "in_normal"),
                (self._uv_vbo, "2f", "in_uv"),
            ],
        )
        self._grid_vertex_count = grid_positions.shape[0]
        self._grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [
                (self._grid_vbo, "3f", "in_position"),
            ],
        )
        self._ensure_placeholder_texture()
        self._load_map_layers()
        self._build_static_points()
    def _ensure_placeholder_texture(self) -> None:
        if self._sphere_texture is not None:
            return
        placeholder = Image.new("RGB", (2, 1), (16, 32, 64))
        self._terra_texture_info = "Default placeholder texture"
        self._update_sphere_texture(placeholder, description=self._terra_texture_info, update_timestamp=False)

    def _current_terra_layer(self) -> TerraLayerOption:
        if not self._terra_layers:
            raise RuntimeError('No Terra layers configured')
        index = max(0, min(self._selected_layer_index, len(self._terra_layers) - 1))
        return self._terra_layers[index]

    def _current_layer_bounds(
        self,
        layer: TerraLayerOption,
        domain: Optional[TimeDomainInfo],
    ) -> tuple[dt.date, dt.date]:
        today = dt.date.today()
        max_date = min(today, layer.max_date or today)
        if domain is not None and domain.max_date is not None:
            max_date = min(max_date, domain.max_date)
        min_candidate = layer.min_date or (max_date - dt.timedelta(days=self._terra_date_range_days))
        if domain is not None and domain.min_date is not None:
            min_candidate = max(min_candidate, domain.min_date)
        min_date = max(GLOBAL_MIN_DATE, min_candidate)
        if min_date > max_date:
            min_date = max_date
        return min_date, max_date

    def _selected_date_for_layer(
        self,
        layer: TerraLayerOption,
        min_date: dt.date,
        max_date: dt.date,
    ) -> Optional[dt.date]:
        if layer.time_mode == "none":
            return None
        max_offset = max((max_date - min_date).days, 0)
        effective_offset = min(self._terra_date_offset_days, max_offset)
        selected = max_date - dt.timedelta(days=effective_offset)
        if selected < min_date:
            selected = min_date
        if selected > max_date:
            selected = max_date
        if layer.time_mode == "monthly":
            selected = selected.replace(day=1)
        return selected

    def _build_terra_request(self) -> TerraRequest:
        layer = self._current_terra_layer()
        width = self._terra_resolutions[self._terra_resolution_index]
        height = max(1, width // 2)
        min_date: Optional[dt.date] = None
        max_date: Optional[dt.date] = None
        selected_date: Optional[dt.date] = None
        domain: Optional[TimeDomainInfo] = None
        max_offset = 0

        if layer.time_mode != "none":
            domain = None
            if layer.snapshot_config is not None:
                domain = get_time_domain(layer.snapshot_config.layer)
                if domain is None:
                    domain = get_time_domain(layer.layer_id)
            else:
                domain = get_time_domain(layer.layer_id)
            min_date, max_date = self._current_layer_bounds(layer, domain)
            max_offset = max((max_date - min_date).days, 0)
            self._terra_date_offset_days = max(0, min(max_offset, self._terra_date_offset_days))
            selected_date = self._selected_date_for_layer(layer, min_date, max_date)
            if selected_date is not None and domain is not None:
                normalized_date = domain.normalize(selected_date)
                if normalized_date != selected_date:
                    selected_date = normalized_date
                    normalized_offset = max(0, (max_date - selected_date).days)
                    self._terra_date_offset_days = max(0, min(max_offset, normalized_offset))
        else:
            self._terra_date_offset_days = 0

        self._controls.set_date_info(
            min_date=min_date.isoformat() if min_date is not None else None,
            max_date=max_date.isoformat() if max_date is not None else None,
            selected_date=selected_date.isoformat() if selected_date is not None else None,
            time_mode=layer.time_mode,
            max_offset=max_offset,
            offset_value=self._terra_date_offset_days,
        )

        return TerraRequest(
            layer_id=layer.layer_id,
            date=selected_date,
            width=width,
            height=height,
            time_mode=layer.time_mode,
            layers=layer.layers,
            snapshot=layer.snapshot_config,
        )

    def _request_current_terra_texture(self, *, force: bool = False) -> None:
        params = self._build_terra_request()
        if not force and self._terra_last_request == params:
            return
        self._terra_controller.request(params, force=force)
        self._terra_last_request = params

    def _process_terra_updates(self) -> None:
        result = self._terra_controller.poll()
        if result is None:
            return
        layer = next((layer for layer in self._terra_layers if layer.layer_id == result.params.layer_id), None)
        label = layer.label if layer is not None else result.params.layer_id
        time_part = result.params.time_param() or "static layer"
        origin = "cache" if result.from_cache else "network"
        resolution = f"{result.params.width}x{result.params.height}"
        detail = f"{label} - {time_part} [{origin}, {resolution}]"
        if result.cache_path is not None:
            detail = f"{detail} · {result.cache_path.name}"
        self._update_sphere_texture(result.image, description=detail, update_timestamp=True)

    def _apply_control_state(self, state: ControlState) -> None:
        if self._terra_layers:
            self._selected_layer_index = self._layer_index_for(state.layer_id)
        if self._terra_resolutions:
            self._terra_resolution_index = self._resolution_index_for(state.resolution)
        self._terra_date_offset_days = max(0, state.offset_days)
        self._use_backfill = state.use_backfill

        if self._terra_layers:
            layer = self._terra_layers[self._selected_layer_index]
            self._controls.set_layer_description(layer.label)
            if layer.time_mode == "none":
                self._terra_date_offset_days = 0

    def _layer_index_for(self, layer_id: str) -> int:
        for idx, layer in enumerate(self._terra_layers):
            if layer.layer_id == layer_id:
                return idx
        return min(self._selected_layer_index, len(self._terra_layers) - 1) if self._terra_layers else 0

    def _resolution_index_for(self, resolution: int) -> int:
        for idx, value in enumerate(self._terra_resolutions):
            if value == resolution:
                return idx
        return min(self._terra_resolution_index, len(self._terra_resolutions) - 1) if self._terra_resolutions else 0

    def _apply_backfill_if_needed(
        self,
        image: Image.Image,
        layer: TerraLayerOption | None,
        params: TerraRequest,
    ) -> tuple[Image.Image, bool]:
        if (
            not self._use_backfill
            or layer is None
            or layer.time_mode == "none"
            or params.date is None
        ):
            return image, False
        if not self._has_voids(image):
            return image, False
        blended = False
        candidate = image
        steps = 0
        for prev_date in self._iter_previous_dates(layer, params.date):
            if steps >= self._backfill_max_steps:
                break
            prev_request = TerraRequest(
                layer_id=params.layer_id,
                date=prev_date,
                width=params.width,
                height=params.height,
                time_mode=params.time_mode,
                layers=params.layers,
                snapshot=params.snapshot,
            )
            prev_result = self._terra_controller.fetch_sync(prev_request, use_cache=True)
            steps += 1
            if prev_result is None:
                continue
            candidate, changed = self._blend_with_backfill(candidate, prev_result.image)
            if not changed:
                continue
            blended = True
            if not self._has_voids(candidate):
                break
        return candidate, blended



    def _iter_previous_dates(self, layer: TerraLayerOption, start_date: dt.date):
        min_date = layer.min_date or GLOBAL_MIN_DATE
        domain: Optional[TimeDomainInfo] = None
        if layer.snapshot_config is not None:
            domain = get_time_domain(layer.snapshot_config.layer)
            if domain is None:
                domain = get_time_domain(layer.layer_id)
        else:
            domain = get_time_domain(layer.layer_id)
        if domain is not None and domain.min_date is not None:
            min_date = max(min_date, domain.min_date)
        if layer.time_mode == "daily":
            if domain is not None and domain.explicit_dates:
                for candidate in reversed(domain.explicit_dates):
                    if min_date <= candidate < start_date:
                        yield candidate
                return
            current = start_date - dt.timedelta(days=1)
            while current >= min_date:
                if domain is None or domain.contains(current):
                    yield current
                current -= dt.timedelta(days=1)
        elif layer.time_mode == "monthly":
            if domain is not None and domain.explicit_dates:
                for candidate in reversed(domain.explicit_dates):
                    if min_date <= candidate < start_date:
                        yield candidate.replace(day=1)
                return
            current = start_date.replace(day=1)
            current = (current - dt.timedelta(days=1)).replace(day=1)
            while current >= min_date:
                normalized = current.replace(day=1)
                if domain is None or domain.contains(normalized):
                    yield normalized
                current = (current - dt.timedelta(days=1)).replace(day=1)

    def _has_voids(self, image: Image.Image) -> bool:
        # Consider both transparency and near-black RGB as voids, regardless of alpha presence
        arr = np.asarray(image.convert("RGBA"))
        if arr.ndim != 3 or arr.shape[2] < 4:
            return False
        rgb_sum = arr[..., :3].sum(axis=2)
        alpha = arr[..., 3]
        mask = (alpha == 0) | (rgb_sum <= BACKFILL_VOID_THRESHOLD)
        return bool(np.any(mask))

    def _blend_with_backfill(
        self,
        base: Image.Image,
        fallback: Image.Image,
    ) -> tuple[Image.Image, bool]:
        if fallback.size != base.size:
            fallback = fallback.resize(base.size, Image.BILINEAR)
        base_arr = np.asarray(base.convert("RGBA"), dtype=np.uint8).astype(np.int16)
        fallback_arr = np.asarray(fallback.convert("RGBA"), dtype=np.uint8).astype(np.int16)
        rgb_sum = base_arr[..., :3].sum(axis=2)
        alpha = base_arr[..., 3]
        mask = (alpha == 0) | (rgb_sum <= BACKFILL_VOID_THRESHOLD)
        if not np.any(mask):
            return base, False
        blended = base_arr.copy()
        blended[mask] = fallback_arr[mask]
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        result = Image.fromarray(blended)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result, True

    def _process_terra_updates_with_backfill(self) -> None:
        result = self._terra_controller.poll()
        if result is None:
            return
        layer = next((layer for layer in self._terra_layers if layer.layer_id == result.params.layer_id), None)
        label = layer.label if layer is not None else result.params.layer_id
        time_part = result.params.time_param() or "static layer"
        origin = "cache" if result.from_cache else "network"
        resolution = f"{result.params.width}x{result.params.height}"
        image = result.image
        blended = False
        try:
            image, blended = self._apply_backfill_if_needed(image, layer, result.params)
        except Exception as exc:
            LOGGER.warning("Backfill step failed: %s", exc)
        detail = f"{label} - {time_part} [{origin}, {resolution}]" + (" + backfilled" if blended else "")
        if result.cache_path is not None:
            try:
                detail = f"{detail} {result.cache_path.name}"
            except Exception:
                pass
        self._update_sphere_texture(image, description=detail, update_timestamp=True)

    def _ingest_analysis_requests(self) -> None:
        if not hasattr(self._controls, "pop_analysis_request"):
            return
        while True:
            request = self._controls.pop_analysis_request()
            if request is None:
                break
            self._analysis_pending.append(request)

    def _service_analysis_tasks(self) -> None:
        self._check_analysis_completion()
        if self._analysis_future is not None:
            return
        if not self._analysis_pending:
            return
        next_request = self._analysis_pending.popleft()
        self._start_analysis_task(next_request)

    def _start_analysis_task(self, request: AnalysisRequest) -> None:
        layer = self._layer_by_id(request.layer_id)
        if layer is None:
            if hasattr(self._controls, "set_analysis_status"):
                self._controls.set_analysis_status(f"Layer {request.layer_id} unavailable for analysis.")
            return
        self._analysis_active_request = request
        self._update_roi_overlay(request.bbox)
        if hasattr(self._controls, "set_analysis_status"):
            self._controls.set_analysis_status(
                f"Running {request.kind.value} analysis {request.start_date.isoformat()} -> {request.end_date.isoformat()}..."
            )
        
        # Показуємо прогрес-бар
        if hasattr(self._controls, "show_analysis_progress"):
            self._controls.show_analysis_progress(f"Starting {request.kind.value} analysis...")
        
        self._analysis_future = self._analysis_executor.submit(self._run_analysis_task, request, layer)

    def _check_analysis_completion(self) -> None:
        if self._analysis_future is None:
            return
        if not self._analysis_future.done():
            return
        future = self._analysis_future
        self._analysis_future = None
        try:
            request, summary, preview = future.result()
        except Exception as exc:
            LOGGER.exception("Analysis failed: %s", exc)
            
            # Приховуємо прогрес-бар при помилці
            if hasattr(self._controls, "hide_analysis_progress"):
                self._controls.hide_analysis_progress()
            
            if hasattr(self._controls, "set_analysis_status"):
                self._controls.set_analysis_status(f"Analysis error: {exc}")
            if hasattr(self._controls, "set_analysis_result"):
                self._controls.set_analysis_result("")
            if hasattr(self._controls, "set_analysis_preview"):
                self._controls.set_analysis_preview(None)
            self._analysis_active_request = None
            return
        self._analysis_active_request = None
        
        # Приховуємо прогрес-бар
        if hasattr(self._controls, "hide_analysis_progress"):
            self._controls.hide_analysis_progress()
        
        if hasattr(self._controls, "set_analysis_status"):
            self._controls.set_analysis_status(
                f"{request.kind.value.capitalize()} analysis complete {request.start_date.isoformat()} -> {request.end_date.isoformat()}"
            )
        if hasattr(self._controls, "set_analysis_result"):
            self._controls.set_analysis_result(summary)
        if hasattr(self._controls, "set_analysis_preview"):
            self._controls.set_analysis_preview(preview)
        
        # Оновлюємо інформацію про дати аналізу
        if hasattr(self._controls, "set_analysis_date_info"):
            self._controls.set_analysis_date_info(request.start_date.isoformat(), request.end_date.isoformat())

    def _run_analysis_task(self, request: AnalysisRequest, layer: TerraLayerOption) -> tuple[AnalysisRequest, str, Optional[Image.Image]]:
        preview: Optional[Image.Image] = None
        if request.kind is AnalysisKind.FIRE:
            result = self._fire_analyzer.analyze(self._terra_controller, layer, request)
            summary = format_fire_analysis(result)
            # Зберігаємо інформацію про аналіз
            self._current_analysis_kind = AnalysisKind.FIRE
            self._current_analysis_start_date = request.start_date.strftime('%Y-%m-%d')
            self._current_analysis_end_date = request.end_date.strftime('%Y-%m-%d')
            try:
                preview = fire_heatmap_image(result)
                # Створюємо heatmap текстуру для глобуса
                self._create_analysis_heatmap(result.density_map, result.roi_mask, result.width, result.height, request.bbox)
            except Exception as exc:
                LOGGER.debug("Failed to build fire heatmap preview: %s", exc)
                preview = None
        elif request.kind is AnalysisKind.FLUX:
            result = self._flux_analyzer.analyze(self._terra_controller, layer, request)
            summary = format_flux_analysis(result)
            # Зберігаємо інформацію про аналіз
            self._current_analysis_kind = AnalysisKind.FLUX
            self._current_analysis_start_date = request.start_date.strftime('%Y-%m-%d')
            self._current_analysis_end_date = request.end_date.strftime('%Y-%m-%d')
            try:
                preview = flux_heatmap_image(result)
                # Створюємо heatmap текстуру для глобуса
                self._create_analysis_heatmap(result.mean_map, result.roi_mask, result.width, result.height, request.bbox)
            except Exception as exc:
                LOGGER.debug("Failed to build flux heatmap preview: %s", exc)
                preview = None
        else:
            raise RuntimeError(f"Unsupported analysis kind {request.kind}")
        return request, summary, preview

    def _create_analysis_heatmap(self, data: np.ndarray, mask: np.ndarray, width: int, height: int, bbox: GeoBoundingBox | None = None) -> None:
        """Створює heatmap текстуру з результатів аналізу."""
        print(f"🔍 _create_analysis_heatmap: data.shape={data.shape}, mask.shape={mask.shape}, bbox={bbox}")
        if data.size == 0 or not np.any(mask):
            return
        
        try:
            # Зберігаємо дані для подальшого використання
            self._current_analysis_data = data
            self._current_analysis_mask = mask
            
            # Оновлюємо colormap візуалізатора
            self._analysis_visualizer.colormap = self._analysis_colormap
            
            # Створюємо правильну географічну проекцію для ROI
            # Використовуємо розмір, пропорційний до розміру ROI
            if bbox is not None:
                # Розраховуємо розмір на основі розміру ROI
                lon_range = bbox.max_lon - bbox.min_lon
                lat_range = bbox.max_lat - bbox.min_lat
                
                # Пропорційний розмір з мінімальним розміром для якості
                base_size = 512  # Збільшуємо базовий розмір для кращої якості
                aspect_ratio = lon_range / lat_range if lat_range > 0 else 1.0
                
                if aspect_ratio > 1:
                    roi_width = int(base_size * aspect_ratio)
                    roi_height = base_size
                else:
                    roi_width = base_size
                    roi_height = int(base_size / aspect_ratio)
                
                # Обмежуємо максимальний розмір для продуктивності
                roi_width = min(roi_width, 2048)
                roi_height = min(roi_height, 1024)
                
                # Обмежуємо мінімальний розмір для якості
                roi_width = max(roi_width, 256)
                roi_height = max(roi_height, 128)
            else:
                # Fallback до оригінальних розмірів
                roi_width = max(width, 512)
                roi_height = max(height, 256)
            
            # Створюємо heatmap текстуру з правильним масштабуванням
            # Використовуємо стандартний метод для відображення на глобусі (з прозорістю)
            heatmap_img = self._analysis_visualizer.create_heatmap_texture(
                data, mask, width=roi_width, height=roi_height, 
                show_legend=self._show_analysis_legend
            )
            
            # Конвертуємо в OpenGL текстуру
            self._update_analysis_heatmap_texture(heatmap_img)
            
        except Exception as exc:
            LOGGER.debug("Failed to create analysis heatmap: %s", exc)
    
    def _update_analysis_heatmap_texture(self, image: Image.Image) -> None:
        """Оновлює текстуру heatmap аналізу."""
        if self._analysis_heatmap_texture is not None:
            self._analysis_heatmap_texture.release()
        
        # Конвертуємо зображення в OpenGL текстуру
        width, height = image.size
        texture = self.ctx.texture((width, height), 4, image.tobytes())
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = True
        texture.repeat_y = False
        
        self._analysis_heatmap_texture = texture
    
    def export_analysis_results(self, filename: str) -> bool:
        """Експортує результати аналізу як зображення."""
        if self._current_analysis_data is None or self._current_analysis_mask is None:
            return False
        
        try:
            # Визначаємо розмір на основі ROI
            if self._roi_bbox:
                # Розраховуємо розмір на основі розміру ROI
                lon_range = self._roi_bbox.max_lon - self._roi_bbox.min_lon
                lat_range = self._roi_bbox.max_lat - self._roi_bbox.min_lat
                
                # Пропорційний розмір з мінімальним розміром для якості
                base_size = 512
                aspect_ratio = lon_range / lat_range if lat_range > 0 else 1.0
                
                if aspect_ratio > 1:
                    export_width = int(base_size * aspect_ratio)
                    export_height = base_size
                else:
                    export_width = base_size
                    export_height = int(base_size / aspect_ratio)
                
                # Обмежуємо максимальний розмір
                export_width = min(export_width, 2048)
                export_height = min(export_height, 1024)
            else:
                export_width, export_height = 1024, 1024
            
            # Підготовлюємо інформацію про аналіз
            analysis_info = {
                'analysis_type': 'Fire Analysis' if hasattr(self, '_current_analysis_kind') and self._current_analysis_kind == AnalysisKind.FIRE else 'Flux Analysis',
                'start_date': getattr(self, '_current_analysis_start_date', 'N/A'),
                'end_date': getattr(self, '_current_analysis_end_date', 'N/A'),
                'resolution': 'High Resolution'
            }
            
            # Створюємо heatmap з правильною якістю для експорту
            # Використовуємо спеціальний метод для експорту з ROI на повний розмір
            heatmap = self._analysis_visualizer.create_heatmap_for_export(
                self._current_analysis_data,
                self._current_analysis_mask,
                self._roi_bbox or FULL_EARTH_BBOX,
                width=export_width, 
                height=export_height,
                show_legend=True,  # З легендою
                show_grid=True,    # З сіткою координат
                analysis_info=analysis_info  # З інформацією про аналіз
            )
            
            # Зберігаємо зображення
            heatmap.save(filename)
            
            # Оновлюємо інформацію про папку збереження в GUI
            import os
            full_path = os.path.abspath(filename)
            if hasattr(self._controls, 'set_save_location'):
                self._controls.set_save_location(os.path.dirname(full_path))
            
            return True
            
        except Exception as exc:
            LOGGER.error("Failed to export analysis results: %s", exc)
            return False
    
    def set_analysis_colormap(self, colormap_name: str) -> None:
        """Встановлює colormap для аналізу."""
        if colormap_name == "hot":
            self._analysis_colormap = ColorMap.hot()
        elif colormap_name == "cool":
            self._analysis_colormap = ColorMap.cool()
        elif colormap_name == "viridis":
            self._analysis_colormap = ColorMap.viridis()
        elif colormap_name == "plasma":
            self._analysis_colormap = ColorMap.plasma()
        else:
            self._analysis_colormap = ColorMap.hot()
        
        # Оновлюємо поточну візуалізацію, якщо є дані
        if self._current_analysis_data is not None and self._current_analysis_mask is not None:
            self._create_analysis_heatmap(
                self._current_analysis_data,
                self._current_analysis_mask,
                1024, 512
            )

    def export_globe_image(self) -> None:
        """Експортує фото глобуса з діалогом вибору папки."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            import datetime
            import os
            
            # Відкриваємо діалог вибору папки
            save_dir = filedialog.askdirectory(
                title="Select folder to save globe image",
                initialdir=os.getcwd()
            )
            
            if not save_dir:
                return
            
            # Створюємо скріншот
            globe_img = self._capture_globe_screenshot()
            if globe_img is None:
                print("Failed to capture globe screenshot")
                return
            
            # Генеруємо ім'я файлу з timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"globe_image_{timestamp}.png"
            full_path = os.path.join(save_dir, filename)
            
            # Зберігаємо зображення
            globe_img.save(full_path)
            print(f"Globe image exported to {full_path}")
            
            # Оновлюємо інформацію про папку збереження в GUI
            if hasattr(self._controls, 'set_save_location'):
                self._controls.set_save_location(save_dir)
            
            # Якщо є аналіз, додаємо його як overlay
            if self._current_analysis_data is not None and self._current_analysis_mask is not None:
                self._add_analysis_overlay_to_globe_image(globe_img, full_path)
                    
        except Exception as e:
            print(f"Error exporting globe image: {e}")
    
    def _capture_globe_screenshot(self, width: int = None, height: int = None) -> Image.Image:
        """Створює скріншот глобуса."""
        try:
            # Отримуємо розмір вікна
            if width is None or height is None:
                fb_width, fb_height = glfw.get_framebuffer_size(self.window)
                width = width or fb_width
                height = height or fb_height
            
            # Читаємо пікселі з OpenGL буфера
            pixels = self.ctx.screen.read(components=3, dtype='f1')
            
            # Створюємо зображення з пікселів
            img_array = np.frombuffer(pixels, dtype=np.uint8)
            img_array = img_array.reshape((height, width, 3))
            
            # Перевертаємо зображення (OpenGL має інвертовану Y-координату)
            img_array = np.flipud(img_array)
            
            # Конвертуємо в PIL Image
            result = Image.fromarray(img_array)
            if result.mode != 'RGB':
                result = result.convert('RGB')
            return result
                    
        except Exception as e:
            print(f"Error capturing globe screenshot: {e}")
            return None

    def _quick_export_globe(self) -> None:
        """Швидкий експорт фото глобуса без діалогу."""
        try:
            # Створюємо скріншот
            globe_img = self._capture_globe_screenshot()
            if globe_img is None:
                print("Failed to capture globe screenshot")
                return
            
            # Генеруємо ім'я файлу з timestamp
            import datetime
            import os
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"globe_image_{timestamp}.png"
            full_path = os.path.abspath(filename)
            
            # Зберігаємо зображення
            globe_img.save(filename)
            print(f"Globe image exported to {full_path}")
            
            # Оновлюємо інформацію про папку збереження в GUI
            if hasattr(self._controls, 'set_save_location'):
                self._controls.set_save_location(os.path.dirname(full_path))
            
            # Якщо є аналіз, додаємо його як overlay
            if self._current_analysis_data is not None and self._current_analysis_mask is not None:
                self._add_analysis_overlay_to_globe_image(globe_img, filename)
                    
        except Exception as e:
            print(f"Error in quick export: {e}")
    
    def _add_analysis_overlay_to_globe_image(self, globe_img: Image.Image, filename: str) -> None:
        """Додає overlay аналізу до зображення глобуса."""
        try:
            # Створюємо heatmap аналізу
            heatmap_img = self._analysis_visualizer.create_heatmap_texture(
                self._current_analysis_data,
                self._current_analysis_mask,
                width=globe_img.width,
                height=globe_img.height,
                show_legend=self._show_analysis_legend
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
            
            result_img.save(overlay_filename)
            print(f"Globe image with analysis overlay exported to {overlay_filename}")
            
        except Exception as e:
            print(f"Error adding analysis overlay: {e}")
    
    def toggle_analysis_legend(self) -> None:
        """Перемикає відображення легенди аналізу."""
        self._show_analysis_legend = not self._show_analysis_legend
        
        # Оновлюємо поточну візуалізацію, якщо є дані
        if self._current_analysis_data is not None and self._current_analysis_mask is not None:
            self._create_analysis_heatmap(
                self._current_analysis_data,
                self._current_analysis_mask,
                1024, 512
            )

    def _layer_by_id(self, layer_id: str) -> TerraLayerOption | None:
        for layer in self._terra_layers:
            if layer.layer_id == layer_id:
                return layer
        for candidate in TERRA_LAYER_OPTIONS:
            if candidate.layer_id == layer_id:
                return candidate
        return None

    def _update_sphere_texture(self, image: Image.Image, *, description: str, update_timestamp: bool = False) -> None:
        # Ensure RGBA then optionally turn near-black to transparent
        converted = image.convert("RGBA")
        if bool(getattr(self.cfg, 'transparentize_black', True)):
            try:
                arr = np.array(converted, dtype=np.uint8, copy=True)
                thr = 10  # per-channel threshold for "near black"
                black_mask = (arr[..., 0] <= thr) & (arr[..., 1] <= thr) & (arr[..., 2] <= thr)
                # Only apply if non-trivial amount present
                if np.mean(black_mask) >= 0.002:
                    arr[..., 3] = np.where(black_mask, 0, arr[..., 3])
                    converted = Image.fromarray(arr)
                    if converted.mode != "RGBA":
                        converted = converted.convert("RGBA")
            except Exception:
                pass
        # Optionally flip vertically so north appears at the top
        if bool(self.cfg.mirror_texture_v):
            converted = ImageOps.flip(converted)
        width, height = converted.size
        texture = self.ctx.texture((width, height), 4, converted.tobytes())
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            texture.anisotropy = 4.0
        except AttributeError:
            pass
        texture.repeat_x = True
        texture.repeat_y = False
        if self._sphere_texture is not None:
            self._sphere_texture.release()
        self._sphere_texture = texture
        self._terra_texture_info = description
        if update_timestamp:
            self._terra_last_update = dt.datetime.now()

    def _load_map_layers(self) -> None:
        self._release_map_layers()
        try:
            self._geodata = load_base_layers(self.cfg.geodata_root)
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("Failed to load base geodata: %s", exc)
            self._geodata = None
            return
        if not self._geodata:
            return
        coast_vbo, coast_vao, coast_offsets = self._build_polyline_layer(
            self._geodata.continents,
            radius=1.0015,
        )
        ukraine_vbo, ukraine_vao, ukraine_offsets = self._build_polyline_layer(
            self._geodata.ukraine,
            radius=1.0025,
        )
        self._coast_vbo, self._coast_vao, self._coast_offsets = coast_vbo, coast_vao, coast_offsets
        self._ukraine_vbo, self._ukraine_vao, self._ukraine_offsets = ukraine_vbo, ukraine_vao, ukraine_offsets

    def _build_polyline_layer(
        self,
        paths: List[List[Tuple[float, float]]],
        *,
        radius: float,
    ) -> Tuple[moderngl.Buffer | None, moderngl.VertexArray | None, List[Tuple[int, int]]]:
        segments: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []
        start = 0
        for path in paths or []:
            if len(path) < 2:
                continue
            coords = []
            for lat, lon in path:
                theta = math.radians(90.0 - lat)
                phi = math.radians(lon)
                x, y, z = spherical_to_cartesian(theta, phi)
                coords.append((radius * x, radius * y, radius * z))
            if len(coords) < 2:
                continue
            arr = np.asarray(coords, dtype=np.float32)
            segments.append(arr)
            offsets.append((start, arr.shape[0]))
            start += arr.shape[0]
        if not segments:
            return None, None, []
        data = np.vstack(segments)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.grid_prog,
            [
                (vbo, "3f", "in_position"),
            ],
        )
        return vbo, vao, offsets

    def _render_polyline_layer(self, vao: moderngl.VertexArray | None, offsets: List[Tuple[int, int]]) -> None:
        if vao is None or not offsets:
            return
        for start, count in offsets:
            vao.render(mode=moderngl.LINE_STRIP, first=start, vertices=count)

    def _update_roi_overlay(self, bbox: GeoBoundingBox | None) -> None:
        self._release_roi_overlay()
        self._roi_bbox = bbox
        if bbox is None:
            return
        try:
            bbox.validate()
        except ValueError:
            return
        
        # Використовуємо новий ROI візуалізатор
        segments_3d, offsets = self._roi_visualizer.create_roi_overlay_geometry(
            bbox, self._roi_type, segments=90
        )
        
        if not segments_3d:
            return
        
        # Створюємо VBO та VAO для ROI
        data = np.vstack(segments_3d)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.grid_prog,
            [(vbo, "3f", "in_position")],
        )
        
        self._roi_overlay_vbo = vbo
        self._roi_overlay_vao = vao
        self._roi_overlay_offsets = offsets

    def _release_roi_overlay(self) -> None:
        if self._roi_overlay_vao is not None:
            self._roi_overlay_vao.release()
        if self._roi_overlay_vbo is not None:
            self._roi_overlay_vbo.release()
        self._roi_overlay_vbo = None
        self._roi_overlay_vao = None
        self._roi_overlay_offsets = []

    def _release_analysis_heatmap(self) -> None:
        """Звільняє ресурси heatmap аналізу."""
        if self._analysis_heatmap_texture is not None:
            self._analysis_heatmap_texture.release()
        if self._analysis_heatmap_vao is not None:
            self._analysis_heatmap_vao.release()
        if self._analysis_heatmap_vbo is not None:
            self._analysis_heatmap_vbo.release()
        self._analysis_heatmap_texture = None
        self._analysis_heatmap_vao = None
        self._analysis_heatmap_vbo = None

    def _roi_path_from_bbox(self, bbox: GeoBoundingBox, segments: int = 90) -> List[Tuple[float, float]]:
        segments = max(4, segments)
        lons = np.linspace(bbox.min_lon, bbox.max_lon, segments, dtype=np.float32)
        lats = np.linspace(bbox.min_lat, bbox.max_lat, segments, dtype=np.float32)
        path: List[Tuple[float, float]] = []
        for lon in lons:
            path.append((float(bbox.min_lat), float(lon)))
        for lat in lats[1:]:
            path.append((float(lat), float(bbox.max_lon)))
        for lon in lons[-2::-1]:
            path.append((float(bbox.max_lat), float(lon)))
        for lat in lats[-2:0:-1]:
            path.append((float(lat), float(bbox.min_lon)))
        if path:
            path.append(path[0])
        return path

    def _release_map_layers(self) -> None:
        for vao, vbo in ((self._coast_vao, self._coast_vbo), (self._ukraine_vao, self._ukraine_vbo)):
            if vao is not None:
                vao.release()
            if vbo is not None:
                vbo.release()
        self._coast_vbo = None
        self._coast_vao = None
        self._coast_offsets = []
        self._ukraine_vbo = None
        self._ukraine_vao = None
        self._ukraine_offsets = []

    def _build_static_points(self) -> None:
        self._release_points()
        def make_point(label: str, lat: float, lon: float) -> PointLabel | None:
            data = self._prepare_city_marker(lat, lon)
            if data is None:
                return None
            line_vbo = self.ctx.buffer(data.line_vertices.tobytes()) if data.line_vertices.size else None
            line_vao = self.ctx.vertex_array(self.grid_prog, [(line_vbo, "3f", "in_position")]) if line_vbo else None
            line_count = int(data.line_vertices.shape[0]) if data.line_vertices.size else 0
            tri_vbo = self.ctx.buffer(data.triangle_vertices.tobytes()) if data.triangle_vertices.size else None
            tri_vao = self.ctx.vertex_array(self.marker_prog, [(tri_vbo, "3f", "in_position")]) if tri_vbo else None
            tri_count = int(data.triangle_vertices.shape[0]) if data.triangle_vertices.size else 0
            texture, size = self._build_label_texture(label)
            label_vbo = self.ctx.buffer(reserve=6 * 5 * 4) if texture is not None else None
            label_vao = self.ctx.vertex_array(self.label_prog, [(label_vbo, "3f 2f", "in_position", "in_uv")]) if label_vbo else None
            label_size_px = np.array(size if texture is not None else (0, 0), dtype=np.float32)
            normal, east, north = self._city_basis(lat, lon)
            return PointLabel(
                label=label,
                lat=lat,
                lon=lon,
                normal=normal,
                east=east,
                north=north,
                label_anchor=data.label_anchor,
                line_vbo=line_vbo,
                line_vao=line_vao,
                line_vertex_count=line_count,
                tri_vbo=tri_vbo,
                tri_vao=tri_vao,
                tri_vertex_count=tri_count,
                label_texture=texture,
                label_vbo=label_vbo,
                label_vao=label_vao,
                label_size_px=label_size_px,
                label_world_size=label_size_px * self._label_cfg.pixel_scale,
            )

        kyiv = make_point(KYIV_LABEL, *KYIV_COORDS)
        tokyo = make_point("TOKYO", 35.6762, 139.6503)
        self._points = [p for p in (kyiv, tokyo) if p is not None]

    def _release_points(self) -> None:
        for p in self._points:
            p.release()
        self._points = []

    def _prepare_city_marker(self, lat: float, lon: float) -> MarkerData | None:
        normal, east, north = self._city_basis(lat, lon)
        surface_point = self._latlon_to_cartesian(lat, lon, radius=MARKER_SURFACE_RADIUS)
        dot_center = (surface_point + normal * MARKER_DOT_ELEVATION).astype(np.float32)
        label_anchor = surface_point + normal * MARKER_LABEL_NORMAL_OFFSET

        triangle_vertices: list[np.ndarray] = []
        outline_points: list[np.ndarray] = []
        if MARKER_DOT_SEGMENTS >= 3:
            points: list[np.ndarray] = []
            for idx in range(MARKER_DOT_SEGMENTS):
                angle = 2.0 * math.pi * idx / MARKER_DOT_SEGMENTS
                direction = math.cos(angle) * east + math.sin(angle) * north
                point = dot_center + direction * MARKER_DOT_RADIUS
                points.append(point.astype(np.float32))
            # Store outline ring points (closed loop)
            outline_points = [p.copy() for p in points]
            outline_points.append(points[0].copy())
            for idx in range(len(points)):
                triangle_vertices.append(dot_center.copy())
                triangle_vertices.append(points[idx])
                triangle_vertices.append(points[(idx + 1) % len(points)])

        triangle_array = (
            np.asarray(triangle_vertices, dtype=np.float32)
            if triangle_vertices
            else np.empty((0, 3), dtype=np.float32)
        )
        return MarkerData(
            normal=normal,
            label_anchor=label_anchor.astype(np.float32),
            line_vertices=(
                np.asarray(outline_points, dtype=np.float32)
                if outline_points
                else np.empty((0, 3), dtype=np.float32)
            ),
            triangle_vertices=triangle_array,
        )

    @staticmethod
    def _city_basis(lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = math.radians(90.0 - lat)
        phi = math.radians(lon)
        normal = np.array(spherical_to_cartesian(theta, phi), dtype=np.float32)
        normal = normal / np.linalg.norm(normal)

        east = np.array(
            (
                math.sin(theta) * math.cos(phi),
                0.0,
                -math.sin(theta) * math.sin(phi),
            ),
            dtype=np.float32,
        )
        if np.linalg.norm(east) < 1e-6:
            east = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), normal)
        if np.linalg.norm(east) < 1e-6:
            east = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        east = east / np.linalg.norm(east)
        north = np.cross(normal, east)
        north = north / np.linalg.norm(north)
        return normal, east, north

    def _build_label_texture(self, text: str) -> tuple[moderngl.Texture | None, tuple[int, int]]:
        text = text.strip()
        if not text:
            return None, (0, 0)

        # Adjust this value to change the base font size of labels
        font_size = 48  # Smaller value = smaller text (try values between 32-64)
        try:
            # Try to load the system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            try:
                # Fallback to DejaVu if available
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:
                # Last resort - default font
                font = ImageFont.load_default()

        # Create a temporary image to measure text size
        temp_image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_image)
        text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        
        # Add padding for better visibility
        padding = 4
        text_width = text_bbox[2] - text_bbox[0] + padding * 2
        text_height = text_bbox[3] - text_bbox[1] + padding * 2

        # Calculate final image dimensions with padding
        img_width = max(2, text_width + 2 * LABEL_PIXEL_PADDING)
        img_height = max(2, text_height + 2 * LABEL_PIXEL_PADDING)

        # Create image with transparent background
        image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Calculate text position with padding
        x = LABEL_PIXEL_PADDING + padding - text_bbox[0]
        y = LABEL_PIXEL_PADDING + padding - text_bbox[1]
        
        # Draw text with white color for better visibility
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        # Convert to numpy array and flip image for OpenGL coordinates
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_data = np.array(image, dtype=np.uint8)
        
        # Create texture directly from RGBA data
        texture = self.ctx.texture(image.size, 4, image_data.tobytes())
        # Use high-quality filtering with mipmaps to avoid pixelation at small on-screen sizes
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            max_aniso = getattr(self.ctx, "max_anisotropy", 8.0)
            texture.anisotropy = float(max(4.0, max_aniso))
        except Exception:
            pass
        try:
            texture.repeat_x = False
            texture.repeat_y = False
        except Exception:
            pass
        texture.build_mipmaps()
        return texture, (img_width, img_height)

    def _update_label_geometry_for_point(
        self,
        point: PointLabel,
        model: np.ndarray,
        proj_view: np.ndarray,
        viewport: tuple[int, int],
        camera_position: np.ndarray,
    ) -> bool:
        if point.label_vbo is None or point.label_texture is None or point.label_world_size[0] <= 0:
            return False

        anchor_local = np.append(point.label_anchor, 1.0).astype(np.float32)
        anchor_world = (model @ anchor_local)[:3].astype(np.float32)

        normal_local = np.append(point.normal, 0.0).astype(np.float32)
        east_local = np.append(point.east, 0.0).astype(np.float32)
        north_local = np.append(point.north, 0.0).astype(np.float32)
        normal_world = (model @ normal_local)[:3].astype(np.float32)
        right = (model @ east_local)[:3].astype(np.float32)
        up = (model @ north_local)[:3].astype(np.float32)
        normal_world = normal_world / (np.linalg.norm(normal_world) or 1.0)
        right = right / (np.linalg.norm(right) or 1.0)
        up = up / (np.linalg.norm(up) or 1.0)

        cfg = self._label_cfg
        anchor_world = anchor_world + normal_world * cfg.surface_normal_offset + right * cfg.right_offset + up * cfg.up_offset

        to_camera = camera_position - anchor_world
        distance = np.linalg.norm(to_camera)
        if distance <= 1e-6:
            return False
        view_dir = to_camera / distance
        if np.dot(normal_world, view_dir) <= 0.0:
            return False

        height_world = float(point.label_world_size[1])

        def _project(p: np.ndarray) -> np.ndarray:
            v = np.append(p, 1.0).astype(np.float32)
            clip = proj_view @ v
            if abs(clip[3]) < 1e-6:
                return np.array([0.0, 0.0], dtype=np.float32)
            ndc = clip[:3] / clip[3]
            x = (ndc[0] * 0.5 + 0.5) * viewport[0]
            y = (ndc[1] * 0.5 + 0.5) * viewport[1]
            return np.array([x, y], dtype=np.float32)

        half_width = 0.5 * float(point.label_world_size[0])
        height = height_world
        bottom_center = anchor_world
        top_center = bottom_center + up * height

        bottom_left = bottom_center - right * half_width
        bottom_right = bottom_center + right * half_width
        top_left = top_center - right * half_width
        top_right = top_center + right * half_width

        vertex_data = np.array(
            [
                [*top_left, 0.0, 1.0],
                [*bottom_left, 0.0, 0.0],
                [*bottom_right, 1.0, 0.0],
                [*top_left, 0.0, 1.0],
                [*bottom_right, 1.0, 0.0],
                [*top_right, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        point.label_vbo.orphan()
        point.label_vbo.write(vertex_data.tobytes())
        return True

    @staticmethod
    def _latlon_to_cartesian(lat: float, lon: float, *, radius: float = 1.0) -> np.ndarray:
        theta = math.radians(90.0 - lat)
        phi = math.radians(lon)
        x, y, z = spherical_to_cartesian(theta, phi)
        return np.asarray((radius * x, radius * y, radius * z), dtype=np.float32)

    def run(self) -> None:
        previous_time = glfw.get_time()
        try:
            while not glfw.window_should_close(self.window):
                current_time = glfw.get_time()
                delta_time = current_time - previous_time
                previous_time = current_time

                if not self._controls.poll():
                    glfw.set_window_should_close(self.window, True)
                    break

                glfw.poll_events()

                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(self.window, True)

                changed, force, state = self._controls.consume_changes()
                if changed or force:
                    self._apply_control_state(state)
                    self._request_current_terra_texture(force=force)

                self._ingest_analysis_requests()
                self._service_analysis_tasks()
                self._process_terra_updates_with_backfill()
                fps_text = f"FPS: {1.0 / delta_time:0.1f}" if delta_time > 1e-6 else "FPS: ∞"
                self._controls.set_fps(fps_text)
                updated_str = (
                    self._terra_last_update.strftime("Updated: %Y-%m-%d %H:%M:%S")
                    if self._terra_last_update is not None
                    else None
                )
                self._controls.update_status(
                    self._terra_controller.status,
                    self._terra_controller.status_detail,
                    self._terra_texture_info,
                    updated_str,
                )

                self._render_frame(delta_time)
                glfw.swap_buffers(self.window)
        finally:
            self._cleanup()


    def _render_frame(self, _delta_time: float) -> None:
        self.ctx.clear(*THEME_BACKGROUND, 1.0)
        width, height = glfw.get_framebuffer_size(self.window)
        if height == 0:
            return
        self.ctx.viewport = (0, 0, width, height)

        proj = perspective(math.radians(45.0), width / height, 0.1, 20.0)
        view = look_at(self._camera_position, self._camera_target, self._camera_up)
        
        # Обробляємо анімацію глобуса
        if self._animation_active:
            import time
            current_time = time.time()
            elapsed_time = current_time - self._animation_start_time
            # Додаємо обертання по осі Y (yaw) з часом
            animation_yaw = elapsed_time * self._animation_speed
            model = rotation_x(self.rotation_pitch) @ rotation_y(self.rotation_yaw + animation_yaw)
        else:
            model = rotation_x(self.rotation_pitch) @ rotation_y(self.rotation_yaw)
        mvp = proj @ view @ model
        normal_matrix = model[:3, :3]

        self.sphere_prog["mvp"].write(mvp.astype("f4").T.tobytes())
        self.sphere_prog["normal_matrix"].write(normal_matrix.astype("f4").T.tobytes())
        self.sphere_prog["light_dir"].value = (0.35, 0.55, 0.85)
        has_texture = self._sphere_texture is not None
        self.sphere_prog["use_texture"].value = bool(has_texture)
        self.sphere_prog["base_color"].value = THEME_SPHERE
        if has_texture:
            self._sphere_texture.use(location=0)
        self._sphere_vao.render()

        self.grid_prog["mvp"].write(mvp.astype("f4").T.tobytes())
        self.grid_prog["line_color"].value = THEME_GRID
        self._grid_vao.render(mode=moderngl.LINES, vertices=self._grid_vertex_count)

        if self._coast_vao is not None:
            self.grid_prog["line_color"].value = THEME_COASTLINE
            self._render_polyline_layer(self._coast_vao, self._coast_offsets)
        if self._ukraine_vao is not None:
            self.grid_prog["line_color"].value = THEME_UKRAINE
            self._render_polyline_layer(self._ukraine_vao, self._ukraine_offsets)
        if self._roi_overlay_vao is not None and self._roi_overlay_offsets:
            self.grid_prog["line_color"].value = self._roi_style.line_color
            self._render_polyline_layer(self._roi_overlay_vao, self._roi_overlay_offsets)
        
        # Рендеримо heatmap аналізу як додаткову текстуру на сфері
        if self._analysis_heatmap_texture is not None:
            self._render_analysis_heatmap(mvp)
        
        # Render points (dot, outline, label) for each labeled location
        if self._points:
            for p in self._points:
                if p.tri_vao is not None and p.tri_vertex_count:
                    self.marker_prog["mvp"].write(mvp.astype("f4").T.tobytes())
                    self.marker_prog["fill_color"].value = THEME_MARKER_FILL
                    p.tri_vao.render(mode=moderngl.TRIANGLES, vertices=p.tri_vertex_count)
                if p.line_vao is not None and p.line_vertex_count:
                    self.grid_prog["mvp"].write(mvp.astype("f4").T.tobytes())
                    self.grid_prog["line_color"].value = THEME_MARKER_OUTLINE
                    p.line_vao.render(mode=moderngl.LINE_STRIP, vertices=p.line_vertex_count)
            # Labels rendered in view space so they billboard to camera
            label_mvp = proj @ view
            self.label_prog["mvp"].write(label_mvp.astype("f4").T.tobytes())
            for p in self._points:
                if p.label_vao is None or p.label_texture is None:
                    continue
                visible = self._update_label_geometry_for_point(p, model, label_mvp, (width, height), self._camera_position)
                if not visible:
                    continue
                # Draw labels without depth test so they never sink into the sphere
                self.ctx.disable(moderngl.DEPTH_TEST)
                p.label_texture.use(location=0)
                p.label_vao.render(mode=moderngl.TRIANGLES, vertices=6)
                self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_analysis_heatmap(self, mvp: np.ndarray) -> None:
        """Рендерить heatmap аналізу на сфері."""
        if self._analysis_heatmap_texture is None:
            return
        
        # Використовуємо той же шейдер, що й для основної текстури сфери
        # але з додатковим блендингом
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        self.sphere_prog["mvp"].write(mvp.astype("f4").T.tobytes())
        self.sphere_prog["normal_matrix"].write(np.eye(3, dtype=np.float32).T.tobytes())
        self.sphere_prog["light_dir"].value = (0.35, 0.55, 0.85)
        self.sphere_prog["use_texture"].value = True
        self.sphere_prog["base_color"].value = (1.0, 1.0, 1.0)  # Білий базовий колір для heatmap
        
        # Використовуємо heatmap текстуру
        self._analysis_heatmap_texture.use(location=0)
        self._sphere_vao.render()
        
        self.ctx.disable(moderngl.BLEND)

    def _on_framebuffer_resize(self, _window: glfw._GLFWwindow, width: int, height: int) -> None:
        self._refresh_viewport(width, height)

    def _refresh_viewport(self, width: int, height: int) -> None:
        width = max(1, width)
        height = max(1, height)
        self.ctx.viewport = (0, 0, width, height)

    def _on_mouse_button(self, window: glfw._GLFWwindow, button: int, action: int, _mods: int) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            self._drag_active = True
            self._last_cursor = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self._drag_active = False
            self._last_cursor = None

    def _on_scroll(self, _window: glfw._GLFWwindow, _xoffset: float, yoffset: float) -> None:
        # Zoom towards/away from target along the current view direction
        to_target = self._camera_position - self._camera_target
        distance = float(np.linalg.norm(to_target))
        if distance <= 1e-6:
            return
        direction = to_target / distance
        distance = float(np.clip(distance - yoffset * CAMERA_ZOOM_STEP, CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX))
        self._camera_position = self._camera_target + direction * distance

    def _on_cursor_move(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        if not self._drag_active:
            return
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return
        last_x, last_y = self._last_cursor
        dx = xpos - last_x
        dy = ypos - last_y
        self._last_cursor = (xpos, ypos)

        sensitivity = 0.005
        self.rotation_yaw += dx * sensitivity
        self.rotation_pitch += dy * sensitivity
        self.rotation_pitch = float(np.clip(self.rotation_pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01))

    def _on_key_press(self, window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:
        """Обробляє натискання клавіш."""
        if action != glfw.PRESS:
            return
        
        if key == glfw.KEY_L:
            # Перемикаємо легенду аналізу
            self.toggle_analysis_legend()
        elif key == glfw.KEY_1:
            # Встановлюємо hot colormap
            self.set_analysis_colormap("hot")
        elif key == glfw.KEY_2:
            # Встановлюємо cool colormap
            self.set_analysis_colormap("cool")
        elif key == glfw.KEY_3:
            # Встановлюємо viridis colormap
            self.set_analysis_colormap("viridis")
        elif key == glfw.KEY_4:
            # Встановлюємо plasma colormap
            self.set_analysis_colormap("plasma")
        elif key == glfw.KEY_E:
            # Експортуємо результати аналізу
            if self._current_analysis_data is not None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_results_{timestamp}.png"
                if self.export_analysis_results(filename):
                    print(f"Analysis results exported to {filename}")
                else:
                    print("Failed to export analysis results")
        elif key == glfw.KEY_G:
            # Швидкий експорт фото глобуса
            self._quick_export_globe()
        elif key == glfw.KEY_F:
            # Розширений експорт фото глобуса з діалогом
            self.export_globe_image()
        elif key == glfw.KEY_R:
            # Відкриваємо ROI селектор
            self._open_roi_selector()
        elif key == glfw.KEY_D:
            # Відкриваємо детальний дисплей аналізу
            self._open_analysis_display()
        elif key == glfw.KEY_X:
            # Відкриваємо діалог експорту
            self._open_export_dialog()
        elif key == glfw.KEY_A:
            # Перемикаємо анімацію глобуса
            self._toggle_animation()
        elif key == glfw.KEY_H:
            # Показуємо довідку
            self._show_help()

    def _toggle_animation(self) -> None:
        """Перемикає анімацію глобуса."""
        import time
        
        if self._animation_active:
            # Зупиняємо анімацію
            self._animation_active = False
            print("Animation stopped")
        else:
            # Запускаємо анімацію
            self._animation_active = True
            self._animation_start_time = time.time()
            # Скидаємо камеру на нуль
            self._reset_camera_to_zero()
            print("Animation started - Globe will rotate slowly")
    
    def _reset_camera_to_zero(self) -> None:
        """Скидає камеру на початкову позицію."""
        self._camera_position = np.array([0.0, 0.0, 3.2], dtype=np.float32)
        self._camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.rotation_pitch = 0.0
        self.rotation_yaw = 0.0

    def _show_help(self) -> None:
        """Показує довідку по клавішах."""
        help_text = """
Terra Tools - Keyboard Shortcuts:
================================
A - Toggle globe animation (slow rotation)
L - Toggle analysis legend
1 - Hot colormap
2 - Cool colormap  
3 - Viridis colormap
4 - Plasma colormap
E - Export analysis results
G - Quick export globe image
F - Advanced export globe image
R - Open ROI selector
D - Open detailed analysis display
X - Open export dialog
H - Show this help
ESC - Exit
        """
        print(help_text)

    def _open_roi_selector(self) -> None:
        """Відкриває діалог вибору ROI."""
        try:
            # Створюємо тимчасове Tkinter вікно для діалогу
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Ховаємо головне вікно
            
            # Відкриваємо діалог вибору ROI
            selection = create_roi_selector_dialog(root, self._roi_bbox)
            
            if selection and selection.is_valid:
                # Оновлюємо ROI
                self._roi_bbox = selection.bbox
                self._roi_type = selection.roi_type
                self._update_roi_overlay(selection.bbox)
                print(f"ROI updated: {selection.bbox.min_lon:.2f}, {selection.bbox.min_lat:.2f} -> {selection.bbox.max_lon:.2f}, {selection.bbox.max_lat:.2f}")
            
            root.destroy()
            
        except Exception as exc:
            print(f"Failed to open ROI selector: {exc}")

    def _open_analysis_display(self) -> None:
        """Відкриває детальний дисплей аналізу."""
        if self._current_analysis_data is None or self._current_analysis_mask is None:
            print("No analysis data available. Run an analysis first.")
            return
        
        try:
            # Створюємо тимчасове Tkinter вікно для діалогу
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Ховаємо головне вікно
            
            # Підготовлюємо статистику
            masked_data = np.where(self._current_analysis_mask, self._current_analysis_data, np.nan)
            valid_data = masked_data[np.isfinite(masked_data)]
            
            stats = {}
            if valid_data.size > 0:
                stats = {
                    "Valid Pixels": int(valid_data.size),
                    "Data Range": f"{float(valid_data.min()):.6f} to {float(valid_data.max()):.6f}",
                    "Mean Value": f"{float(valid_data.mean()):.6f}",
                    "Standard Deviation": f"{float(valid_data.std()):.6f}",
                }
            
            # Відкриваємо діалог дисплею аналізу
            create_analysis_display_dialog(
                root,
                self._current_analysis_data,
                self._current_analysis_mask,
                self._roi_bbox or FULL_EARTH_BBOX,
                "Analysis Results",
                stats
            )
            
            root.destroy()
            
        except Exception as exc:
            print(f"Failed to open analysis display: {exc}")

    def _open_export_dialog(self) -> None:
        """Відкриває діалог експорту аналізу."""
        if self._current_analysis_data is None or self._current_analysis_mask is None:
            print("No analysis data available. Run an analysis first.")
            return
        
        try:
            import tkinter as tk
            from tkinter import filedialog
            import datetime
            import os
            
            # Відкриваємо діалог вибору папки
            save_dir = filedialog.askdirectory(
                title="Select folder to save analysis results",
                initialdir=os.getcwd()
            )
            
            if not save_dir:
                return
            
            # Генеруємо ім'я файлу з timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.png"
            full_path = os.path.join(save_dir, filename)
            
            # Експортуємо результати
            if self.export_analysis_results(full_path):
                print(f"Analysis results exported to {full_path}")
                
                # Оновлюємо інформацію про папку збереження в GUI
                if hasattr(self._controls, 'set_save_location'):
                    self._controls.set_save_location(save_dir)
            else:
                print("Failed to export analysis results")
            
        except Exception as exc:
            print(f"Failed to export analysis: {exc}")

    def _cleanup(self) -> None:
        self._terra_controller.shutdown()
        if self._analysis_future is not None:
            self._analysis_future.cancel()
        self._analysis_future = None
        self._analysis_pending.clear()
        self._analysis_executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "_controls"):
            self._controls.destroy()
        self._release_roi_overlay()
        self._release_analysis_heatmap()
        self._release_map_layers()
        self._release_points()

        glfw.set_framebuffer_size_callback(self.window, None)
        glfw.set_cursor_pos_callback(self.window, None)
        glfw.set_mouse_button_callback(self.window, None)
        glfw.set_scroll_callback(self.window, None)
        glfw.set_key_callback(self.window, None)

        if self._sphere_vao is not None:
            self._sphere_vao.release()
        if self._grid_vao is not None:
            self._grid_vao.release()
        if self._sphere_vbo is not None:
            self._sphere_vbo.release()
        if self._normal_vbo is not None:
            self._normal_vbo.release()
        if self._uv_vbo is not None:
            self._uv_vbo.release()
        if self._grid_vbo is not None:
            self._grid_vbo.release()
        if self._sphere_texture is not None:
            self._sphere_texture.release()

        if hasattr(self, "sphere_prog"):
            self.sphere_prog.release()
        if hasattr(self, "grid_prog"):
            self.grid_prog.release()
        if hasattr(self, "marker_prog"):
            self.marker_prog.release()
        if hasattr(self, "label_prog"):
            self.label_prog.release()

        self.ctx.release()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def _build_sphere_mesh(self, lat_segments: int, lon_segments: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices: list[list[float]] = []
        normals: list[list[float]] = []
        uvs: list[list[float]] = []
        for lat in range(lat_segments):
            frac0 = lat / lat_segments
            frac1 = (lat + 1) / lat_segments
            theta0 = math.pi * frac0
            theta1 = math.pi * frac1
            # Standard V mapping: north (v=0) -> south (v=1)
            v0 = frac0
            v1 = frac1
            for lon in range(lon_segments):
                lon_frac0 = lon / lon_segments
                lon_frac1 = (lon + 1) / lon_segments
                phi0 = lon_frac0 * 2.0 * math.pi - math.pi
                phi1 = lon_frac1 * 2.0 * math.pi - math.pi
                u0 = lon_frac0
                u1 = lon_frac1 if lon < lon_segments - 1 else 1.0

                p0 = spherical_to_cartesian(theta0, phi0)
                p1 = spherical_to_cartesian(theta1, phi0)
                p2 = spherical_to_cartesian(theta1, phi1)
                p3 = spherical_to_cartesian(theta0, phi1)

                vertices.extend((p0, p1, p2, p0, p2, p3))
                normals.extend((p0, p1, p2, p0, p2, p3))
                uvs.extend((
                    (u0, v0),
                    (u0, v1),
                    (u1, v1),
                    (u0, v0),
                    (u1, v1),
                    (u1, v0),
                ))
        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(normals, dtype=np.float32),
            np.asarray(uvs, dtype=np.float32),
        )

    @staticmethod
    def _build_grid(lat_step: int, lon_step: int) -> np.ndarray:
        segments: list[list[float]] = []
        radius = 1.002  # lift slightly above the surface to avoid z-fighting

        # Parallels
        for lat in range(-90 + lat_step, 90, lat_step):
            theta = math.radians(90 - lat)
            ring_points = _sample_circle(theta, 256)
            for p0, p1 in zip(ring_points, ring_points[1:]):
                segments.append((radius * p0[0], radius * p0[1], radius * p0[2]))
                segments.append((radius * p1[0], radius * p1[1], radius * p1[2]))

        # Meridians
        for lon in range(0, 360, lon_step):
            phi = math.radians(lon)
            meridian_points = _sample_meridian(phi, 192)
            for p0, p1 in zip(meridian_points, meridian_points[1:]):
                segments.append((radius * p0[0], radius * p0[1], radius * p0[2]))
                segments.append((radius * p1[0], radius * p1[1], radius * p1[2]))

        return np.asarray(segments, dtype=np.float32)


def spherical_to_cartesian(theta: float, phi: float) -> Tuple[float, float, float]:
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    x = sin_theta * sin_phi
    y = cos_theta
    z = sin_theta * cos_phi
    return (x, y, z)


def _sample_circle(theta: float, samples: int) -> list[Tuple[float, float, float]]:
    data: list[Tuple[float, float, float]] = []
    for i in range(samples + 1):  # close the loop
        phi = 2.0 * math.pi * i / samples
        data.append(spherical_to_cartesian(theta, phi))
    return data


def _sample_meridian(phi: float, samples: int) -> list[Tuple[float, float, float]]:
    data: list[Tuple[float, float, float]] = []
    for i in range(samples + 1):
        theta = math.pi * i / samples
        data.append(spherical_to_cartesian(theta, phi))
    return data


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def perspective(fovy: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy / 2.0)
    return np.asarray([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_near - z_far), (2.0 * z_far * z_near) / (z_near - z_far)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view



if __name__ == "__main__":
    # Adjust LabelConfig values here to fine-tune label placement and size.
    custom_label_cfg = LabelConfig(
        surface_normal_offset=-0.04,
        right_offset=0.02,
        up_offset=0.005,
        min_screen_px=16.0,
        max_screen_px=64.0,
        pixel_scale=0.0003,
    )
    viewer_config = ViewerConfig(
        label_config=custom_label_cfg,
        enable_backfill=False,
        backfill_max_lookups=1,
    )
    viewer = SphereViewer(viewer_config)
    viewer.run()
