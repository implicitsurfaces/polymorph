from __future__ import annotations  # For forward refs in type annotations

import logging
import os
import sys
from collections import Counter, namedtuple
from functools import lru_cache
from pathlib import Path

import glfw
import imgui
import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import polymorph_s2df as s2df
from imgui.integrations.glfw import GlfwRenderer
from jaxtyping import Array, Float, UInt8
from PIL import Image
from polymorph_num.ops import grid_gen
from polymorph_num.unit import CompiledUnit, Unit
from polymorph_s2df import geometric_properties

from .icons import ICON_MAX_VALUE, ICON_MIN_VALUE, ICON_PATH, Icon
from .scenes import scene_dict
from .sketch import Sketch
from .tools import BoxTool, CircleTool, PolygonTool
from .types import ScreenPos, WorldPos
from .util import log_perf, perf_logging

type Rgb3D = UInt8[Array, "h w 3"]  # noqa: F722
type Rgba3D = UInt8[Array, "h w 4"]  # noqa: F722
type FloatBitmap1D = Float[Array, "pix_count"]  # noqa: F821


INITIAL_WINDOW_SIZE = (800, 600)

FONTS_DIR = Path(__file__).parent.parent.parent.joinpath("fonts")
INCONSOLATA_PATH = str(FONTS_DIR.joinpath("Inconsolata.ttf"))


def fb_to_window_factor(window):
    win_w, win_h = glfw.get_window_size(window)
    fb_w, fb_h = glfw.get_framebuffer_size(window)

    return max(float(fb_w) / win_w, float(fb_h) / win_h)


def load_fonts(scaling_factor):
    io = imgui.get_io()
    io.fonts.clear()

    io.font_global_scale = 1.0 / scaling_factor

    io.fonts.add_font_from_file_ttf(INCONSOLATA_PATH, 12 * scaling_factor)

    lucid_range = imgui.GlyphRanges((ICON_MIN_VALUE, ICON_MAX_VALUE, 0))  # type: ignore
    lucid_config = imgui.FontConfig(merge_mode=True)

    io.fonts.add_font_from_file_ttf(
        ICON_PATH, 10 * scaling_factor, lucid_config, lucid_range
    )


# Maps from a glfw key constant (https://www.glfw.org/docs/3.3/group__keys.html)
# to a tool constructor.
TOOL_HOTKEYS = {
    glfw.KEY_C: CircleTool,
    glfw.KEY_B: BoxTool,
    glfw.KEY_P: PolygonTool,
    # glfw.KEY_R: rectangle
}

COLOR_FILL = (255, 255, 255)
COLOR_OUTLINE = (128, 171, 228)
COLOR_INSIDE = (203, 195, 227)
COLOR_OUTSIDE = (144, 238, 144)

Transform = namedtuple("Transform", ["translation", "scale"])

logging.basicConfig(level=logging.WARNING)

render_log = logging.getLogger("render")
compile_log = logging.getLogger("compile")
geom_log = logging.getLogger("geom")

# Turn on logging via the POLYMORPH_DEBUG environment variable.
# Examples:
#
#     POLYMORPH_DEBUG=all
#     POLYMORPH_DEBUG=render,compile
debug_key = os.environ.get("POLYMORPH_DEBUG", "")
if debug_key in ["1", "all"]:
    logging.basicConfig(level=logging.DEBUG, force=True)
elif debug_key != "":
    for k in debug_key.split(","):
        logging.getLogger(k).setLevel(logging.DEBUG)


def memoize(fn):
    return lru_cache(maxsize=1)(fn)


@memoize
def empty_bitmap(size: tuple[int, int]):
    return jnp.zeros(size[0] * size[1], dtype=jnp.float32)


@memoize
def empty_rgba3d(size: tuple[int, int]) -> Rgba3D:
    return jnp.zeros((size[1], size[0], 4), jnp.uint8)


def new_render_target(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size)


@log_perf(render_log)
def to_rgba3d(
    bitmap: FloatBitmap1D, size: tuple[int, int], color: tuple[int, int, int]
) -> Rgba3D:
    w, h = size
    color_and_alpha = jnp.array([*color, 255])
    bitmap_2d = bitmap.reshape(h, w)
    return (bitmap_2d[:, :, jnp.newaxis] * color_and_alpha).astype(jnp.uint8)


@log_perf(render_log)
def composite_layers(out: Image.Image, layers: list[Rgb3D]):
    images = (Image.fromarray(np.asarray(layer)) for layer in layers)
    for img in images:
        out.alpha_composite(img)


@log_perf(render_log)
def render_scene(
    fills: tuple[FloatBitmap1D],
    outlines: tuple[FloatBitmap1D] | None,
    contour: FloatBitmap1D | None,
    metric_aberration: FloatBitmap1D | None,
    viewport_size: tuple[int, int],
) -> Rgba3D:
    if len(fills) == 0:
        return empty_rgba3d(viewport_size)

    if outlines is None and contour is None and metric_aberration is None:
        # Naive (but cheap) rendering — just merge all the bitmaps.
        combined_bitmaps = jnp.max(jnp.array(fills), axis=0)
        return to_rgba3d(combined_bitmaps, viewport_size, COLOR_FILL)

    dest = new_render_target(viewport_size)

    assert outlines
    outlines_3d = [
        to_rgba3d(bitmap, viewport_size, COLOR_OUTLINE) for bitmap in outlines
    ]

    if metric_aberration is not None:
        aberration_3d = to_rgba3d(metric_aberration, viewport_size, COLOR_FILL)
        composite_layers(dest, [aberration_3d] + outlines_3d)
        return jnp.array(np.array(dest))

    if contour is not None:
        c = jnp.asarray(contour)
        inside = jnp.where(c > 0, c, 0)
        outside = -jnp.where(c <= 0, c, 0)

        composite_layers(
            dest,
            [
                to_rgba3d(outside, viewport_size, COLOR_INSIDE),
                to_rgba3d(inside, viewport_size, COLOR_OUTSIDE),
            ]
            + outlines_3d,
        )
        return jnp.array(np.array(dest))

    fills_3d = [to_rgba3d(bitmap, viewport_size, COLOR_FILL) for bitmap in fills]
    composite_layers(dest, fills_3d + outlines_3d)
    return jnp.array(np.array(dest))


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D tex;

void main()
{
    FragColor = texture(tex, TexCoord);
}
"""


def create_window():
    width, height = INITIAL_WINDOW_SIZE
    window_name = "Polymorph"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


def init_quad(ctx):
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # fmt: off
    # Four vertices (x, y, z, u, v) for a quad that will display a texture.
    # Note that the v coordinate is flipped to account for the difference b/w
    # world space (y+ up) and screen space (y+ down).
    vbo = ctx.buffer(np.array([
        -1.0, -1.0, 0.0, 0.0, 0.0,
         1.0, -1.0, 0.0, 1.0, 0.0,
         1.0,  1.0, 0.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 0.0, 1.0
    ], dtype=np.float32))
    # fmt: on
    vao = ctx.vertex_array(prog, [(vbo, "3f 2f", "aPos", "aTexCoord")])

    # Return a render() function.
    return lambda: vao.render(moderngl.TRIANGLE_FAN)


def render_devtools(vm):
    w = 100
    window_width, _ = vm.window_size
    imgui.set_next_window_position(window_width - w - 10, 10)  # Top right
    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    with imgui.begin(
        "FPS Meter",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
    ):
        imgui.text(
            f"{Icon.zap} FPS: {imgui.get_io().framerate:.2f}",
        )
        changed, vm.vsync_enabled = imgui.checkbox("VSync", vm.vsync_enabled)
        if changed:
            glfw.swap_interval(vm.vsync_enabled)
        _, vm.show_outlines = imgui.checkbox("Outlines", vm.show_outlines)
        imgui.separator()
        _, vm.shape_debug = imgui.checkbox(f"{Icon.waves} Inspect", vm.shape_debug)
        if vm.shape_debug and vm.sketch.shapes:
            render_shape_selection(vm, w - 10)
            _, vm.metric_debug = imgui.checkbox("Show gradient", vm.metric_debug)

    imgui.pop_style_var()


def render_shape_stats(vm, areas=(), **kargs):
    w = 100

    window_width, window_height = vm.window_size

    # TODO: how can I calculate actual line height from imgui?
    line_height = 20

    # Align group to bottom right
    imgui.set_next_window_position(
        window_width - w - 10, window_height - 50 - line_height * len(areas)
    )

    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    with imgui.begin(
        "Shape stats",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
    ):
        imgui.text("Areas: ")
        for area in areas:
            imgui.text(
                f"  {area:.2E}",
            )
    imgui.pop_style_var()


def render_shapes_tree(vm, pos: tuple[int, int]):
    _, window_height = vm.window_size
    width = 200

    spacing = imgui.get_style().item_spacing
    padding = imgui.get_style().window_padding
    content_width = width - 2 * padding.x

    imgui.set_next_window_size(width, window_height)
    imgui.set_next_window_position(*pos)
    with imgui.begin(
        "Shapes", closable=False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
    ):
        for s in vm.sketch:
            if imgui.tree_node(s.classname(), imgui.TREE_NODE_LEAF):
                imgui.tree_pop()

        # Calculate the height required for the stack of widgets at the bottom.
        num_rows = len(vm.vars) + len(vm.actions) + 1
        stack_height = num_rows * imgui.get_frame_height() + (num_rows - 1) * spacing.y
        x = imgui.get_cursor_pos_x()
        y = window_height - stack_height - padding.y
        imgui.set_cursor_pos((x, y))

        # Render the stack.
        render_vars_and_actions(vm, content_width)
        render_scene_selector(vm, content_width)


def render_scene_selector(vm: ViewModel, width: int):
    imgui.set_next_item_width(width)

    default_label = "Load scene"
    preview_value = vm.scene.name if vm.scene else default_label

    with imgui.begin_combo("", preview_value) as combo:
        if combo.opened:
            for name in vm.scene_names:
                is_selected = name == vm.scene.name if vm.scene else False

                did_select, _ = imgui.selectable(name, is_selected)
                if did_select:
                    vm.load_scene(name)

                # Set the initial focus when opening the combo
                if is_selected:
                    imgui.set_item_default_focus()


def render_shape_selection(vm: ViewModel, width: int):
    imgui.set_next_item_width(width)

    default_label = "No contour"
    preview_value = (
        default_label
        if vm.shape_debug_index is None
        else vm.sketch.shapes[vm.shape_debug_index].classname()
    )

    all_names = Counter(shape.classname() for shape in vm.sketch)
    name_index = {}

    with imgui.begin_combo("", preview_value) as combo:
        if combo.opened:
            for i, shape in enumerate(vm.sketch):
                is_selected = i == vm.shape_debug_index
                name = shape.classname()
                if all_names[name] > 1:
                    if name not in name_index:
                        name_index[name] = 0
                    else:
                        name_index[name] += 1
                        name = f"{name} {name_index[name]}"
                if is_selected:
                    name = f"{Icon.check} {name}"

                _, did_select = imgui.selectable(name, is_selected)
                if did_select:
                    vm.shape_debug_index = i

                if is_selected:
                    imgui.set_item_default_focus()


def render_vars_and_actions(vm: ViewModel, width: int):
    col_width = (width - imgui.get_style().item_spacing.x) / 2

    for name, value in vm.vars.items():
        imgui.set_next_item_width(col_width)
        changed, new_val = imgui.input_text(
            name,
            str(value),
            -1,
            flags=imgui.INPUT_TEXT_CHARS_DECIMAL
            | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            | imgui.INPUT_TEXT_AUTO_SELECT_ALL,
        )
        if changed:
            vm.vars[name] = float(new_val)

    for name in vm.actions:
        imgui.set_next_item_width(col_width)
        if imgui.button(name):
            getattr(vm.scene, name)(vm.sketch)


def render_overlay(vm, params, centroids=()):
    # Set the window position and size to cover the entire viewport
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_position(*viewport.pos)
    imgui.set_next_window_size(*viewport.size)

    with imgui.begin(
        "Overlay",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_BACKGROUND
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
        | imgui.WINDOW_NO_MOUSE_INPUTS,
    ):
        draw_list = imgui.get_window_draw_list()

        for centroid in centroids:
            x_screen, y_screen = vm.world_to_screen(centroid)
            draw_list.add_circle_filled(
                x_screen, y_screen, 4, imgui.get_color_u32_rgba(1, 0, 0, 1)
            )

        if params is not None:
            x_screen, y_screen = vm.world_to_screen(params)
            draw_list.add_circle_filled(
                x_screen, y_screen, 4, imgui.get_color_u32_rgba(1, 0, 0, 1)
            )

        if vm.tool:
            vm.tool.render_feedback(vm, draw_list)


def render_ui(renderer, vm, params, stats=None):
    if stats is None:
        stats = {}
    imgui.new_frame()

    _, window_height = vm.window_size

    render_overlay(
        vm,
        params,
        centroids=stats.get("centroids", []),
    )
    render_shapes_tree(vm, (0, 0))
    render_devtools(vm)
    render_shape_stats(vm, **stats)

    imgui.render()
    renderer.render(imgui.get_draw_data())


def handle_hotkeys(vm: ViewModel):
    for key, tool in TOOL_HOTKEYS.items():
        if imgui.is_key_pressed(key):
            vm.tool = tool(vm)


class ViewModel:
    def __init__(self, window):
        # Window/rendering stuff
        self.vsync_enabled = True
        self._update_transforms(window)
        self.show_outlines = False

        self.shape_debug = False
        self.shape_debug_index = 0
        self.metric_debug = False

        # User-level state
        self.sketch = Sketch()
        self.tool = None
        self.current_unit: None | CompiledUnit = None
        self.cursor_world = self.screen_to_world(glfw.get_cursor_pos(window))

        self.scene_names = list(scene_dict.keys())

        # Scene-specific stuff
        self.scene = None
        self.vars = {}
        self.actions = []

    def _update_transforms(self, window):
        # About coordinate frames:
        # - Screen space (GLFW): (0, 0) is top left, y+ down.
        # - World space (SDF): (0, 0) is center, y+ up.
        # - ImGui coordinates are the same as GLFW.
        # - On retina displays, glfw.get_framebuffer_size() returns 2x window
        #   size, but we don't care about that.
        self.window_size = glfw.get_window_size(window)
        hw, hh = self.window_size[0] / 2, self.window_size[1] / 2
        self.world_transform = Transform(
            translation=jnp.array([-hw, hh]),
            scale=jnp.array([1, -1]),
        )

    def handle_frame(self, window):
        self._update_transforms(window)
        self.cursor_world = self.screen_to_world(glfw.get_cursor_pos(window))

    def world_to_screen(self, pos: WorldPos) -> ScreenPos:
        x, y = (
            pos.as_array() - self.world_transform.translation
        ) / self.world_transform.scale
        return (x.item(), y.item())

    def screen_to_world(self, pos: ScreenPos) -> WorldPos:
        x, y = (
            jnp.array(pos) * self.world_transform.scale
            + self.world_transform.translation
        )
        return WorldPos(x.item(), y.item())

    def current_obs_dict(self) -> dict[str, float]:
        x, y = self.cursor_world.as_array()
        return {"mouse_x": x, "mouse_y": y, **self.vars}

    def observation_names(self):
        return frozenset(self.current_obs_dict().keys())

    def load_scene(self, name: str):
        scene = scene_dict[name]

        self.sketch = scene.load()
        self.vars = dict(scene.vars)  # Make a copy
        self.actions = list(scene.actions)  # Make a copy
        self.scene = scene

        self.shape_debug = False
        self.shape_debug_index = 0


def main():
    window = create_window()

    gl_context = moderngl.create_context(require=330)
    gl_context.gc_mode = (
        "auto"  # https://moderngl.readthedocs.io/en/latest/topics/gc.html
    )

    imgui.create_context()
    imgui_glfw_renderer = GlfwRenderer(window)

    load_fonts(fb_to_window_factor(window))
    imgui_glfw_renderer.refresh_font_texture()

    view_model = ViewModel(window)

    render_quad = init_quad(gl_context)

    @memoize
    def get_sdf_texture(size):
        """
        Allocate the texture into which we render the SDF.
        Memoized to account for changes to the framebuffer size.
        """
        return gl_context.texture(size, components=4)

    @memoize
    @log_perf(compile_log)
    def compile_unit(
        sdfs: tuple[s2df.Shape], size: tuple[int, int], obs_names: frozenset[str]
    ) -> CompiledUnit:
        unit = Unit(obs_names)
        unit.registerLoss(view_model.sketch.total_loss(size=size))

        pg = grid_gen(*size)
        for i, sdf in enumerate(sdfs):
            unit.register(f"is_inside{i}", sdf.is_inside(*pg))
            unit.register(f"is_on_boundary{i}", sdf.is_on_boundary(*pg))
            distance = sdf.distance(*pg)

            dist_x = sdf.distance(pg[0] + 1, pg[1])
            dist_y = sdf.distance(pg[0], pg[1] + 1)

            diff_x = dist_x - distance
            diff_y = dist_y - distance

            grad = (diff_x * diff_x + diff_y * diff_y).sqrt()

            unit.register(f"grad_distance{i}", grad)

            mod_distance = (distance % 10) / 10
            unit.register(
                f"contour{i}",
                distance.sign() * (1 - mod_distance * mod_distance * mod_distance),
            )
            unit.register(f"area{i}", geometric_properties.area_monte_carlo(sdf))
            centroid = geometric_properties.centroid_monte_carlo(sdf)
            unit.register(f"centroid{i}", centroid)
        return unit.compile()

    @log_perf(render_log)
    def render_sdfs(
        unit: CompiledUnit, window_size: tuple[int, int], count: int
    ) -> tuple[FloatBitmap1D]:
        return tuple(unit.evaluate(f"is_inside{i}") for i in range(count))

    @log_perf(render_log)
    def render_outlines(
        unit: CompiledUnit, window_size: tuple[int, int], count: int
    ) -> tuple[FloatBitmap1D]:
        return tuple(unit.evaluate(f"is_on_boundary{i}") for i in range(count))

    def render_contour(
        unit: CompiledUnit, window_size: tuple[int, int], index: int
    ) -> FloatBitmap1D:
        return unit.evaluate(f"contour{index}")

    def render_metric_aberration(
        unit: CompiledUnit, window_size: tuple[int, int], index: int
    ) -> FloatBitmap1D:
        return unit.evaluate(f"grad_distance{index}")

    while not glfw.window_should_close(window):
        ###############
        ## Model update

        glfw.poll_events()  # Process event queue & run GLFW callbacks
        imgui_glfw_renderer.process_inputs()  # Update ImGui IO state

        handle_hotkeys(view_model)

        view_model.handle_frame(window)
        if view_model.tool and not imgui.get_io().want_capture_mouse:
            view_model.tool.handle_frame()

        sdfs = view_model.sketch.cached_sdfs
        obs_dict = view_model.current_obs_dict()
        unit = compile_unit(
            sdfs, view_model.window_size, view_model.observation_names()
        )
        unit = unit.observe(obs_dict).minimize()
        view_model.current_unit = unit

        ###########
        ## Render

        gl_context.viewport = (0, 0, *glfw.get_framebuffer_size(window))
        gl_context.clear()

        # Render SDFs
        count = len(sdfs)
        bitmaps = render_sdfs(unit, view_model.window_size, count)

        contour = None
        if (
            sdfs
            and view_model.shape_debug
            and not view_model.metric_debug
            and view_model.shape_debug_index is not None
        ):
            contour = render_contour(
                unit, view_model.window_size, view_model.shape_debug_index
            )

        metric_aberration = None
        if (
            sdfs
            and view_model.shape_debug
            and view_model.metric_debug
            and view_model.shape_debug_index is not None
        ):
            metric_aberration = render_metric_aberration(
                unit, view_model.window_size, view_model.shape_debug_index
            )

        outlines = None
        if view_model.show_outlines or view_model.shape_debug:
            outlines = render_outlines(unit, view_model.window_size, count)

        buf = render_scene(
            bitmaps, outlines, contour, metric_aberration, view_model.window_size
        )

        texture = get_sdf_texture(view_model.window_size)
        texture.write(jax.device_get(buf))
        texture.use(0)
        render_quad()

        # Calculate stats
        with perf_logging(geom_log, "area"):
            areas = tuple(unit.evaluate(f"area{i}") for i in range(len(sdfs)))

        with perf_logging(geom_log, "centroids"):
            centroids = tuple(
                WorldPos(*unit.evaluate(f"centroid{i}")) for i in range(len(sdfs))
            )

        # Render ImGui

        render_ui(
            imgui_glfw_renderer,
            view_model,
            None,
            stats={
                "areas": areas,
                "centroids": centroids,
            },
        )

        glfw.swap_buffers(window)

    imgui_glfw_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
