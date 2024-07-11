import logging
import multiprocessing
import os
import sys
from collections import namedtuple
from functools import lru_cache
from typing import FrozenSet

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
from polymorph_num.expr import Observation, as_expr
from polymorph_num.unit import CompiledUnit, Unit

from .graph import Graph
from .solve import async_solver
from .tools import BoxTool, CircleTool, PolygonTool
from .types import ScreenPos, WorldPos
from .util import log_perf

type Rgb3D = UInt8[Array, "h w 3"]  # noqa: F722
type Rgba3D = UInt8[Array, "h w 4"]  # noqa: F722
type FloatBitmap1D = Float[Array, "pix_count"]  # noqa: F821


INITIAL_WINDOW_SIZE = (800, 600)


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

Transform = namedtuple("Transform", ["translation", "scale"])


render_log = logging.getLogger("render")
log = logging.getLogger(__name__)  # A generic logger

# Turn on debug logging if POLYMORPH_DEBUG is set to a non-empty value.
if os.environ.get("POLYMORPH_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)


def memoize(fn):
    return lru_cache(maxsize=1)(fn)


@memoize
def pixel_grid(size):
    """
    Allocates a uniform grid of points for sampling the SDFs.
    Memoized to handle changes to the viewport size.
    """
    half_width = size[0] / 2
    half_height = size[1] / 2

    yy, xx = jnp.mgrid[-half_height:half_height, -half_width:half_width]
    return as_expr(xx.ravel()), as_expr(yy.ravel())


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
    viewport_size: tuple[int, int],
) -> Rgba3D:
    if len(fills) == 0:
        return empty_rgba3d(viewport_size)

    if outlines is None:
        # Naive (but cheap) rendering — just merge all the bitmaps.
        combined_bitmaps = jnp.max(jnp.array(fills), axis=0)
        return to_rgba3d(combined_bitmaps, viewport_size, COLOR_FILL)

    dest = new_render_target(viewport_size)
    fills_3d = [to_rgba3d(bitmap, viewport_size, COLOR_FILL) for bitmap in fills]
    outlines_3d = [
        to_rgba3d(bitmap, viewport_size, COLOR_OUTLINE) for bitmap in outlines
    ]
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
    imgui.begin(
        "FPS Meter",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
    )

    imgui.text(
        f"FPS: {imgui.get_io().framerate:.2f}",
    )
    changed, vm.vsync_enabled = imgui.checkbox("VSync", vm.vsync_enabled)
    if changed:
        glfw.swap_interval(vm.vsync_enabled)
    _, vm.show_outlines = imgui.checkbox("Outlines", vm.show_outlines)

    imgui.end()
    imgui.pop_style_var()


def render_shape_stats(vm, areas=None):
    w = 100

    window_width, window_height = vm.window_size

    # TODO: how can I calculate actual line height from imgui?
    line_height = 20

    # Align group to bottom right
    imgui.set_next_window_position(
        window_width - w - 10, window_height - 50 - line_height * len(areas)
    )

    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    imgui.begin(
        "Shape stats",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
    )

    imgui.text("Areas: ")
    for area in areas:
        imgui.text(
            f"  {area:.2E}",
        )
    imgui.end()
    imgui.pop_style_var()


def render_shapes_tree(vm):
    _, window_height = vm.window_size
    imgui.set_next_window_size(200, window_height)
    imgui.set_next_window_position(0, 0)
    imgui.begin(
        "Shapes", closable=False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
    )
    for s in vm.graph.nodes:
        if imgui.tree_node(s.classname(), imgui.TREE_NODE_LEAF):
            imgui.tree_pop()
    imgui.end()


def render_overlay(vm, params):
    # Set the window position and size to cover the entire viewport
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_position(*viewport.pos)
    imgui.set_next_window_size(*viewport.size)

    imgui.begin(
        "Overlay",
        None,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_BACKGROUND
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
        | imgui.WINDOW_NO_MOUSE_INPUTS,
    )

    draw_list = imgui.get_window_draw_list()
    if params is not None:
        x_screen, y_screen = vm.world_to_screen((params[0], params[1]))
        draw_list.add_circle_filled(
            x_screen, y_screen, 4, imgui.get_color_u32_rgba(1, 0, 0, 1)
        )

    if vm.tool:
        vm.tool.render_feedback(vm, draw_list)

    imgui.end()


def render_ui(renderer, vm, params, stats=None):
    imgui.new_frame()

    render_overlay(vm, params)
    render_shapes_tree(vm)
    render_devtools(vm)
    render_shape_stats(vm, **stats)

    imgui.render()
    renderer.render(imgui.get_draw_data())


class ViewModel:
    def __init__(self, window):
        glfw.set_key_callback(window, self.on_key)
        glfw.set_mouse_button_callback(window, self.on_mouse_button)

        # Window/rendering stuff
        self.vsync_enabled = True
        self._update_transforms(window)
        self.show_outlines = False

        class Observations:
            mouse_x = Observation("mouse_x")
            mouse_y = Observation("mouse_y")

        # User-level state
        self.graph = Graph()
        self.tool = None
        self.cursor_world = self.screen_to_world(glfw.get_cursor_pos(window))
        self.observations = Observations()

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

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key in TOOL_HOTKEYS:
            self.tool = TOOL_HOTKEYS[key](self)
        if self.tool:
            self.tool.handle_key(key, action, mods)

    def on_mouse_button(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse or button != glfw.MOUSE_BUTTON_LEFT:
            return

        if self.tool:
            pos = self.screen_to_world(glfw.get_cursor_pos(window))
            self.tool.handle_mouse_button(pos, action)

    def on_frame(self, window):
        self._update_transforms(window)
        self.cursor_world = self.screen_to_world(glfw.get_cursor_pos(window))
        if self.tool:
            self.tool.handle_frame()

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
        return {"mouse_x": x, "mouse_y": y}

    def observation_names(self):
        return frozenset(self.current_obs_dict().keys())


def main(solver):
    window = create_window()

    gl_context = moderngl.create_context(require=330)
    gl_context.gc_mode = (
        "auto"  # https://moderngl.readthedocs.io/en/latest/topics/gc.html
    )

    imgui.create_context()
    imgui_glfw_renderer = GlfwRenderer(window)

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
    @log_perf(log)
    def compile_unit(
        sdfs: tuple[s2df.Shape], size: tuple[int, int], obs_names: FrozenSet[str]
    ) -> CompiledUnit:
        unit = Unit(obs_names)
        unit.registerLoss(view_model.graph.total_loss())
        pg = pixel_grid(size)
        for i, sdf in enumerate(sdfs):
            unit.register(f"is_inside{i}", sdf.is_inside(*pg))
            unit.register(f"is_on_boundary{i}", sdf.is_on_boundary(*pg))
            unit.register(f"area{i}", sdf.area(*pg, size[0] * size[1]))
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

    while not glfw.window_should_close(window):
        ###############
        ## Model update

        glfw.poll_events()  # Process event queue & run GLFW callbacks
        imgui_glfw_renderer.process_inputs()  # Update ImGui IO state

        view_model.on_frame(window)

        sdfs = view_model.graph.cached_sdfs
        obs_dict = view_model.current_obs_dict()
        unit = compile_unit(
            sdfs, view_model.window_size, view_model.observation_names()
        )
        unit = unit.observe(obs_dict).minimize()

        ###########
        ## Render

        gl_context.clear()

        # Render SDFs
        count = len(sdfs)
        bitmaps = render_sdfs(unit, view_model.window_size, count)

        outlines = None
        if view_model.show_outlines:
            outlines = render_outlines(unit, view_model.window_size, count)
        buf = render_scene(bitmaps, outlines, view_model.window_size)

        texture = get_sdf_texture(view_model.window_size)
        texture.write(jax.device_get(buf))
        texture.use(0)
        render_quad()

        # Calculate stats
        areas = tuple(unit.evaluate(f"area{i}") for i in range(len(sdfs)))

        # Render ImGui

        render_ui(imgui_glfw_renderer, view_model, None, stats={"areas": areas})

        glfw.swap_buffers(window)

    imgui_glfw_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    with multiprocessing.Pool(1) as pool:
        main(async_solver(pool))
