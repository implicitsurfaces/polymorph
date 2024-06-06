import moderngl

import numpy as np
import jax
import jax.numpy as jnp

from polymorph_s2df import *

from collections import namedtuple
import glfw
import imgui
from functools import lru_cache, reduce
from imgui.integrations.glfw import GlfwRenderer
import sys

import optimistix
from timeit import default_timer as timer

INITIAL_WINDOW_SIZE = (800, 600)


# Maps from a glfw key constant (https://www.glfw.org/docs/3.3/group__keys.html)
# to a s2df constructor.
TOOL_HOTKEYS = {glfw.KEY_C: Circle, glfw.KEY_B: Box}


Gesture = namedtuple("Gesture", ["start_pos", "shapes"])
Shape = namedtuple("Shape", ["name", "sdf"])


def memoize(fn):
    return lru_cache(maxsize=1)(fn)


@memoize
def pixel_grid(size):
    """
    Allocates a uniform grid of points for sampling the SDFs.
    Memoized to handle changes to the viewport size.
    """
    yy, xx = jnp.mgrid[0 : size[1], 0 : size[0]]
    return jnp.column_stack((xx.ravel(), yy.ravel()))


def render_scene(sdf, size):
    # start = time.time()
    ans = sdf.is_inside(pixel_grid(size))

    # Convert the float entries for each pixel into a (4,) of uint8.
    ans_3d = jnp.repeat(
        255 * ans.reshape((size[1], size[0], 1)).astype(jnp.uint8), 4, axis=2
    )
    # print(f"rendered in {time.time() - start}s")
    return ans_3d


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
    vbo = ctx.buffer(np.array([
        -1.0, -1.0, 0.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 1.0, 1.0,
         1.0,  1.0, 0.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 0.0, 0.0
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

    imgui.end()
    imgui.pop_style_var()


def render_shapes_tree(vm):
    _, window_height = vm.window_size
    imgui.set_next_window_size(200, window_height)
    imgui.set_next_window_position(0, 0)
    imgui.begin(
        "Shapes", closable=False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
    )
    for s in vm.shapes:
        if imgui.tree_node(s.name, imgui.TREE_NODE_LEAF):
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
    draw_list.add_circle_filled(
        params[0], params[1], 4, imgui.get_color_u32_rgba(1, 0, 0, 1)
    )

    imgui.end()


def render_ui(renderer, vm, params):
    imgui.new_frame()

    render_overlay(vm, params)
    render_shapes_tree(vm)
    render_devtools(vm)

    imgui.render()
    renderer.render(imgui.get_draw_data())


class ViewModel:
    def __init__(self, window):
        glfw.set_key_callback(window, self.on_key)
        glfw.set_mouse_button_callback(window, self.on_mouse_button)

        # User-level state
        self.gesture = None
        self.shapes = []
        self.tool = None
        self.cursor_world = p(*glfw.get_cursor_pos(window))

        # Window stuff
        self.window_size = glfw.get_window_size(window)
        self.vsync_enabled = True

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key in TOOL_HOTKEYS:
            self.tool = TOOL_HOTKEYS[key]

    def on_mouse_button(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse or button != glfw.MOUSE_BUTTON_LEFT:
            return

        if action == glfw.PRESS:
            pos = glfw.get_cursor_pos(window)
            self.gesture = Gesture(p(*pos), [])
        elif action == glfw.RELEASE:
            self.shapes = self.shapes + self.gesture.shapes
            self.gesture = None

    def on_frame(self, window):
        self.window_size = glfw.get_window_size(window)
        self.cursor_world = p(*glfw.get_cursor_pos(window))
        gesture, tool = (self.gesture, self.tool)
        if gesture and tool:
            current_pos = self.cursor_world
            start_pos = gesture.start_pos
            if tool == Circle:
                r = jnp.linalg.norm(current_pos - start_pos, axis=-1)
                self.gesture = Gesture(
                    start_pos, [Shape("Circle", Circle(r).translate(start_pos))]
                )
            elif self.tool == Box:
                w, h = (jnp.abs(current_pos - start_pos)) * 2
                self.gesture = Gesture(
                    start_pos, [Shape("Box", Box(w, h).translate(start_pos))]
                )

    @property
    def scene(self):
        """Returns an SDF representing the current scene to be rendered."""
        all_shapes = self.shapes + (self.gesture.shapes if self.gesture else [])
        return reduce(
            lambda a, b: Union(a, b), map(lambda s: s.sdf, all_shapes), Circle(0)
        )


def optimize_params(cost, params, scene):
    solver = optimistix.BFGS(rtol=1e-5, atol=1e-6)
    start = timer()
    solution = optimistix.minimise(cost, solver, params, scene, throw=False)
    elapsed = timer() - start
    print(
        "{0} steps in {1:.3f} seconds".format(solution.stats.get("num_steps"), elapsed)
    )
    return solution.value


def get_params(cursor, scene):
    return cursor


def solve(params, scene):
    def cost(params, scene):
        shape = scene
        target_distance = 0.5
        cost_distance = (
            shape.distance(params[jnp.newaxis, 0:2]) - target_distance
        ) ** 2
        return cost_distance[0]

    return optimize_params(cost, params, scene)


def main():
    window = create_window()

    gl_context = moderngl.create_context(require=330)
    gl_context.gc_mode = (
        "context_gc"  # https://moderngl.readthedocs.io/en/latest/topics/gc.html
    )

    imgui.create_context()
    imgui_glfw_renderer = GlfwRenderer(window)

    view_model = ViewModel(window)

    render_quad = init_quad(gl_context)

    @memoize
    def get_sdf_texture(framebuffer_size):
        """
        Allocate the texture into which we render the SDF.
        Memoized to account for changes to the framebuffer size.
        """
        gl_context.gc()  # Ensure the previous texture is released.
        return gl_context.texture(framebuffer_size, components=4)

    while not glfw.window_should_close(window):
        ###############
        ## Model update

        glfw.poll_events()  # Process event queue & run GLFW callbacks
        imgui_glfw_renderer.process_inputs()  # Update ImGui IO state

        view_model.on_frame(window)

        initial_params = get_params(view_model.cursor_world, view_model.scene)
        params = solve(initial_params, view_model.scene)

        ###########
        ## Render

        gl_context.clear()

        # Render SDFs
        fb_size = glfw.get_framebuffer_size(window)
        texture = get_sdf_texture(fb_size)
        buf = render_scene(view_model.scene, fb_size)
        texture.write(jax.device_get(buf))
        texture.use(0)
        render_quad()

        # Render ImGui
        render_ui(imgui_glfw_renderer, view_model, params)

        glfw.swap_buffers(window)

    gl_context.gc()
    imgui_glfw_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
