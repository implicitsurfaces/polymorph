import moderngl

import numpy as np
import jax
import jax.numpy as jnp

from polymorph_s2df import *

from collections import namedtuple
import glfw
import imgui
from functools import reduce
from imgui.integrations.glfw import GlfwRenderer
import sys

import optimistix
from timeit import default_timer as timer

WIDTH = 800
HEIGHT = 600


# Maps from a glfw key constant (https://www.glfw.org/docs/3.3/group__keys.html)
# to a s2df constructor.
TOOL_HOTKEYS = {glfw.KEY_C: Circle, glfw.KEY_B: Box}


Gesture = namedtuple("Gesture", ["start_pos", "shapes"])
Shape = namedtuple("Shape", ["name", "sdf"])


def gen_pixel_grid():
    yy, xx = jnp.mgrid[0:HEIGHT, 0:WIDTH]
    return jnp.column_stack((xx.ravel(), yy.ravel()))


pixel_grid = gen_pixel_grid()


def render_scene(sdf):
    # start = time.time()
    ans = sdf.is_inside(pixel_grid)

    # Convert the float entries for each pixel into a (4,) of uint8.
    ans_3d = jnp.repeat(
        255 * ans.reshape((HEIGHT, WIDTH, 1)).astype(jnp.uint8), 4, axis=2
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
    width, height = 800, 600
    window_name = "Polymorph"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


def init_quad(ctx):
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # fmt: off
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
    imgui.set_next_window_position(WIDTH - w - 10, 10)  # Top right
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


def render_shapes_tree(shapes):
    imgui.set_next_window_size(200, HEIGHT)
    imgui.set_next_window_position(0, 0)
    imgui.begin(
        "Shapes", closable=False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
    )
    for s in shapes:
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
    render_shapes_tree(vm.shapes)
    render_devtools(vm)

    imgui.render()
    renderer.render(imgui.get_draw_data())


class ViewModel:
    def __init__(self, window):
        glfw.set_key_callback(window, self.handle_key)
        glfw.set_mouse_button_callback(window, self.handle_mouse_button)

        self.gesture = None
        self.shapes = []
        self.tool = None
        self.cursor_world = p(*glfw.get_cursor_pos(window))

        self.vsync_enabled = True

    def handle_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key in TOOL_HOTKEYS:
            self.tool = TOOL_HOTKEYS[key]

    def handle_mouse_button(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse or button != glfw.MOUSE_BUTTON_LEFT:
            return

        if action == glfw.PRESS:
            pos = glfw.get_cursor_pos(window)
            self.gesture = Gesture(p(*pos), [])
        elif action == glfw.RELEASE:
            self.shapes = self.shapes + self.gesture.shapes
            self.gesture = None

    def handle_frame(self, window):
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

    texture = gl_context.texture((WIDTH, HEIGHT), components=4)
    texture.use(0)
    render_quad = init_quad(gl_context)

    while not glfw.window_should_close(window):
        ###############
        ## Model update

        glfw.poll_events()  # Process event queue & run GLFW callbacks
        imgui_glfw_renderer.process_inputs()  # Update ImGui IO state

        view_model.handle_frame(window)

        initial_params = get_params(view_model.cursor_world, view_model.scene)
        params = solve(initial_params, view_model.scene)

        ###########
        ## Render

        gl_context.clear()

        # Render SDFs
        buf = render_scene(view_model.scene)
        texture.write(jax.device_get(buf))
        render_quad()

        # Render ImGui
        render_ui(imgui_glfw_renderer, view_model, params)

        glfw.swap_buffers(window)

    gl_context.gc()
    imgui_glfw_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
