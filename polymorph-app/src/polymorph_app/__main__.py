from collections import namedtuple
from functools import reduce
import jax
import jax.numpy as jnp
import pygame
import time

from pygame_gui import UIManager, UI_BUTTON_PRESSED
from pygame_gui.elements import *
from pygame import surfarray
from polymorph_s2df import *

# Constants
FPS = 60
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)  # White background

# Maps from a pygame key constant (https://www.pygame.org/docs/ref/key.html#key-constants-label)
# to a s2df constructor.
TOOL_HOTKEYS = {pygame.K_c: Circle, pygame.K_b: Box, pygame.K_t: Triangle}


Gesture = namedtuple("Gesture", ["start_pos", "shapes"])


def gen_pixel_grid():
    xx, yy = jnp.mgrid[0:WIDTH, 0:HEIGHT]
    return jnp.column_stack((xx.ravel(), yy.ravel()))


pixel_grid = gen_pixel_grid()


def render_scene(sdf):
    # start = time.time()
    ans = sdf.is_inside(pixel_grid)

    # Convert the float entries for each pixel into a (3,) of uint8.
    ans_3d = jnp.repeat(
        255 * ans.reshape((WIDTH, HEIGHT, 1)).astype(jnp.uint8), 3, axis=2
    )
    # print(f"rendered in {time.time() - start}s")
    return ans_3d


if __name__ == "__main__":
    window_dims = (WIDTH, HEIGHT)
    pygame.init()

    pygame.display.set_caption("Quick Start")
    display_surface = pygame.display.set_mode(window_dims)
    ui_manager = UIManager(window_dims)

    button_layout_rect = pygame.Rect(0, 0, 100, 20)
    button_layout_rect.bottomright = (-30, -20)

    tool_label = UILabel(
        relative_rect=button_layout_rect,
        text="",
        manager=ui_manager,
        container=ui_manager.root_container,
        anchors={"right": "right", "bottom": "bottom"},
    )

    clock = pygame.time.Clock()
    is_running = True

    gesture = None
    shapes = []
    tool = None

    while is_running:
        time_delta = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gesture = Gesture(p(*event.pos), [])
            elif event.type == pygame.MOUSEBUTTONUP:
                shapes = shapes + gesture.shapes
                gesture = None
            elif event.type == pygame.KEYDOWN:
                if event.key in TOOL_HOTKEYS:
                    tool = TOOL_HOTKEYS[event.key]
            ui_manager.process_events(event)
        ui_manager.update(time_delta)
        tool_label.set_text(tool.__name__ if tool else "None")

        keys = pygame.key.get_pressed()

        if gesture and tool:
            current_pos = p(*pygame.mouse.get_pos())
            if tool == Circle:
                r = jnp.linalg.norm(current_pos - gesture.start_pos, axis=-1)
                gesture = Gesture(
                    gesture.start_pos, [tool(r).translate(gesture.start_pos)]
                )
            elif tool == Box:
                w, h = (jnp.abs(current_pos - gesture.start_pos)) * 2
                gesture = Gesture(
                    gesture.start_pos, [tool(w, h).translate(gesture.start_pos)]
                )

        if keys[pygame.K_c]:
            pass

        gesture_shapes = gesture.shapes if gesture else []
        scene = reduce(lambda a, b: Union(a, b), shapes + gesture_shapes, Circle(0))

        surfarray.blit_array(display_surface, render_scene(scene))
        ui_manager.draw_ui(display_surface)

        pygame.display.update()
