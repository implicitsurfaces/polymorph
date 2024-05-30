import jax
import jax.numpy as jnp
import pygame
import time

from pygame_gui import UIManager, UI_BUTTON_PRESSED
from pygame_gui.elements import UIButton
from pygame import surfarray
from polymorph_s2df import *

# Constants
FPS = 60
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)  # White background


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

    hello_button = UIButton((350, 280), "Hello")

    clock = pygame.time.Clock()
    is_running = True

    while is_running:
        time_delta = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == UI_BUTTON_PRESSED:
                if event.ui_element == hello_button:
                    print("Hello World!")
            ui_manager.process_events(event)
        ui_manager.update(time_delta)

        mouse_pos = pygame.mouse.get_pos()

        scene = (
            Circle(200)
            .substract(Circle(50))
            .substract(Box(230, 20).translate(p(100, 0)))
            .translate(p(*mouse_pos))
        )

        surfarray.blit_array(display_surface, render_scene(scene))
        ui_manager.draw_ui(display_surface)

        pygame.display.update()
