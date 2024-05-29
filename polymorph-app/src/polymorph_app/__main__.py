import pygame

from pygame_gui import UIManager, UI_BUTTON_PRESSED
from pygame_gui.elements import UIButton

# Constants
FPS = 60
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)  # White background


def render_scene(surface):
    surface.fill(BACKGROUND_COLOR)
    color = (255, 0, 0)
    pos = (WIDTH / 2, HEIGHT / 2)
    radius = 100
    pygame.draw.circle(surface, color, pos, radius)


if __name__ == "__main__":
    pygame.init()

    pygame.display.set_caption("Quick Start")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ui_manager = UIManager((800, 600))

    background = pygame.Surface((800, 600))
    background.fill(ui_manager.ui_theme.get_colour("dark_bg"))

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

        render_scene(screen)
        ui_manager.draw_ui(screen)

        pygame.display.update()
