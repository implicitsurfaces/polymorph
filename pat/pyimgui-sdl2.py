#!/usr/bin/env python

import ctypes
import sys

import imgui
import OpenGL.GL as gl
import sdl2
from imgui.integrations.sdl2 import SDL2Renderer
from testwindow import show_test_window


def main():
    window, gl_context = impl_pysdl2_init()
    imgui.create_context()
    impl = SDL2Renderer(window)

    show_custom_window = True

    running = True
    event = sdl2.SDL_Event()
    while running:
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            impl.process_event(event)
        impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        show_test_window()
        # imgui.show_test_window()

        if show_custom_window:
            is_expand, show_custom_window = imgui.begin("Custom window", True)
            if is_expand:
                imgui.text("Bars")
                imgui.text_colored("Eggs", 0.2, 1.0, 0.0)
            imgui.end()

        gl.glClearColor(1.0, 1.0, 1.0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        sdl2.SDL_GL_SwapWindow(window)

    impl.shutdown()
    sdl2.SDL_GL_DeleteContext(gl_context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()


def impl_pysdl2_init():
    width, height = 1280, 720
    window_name = "minimal ImGui/SDL2 example"

    if sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING) < 0:
        print(
            "Error: SDL could not initialize! SDL Error: "
            + sdl2.SDL_GetError().decode("utf-8")
        )
        sys.exit(1)

    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_STENCIL_SIZE, 8)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_ACCELERATED_VISUAL, 1)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLEBUFFERS, 1)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLESAMPLES, 8)
    sdl2.SDL_GL_SetAttribute(
        sdl2.SDL_GL_CONTEXT_FLAGS, sdl2.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG
    )
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)
    sdl2.SDL_GL_SetAttribute(
        sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_CORE
    )

    sdl2.SDL_SetHint(sdl2.SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b"1")
    sdl2.SDL_SetHint(sdl2.SDL_HINT_VIDEO_HIGHDPI_DISABLED, b"1")

    window = sdl2.SDL_CreateWindow(
        window_name.encode("utf-8"),
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        width,
        height,
        sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE,
    )

    if window is None:
        print(
            "Error: Window could not be created! SDL Error: "
            + sdl2.SDL_GetError().decode("utf-8")
        )
        sys.exit(1)

    gl_context = sdl2.SDL_GL_CreateContext(window)
    if gl_context is None:
        print(
            "Error: Cannot create OpenGL Context! SDL Error: "
            + sdl2.SDL_GetError().decode("utf-8")
        )
        sys.exit(1)

    sdl2.SDL_GL_MakeCurrent(window, gl_context)
    if sdl2.SDL_GL_SetSwapInterval(1) < 0:
        print(
            "Warning: Unable to set VSync! SDL Error: "
            + sdl2.SDL_GetError().decode("utf-8")
        )
        sys.exit(1)

    return window, gl_context


if __name__ == "__main__":
    main()
