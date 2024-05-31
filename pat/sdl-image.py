import sys
import sdl2
import sdl2.ext
from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
import jax.numpy as jnp

import imgui
from imgui.integrations.sdl2 import SDL2Renderer

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


def create_gl_context(sdl_window):
    assert sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 3) == 0
    assert sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 3) == 0
    assert (
        sdl2.SDL_GL_SetAttribute(
            sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_CORE
        )
        == 0
    )

    ctx = sdl2.SDL_GL_CreateContext(sdl_window)
    assert ctx, "Failed to create OpenGL context"
    return ctx


def create_quad():
    # fmt: off
    vertices = np.array([
        -1.0, -1.0, 0.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 1.0, 1.0,
         1.0,  1.0, 0.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    # fmt: on

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0)
    )
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        5 * vertices.itemsize,
        ctypes.c_void_p(3 * vertices.itemsize),
    )
    glEnableVertexAttribArray(1)
    glBindVertexArray(vao)

    # Return draw() function.
    return lambda: glDrawArrays(GL_TRIANGLE_FAN, 0, 4)


def ndarray_to_texture(arr):
    # Generate texture object
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload numpy array to texture
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        arr.shape[1],
        arr.shape[0],
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        arr,
    )

    return tex


def main():
    sdl2.ext.init()
    window = sdl2.ext.Window("OpenGL Texture", size=(800, 600))
    window.show()

    gl_context = create_gl_context(window.window)
    draw_quad = create_quad()
    shader_program = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )

    array = np.random.randint(0, 256, size=(600, 800, 4), dtype=jnp.uint8)
    texture = ndarray_to_texture(array)

    # Initialize ImGui
    imgui.create_context()
    impl = SDL2Renderer(window.window)

    running = True
    while running:
        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            impl.process_event(event)
        impl.process_inputs()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        glUseProgram(shader_program)
        glBindTexture(GL_TEXTURE_2D, texture)
        draw_quad()

        imgui.new_frame()
        imgui.begin("Hello, ImGui!")
        imgui.text("Hello, World!")
        imgui.button("Click Me!")
        imgui.end()

        imgui.render()

        impl.render(imgui.get_draw_data())

        sdl2.SDL_GL_SwapWindow(window.window)

    sdl2.SDL_GL_DeleteContext(gl_context)
    window.close()
    sdl2.ext.quit()


if __name__ == "__main__":
    main()
