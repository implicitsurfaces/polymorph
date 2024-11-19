# Overview

Typical GPU workloads involve a fixed computation, with dynamic data provided through buffers. A _really_ dynamic approach is to put bytecode into a buffer, and write a shader program that interprets the bytecode.

We prototyped a GPU-based bytecode interpreter for a research project we're working on. This page describes our approach and some of the things we learned.

## Why would you do this?

Our project involves an interactive drawing app using [signed distance functions](https://en.wikipedia.org/wiki/Signed_distance_function). It's built on [a Rust library called Fidget](https://github.com/mkeeter/fidget), which provides an API for building scalar compute graphs and efficiently evaluating them on the CPU.

Fidget's author, Matthew Keeter, published [a technique for efficiently rendering SDFs on the GPU][mpr], but hasn't (hadn't) yet implemented a GPU-based interpreter in Fidget. Also, his implementation used CUDA, which is only available for NVIDIA hardware. We're interested in a cross-vendor, cross-platform approach, so we created our prototype using [wgpu][] a Rust library based on the WebGPU standard.

[wgpu]: https://github.com/gfx-rs/wgpu

[mpr]: https://www.mattkeeter.com/research/mpr/

<!--
## Graveyard

If the structure of the computation changes, you can  either load a new shader program — maybe dropping frames as it's compiled…
-->











<!-- Hi Pat, here's my sketch of a post outline. Happy to discuss, and feel free to take whatever parts you want and integrate directly into a canonical version. -->

# Bulk evaluation of arbitrary functions and their derivatives on the GPU


We recently needed to bulk evaluate user-input mathematical functions and their derivatives at runtime to make interactive applications like this tiny solar system.

[DEMO]

The colors show the gravitational potential everywhere in the viewport, and the objects move based on the derivative of this potential.
The sun follows your mouse cursor (try it!)


This is powered by a bytecode interpreter running on the GPU. If those words don't make sense together (or at all), read on for how we ended up here =D


## Why GPU?

We wanted to use the whole computer we paid for, which means doing some computation on the Graphics Processing Unit (GPU), which provides most of the number-crunching horsepower for modern computers.
(E.g., on my M1 MacBook Air there are XXX CPUs with XXX cores, which can do XXX floating point operations per second (FLOPS); its GPU can do XXX FLOPS, about a factor XXX more compute.)

We know how to evaluate a mathematical function using the CPU --- just write it in Python/Rust/C and run the resulting program --- but how can we use the GPU?
There are two main ways we're aware of (i.e., learned about in this course of this exploration =P):

1. GPU-platform-specific compute APIs like XXXX.
2. Shaders, which are programs written in a special programming language that run on the GPU.

We investigated GPU-platform-specific APIs indirectly through the [JAX](https://jax.readthedocs.io/) machine learning library, but the compile times were too long for our dynamic/interactive use case (multiple seconds).
Plus we wanted something that'd work in a browser, so we turned to...


## Shaders

Roughly, there are three types of GPU shader: vertex, fragment, and compute.

They run within the context of *render pipeline*, which describes where each *shader pass* gets its input data and stores is result.

For example, in the traditional graphics pipeline, the programmer provides (from the CPU) a list of vertices (points in 3d space), every three of which define the face of a triangle.
Then on the GPU:

- the vertex shader runs for every input vertex to compute various attributes (position/transformation, color, etc.), and
- the fragment shader runs for every pixel on the triangular faces and return the pixel's color.

As people have realized GPUs are useful for general number crunching, *compute shaders* have become available, which run over abstract indices and compute whatever you want.

Conceptually, each *shader pass* [?] runs in parallel and each [instance? invocation? what's the word here?] works on just its vertex/pixel/index input and cannot share data with its siblings.


## What shader language?

To use a given CPU or GPU, we have to translate our human-readable program into the specific interface provided by the hardware: it's instruction set.
Just like eventually our Rust or JavaScript code has to become x86 or ARM assembly, our shader code eventually has to become, uh, something something Apple / Nvidia.

We have a lot of options for a "GPU Stack" [XXX better word?] that allow us to write shaders and define render pipelines:

- Metal, Apple's framework
- OpenGL, a classic from the 90's (with lots of problems?)
- Rust, compiling it to run on the GPU with XXXX
- SPIR-V
- Vulkan [XXX maybe?]
- WebGPU

We choose WebGPU because it was designed to target modern GPU features (it first became available in [XXX Year]) and works across Mac, Linux, and in the browser.


## Dynamism

GPU Stacks can compile shaders at runtime (from strings of program text) and sending them to the GPU as part of a new render pipeline.
This can even be done in the browser by WebGPU or by [XXX ????], which is how ShaderToy works.

However, we wanted to explore a different approach: Implement a language *interpreter* which can run on the GPU.

[XXX discuss background on interpreters; compare to Python/JS REPL.]


## Language

Since we wanted to evaluate mathematical expressions and their derivatives, the "language" our shader needs to interpret is [Fidget](XXX), designed by computer graphics enthusiast [Matt Keeter](XXX).

Fidget was appealed to us in particular for two reasons:

- expressions can be automatically differentiated (not the [numerical method](https://en.wikipedia.org/wiki/Finite_difference_method) of approximating the slope from two "nearby" points, but an exact method based on analyzing the expression itself; see [this overview](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation) for more).
- expressions can be simplified for a given evaluation domain via an interval arithmetic analysis (see Keeter's preso/paper for more [XXX])


## Recap

With all that context, let's recap how the demo works on the top of the page.
Here's what happens when you type in a new mathematical expression (which we've defaulted to be the gravitational potential):

- The string you typed is parsed into a Fidget expression.
- The derivatives of this expression with respect to the x and y variables are calculated.
- These three expressions are then compiled to bytecode and sent to the GPU.
- Our compute shader evaluates the potential everywhere in the viewport.
- Our fragment shader maps these results into colors
- Our other compute shader evaluates the derivatives (i.e., the forces on the planets)
- Our CPU takes these results and draws the planets using <div>s [XXX or we do this on the GPU too? TBD]


## Next steps / questions

This was a research exploration.
While we learned a lot, there's a lot more engineering work to be done is even a Good Idea (tm).

In particular we haven't rigorously explored the performance characteristics:
- Comparing the render time: Once everything is loaded on the GPU, how long does it take to render a frame?
- Comparing the "expression update latency": Given a new expression at runtime, how long does it take the traditional approach (compile a shader, load new render pipeline, render a frame) compared to our approach (compile expression to bytecode, load onto existing render pipeline, render)?
- How does the memory/size compare?
- How does the faster shader approach compare to the GPU-platform-specific compute API as exposed by, e.g., JAX

