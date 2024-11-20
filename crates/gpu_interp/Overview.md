# Overview

On a recent research project, we were building different kinds of physical models. They all involved:

- repeatedly evaluating the same arithmetic expression with different inputs (aka _bulk evaluation_)
- allowing the expressions to be adjusted interactively (think dragging shapes around on a canvas)
- visualizing the results at 60 fps.

Here's an example:

[DEMO]

The colors show the gravitational potential everywhere in the viewport, and the objects move based on the derivative of this potential.

The sun follows your mouse cursor (try it!

This is powered by a bytecode interpreter running on the GPU. If those words don't make sense together (or at all), read on for how we ended up here =D

## Why GPU?

Well, we wanted to use the whole computer we paid for.

In terms of raw numerical compute power, the GPU in modern consumer laptops is often 2-2.5x more powerful than the CPU. Take the M1 MacBook Air from 2020: it comes with an 8-core CPU and 7-core GPU. From Philip Turner's excellent [Metal Benchmarks](https://github.com/philipturner/metal-benchmarks) repo, we learned that:

- Apple's GPU cores are identical to the CPU cores in both transistor count.
- A single GPU core has ~⅓ the frequency and ~8x the parallelism.

Multiplying those numbers together, a GPU core is ~2.67x more powerful than a CPU core, and the MacBook Air has more than 2x the compute power on the GPU.

We know how to evaluate a mathematical function using the CPU --- just write it in Python/Rust/C and run the resulting program --- but how can we use the GPU?
There are two main ways we're aware of (i.e., learned about in this course of this exploration =P):

1. GPU-platform-specific compute APIs like XXXX.
2. Shaders, which are programs written in a special programming language that run on the GPU.

We investigated GPU-platform-specific APIs indirectly through the [JAX](https://jax.readthedocs.io/) machine learning library, but the compile times were too long for our dynamic/interactive use case (multiple seconds).
Plus we wanted something that'd work in a browser, so we turned to...


## Shaders

Long ago, the term _shader_ meant something like: a function that runs over every pixel of an input image, producing an output image. The main use case was adding realistic lighting to 3D scenes, a process known as _shading_.

Nowadays, a modern rendering pipeline has a number of different stages, and most of them have nothing to do with lighting or shading. But if you squint, they follow a similar pattern: parallel processing of a large number of input elements on the GPU, producing the outputs for the next stage. So the name _shader_ has stuck.

A typical rendering pipeline consists of at least two shader stages:

- a _vertex shader_ runs for every input vertex, doing things like transforming the 3D position to 2D screen coordinates.
- a _pixel shader_ (aka _fragment shader_) runs for every pixel on the triangular faces and return the pixel's color.

Over the years, people also noticed that GPUs are pretty great for general number crunching (i.e., nothing to do  with graphics). That led to _compute shaders_, which give a more general way to run parallel programs on the GPU, even if you're not rendering anything.

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
