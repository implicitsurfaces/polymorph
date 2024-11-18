# Overview

Typical GPU workloads involve a fixed computation, with dynamic data provided through buffers. A _really_ dynamic approach is to put bytecode into a buffer, and write a shader program that interprets the bytecode.

We prototyped a GPU-based bytecode interpreter for a research project we're working on. This page describes our approach and some of the things we learned.

## Why would you do this?

Our project involves an interactive drawing app using [signed distance functions](https://en.wikipedia.org/wiki/Signed_distance_function). It's built on [a Rust library called Fidget](https://github.com/mkeeter/fidget), which provides an API for building scalar compute graphs and efficiently evaluating them on the CPU.

Fidget's author, Matthew Keeter, published [a technique for efficiently rendering SDFs on the GPU][mpr], but hasn't (hadn't) yet implemented a GPU-based interpreter in Fidget. Also, his implementation used CUDA, which is only available for NVIDIA hardware. We're interested in a cross-vendor, cross-platform approach, so we created our prototype using [wgpu][] a Rust library based on the WebGPU standard.

[wgpu]: [WebGPU standard](https://github.com/gfx-rs/wgpu)

[mpr]: https://www.mattkeeter.com/research/mpr/

<!--
## Graveyard

If the structure of the computation changes, you can  either load a new shader program — maybe dropping frames as it's compiled…
-->
