# GPU-based rendering for Fidget

A 2D renderer for Fidget expressions that uses the GPU.

## What's in here

- All of the Rust rendering stuff in src/lib.rs
- The interpreter itself is defined in shader-in.wgsl.
- Unit tests in tests/tests.rs
- A native application demo (src/bin/win.rs)
- A web app demo (src/web)

## Building and running

- Build the non-web stuff: `cargo build`
- Run the unit tests: `cargo test`
- Run the native demo: `cargo run --bin win`

### Web

The web demo is built with [trunk](https://trunkrs.dev/). Because there are a number of ways to install trunk, we use `direnv` to make sure we all use a consistent version.

#### Installing direnv and trunk (recommended)

Make sure you're in this directory:

    cd gpu_interp

Install [direnv](https://direnv.net/) so your shell path always refers to correct version of trunk.

Install trunk:

    cargo install --locked --root ".cargo-installed/" --version 0.21.4 --no-default-features --features rustls trunk

You can also install trunk another way, but things may not work due to version incompatibilities!

#### Building

    trunk serve --release --features "web"

## Implementation notes

At a high level, there are two ways to use this:

1. `evaluate()` initializes all the wgpu machinery (pipelines, buffers, etc.) and then discards it. This is roughly analogous to Fidget's [RenderConfig::run](https://docs.rs/fidget/latest/fidget/render/struct.RenderConfig.html#method.run).
2. The fine-grained APIs (`create_device`, `create_pipeline_layout`, etc.), as src/bin/win.rs and src/web/main.rs do.

Using the fine-grained APIs is recommended for an interactive application, because it avoids a lot of overheading with creating and freeing the buffers, multiple round-trips between main memory and GPU memory, etc.

<img width="801" alt="shapes at 24-11-28 15 08 21" src="https://github.com/user-attachments/assets/998011ee-87f4-4657-83e1-0cb182b7a33e">

The diagram above shows the high-level flow, with some aspects simplified. Here are some more details:

- We start by turning the Fidget shape into a `Vec<RegOp>`. This is the same representation Fidget's JIT evaluator uses to generate native code.
  - **Optimization:** we split the image up into tiles, and use Fidget's tape simplification to compute a simplified tape for each tile. (Basically the same approach Fidget uses.) We concatenate all these together into the same Vec.
- `GPUExpression::tape_bytes` does a custom serialization of a `Vec<RegOp>` into a `Vec<u8>`.
  - **Note:** we did it this way because Fidget didn't have a built-in way to serialize RegOps. In the meantime, Matt implemented a canonical bytecode format on his [wgpu-bytecode branch][wgpu-bytecode]. We could switch to that.
- `update_buffers` takes the bytecode and stuffs it into a GPU storage buffer (named `bytecode` in the shader). It also writes some uniforms with info about where the simplified tape for each tile is found.
- Then we dispatch the compute shader. It operates on vec4<f32>, so a single invocation computes the result for four pixels (along the x axis) at once. It writes the results to another storage buffer.
- In the web and native demos, we then have a trivial vertex shader (just two triangles covering the screen) and then a fragment shader which reads from the output buffer.

[wgpu-bytecode]: https://github.com/mkeeter/fidget/tree/wgpu-bytecode
[mpr]: https://www.mattkeeter.com/research/mpr/keeter_mpr20.pdf

### Workgroup size

We didn't put a lot  of thought into the workgroup size. Some quick experimention didn't show any big differences. In theory, we should choose a combination of workgroup size and tile size that ensures that a single SIMD group (32 invocations) is always executing the same bytecode.

### Tiling

There's an optimization possible in the tiling, that if we know we are rendering an SDF (as opposed to some arbitrary function), if the tile is entirely inside or outside the shape, we don't need to evaluate anything. We don't current do that however.

### Missing opcodes

Not all Fidget opcodes are implemented! When an unimplemented opcode is encountered, the compute shader returns NaN. If this happens in a test, it will be flagged. We could consider adding a flag to the GPURenderConfig or something to be able to detect this scenario more easily in other cases.

### Opcode constants

When we first wrote the compute shader, we didn't have a good way to define constants for each bytecode in the shader code, so we just hardcoded the constants. Matt later implemented a solution on his [wgpu-bytecode branch][]. It would make sense to use that.
