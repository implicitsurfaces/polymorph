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
