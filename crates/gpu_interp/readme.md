# GPU-based rendering for Fidget

A 2D renderer for Fidget expressions that uses the GPU.

## What's in here

- The implementation of the GPU-based renderer in src/lib.rs
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
