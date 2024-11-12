## Install

Make sure you're in this directory:

    cd gpu_interp

Install [direnv](https://direnv.net/) so your shell path always refers to correct CLI binaries.
Install the CLI binaries:

    cargo install --locked --root ".cargo-installed/" --version 0.21.4 --no-default-features --features rustls trunk


## Build Web UI

    trunk serve --release
