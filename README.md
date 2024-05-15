# polymorph

## Prerequisites

Install [Rye](https://rye-up.com/guide/installation/). Then run:

    rye sync

## Things that work

To run the iceberg simulation:

    cargo run --bin iceberg

To see the buoyancy curve (centers of buoyancy at various heel angles):

    cargo run --release --bin visualize_buoyancy_curve

To sample the cost function for visualization:

    cargo run --release --bin sample_cost_fn > cost_function_data.csv

This will write `cost_function_data.csv` to the project root.

To visualize the cost function data:

    python script/equilibrium_viz.py cost_function_data.csv

or, for the 3D visualization:

    python script/equilibrium_viz.py --3d cost_function_data.csv

To run the jupyter notebook, you need to have the python env installed. You
need to [install rye](https://rye-up.com/) and then run:

    rye sync

Then run:

    rye run jupyter lab

## Architecture + overview

Most of the codebase is written in Rust and setup within a [Cargo Workspace](https://doc.rust-lang.org/cargo/reference/workspaces.html) so it can be checked easily in development via `cargo check` and `cargo clippy` at the root directory.

Code in `crates/` should be kept running and can be checked via the `script/check_all.sh` script. To avoid accidentally break anything for others, you may want to have gut run this as a pre-commit hook --- you can set this up by running

    git config core.hooksPath script/git-hooks

Our named folders (`kevin/`, etc.) are for scratch work and aren't automatically checked or assumed to be working.
