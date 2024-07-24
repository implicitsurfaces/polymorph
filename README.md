# polymorph

## Prerequisites

Install [Rye](https://rye-up.com/guide/installation/). Then run:

    rye sync

Setup pre-commit hooks:

    git config --unset-all core.hooksPath && rye run pre-commit install

## Things that work

Run

    rye run jupyter lab

to open various experiments/explorations in `notebooks/`.
You can also use VS Code's Jupyter extension, which will detect the rye-managed Python kernel in `.venv`.

To run the Python app UI:

    rye run polymorph


To run tests / benchmark:

    rye test --package polymorph_num

To check types:

    rye run pyright


## Rust experiments (Apr/May)

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

## Architecture + overview

The original Rust code is setup within a [Cargo Workspace](https://doc.rust-lang.org/cargo/reference/workspaces.html) so it can be checked easily in development via `cargo check` and `cargo clippy` at the root directory.

Our named folders (`kevin/`, etc.) are for scratch work and aren't automatically checked or assumed to be working.
