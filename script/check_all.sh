#!/usr/bin/env bash

set -euo pipefail

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$DIR/../"

# Keep people from accidentally pushing unformatted code.
cargo fmt --check

cargo --locked clippy --                   \
      -D warnings                          \
      -A clippy::new-without-default

if type rye >/dev/null 2>&1; then
    rye run ruff check --fix script/*.py
else
    echo "rye not found, skipping python linting"
fi
