#!/usr/bin/env bash

set -euo pipefail

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$DIR/../"

# # Keep people from accidentally pushing unformatted code.
# cargo fmt --check

# cargo --locked clippy --                   \
#       -D warnings                          \
#       -A clippy::new-without-default

# Keep people from accidentally pushing unformatted Python code.
rye lint .
rye fmt --check .
