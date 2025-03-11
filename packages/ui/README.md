# The Polymorph ui TypeScript package

This package implements a UI for the Polymorph project using React.

## How to run locally?

As this depends on `fidget`, you first need to install some Rust and Wasm tools
if not done already. For example on Ubuntu:

```
# Install recent version of Rust and wasm-pack
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
cargo install wasm-pack
```

Then build the fidget, sketch, and draw-api packages:

```
cd polymorph/packages
cd fidget && npm install && npm run build && cd ..
cd sketch && npm install && npm run build && cd ..
cd draw-api && npm install && npm run build && cd ..
```

You can finally run the UI application with:

```
cd polymorph/packages/ui
npm install
npm run dev
```

Then open the shown link in a browser.
