# Fidget

This package contains the bindings to the (rust based) fidget library.

## Building

In order to build the package, you need to have the rust compiler installed, as
well as node.

You can then build a new version of the package by running:

```bash
npm run build
```

This is a two step process, first the rust code is compiled to wasm, and then
a Typescript library is compiled.
