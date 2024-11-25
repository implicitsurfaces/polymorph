/**
 * A simple fidget wrapper extension
 * @module fidget
 */

import init, { Context, Var, Node } from "./pkg/fidget.js";

export async function initLib() {
  let lib;
  try {
    lib = await init();
  } catch (e) {
    const { readFile } = await import("node:fs/promises");
    const wasmUrl = new URL("./pkg/fidget_bg.wasm", import.meta.url);
    const buffer = await readFile(wasmUrl);
    lib = await init(buffer);
  }

  return lib;
}

const CACHED_LIB = { current: null };
export async function createContext() {
  if (!CACHED_LIB.current) {
    const lib = await initLib();
    CACHED_LIB.current = lib;
  }
  return new Context();
}

export { Var, Node, Context };
