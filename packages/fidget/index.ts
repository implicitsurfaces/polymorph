/**
 * A simple fidget wrapper extension
 * @module fidget
 */

import init, { FidgetContext, FidgetVar, FidgetNode } from "./pkg/fidget.js";

function c_string(buffer, offset) {
  const m = new DataView(buffer);
  let result = "";
  for (let i = 0; m.getUint8(offset + i) !== 0; i++) {
    result += String.fromCharCode(m.getUint8(offset + i));
  }
  return result;
}

type FidgetWasm = Awaited<ReturnType<typeof init>>;

export type FNode = number;

export class Context {
  private handle: number;
  private fidget: FidgetWasm;

  constructor(fidgetInstance: FidgetWasm) {
    this.fidget = fidgetInstance;
    this.handle = this.fidget.new_context();
  }
  constant(a: number): FNode {
    return this.fidget.ctx_constant(this.handle, a);
  }
  x(): FNode {
    return this.fidget.ctx_x(this.handle);
  }
  y(): FNode {
    return this.fidget.ctx_y(this.handle);
  }
  z(): FNode {
    return this.fidget.ctx_y(this.handle);
  }
  var(): FNode {
    return this.fidget.ctx_var(this.handle);
  }
  add(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_add(this.handle, a, b);
  }
  sub(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_sub(this.handle, a, b);
  }
  mul(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_mul(this.handle, a, b);
  }
  div(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_div(this.handle, a, b);
  }
  max(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_max(this.handle, a, b);
  }
  min(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_min(this.handle, a, b);
  }
  neg(a: FNode): FNode {
    return this.fidget.ctx_neg(this.handle, a);
  }
  square(a: FNode): FNode {
    return this.fidget.ctx_square(this.handle, a);
  }
  sqrt(a: FNode): FNode {
    return this.fidget.ctx_sqrt(this.handle, a);
  }

  and(a: FNode, b: FNode) {
    return this.fidget.ctx_and(this.handle, a, b);
  }
  or(a: FNode, b: FNode) {
    return this.fidget.ctx_or(this.handle, a, b);
  }
  not(a: FNode) {
    return this.fidget.ctx_not(this.handle, a);
  }
  recip(a: FNode) {
    return this.fidget.ctx_recip(this.handle, a);
  }
  abs(a: FNode) {
    return this.fidget.ctx_abs(this.handle, a);
  }
  sin(a: FNode) {
    return this.fidget.ctx_sin(this.handle, a);
  }
  cos(a: FNode) {
    return this.fidget.ctx_cos(this.handle, a);
  }
  tan(a: FNode) {
    return this.fidget.ctx_tan(this.handle, a);
  }
  asin(a: FNode) {
    return this.fidget.ctx_asin(this.handle, a);
  }
  acos(a: FNode) {
    return this.fidget.ctx_acos(this.handle, a);
  }
  atan(a: FNode) {
    return this.fidget.ctx_atan(this.handle, a);
  }
  atan2(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_atan2(this.handle, a, b);
  }
  exp(a: FNode): FNode {
    return this.fidget.ctx_exp(this.handle, a);
  }
  ln(a: FNode): FNode {
    return this.fidget.ctx_ln(this.handle, a);
  }
  compare(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_compare(this.handle, a, b);
  }
  mod(a: FNode, b: FNode): FNode {
    return this.fidget.ctx_mod(this.handle, a, b);
  }

  deriv(n: FNode, v: FNode): FNode {
    return this.fidget.ctx_deriv(this.handle, n, v);
  }

  eval(n: FNode): number {
    return this.fidget.ctx_eval_node(this.handle, n);
  }

  render(n: FNode, imageSize = 50) {
    return this.fidget.ctx_render_node(this.handle, n, imageSize);
  }

  to_graphviz() {
    const offset = this.fidget.ctx_to_graphviz(this.handle);
    return c_string(this.fidget.memory.buffer, offset);
  }
}

export async function initLib() {
  let lib;
  try {
    lib = await init();
  } catch (e) {
    // @ts-expect-error we use this as a way to fake being in a browserj
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
  return new FidgetContext();
}

export { FidgetVar, FidgetNode, FidgetContext };
