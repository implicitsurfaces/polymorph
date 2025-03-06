import type { BinaryOperation, UnaryOperation } from "./num-tree";
import {
  NumNode,
  LiteralNum,
  BinaryOp,
  UnaryOp,
  Variable,
  ZERO_NODE,
  ONE_NODE,
  TWO_NODE,
  NEG_ONE_NODE,
} from "./num-tree";
import { compressNum } from "./num-dag-tools/compress-num";
import { renderNodeAsDot } from "./eval-num/dot-eval";
import { simplify } from "./num-dag-tools/simplify-num";

export function asNum(n: number | Num): Num {
  if (n instanceof Num) {
    return n;
  }
  return new Num(new LiteralNum(n));
}

export function binaryOpNum(op: BinaryOperation, a: Num, b: Num): Num {
  return new Num(new BinaryOp(op, a.n, b.n));
}

export function unaryOpNum(op: UnaryOperation, a: Num): Num {
  return new Num(new UnaryOp(op, a.n));
}

export class Num {
  readonly n: NumNode;
  constructor(n: NumNode) {
    this.n = n;
  }
  add(other: Num | number) {
    return binaryOpNum("ADD", this, asNum(other));
  }
  sub(other: Num | number) {
    return binaryOpNum("SUB", this, asNum(other));
  }
  mul(other: Num | number) {
    return binaryOpNum("MUL", this, asNum(other));
  }
  div(other: Num | number) {
    return binaryOpNum("DIV", this, asNum(other));
  }
  sqrt() {
    return unaryOpNum("SQRT", this);
  }
  safeSqrt() {
    return this.max(0).sqrt();
  }
  cbrt() {
    return this.abs().log().div(3).exp().mul(this.sign());
  }
  neg() {
    return unaryOpNum("NEG", this);
  }
  inv() {
    return binaryOpNum("DIV", asNum(1), this);
  }
  sign() {
    return unaryOpNum("SIGN", this);
  }
  abs() {
    return unaryOpNum("ABS", this);
  }
  smoothabs() {
    return this.mul(unaryOpNum("TANH", this.mul(10)));
  }
  log1p() {
    return unaryOpNum("LOG1P", this);
  }
  softplus() {
    // This implementation is based on the jax implementation of softplus.
    // It uses the log-sum-exp trick to avoid numerical instability.
    const factor = 50;
    const val = this.mul(factor);
    const amax = binaryOpNum("MAX", val, asNum(0));
    return val.abs().neg().exp().log1p().add(amax).div(factor);
  }
  softminus() {
    return this.sub(this.softplus());
  }
  mod(other: Num) {
    return binaryOpNum("MOD", this, other);
  }
  cos() {
    return unaryOpNum("COS", this);
  }
  acos() {
    return unaryOpNum("ACOS", this);
  }
  sin() {
    return unaryOpNum("SIN", this);
  }
  asin() {
    return unaryOpNum("ASIN", this);
  }
  tan() {
    return unaryOpNum("TAN", this);
  }
  atan() {
    return unaryOpNum("ATAN", this);
  }
  exp() {
    return unaryOpNum("EXP", this);
  }
  tanh() {
    return unaryOpNum("TANH", this);
  }
  log() {
    return unaryOpNum("LOG", this);
  }
  square() {
    return binaryOpNum("MUL", this, this);
  }

  compare(other: Num | number) {
    return binaryOpNum("COMPARE", this, asNum(other));
  }
  and(other: Num | number) {
    return binaryOpNum("AND", this, asNum(other));
  }
  or(other: Num | number) {
    return binaryOpNum("OR", this, asNum(other));
  }
  not() {
    return unaryOpNum("NOT", this);
  }
  max(other: Num | number) {
    return binaryOpNum("MAX", this, asNum(other));
  }
  min(other: Num | number) {
    return binaryOpNum("MIN", this, asNum(other));
  }
  equals(other: Num | number) {
    return asNum(other).compare(this).not();
  }
  lessThan(other: Num | number) {
    return asNum(other).compare(this).max(ZERO);
  }
  lessThanOrEqual(other: Num | number) {
    return asNum(other).compare(this).add(ONE).min(ONE);
  }
  greaterThan(other: Num | number) {
    return asNum(other).lessThan(this);
  }
  greaterThanOrEqual(other: Num | number) {
    return asNum(other).lessThanOrEqual(this);
  }

  compress(): Num {
    return new Num(compressNum(this.n));
  }
  simplify(): Num {
    return new Num(simplify(this.n));
  }

  asDot(): string {
    return renderNodeAsDot(this.n);
  }
}

export const ZERO = new Num(ZERO_NODE);
export const NEG_ONE = new Num(NEG_ONE_NODE);
export const ONE = new Num(ONE_NODE);
export const TWO = new Num(TWO_NODE);

export const variable = (name: string) => new Num(new Variable(name));

export const NumX = new Num(new Variable("x"));
export const NumY = new Num(new Variable("y"));
export const NumZ = new Num(new Variable("z"));
