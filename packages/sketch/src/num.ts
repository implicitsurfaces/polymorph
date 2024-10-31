import type { BinaryOperation, UnaryOperation } from "./num-tree";
import { NumNode, LiteralNum, BinaryOp, UnaryOp } from "./num-tree";

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
}

export const ZERO = new Num(new LiteralNum(0));
export const ONE = new Num(new LiteralNum(1));
export const TWO = new Num(new LiteralNum(2));
