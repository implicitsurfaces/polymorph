import type { BinaryOperation, UnaryOperation } from "./num-tree";
import { NumNode, LiteralNum, BinaryOp, UnaryOp } from "./num-tree";

export function as_num(n: number | Num): Num {
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
    return binaryOpNum("ADD", this, as_num(other));
  }
  sub(other: Num | number) {
    return binaryOpNum("SUB", this, as_num(other));
  }
  mul(other: Num | number) {
    return binaryOpNum("MUL", this, as_num(other));
  }
  div(other: Num | number) {
    return binaryOpNum("DIV", this, as_num(other));
  }
  sqrt() {
    return unaryOpNum("SQRT", this);
  }
  neg() {
    return unaryOpNum("NEG", this);
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
    const amax = binaryOpNum("MAX", val, as_num(0));
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
}
