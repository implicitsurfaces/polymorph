// Use numeric codes instead of strings.
const OP_ADD = 0;
const OP_SUB = 1;
const OP_MUL = 2;
const OP_DIV = 3;
const OP_SIN = 4;
const OP_COS = 5;
const OP_TAN = 6;
const OP_ASIN = 7;
const OP_ACOS = 8;
const OP_ATAN = 9;
const OP_ATAN2 = 10;
const OP_NEG = 11;
const OP_EXP = 12;
const OP_SQRT = 13;
const OP_LOG = 14;
const OP_POW = 15;

export class AutoDiff {
  constructor() {
    this.params = [];
    this.trace = [];
    this.assertions = [];
    this.traceStateStore = [];
  }

  saveTrace() {
    this.traceStateStore.push(this.trace.length);
  }

  restoreTrace() {
    // Instead of using splice (which creates a new array), simply reset the length.
    this.trace.length = this.traceStateStore.pop();
  }

  num(x) {
    return new AutoDiffNum(x, this);
  }

  param(x) {
    const newParam = this.num(x);
    newParam.index = this.params.length;
    this.params.push(newParam);
    return newParam;
  }

  evalGrad() {
    // Compute gradients via reverse-mode autodiff.
    return evalGradReverse(this).map((n) => n.grad);
  }

  resetGrad() {
    // Reset gradients for all parameters and intermediate results.
    for (let i = 0, n = this.params.length; i < n; i++) {
      this.params[i].grad = 0;
    }
    for (let i = 0, n = this.trace.length; i < n; i++) {
      this.trace[i].result.grad = 0;
    }
  }

  assert(exp) {
    this.assertions.push(exp);
  }

  values() {
    this.saveTrace();
    const result = new Array(this.assertions.length);
    for (let i = 0, n = this.assertions.length; i < n; i++) {
      result[i] = this.assertions[i]().val;
    }
    this.restoreTrace();
    return result;
  }
}

export class AutoDiffNum {
  constructor(x, ad) {
    this.val = x;
    this.grad = 0;
    this._ad = ad;
  }

  add(other) {
    const otherVal = getValue(other);
    const result = new AutoDiffNum(this.val + otherVal, this._ad);
    this._ad.trace.push({
      op: OP_ADD,
      arg1: this,
      arg2: other,
      result,
      // Cache the input values to avoid recomputation during the reverse pass.
      vals: [this.val, otherVal],
    });
    return result;
  }

  sub(other) {
    const otherVal = getValue(other);
    const result = new AutoDiffNum(this.val - otherVal, this._ad);
    this._ad.trace.push({
      op: OP_SUB,
      arg1: this,
      arg2: other,
      result,
      vals: [this.val, otherVal],
    });
    return result;
  }

  mul(other) {
    const otherVal = getValue(other);
    const result = new AutoDiffNum(this.val * otherVal, this._ad);
    this._ad.trace.push({
      op: OP_MUL,
      arg1: this,
      arg2: other,
      result,
      vals: [this.val, otherVal],
    });
    return result;
  }

  div(other) {
    const otherVal = getValue(other);
    const result = new AutoDiffNum(
      zeroSafeDivide(this.val, otherVal),
      this._ad,
    );
    this._ad.trace.push({
      op: OP_DIV,
      arg1: this,
      arg2: other,
      result,
      vals: [this.val, otherVal],
    });
    return result;
  }

  sin() {
    const result = new AutoDiffNum(Math.sin(this.val), this._ad);
    this._ad.trace.push({
      op: OP_SIN,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  cos() {
    const result = new AutoDiffNum(Math.cos(this.val), this._ad);
    this._ad.trace.push({
      op: OP_COS,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  tan() {
    const result = new AutoDiffNum(Math.tan(this.val), this._ad);
    this._ad.trace.push({
      op: OP_TAN,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  asin() {
    const result = new AutoDiffNum(Math.asin(this.val), this._ad);
    this._ad.trace.push({
      op: OP_ASIN,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  acos() {
    const result = new AutoDiffNum(Math.acos(this.val), this._ad);
    this._ad.trace.push({
      op: OP_ACOS,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  atan() {
    const result = new AutoDiffNum(Math.atan(this.val), this._ad);
    this._ad.trace.push({
      op: OP_ATAN,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  atan2(other) {
    const otherVal = getValue(other);
    // Note: Math.atan2 expects (y, x); here we treat `this.val` as y and otherVal as x.
    const result = new AutoDiffNum(Math.atan2(this.val, otherVal), this._ad);
    this._ad.trace.push({
      op: OP_ATAN2,
      arg1: this, // y value
      arg2: other, // x value
      result,
      vals: [this.val, otherVal],
    });
    return result;
  }

  neg() {
    const result = new AutoDiffNum(-this.val, this._ad);
    this._ad.trace.push({
      op: OP_NEG,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  exp() {
    const result = new AutoDiffNum(Math.exp(this.val), this._ad);
    this._ad.trace.push({
      op: OP_EXP,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  sqrt() {
    const result = new AutoDiffNum(Math.sqrt(this.val), this._ad);
    this._ad.trace.push({
      op: OP_SQRT,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  log() {
    const result = new AutoDiffNum(Math.log(this.val), this._ad);
    this._ad.trace.push({
      op: OP_LOG,
      arg1: this,
      result,
      vals: [this.val],
    });
    return result;
  }

  pow(other) {
    const otherVal = getValue(other);
    const result = new AutoDiffNum(Math.pow(this.val, otherVal), this._ad);
    this._ad.trace.push({
      op: OP_POW,
      arg1: this,
      arg2: other,
      result,
      vals: [this.val, otherVal],
    });
    return result;
  }
}

function getValue(x) {
  return x instanceof AutoDiffNum ? x.val : x;
}

function zeroSafeDivide(x, y) {
  return x / (y === 0 ? 1e-15 : y);
}

export function evalGradReverse(ad) {
  const tape = ad.trace;
  for (let i = tape.length - 1; i >= 0; i--) {
    const entry = tape[i];
    switch (entry.op) {
      case OP_ADD:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad;
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad += entry.result.grad;
        break;
      case OP_SUB:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad;
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad -= entry.result.grad;
        break;
      case OP_MUL:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad * entry.vals[1];
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad += entry.result.grad * entry.vals[0];
        break;
      case OP_DIV: {
        const xVal = entry.vals[0];
        const yVal = entry.vals[1];
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad = zeroSafeDivide(
            entry.result.grad,
            yVal,
          );
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad += zeroSafeDivide(
            -entry.result.grad * xVal,
            yVal * yVal,
          );
        break;
      }
      case OP_SIN:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad * Math.cos(entry.vals[0]);
        break;
      case OP_COS:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += -entry.result.grad * Math.sin(entry.vals[0]);
        break;
      case OP_TAN: {
        const cosVal = Math.cos(entry.vals[0]);
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad = zeroSafeDivide(
            entry.result.grad,
            cosVal * cosVal,
          );
        break;
      }
      case OP_ASIN:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += zeroSafeDivide(
            entry.result.grad,
            Math.sqrt(1 - entry.vals[0] * entry.vals[0]),
          );
        break;
      case OP_ACOS:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad +=
            -entry.result.grad *
            zeroSafeDivide(1, Math.sqrt(1 - entry.vals[0] * entry.vals[0]));
        break;
      case OP_ATAN:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad = zeroSafeDivide(
            entry.result.grad,
            1 + entry.vals[0] * entry.vals[0],
          );
        break;
      case OP_ATAN2: {
        // In our implementation, arg1 corresponds to the first argument (y) and arg2 to the second (x).
        const yVal = entry.vals[0];
        const xVal = entry.vals[1];
        const denom = xVal * xVal + yVal * yVal;
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad * zeroSafeDivide(xVal, denom);
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad += -entry.result.grad * zeroSafeDivide(yVal, denom);
        break;
      }
      case OP_NEG:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad -= entry.result.grad;
        break;
      case OP_EXP:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += entry.result.grad * entry.result.val;
        break;
      case OP_SQRT:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad += zeroSafeDivide(
            entry.result.grad,
            2 * entry.result.val,
          );
        break;
      case OP_LOG:
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad +=
            entry.result.grad * zeroSafeDivide(1, entry.vals[0]);
        break;
      case OP_POW: {
        const xVal = entry.vals[0];
        const yVal = entry.vals[1];
        if (entry.arg1 instanceof AutoDiffNum)
          entry.arg1.grad +=
            entry.result.grad * yVal * Math.pow(xVal, yVal - 1);
        if (entry.arg2 instanceof AutoDiffNum)
          entry.arg2.grad +=
            entry.result.grad * entry.result.val * Math.log(xVal);
        break;
      }
      default:
        throw new Error("Unknown op code in reverse mode: " + entry.op);
    }
  }
  return ad.params;
}
