import { Num, ONE, asNum, binaryOpNum } from "./num";
import { fullDerivative } from "./num-diff";
import { partialDerivative, replaceVariable } from "./num-tree";

export function max(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("MAX", tree, asNum(n));
  });
  return tree;
}

export function min(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("MIN", tree, asNum(n));
  });
  return tree;
}

export function atan2(a: Num | number, b: Num | number): Num {
  return binaryOpNum("ATAN2", asNum(a), asNum(b));
}

export function ifTruthyElse(
  condition: Num | number,
  if_non_zero: Num | number,
  if_zero: Num | number,
): Num {
  const cond = asNum(condition);
  const lhs = cond.and(if_non_zero);
  const n_condition = cond.not();
  const rhs = n_condition.and(if_zero);
  return lhs.or(rhs);
}

export function hypot(a: Num | number, b: Num | number): Num {
  return asNum(a).square().add(asNum(b).square()).sqrt();
}

export function clamp(
  a: Num | number,
  minVal: Num | number,
  maxVal: Num | number,
): Num {
  const bottomClamped = max(minVal, a);
  return min(maxVal, bottomClamped);
}

export function sigmoid(a: Num | number): Num {
  const v = asNum(a);

  const posExpr = ONE.div(v.neg().exp().add(ONE));
  const negExpr = v.exp().div(v.exp().add(ONE));

  const vGT0 = v.greaterThan(0);

  return ifTruthyElse(vGT0, posExpr, negExpr);
}

export function diff(num: Num): Num {
  return new Num(fullDerivative(num.compress().n)).compress();
}

export function gradientAt(num: Num, point: [string, Num | number][]): Num[] {
  const diffNum = fullDerivative(num.compress().n);
  const diffAtPoint = replaceVariable(
    diffNum,
    new Map(point.map(([k, v]) => [k, asNum(v).n])),
  );

  const grad = point.map(([k]) => {
    return new Num(partialDerivative(diffAtPoint, k)).compress();
  });
  return grad;
}
