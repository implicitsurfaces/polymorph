import { Num, asNum, binaryOpNum, unaryOpNum } from "./num";

export function add(a: Num | number, b: Num | number): Num {
  return binaryOpNum("ADD", asNum(a), asNum(b));
}

export function sub(a: Num | number, b: Num | number): Num {
  return binaryOpNum("SUB", asNum(a), asNum(b));
}

export function mul(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MUL", asNum(a), asNum(b));
}

export function div(a: Num | number, b: Num | number): Num {
  return binaryOpNum("DIV", asNum(a), asNum(b));
}

export function mod(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MOD", asNum(a), asNum(b));
}

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

export function compare(a: Num | number, b: Num | number): Num {
  return binaryOpNum("COMPARE", asNum(a), asNum(b));
}

export function and(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("AND", tree, asNum(n));
  });
  return tree;
}

export function or(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("OR", tree, asNum(n));
  });
  return tree;
}

export function not(a: Num | number): Num {
  return unaryOpNum("NOT", asNum(a));
}

export function lessThan(a: Num | number, b: Num | number): Num {
  const cmp = compare(b, a);
  return max(0, cmp);
}

export function lessThanOrEqual(a: Num | number, b: Num | number): Num {
  const cmp = compare(b, a);
  const shift = cmp.add(1.0);
  return min(shift, 1.0);
}

export function greaterThan(a: Num | number, b: Num | number): Num {
  return lessThan(b, a);
}

export function greaterThanOrEqual(a: Num | number, b: Num | number): Num {
  return lessThanOrEqual(b, a);
}

export function ifTruthyElse(
  condition: Num | number,
  if_non_zero: Num | number,
  if_zero: Num | number,
): Num {
  const lhs = and(condition, if_non_zero);
  const n_condition = not(condition);
  const rhs = and(n_condition, if_zero);
  return or(lhs, rhs);
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
