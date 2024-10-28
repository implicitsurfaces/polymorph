import { Num, as_num, binaryOpNum, unaryOpNum } from "./num";

export function add(a: Num | number, b: Num | number): Num {
  return binaryOpNum("ADD", as_num(a), as_num(b));
}

export function sub(a: Num | number, b: Num | number): Num {
  return binaryOpNum("SUB", as_num(a), as_num(b));
}

export function mul(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MUL", as_num(a), as_num(b));
}

export function div(a: Num | number, b: Num | number): Num {
  return binaryOpNum("DIV", as_num(a), as_num(b));
}

export function mod(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MOD", as_num(a), as_num(b));
}

export function max(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MAX", as_num(a), as_num(b));
}

export function min(a: Num | number, b: Num | number): Num {
  return binaryOpNum("MIN", as_num(a), as_num(b));
}

export function atan2(a: Num | number, b: Num | number): Num {
  return binaryOpNum("ATAN2", as_num(a), as_num(b));
}

export function compare(a: Num | number, b: Num | number): Num {
  return binaryOpNum("COMPARE", as_num(a), as_num(b));
}

export function and(a: Num | number, b: Num | number): Num {
  return binaryOpNum("AND", as_num(a), as_num(b));
}

export function or(a: Num | number, b: Num | number): Num {
  return binaryOpNum("OR", as_num(a), as_num(b));
}

export function not(a: Num | number): Num {
  return unaryOpNum("NOT", as_num(a));
}

export function less_than(a: Num | number, b: Num | number): Num {
  const cmp = compare(b, a);
  return max(0, cmp);
}

export function less_than_or_equal(a: Num | number, b: Num | number): Num {
  const cmp = compare(b, a);
  const shift = cmp.add(1.0);
  return min(shift, 1.0);
}

export function greater_than(a: Num | number, b: Num | number): Num {
  return less_than(b, a);
}

export function greater_than_or_equal(a: Num | number, b: Num | number): Num {
  return less_than_or_equal(b, a);
}

export function if_non_zero_else(
  condition: Num | number,
  if_non_zero: Num | number,
  if_zero: Num | number,
): Num {
  const lhs = and(condition, if_non_zero);
  const n_condition = not(condition);
  const rhs = and(n_condition, if_zero);
  return or(lhs, rhs);
}
