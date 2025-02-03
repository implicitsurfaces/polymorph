import { Num, TWO } from "../num";
import { ifTruthyElse } from "../num-ops";

export function solveQuadratic(a: Num, b: Num, c: Num): [Num, Num] {
  const zeroASolution = c.neg().div(b);
  const discriminant = b.square().sub(a.mul(c).mul(4));

  const bNeg = b.neg();
  const _2a = a.mul(TWO);

  return [
    ifTruthyElse(a, bNeg.add(discriminant.sqrt()).div(_2a), zeroASolution),
    ifTruthyElse(a, bNeg.sub(discriminant.sqrt()).div(_2a), zeroASolution),
  ];
}
