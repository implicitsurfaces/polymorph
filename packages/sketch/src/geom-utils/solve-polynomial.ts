import { Num, ONE, TWO, ZERO } from "../num";
import { atan2, ifTruthyElse } from "../num-ops";

function applySign(from: Num, to: Num): Num {
  return ifTruthyElse(from.lessThan(ZERO), to.neg(), to);
}

export function solveQuadratic(c2: Num, c1: Num, c0: Num): [Num, Num] {
  const zeroASolution = c0.neg().div(c1);
  const discriminant = c1.square().sub(c2.mul(c0).mul(4));

  const bNeg = c1.neg();
  const _2a = c2.mul(TWO);

  return [
    ifTruthyElse(c2, bNeg.add(discriminant.sqrt()).div(_2a), zeroASolution),
    ifTruthyElse(c2, bNeg.sub(discriminant.sqrt()).div(_2a), zeroASolution),
  ];
}

export function solveCubic(
  c3in: Num,
  c2in: Num,
  c1in: Num,
  c0in: Num,
): [Num, Num, Num] {
  const [zero3Solution1, zero3Solution2] = solveQuadratic(c2in, c1in, c0in);
  const zero3Solution = [zero3Solution1, zero3Solution2, zero3Solution2];

  const c2 = c2in.div(c3in.mul(3));
  const c1 = c1in.div(c3in.mul(3));
  const c0 = c0in.div(c3in);

  const d0 = c1.sub(c2.square());
  const d1 = c0.sub(c1.mul(c2));
  const d2 = c2.mul(c0).sub(c1.square());

  const d = d0.mul(4).mul(d2).sub(d1.square());
  const de = d1.sub(c2.mul(d0).mul(2));

  /* solution for d < 0 */
  const sq = d.mul(-0.25).sqrt();
  const r = de.mul(-0.5);
  const dNegSolution1 = r.add(sq).cbrt().add(r.sub(sq).cbrt());
  const dNegSolution = [dNegSolution1, dNegSolution1, dNegSolution1];

  /* solution for d == 0 */
  const d0Solution1 = applySign(de, d0.neg().sqrt());
  const d0Solution2 = d0Solution1.mul(-2);
  const dZeroSolution = [d0Solution1, d0Solution2, d0Solution2];

  /* solution for d > 0 */
  const th = atan2(d.sqrt(), de.neg()).div(3);
  const r0 = th.cos();
  const ss3 = th.sin().mul(Math.sqrt(3));
  const r1 = r0.neg().add(ss3).mul(0.5);
  const r2 = r0.neg().sub(ss3).mul(0.5);
  const t = d0.neg().sqrt().mul(2);
  const dPosSolution1 = t.mul(r0);
  const dPosSolution2 = t.mul(r1);
  const dPosSolution3 = t.mul(r2);

  const dPosSolution = [dPosSolution1, dPosSolution2, dPosSolution3];

  return dZeroSolution.map((zeroSolution, i) => {
    let solution = ifTruthyElse(d, dPosSolution[i], zeroSolution);
    solution = ifTruthyElse(d.lessThan(ZERO), dNegSolution[i], solution);
    solution = solution.sub(c2);

    return ifTruthyElse(c3in, solution, zero3Solution[i]);
  }) as [Num, Num, Num];
}

export function solveQuartic(
  c4: Num,
  c3: Num,
  c2: Num,
  c1: Num,
  c0: Num,
): [Num, Num, Num, Num] {
  const p = c3.div(c4);
  const q = c2.div(c4);
  const r = c1.div(c4);
  const s = c0.div(c4);

  const p2 = p.square();

  // Coefficient of the resolvent cubic
  const z2 = q.neg();
  const z1 = p.mul(r).sub(s.mul(4));
  const z0 = s.mul(4).mul(q).sub(p2.mul(s)).sub(r.square());

  const sol = solveCubic(ONE, z2, z1, z0)[0];

  const R2 = sol.add(p2.div(4)).sub(q).max(ZERO);
  const R = R2.sqrt();

  const t = p2.mul(0.75).sub(q.mul(TWO));
  const frac = p.mul(q).mul(4).sub(r.mul(8)).sub(p2.mul(p)).div(R.mul(4));
  const innerSqrt = TWO.mul(sol.square().sub(s.mul(4)));

  const D_Req = t.sub(R2).add(frac);
  const D_Req0 = t.add(innerSqrt);
  const D2 = ifTruthyElse(R, D_Req, D_Req0);
  const D = D2.sqrt();

  const E_Req = t.sub(R2).sub(frac);
  const E_Req0 = t.sub(innerSqrt);
  const E2 = ifTruthyElse(R, E_Req, E_Req0);
  const E = E2.sqrt();

  const DisValid = D2.greaterThan(ZERO);
  const EisValid = E2.greaterThan(ZERO);

  const p4 = p.div(-4);

  const x1 = p4.add(R.add(D).div(TWO));
  const x2 = p4.add(R.sub(D).div(TWO));
  const x3 = p4.sub(R.add(E).div(TWO));
  const x4 = p4.sub(R.sub(E).div(TWO));

  const quarticSolutions = [
    ifTruthyElse(DisValid, x1, x3),
    ifTruthyElse(DisValid, x2, x4),
    ifTruthyElse(EisValid, x3, x1),
    ifTruthyElse(EisValid, x4, x2),
  ];

  const cubicSolutions = solveCubic(c3, c2, c1, c0);

  return quarticSolutions.map((quarticSolution, i) => {
    return ifTruthyElse(c4, quarticSolution, cubicSolutions[i % 3]);
  }) as [Num, Num, Num, Num];
}
