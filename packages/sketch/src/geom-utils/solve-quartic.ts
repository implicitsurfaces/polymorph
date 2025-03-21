/*
 * This solver partiall implement Orellana and Michele's algorithm for solving
 * quartics.
 *
 * This implementation is adapted from
 * https://raw.githubusercontent.com/raphlinus/raphlinus.github.io/master/_posts/2022-09-02-parallel-beziers.md
 */
import { EPS_M } from "../constants";
import { Num, ONE, ZERO, asNum } from "../num";
import { ifTruthyElse } from "../num-ops";
import { solveQuadratic, solveCubic } from "./solve-polynomial";
import { SENTINELLE } from "../constants";

function eps_rel(raw: Num, a: Num) {
  return ifTruthyElse(a, raw.sub(a).div(a), raw).abs();
}

function copysign(a: Num, b: Num) {
  return ifTruthyElse(b.lessThan(ZERO), a.neg(), a);
}

function doubleQuadraticSolutions(
  a1: Num,
  b1: Num,
  a2: Num,
  b2: Num,
  sentinelle: Num,
) {
  const s1 = solveQuadratic(ONE, a1, b1, sentinelle);
  const s2 = solveQuadratic(ONE, a2, b2, sentinelle);

  return [
    ...s1.map((s, i) => ifTruthyElse(s.equals(sentinelle), s2[i], s)),
    ...s2.map((s, i) => ifTruthyElse(s.equals(sentinelle), s1[i], s)),
  ];
}

function newtonForQuadraticCoeffs(
  alpha1Input: Num,
  beta1Input: Num,
  alpha2Input: Num,
  beta2Input: Num,
  a: Num,
  b: Num,
  c: Num,
  d: Num,
  iterations: number,
) {
  let alpha1 = alpha1Input;
  let beta1 = beta1Input;
  let alpha2 = alpha2Input;
  let beta2 = beta2Input;

  for (let i = 0; i < iterations; i++) {
    const f0 = beta1.mul(beta2).sub(d);
    const f1 = beta1.mul(alpha2).add(alpha1.mul(beta2)).sub(c);
    const f2 = beta1.add(alpha1.mul(alpha2)).add(beta2).sub(b);
    const f3 = alpha1.add(alpha2).sub(a);
    const c1 = alpha1.sub(alpha2);
    const detJ = beta1
      .mul(beta1)
      .sub(beta1.mul(alpha2.mul(c1).add(beta2.mul(2))))
      .add(beta2.mul(alpha1.mul(c1).add(beta2)));

    const inv = asNum(1).div(detJ);
    const c2 = beta2.sub(beta1);
    const c3 = beta1.mul(alpha2).sub(alpha1.mul(beta2));

    const dz0 = c1
      .mul(f0)
      .add(c2.mul(f1))
      .add(c3.mul(f2))
      .sub(beta1.mul(c2).add(alpha1.mul(c3)).mul(f3));

    const dz1 = alpha1
      .mul(c1)
      .add(c2)
      .mul(f0)
      .sub(beta1.mul(c1.mul(f1).add(c2.mul(f2)).add(c3.mul(f3))));

    const dz2 = c1
      .mul(f0)
      .neg()
      .sub(c2.mul(f1))
      .sub(c3.mul(f2))
      .add(alpha2.mul(c3).add(beta2.mul(c2)).mul(f3));

    const dz3 = alpha2
      .mul(c1)
      .add(c2)
      .mul(f0)
      .neg()
      .add(beta2.mul(c1.mul(f1).add(c2.mul(f2)).add(c3.mul(f3))));

    const a1 = ifTruthyElse(detJ, alpha1.sub(inv.mul(dz0)), alpha1);
    const b1 = ifTruthyElse(detJ, beta1.sub(inv.mul(dz1)), beta1);
    const a2 = ifTruthyElse(detJ, alpha2.sub(inv.mul(dz2)), alpha2);
    const b2 = ifTruthyElse(detJ, beta2.sub(inv.mul(dz3)), beta2);

    alpha1 = a1;
    beta1 = b1;
    alpha2 = a2;
    beta2 = b2;
  }

  return [alpha1, beta1, alpha2, beta2];
}

export function solveQuartic(
  c4: Num,
  c3: Num,
  c2: Num,
  c1: Num,
  c0: Num,
  sentinelle = SENTINELLE,
) {
  const a = c3.div(c4);
  const b = c2.div(c4);
  const c = c1.div(c4);
  const d = c0.div(c4);

  const disc = a.square().mul(9).sub(b.mul(24));
  const s = ifTruthyElse(
    disc.greaterThanOrEqual(0),
    b.mul(-2).div(a.mul(3).add(disc.safeSqrt().mul(a))),
    a.mul(-0.25),
  );

  const aPrime = ifTruthyElse(a, a.add(s.mul(4)), a);
  const bPrime = ifTruthyElse(a, b.add(s.mul(3).mul(a.add(s.mul(2)))), b);
  const cPrime = ifTruthyElse(
    a,
    c.add(s.mul(b.mul(2).add(s.mul(a.mul(3).add(s.mul(4)))))),
    c,
  );
  const dPrime = ifTruthyElse(
    a,
    d.add(s.mul(c.add(s.mul(b.add(s.mul(a.add(s))))))),
    d,
  );

  const gPrime = aPrime
    .mul(cPrime)
    .sub(dPrime.mul(4))
    .sub(bPrime.mul(bPrime).mul(1 / 3));

  const hPrime = aPrime
    .mul(cPrime)
    .add(dPrime.mul(8))
    .sub(bPrime.mul(bPrime).mul(2 / 9))
    .mul(1 / 3)
    .mul(bPrime)
    .sub(cPrime.mul(cPrime))
    .sub(aPrime.mul(aPrime).mul(dPrime));

  const phi = depressedCubicDominant(gPrime, hPrime);

  const l1 = a.mul(0.5);
  const l3 = b.mul(1 / 6).add(phi.mul(0.5));
  const delt2 = c.sub(a.mul(l3));
  const d2Cand1 = b
    .mul(2 / 3)
    .sub(phi)
    .sub(l1.mul(l1));

  const l2Cand1 = delt2.mul(0.5).div(d2Cand1);
  const l2Cand2 = d.sub(l3.mul(l3)).mul(2).div(delt2);
  const d2Cand2 = delt2.mul(0.5).div(l2Cand2);

  let d2Best = asNum(0);
  let l2Best = asNum(0);
  let epsLBest = asNum(0);

  // we might just pick one to make the logic simpler
  for (let i = 0; i < 3; i++) {
    const d2 = i == 1 ? d2Cand2 : d2Cand1;
    const l2 = i == 0 ? l2Cand1 : l2Cand2;

    const eps0 = eps_rel(d2.add(l1.mul(l1)).add(l3.mul(2)), b);
    const eps1 = eps_rel(d2.mul(l2).add(l1.mul(l3)).mul(2), c);
    const eps2 = eps_rel(d2.mul(l2).mul(l2).add(l3.mul(l3)), d);
    const epsL = eps0.add(eps1).add(eps2);

    if (i == 0) {
      d2Best = d2;
      l2Best = l2;
      epsLBest = epsL;
    } else {
      const currentIsBest = epsL.lessThan(epsLBest);
      d2Best = ifTruthyElse(currentIsBest, d2, d2Best);
      l2Best = ifTruthyElse(currentIsBest, l2, l2Best);
      epsLBest = ifTruthyElse(currentIsBest, epsL, epsLBest);
    }
  }
  const d2 = d2Best;
  const l2 = l2Best;

  // d2 < 0
  const sq = d2.neg().safeSqrt();

  const alpha1Case1 = l1.add(sq);
  const beta1Case1 = l3.add(sq.mul(l2));
  const alpha2Case1 = l1.sub(sq);
  const beta2Case1 = l3.sub(sq.mul(l2));

  // d2 == 0

  const d3 = d.sub(l3.square());

  const alpha1Case2 = l1;
  const beta1Case2 = l3.add(d3.neg().safeSqrt());
  const alpha2Case2 = l1;
  const beta2Case2 = l3.sub(d3.neg().safeSqrt());

  const d2Neg = d2.lessThan(ZERO);

  let alpha1 = ifTruthyElse(d2Neg, alpha1Case1, alpha1Case2);
  let beta1 = ifTruthyElse(d2Neg, beta1Case1, beta1Case2);
  let alpha2 = ifTruthyElse(d2Neg, alpha2Case1, alpha2Case2);
  let beta2 = ifTruthyElse(d2Neg, beta2Case1, beta2Case2);

  const beta1Bigger = beta1.abs().greaterThan(beta2.abs());
  const beta2Bigger = beta2.abs().greaterThan(beta1.abs());

  beta1 = ifTruthyElse(beta2Bigger, d.div(beta2), beta1);
  beta2 = ifTruthyElse(beta1Bigger, d.div(beta1), beta2);

  [alpha1, beta1, alpha2, beta2] = newtonForQuadraticCoeffs(
    alpha1,
    beta1,
    alpha2,
    beta2,
    a,
    b,
    c,
    d,
    1, // This means that we are not using newton iterations. I keep the code around
  );
  const solutions = doubleQuadraticSolutions(
    alpha1,
    beta1,
    alpha2,
    beta2,
    sentinelle,
  );

  const d2Pos = d2.greaterThan(EPS_M);

  const cubicSolutions = solveCubic(c3, c2, c1, c0);

  return solutions
    .map((solution) => {
      return ifTruthyElse(d2Pos, sentinelle, solution);
    })
    .map((solution, i) => {
      const cubicSolution = cubicSolutions[i % 3];
      return ifTruthyElse(c4, solution, cubicSolution);
    });
}

function newtonIteration(
  x0: Num,
  f: (x: Num) => Num,
  df: (x: Num) => Num,
  iterations: number,
) {
  let x = x0;
  let fX = f(x);

  for (let i = 0; i < iterations; i++) {
    const df_x = df(x);
    const newX = x.sub(fX.div(df_x));
    const newFX = f(newX);

    const notConverged = df_x.and(newFX).and(newFX.abs().lessThan(fX.abs()));

    x = ifTruthyElse(notConverged, newX, x);
    fX = ifTruthyElse(notConverged, newFX, fX);
  }
  return x;
}

export function depressedCubicDominant(g: Num, h: Num) {
  const q = g.mul(-1 / 3);
  const r = h.mul(0.5);

  const q3 = q.square().mul(q);
  const discriminant = r.square().sub(q3);

  // positive discriminant
  const sqrtDiscriminant = discriminant.safeSqrt();
  const a = r.neg().sub(copysign(sqrtDiscriminant, r)).cbrt();
  const b = ifTruthyElse(a, q.div(a), ZERO);
  const phi0Pos = a.add(b);

  // negative discriminant
  const t = r.div(q3.safeSqrt());
  const phi0Neg = q
    .safeSqrt()
    .mul(-2)
    .mul(
      copysign(
        t
          .abs()
          .min(1)
          .acos()
          .mul(1 / 3)
          .cos(),
        t,
      ),
    );

  const phi0 = ifTruthyElse(discriminant.lessThan(ZERO), phi0Neg, phi0Pos);

  const fun = (x: Num) => x.square().add(g).mul(x).add(h);
  const dfun = (x: Num) => x.square().mul(3).add(g);

  const value = newtonIteration(phi0, fun, dfun, 0); // no newton iterations, the code is kept here if we need it at some point

  const greatestInput = h.max(phi0.square().mul(phi0)).max(g.mul(phi0));
  const isUnderflow = fun(phi0).lessThan(EPS_M.mul(greatestInput));

  return ifTruthyElse(isUnderflow, phi0, value);
}
