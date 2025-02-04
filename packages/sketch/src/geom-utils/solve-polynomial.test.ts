import { describe, test, expect } from "vitest";

import { naiveEval } from "../num-tree";
import { fidgetEval } from "../num-tree-fidget";
import { solveCubic, solveQuadratic, solveQuartic } from "./solve-polynomial";
import { asNum } from "../num";

//import { treeEval } from "../num-tree";
//import { renderNodeAsDot } from "../utils/num-to-dot";
//import fs from "node:fs";

describe("solveQuadratic", () => {
  const solve = (a: number, b: number, c: number) => {
    const [x1, x2] = solveQuadratic(asNum(a), asNum(b), asNum(c));
    //fs.writeFileSync("error.dot", renderNodeAsDot(treeEval(x1.n)));
    return new Set([naiveEval(x1.n, new Map()), naiveEval(x2.n, new Map())]);
  };

  test("two roots", () => {
    expect(solve(1, -5, 6)).toEqual(new Set([2, 3]));
  });

  test("one root", () => {
    expect(solve(1, -4, 4)).toEqual(new Set([2]));
  });

  test("one root, positive b", () => {
    expect(solve(1, 2, 1)).toEqual(new Set([-1]));
  });

  test("a is zero", () => {
    expect(solve(0, 1, 2)).toEqual(new Set([-2]));
  });

  test("no roots", () => {
    expect(solve(1, 1, 1)).toEqual(new Set([NaN]));
  });

  test("no roots, b is zero", () => {
    expect(solve(1, 0, 1)).toEqual(new Set([NaN]));
  });

  test("c is zero", () => {
    expect(solve(1, 1, 0)).toEqual(new Set([0, -1]));
  });
});

function round(x: number, digits = 6) {
  const factor = Math.pow(10, digits);
  return Math.round(x * factor) / factor;
}

describe("solveCubic", () => {
  const solve = (a: number, b: number, c: number, d: number) => {
    const [x1, x2, x3] = solveCubic(asNum(a), asNum(b), asNum(c), asNum(d));
    //fs.writeFileSync("error.dot", renderNodeAsDot(treeEval(x1.n)));
    return new Set([
      round(naiveEval(x1.n, new Map())),
      round(naiveEval(x2.n, new Map())),
      round(naiveEval(x3.n, new Map())),
    ]);
  };

  test("one multiple root", () => {
    expect(solve(1, -3, 3, -1)).toEqual(new Set([1]));
  });

  test("one single real root", () => {
    expect(solve(1, 0, -1, -1)).toEqual(new Set([round(1.324718)]));
  });

  test("two roots", () => {
    expect(solve(1, -5, 8, -4)).toEqual(new Set([1, 2]));
  });

  test("three roots", () => {
    expect(solve(1, -6, 11, -6)).toEqual(new Set([1, 2, 3]));
  });

  test("one simple root", () => {
    expect(solve(1, 0, 0, 0)).toEqual(new Set([0]));
  });

  test("a quadratic equation", () => {
    expect(solve(0, 1, -5, 6)).toEqual(new Set([2, 3]));
  });

  test("one real roots", () => {
    expect(solve(1, 1, 1, 1)).toEqual(new Set([-1]));
  });

  test("single real root, buggy", () => {
    expect(solve(1, -1, 3, -4)).toEqual(new Set([round(1.222494514)]));
  });
});

describe("solveQuartic", () => {
  const solve = (a: number, b: number, c: number, d: number, e: number) => {
    const [x1, x2, x3, x4] = solveQuartic(
      asNum(a),
      asNum(b),
      asNum(c),
      asNum(d),
      asNum(e),
    );
    return new Set([
      round(naiveEval(x1.n, new Map())),
      round(naiveEval(x2.n, new Map())),
      round(naiveEval(x3.n, new Map())),
      round(naiveEval(x4.n, new Map())),
    ]);
  };
  test("two roots", () => {
    expect(solve(1, -6, 11, -6, 1)).toEqual(
      new Set([round(0.38196601), round(2.618033988)]),
    );
  });

  test("three roots", () => {
    expect(solve(1, -5, 8, -4, 0)).toEqual(new Set([0, 1, 2]));
  });

  test("four roots", () => {
    expect(solve(1, -6, 11, -6, 0)).toEqual(new Set([0, 1, 2, 3]));
    expect(solve(1, -10, 35, -50, 24)).toEqual(new Set([2, 3, 4, 1]));
  });

  test("complex roots only", () => {
    expect(solve(1, 0, 1, 0, 1)).toEqual(new Set([NaN]));
  });

  test("repeated roots", () => {
    expect(solve(1, -4, 6, -4, 1)).toEqual(new Set([1]));
  });

  test("zero coefficient", () => {
    expect(solve(1, 0, -5, 0, 4)).toEqual(new Set([1, -1, 2, -2]));
  });

  test("large coefficients", () => {
    expect(solve(1, -100, 3750, -62500, 390625)).toEqual(new Set([25]));
  });

  test("actually a cubic", () => {
    expect(solve(0, 1, -6, 11, -6)).toEqual(new Set([1, 2, 3]));
  });

  test("two real roots (plus two complex)", () => {
    expect(solve(1, -1, 1, 1, -1)).toEqual(new Set([-0.848375, 0.660993]));
  });
});

describe("solveQuartic with fidget", async () => {
  const solve = async (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
  ) => {
    const [x1, x2, x3, x4] = solveQuartic(
      asNum(a),
      asNum(b),
      asNum(c),
      asNum(d),
      asNum(e),
    );
    return new Set([
      round(await fidgetEval(x1.n)),
      round(await fidgetEval(x2.n)),
      round(await fidgetEval(x3.n)),
      round(await fidgetEval(x4.n)),
    ]);
  };
  test("two roots", async () => {
    expect(await solve(1, -6, 11, -6, 1)).toEqual(
      new Set([round(0.38196601), round(2.618033988)]),
    );
  });

  test("three roots", async () => {
    expect(await solve(1, -5, 8, -4, 0)).toEqual(new Set([0, 1, 2]));
  });

  test("four roots", async () => {
    expect(await solve(1, -6, 11, -6, 0)).toEqual(new Set([0, 1, 2, 3]));
    expect(await solve(1, -10, 35, -50, 24)).toEqual(new Set([2, 3, 4, 1]));
  });

  test("complex roots only", async () => {
    expect(await solve(1, 0, 1, 0, 1)).toEqual(new Set([NaN]));
  });

  test("repeated roots", async () => {
    expect(await solve(1, -4, 6, -4, 1)).toEqual(new Set([1]));
  });

  test("zero coefficient", async () => {
    expect(await solve(1, 0, -5, 0, 4)).toEqual(new Set([1, -1, 2, -2]));
  });

  test("large coefficients", async () => {
    expect(await solve(1, -100, 3750, -62500, 390625)).toEqual(new Set([25]));
  });

  test("actually a cubic", async () => {
    expect(await solve(0, 1, -6, 11, -6)).toEqual(new Set([1, 2, 3]));
  });

  test("two real roots (plus two complex)", async () => {
    expect(await solve(1, -1, 1, 1, -1)).toEqual(
      new Set([-0.848375, 0.660993]),
    );
  });
});
