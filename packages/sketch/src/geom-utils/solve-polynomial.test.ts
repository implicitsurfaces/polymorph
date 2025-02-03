import { describe, test, expect } from "vitest";

import { naiveEval } from "../num-tree";
import { solveQuadratic } from "./solve-polynomial";
import { asNum } from "../num";

//import {  treeEval } from "../num-tree";
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
