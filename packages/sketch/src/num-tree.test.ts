import { describe, test, expect } from "vitest";
import { asNum, variable } from "./num";
import { allVariables, replaceVariable } from "./num-tree";

describe("all variables", () => {
  test("finds no variables in a Num without vars", () => {
    const num = asNum(1).sub(asNum(2)).add(asNum(3));
    expect(allVariables(num.n)).toEqual(new Set([]));
  });

  test("finds a single variable", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variable("x").mul(asNum(5)));
    expect(allVariables(num.n)).toEqual(new Set(["x"]));
  });
});

describe("Replace variable", () => {
  test("replaces a single variable", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variable("x").mul(asNum(5)));

    const replaced = replaceVariable(num.n, new Map([["x", 10]]));
    expect(allVariables(replaced)).toEqual(new Set([]));
  });

  test("replaces keep unspecified variables", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variable("x").mul(variable("y")));

    const replaced = replaceVariable(num.n, new Map([["x", 10]]));
    expect(allVariables(replaced)).toEqual(new Set(["y"]));
  });
});
