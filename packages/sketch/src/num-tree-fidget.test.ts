import { describe, test, expect } from "vitest";

import { fidgetEval, fidgetRender } from "./num-tree-fidget";
import { asNum, Num } from "./num";
import { Box, Circle } from "./profiles";
import { vecFromCartesianCoords } from "./geom";

const expectToBe = async (node: Num, value: number) =>
  expect(await fidgetEval(node.n)).toBeCloseTo(value);

describe("fidgetEval", () => {
  test("simple sum", async () => {
    await expectToBe(asNum(40).add(2), 42);
  });

  test("simple product", async () => {
    await expectToBe(asNum(6).mul(7), 42);
  });

  test("simple division", async () => {
    await expectToBe(asNum(84).div(2), 42);
  });

  test("simple subtraction", async () => {
    await expectToBe(asNum(44).sub(2), 42);
  });

  test("min", async () => {
    expectToBe(asNum(42).min(43), 42);
  });

  test("max", async () => {
    await expectToBe(asNum(42).max(41), 42);
  });

  test("sign", async () => {
    await expectToBe(asNum(42).sign(), 1);
    await expectToBe(asNum(-42).sign(), -1);
    await expectToBe(asNum(0).sign(), 0);
  });

  test("distance to a circle", async () => {
    const circle = new Circle(asNum(3));
    const p1 = vecFromCartesianCoords(asNum(0), asNum(5)).pointFromOrigin();
    const p2 = vecFromCartesianCoords(asNum(0), asNum(1)).pointFromOrigin();

    await expectToBe(circle.distanceTo(p1), 2);
    await expectToBe(circle.distanceTo(p2), -2);
  });

  test("distance to a box", async () => {
    const box = new Box(asNum(3), asNum(4));
    const p1 = vecFromCartesianCoords(asNum(0), asNum(5)).pointFromOrigin();
    const p2 = vecFromCartesianCoords(asNum(1), asNum(0)).pointFromOrigin();

    await expectToBe(box.distanceTo(p1), 3);
    await expectToBe(box.distanceTo(p2), -0.5);
  });
});

describe("fidgetRender", async () => {
  test("distance to a circle", async () => {
    const circle = new Circle(asNum(0.5));

    console.log(await fidgetRender(circle));

    throw new Error("Not implemented");
  });
});
