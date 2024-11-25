import { describe, test, expect } from "vitest";

import { fidgetEval } from "./num-tree-fidget";
import { asNum, Num } from "./num";
import { Box, Circle, ClosedPath, OpenPath } from "./profiles";
import { asVec, vecFromCartesianCoords } from "./geom";
import { expectFidgetRender } from "./test-utils";
import { Dilatation, Translation } from "./sdf-operations";
import { BulgingSegment, LineSegment } from "./segments";
import { Segment } from "./types";

const t = (s: Segment) => new Dilatation(asNum(0.2), new OpenPath([s]));

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
    await expectToBe(asNum(0).min(0), 0);
    await expectToBe(asNum(1).min(0), 0);
    await expectToBe(asNum(0).min(1), 0);
    await expectToBe(asNum(1).min(1), 1);
    await expectToBe(asNum(1).min(-1), -1);
    await expectToBe(asNum(-1).min(1), -1);
  });

  test("max", async () => {
    await expectToBe(asNum(42).max(41), 42);

    await expectToBe(asNum(0).max(0), 0);
    await expectToBe(asNum(1).max(0), 1);
    await expectToBe(asNum(0).max(1), 1);
    await expectToBe(asNum(1).max(1), 1);
    await expectToBe(asNum(1).max(-1), 1);
    await expectToBe(asNum(-1).max(1), 1);
  });

  test("and", async () => {
    await expectToBe(asNum(0).and(0), 0);
    await expectToBe(asNum(1).and(0), 0);
    await expectToBe(asNum(0).and(1), 0);
    await expectToBe(asNum(1).and(1), 1);
    await expectToBe(asNum(1).and(-1), -1);
  });

  test("or", async () => {
    await expectToBe(asNum(0).or(0), 0);
    await expectToBe(asNum(1).or(0), 1);
    await expectToBe(asNum(0).or(1), 1);
    await expectToBe(asNum(1).or(1), 1);
    await expectToBe(asNum(1).or(-1), 1);
    await expectToBe(asNum(-1).or(1), -1);
  });

  test("sign", async () => {
    await expectToBe(asNum(42).sign(), 1);
    await expectToBe(asNum(-42).sign(), -1);
    await expectToBe(asNum(0).sign(), 0);
  });

  test("abs", async () => {
    await expectToBe(asNum(0).abs(), 0);
    await expectToBe(asNum(1.5).abs(), 1.5);
    await expectToBe(asNum(-2).abs(), 2);
  });

  test("compare", async () => {
    await expectToBe(asNum(0).compare(0), 0);
    await expectToBe(asNum(1).compare(0), 1);
    await expectToBe(asNum(0).compare(1), -1);
  });

  test("less than", async () => {
    await expectToBe(asNum(0).lessThan(0), 0);
    await expectToBe(asNum(1).lessThan(0), 0);
    await expectToBe(asNum(0).lessThan(1), 1);
    await expectToBe(asNum(1).lessThan(1), 0);
    await expectToBe(asNum(1).lessThan(-1), 0);
    await expectToBe(asNum(-1).lessThan(1), 1);
  });

  test("less than or equal", async () => {
    await expectToBe(asNum(0).lessThanOrEqual(0), 1);
    await expectToBe(asNum(1).lessThanOrEqual(0), 0);
    await expectToBe(asNum(0).lessThanOrEqual(1), 1);
    await expectToBe(asNum(1).lessThanOrEqual(1), 1);
    await expectToBe(asNum(1).lessThanOrEqual(-1), 0);
    await expectToBe(asNum(-1).lessThanOrEqual(1), 1);
  });

  test("greater than", async () => {
    await expectToBe(asNum(0).greaterThan(0), 0);
    await expectToBe(asNum(1).greaterThan(0), 1);
    await expectToBe(asNum(0).greaterThan(1), 0);
    await expectToBe(asNum(1).greaterThan(1), 0);
    await expectToBe(asNum(1).greaterThan(-1), 1);
    await expectToBe(asNum(-1).greaterThan(1), 0);
  });

  test("greater than or equal", async () => {
    await expectToBe(asNum(0).greaterThanOrEqual(0), 1);
    await expectToBe(asNum(1).greaterThanOrEqual(0), 1);
    await expectToBe(asNum(0).greaterThanOrEqual(1), 0);
    await expectToBe(asNum(1).greaterThanOrEqual(1), 1);
    await expectToBe(asNum(1).greaterThanOrEqual(-1), 1);
    await expectToBe(asNum(-1).greaterThanOrEqual(1), 0);
  });
  test("equals", async () => {
    await expectToBe(asNum(0).equals(0), 1);
    await expectToBe(asNum(1).equals(0), 0);
    await expectToBe(asNum(0).equals(1), 0);
    await expectToBe(asNum(1).equals(1), 1);
    await expectToBe(asNum(1).equals(-1), 0);
    await expectToBe(asNum(-1).equals(1), 0);
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
  test("circle", async () => {
    const circle = new Circle(asNum(0.5));
    (await expectFidgetRender(circle)).toMatchSnapshot();
  });

  test("positionned circle", async () => {
    let circle = new Translation(asVec(-0.5, 0.5), new Circle(asNum(0.1)));
    (await expectFidgetRender(circle)).toMatchSnapshot();

    circle = new Translation(asVec(0.5, 0.5), new Circle(asNum(0.1)));
    (await expectFidgetRender(circle)).toMatchSnapshot();

    circle = new Translation(asVec(0.5, -0.5), new Circle(asNum(0.1)));
    (await expectFidgetRender(circle)).toMatchSnapshot();
  });

  test("box", async () => {
    const box = new Box(asNum(1.9), asNum(0.4));
    (await expectFidgetRender(box)).toMatchSnapshot();
  });

  test("pill", async () => {
    const p0 = asVec(-0.5, -0.3).pointFromOrigin();
    const p1 = asVec(0.3, 0.5).pointFromOrigin();
    const pill = t(new LineSegment(p0, p1));

    (await expectFidgetRender(pill)).toMatchSnapshot();
  });

  test("arc pill", async () => {
    const p0 = asVec(-0.5, -0.3).pointFromOrigin();
    const p1 = asVec(0.3, 0.5).pointFromOrigin();
    const pill = t(new BulgingSegment(p0, p1, asNum(0.9)));

    (await expectFidgetRender(pill)).toMatchSnapshot();

    throw new Error("Does not look right, fix it!");
  });

  test("basic heart shape", async () => {
    const p0 = asVec(0, -0.8).pointFromOrigin();
    const p1 = asVec(-0.9, 0.3).pointFromOrigin();
    const p2 = asVec(0, 0.3).pointFromOrigin();
    const p3 = asVec(0.9, 0.3).pointFromOrigin();

    const heart = new ClosedPath([
      new LineSegment(p0, p1),
      new BulgingSegment(p1, p2, asNum(-0.9)),
      new BulgingSegment(p2, p3, asNum(-0.9)),
      new LineSegment(p3, p0),
    ]);

    (await expectFidgetRender(heart)).toMatchSnapshot();
  });
});
