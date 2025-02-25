import { test, expect } from "vitest";
import { circleConic, ellipseConic, genericConic } from "./conic";
import { expectASCIIDistance } from "./test-utils";
import { asNum } from "./num";
import { angleFromDeg, Point, Vec2 } from "./geom";
import { evaluate } from "./utils/evaluate";

test("circle as a conic", async () => {
  const circle = circleConic(asNum(0.6));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

test("ellipse as a conic", async () => {
  const circle = ellipseConic(asNum(0.6), asNum(0.2));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

/*
test("check points on ellipse", async () => {
  const e = ellipseConic(asNum(0.6), asNum(0.6));
  const candidates = e.candidatePoints(new Point(asNum(0.7), asNum(0.7)));

  console.log(candidates.map((p) => [simpleEval(p.x.n), simpleEval(p.y.n)]));

  const evalP = (p: Point) => [simpleEval(p.x.n), simpleEval(p.y.n)];
  console.log(candidates.map(evalP));
  console.log(evalMatrix(e._matrix));

  throw new Error("stop");
});
*/
test("extracting the radiuses", () => {
  const e = genericConic(
    asNum(0.2),
    asNum(4),
    angleFromDeg(0),
    new Point(asNum(0), asNum(0)),
  );

  expect(evaluate(e.radiuses)).toEqual([4, 0.2]);
});

test("extracting the translation", async () => {
  const center = [2, 1];
  const e = genericConic(
    asNum(0.5),
    asNum(0.6),
    angleFromDeg(0),
    new Point(asNum(center[0]), asNum(center[1])),
  );

  expect(evaluate(e.center)).toEqual(center.map((v) => expect.closeTo(v, 4)));
});

function positiveOrientation(angle: number) {
  return angle < 0 ? angle + 180 : angle;
}

test("extracting the tilt", async () => {
  const center = [0, 0];
  const angle = 55;
  const e = genericConic(
    asNum(3),
    asNum(0.5),
    angleFromDeg(angle),
    new Point(asNum(center[0]), asNum(center[1])),
  );

  expect(positiveOrientation(evaluate(e.tilt))).toBeCloseTo(angle);
});
