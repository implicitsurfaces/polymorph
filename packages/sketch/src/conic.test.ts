import { test } from "vitest";
import { circleConic, ellipseConic } from "./conic";
import { expectASCIIDistance } from "./test-utils";
import { asNum } from "./num";
import { Point } from "./geom";
import { simpleEval } from "./num-tree";

test("circle as a conic", async () => {
  const circle = circleConic(asNum(0.6));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

test("ellipse as a conic", async () => {
  const circle = ellipseConic(asNum(0.6), asNum(0.2));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

function evalMatrix(matrix: Matrix3x3) {
  return [
    simpleEval(matrix.x11.n),
    simpleEval(matrix.x12.n),
    simpleEval(matrix.x13.n),
    simpleEval(matrix.x21.n),
    simpleEval(matrix.x22.n),
    simpleEval(matrix.x23.n),
    simpleEval(matrix.x31.n),
    simpleEval(matrix.x32.n),
    simpleEval(matrix.x33.n),
  ];
}

test("check points on ellipse", async () => {
  const e = ellipseConic(asNum(0.6), asNum(0.6));
  const candidates = e.candidatePoints(new Point(asNum(0.7), asNum(0.7)));

  console.log(candidates.map((p) => [simpleEval(p.x.n), simpleEval(p.y.n)]));

  const evalP = (p: Point) => [simpleEval(p.x.n), simpleEval(p.y.n)];
  console.log(candidates.map(evalP));
  console.log(evalMatrix(e._matrix));

  throw new Error("stop");
});
