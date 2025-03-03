import { writeFileSync } from "node:fs";
import {
  asNum,
  Num,
  NumX,
  NumY,
  ONE,
  TWO,
  NEG_ONE,
  variable,
} from "../src/num";
import { Point } from "../src/geom";
import { numNodeToFPCore } from "../src/utils/num-to-fpcore";

import { circleConic } from "../src/conic";
import { Circle } from "../src/profiles";
import { solveQuadratic } from "../src/geom-utils/solve-polynomial";

function dumpForHerbie(name: string, n: Num) {
  writeFileSync(`${name}.fpcore`, numNodeToFPCore(n.n, name));
}

function distanceToCircleConic() {
  const r = variable("r");
  const conic = circleConic(r);
  return conic.distanceTo(new Point(NumX, NumY));
}

function basicDistanceToCircle() {
  const circle = new Circle(asNum(0.6));
  return circle.distanceTo(new Point(NumX, NumY));
}

function quadraticSolver() {
  const a = variable("a");
  const b = variable("b");
  const c = variable("c");
  return solveQuadratic(a, b, c)[0];
}

dumpForHerbie("quadratic-solver", quadraticSolver());
