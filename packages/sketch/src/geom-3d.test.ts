import { test, describe, expect } from "vitest";

import { simpleEval } from "./num-tree";
import {
  distanceToLine,
  intersectLinePlane,
  ORIGIN,
  Plane,
  Point3D,
  X_AXIS,
  Y_AXIS,
  Z_AXIS,
} from "./geom-3d";
import { asNum } from "./num";

const round = (num: number, places = 5) => {
  const shift = 10 ** places;
  return Math.round(num * shift) / shift;
};

const expectPoint3D = (p: Point3D) => {
  return expect([
    round(simpleEval(p.x.n)),
    round(simpleEval(p.y.n)),
    round(simpleEval(p.z.n)),
  ]);
};

const p = (x: number, y: number, z: number) =>
  new Point3D(asNum(x), asNum(y), asNum(z));

describe("intersect line plane", () => {
  test("simple", () => {
    const plane = new Plane(ORIGIN, Z_AXIS, X_AXIS);

    const lineOrigin = p(0, 0, 2);
    const lineDirection = p(0, 0, -1).vecFromOrigin();

    const intersection = intersectLinePlane(lineOrigin, lineDirection, plane);

    expectPoint3D(intersection).toEqual([0, 0, 0]);
  });

  test("with angle in y", () => {
    const plane = new Plane(ORIGIN, Z_AXIS, X_AXIS);

    const lineOrigin = p(0, 0, 2);
    const lineDirection = p(0, 1, -1).vecFromOrigin();

    const intersection = intersectLinePlane(lineOrigin, lineDirection, plane);
    expectPoint3D(intersection).toEqual([0, 2, 0]);
  });

  test("with angle in x and y", () => {
    const plane = new Plane(ORIGIN, Z_AXIS, X_AXIS);

    const lineOrigin = p(0, 0, 2);
    const lineDirection = p(1, 1, -1).vecFromOrigin();

    const intersection = intersectLinePlane(lineOrigin, lineDirection, plane);
    expectPoint3D(intersection).toEqual([2, 2, 0]);
  });

  test("angled plane", () => {
    const dir = p(1, 1, 1).vecFromOrigin().normalize();
    const plane = new Plane(ORIGIN, dir, dir.cross(Y_AXIS));

    const lineOrigin = p(0, 0, 4);
    const lineDirection = p(0.5, 1, -1).vecFromOrigin();

    const intersection = intersectLinePlane(lineOrigin, lineDirection, plane);
    expectPoint3D(intersection).toEqual([-4, -8, 12]);
  });
});

describe("distanceToLine", () => {
  test("simple", () => {
    const lineOrigin = p(0, 0, 0);
    const lineDirection = p(0, 0, 1).vecFromOrigin().normalize();

    const point = p(1, 1, 1);

    const distance = distanceToLine(lineOrigin, lineDirection, point);
    expect(round(simpleEval(distance.n))).toEqual(1.41421);
  });

  test("point on line", () => {
    const lineOrigin = p(1, 1, 1);
    const lineDirection = p(2, 2, 2).vecFromOrigin().normalize();
    const point = p(1, 1, 1); // Same as origin
    const distance = distanceToLine(lineOrigin, lineDirection, point);
    expect(round(simpleEval(distance.n))).toEqual(0);
  });

  test("perpendicular distance", () => {
    const lineOrigin = p(0, 0, 0);
    const lineDirection = p(1, 0, 0).vecFromOrigin().normalize();
    const point = p(0, 3, 0);
    const distance = distanceToLine(lineOrigin, lineDirection, point);
    expect(round(simpleEval(distance.n))).toEqual(3);
  });

  test("arbitrary angle", () => {
    const lineOrigin = p(1, 1, 1);
    const lineDirection = p(2, 3, 4).vecFromOrigin().normalize();
    const point = p(5, 5, 5);
    const distance = distanceToLine(lineOrigin, lineDirection, point);
    // This value would need to be adjusted based on your actual calculation
    expect(round(simpleEval(distance.n))).toBeCloseTo(1.81944);
  });
});
