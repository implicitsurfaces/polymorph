import { test, expect } from "vitest";
import {
  rotationTransform,
  scalingTransform,
  translationTransform,
} from "./transforms-2d";
import { angleFromDeg, Point, Vec2 } from "./geom";
import { asNum } from "./num";
import { simpleEval } from "./num-tree";

const evalPoint = (point: Point) => {
  return [simpleEval(point.x.n), simpleEval(point.y.n)];
};

const v = (x: number, y: number) => new Vec2(asNum(x), asNum(y));
const p = (x: number, y: number) => new Point(asNum(x), asNum(y));

const c = (input: number[]) => input.map((i) => expect.closeTo(i, 5));

test("translate", () => {
  const translation = translationTransform(v(1, 2));

  const p2 = translation.apply(p(1, 1));
  expect(evalPoint(p2)).toEqual(c([2, 3]));

  const p3 = translation.apply(p(2, 2));
  expect(evalPoint(p3)).toEqual(c([3, 4]));
});

test("translate twice", () => {
  const translation = translationTransform(v(1, 2)).followedBy(
    translationTransform(v(3, 4)),
  );

  const p2 = translation.apply(p(1, 1));
  expect(evalPoint(p2)).toEqual(c([5, 7]));
});

test("rotate", () => {
  const rotation = rotationTransform(angleFromDeg(15));

  const p2 = rotation.apply(p(1, 0));
  expect(evalPoint(p2)).toEqual(c([0.965925, 0.258819]));
});

test("double rotation", () => {
  const rotation = rotationTransform(angleFromDeg(45)).followedBy(
    rotationTransform(angleFromDeg(-30)),
  );

  const p2 = rotation.apply(p(1, 0));
  expect(evalPoint(p2)).toEqual(c([0.965925, 0.258819]));
});

test("translate and rotate", () => {
  const transform = translationTransform(v(1, 2)).followedBy(
    rotationTransform(angleFromDeg(45)),
  );

  const p2 = transform.apply(p(1, 0));
  expect(evalPoint(p2)).toEqual(c([0, 2.828427]));

  const transform2 = rotationTransform(angleFromDeg(45)).followedBy(
    translationTransform(v(1, 2)),
  );
  const p3 = transform2.apply(p(1, 0));
  expect(evalPoint(p3)).toEqual(c([1.707107, 2.707107]));
});

test("scale", () => {
  const transform = scalingTransform(asNum(2), asNum(3));

  const p2 = transform.apply(p(0.5, 1));
  expect(evalPoint(p2)).toEqual(c([1, 3]));
});
