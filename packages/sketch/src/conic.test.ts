import { test, expect } from "vitest";
import {
  circleConic,
  ConicProfile,
  ellipseConic,
  genericEllipseConic,
} from "./conic";
import { expectASCIIDistance } from "./test-utils";
import { angleFromDeg, asVec, Point } from "./geom";
import { asNum } from "./num";
import { evaluate } from "./utils/evaluate";
import {
  rawTransform,
  rotationTransform,
  scalingTransform,
  translationTransform,
} from "./transforms-2d";

test("circle as a conic", async () => {
  const circle = circleConic(asNum(0.6));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

test("zoomed in ellipse as a conic", async () => {
  const circle = ellipseConic(asNum(60), asNum(60));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

test("ellipse as a conic", async () => {
  const circle = ellipseConic(asNum(0.6), asNum(0.2));
  (await expectASCIIDistance(circle)).toMatchSnapshot();
});

test("generic ellipse as a conic", async () => {
  const ellipse = new ConicProfile(
    rawTransform(
      asNum(1.2497450777436905),
      asNum(-0.00002498738060815532),
      asNum(-0.002498990407875218),
      asNum(-0.00009994952243262128),
      asNum(4.99799081070268),
      asNum(-0.09995961631500871),
      asNum(0.0019991923263001747),
      asNum(0.019991923263001747),
      asNum(0.9995961631500874),
    ),
  );

  (await expectASCIIDistance(ellipse)).toMatchSnapshot();
});

test("rotated ellipse", async () => {
  const transform = rotationTransform(angleFromDeg(-15)).followedBy(
    scalingTransform(asNum(2), asNum(5)),
  );
  const ellipse = new ConicProfile(transform);
  (await expectASCIIDistance(ellipse)).toMatchSnapshot();
});

test("translated ellipse", async () => {
  const transform = translationTransform(asVec(-0.4, -0.2)).followedBy(
    scalingTransform(asNum(2), asNum(5)),
  );
  const ellipse = new ConicProfile(transform);
  (await expectASCIIDistance(ellipse)).toMatchSnapshot();
});

test("extracting the radiuses", () => {
  const e = genericEllipseConic(
    asNum(0.2),
    asNum(4),
    angleFromDeg(0),
    new Point(asNum(0), asNum(0)),
  );

  expect(evaluate(e.radiuses)).toEqual([4, 0.2]);
});

test("extracting the translation", async () => {
  const center = [2, 1];
  const e = genericEllipseConic(
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
  const e = genericEllipseConic(
    asNum(3),
    asNum(0.5),
    angleFromDeg(angle),
    new Point(asNum(center[0]), asNum(center[1])),
  );

  expect(positiveOrientation(evaluate(e.tilt))).toBeCloseTo(angle);
});
