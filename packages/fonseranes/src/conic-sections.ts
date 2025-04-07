import { Conic2D, HalfPlane2D, Line2D, Point2D, Segment2D } from "./primitives";
import { rawTransform2D, scalingTransform2D, translate } from "./transforms";

export function circle(radius: number) {
  const transform = scalingTransform2D(1 / radius, 1 / radius);
  return new Conic2D(transform);
}

export function ellipse(xRadius: number, yRadius: number) {
  const transform = scalingTransform2D(1 / xRadius, 1 / yRadius);
  return new Conic2D(transform);
}

export function hyperbola(
  xSemiAxis: number,
  ySemiAxis: number,
  orientation: "horizontal" | "vertical" = "horizontal",
) {
  let transform;

  if (orientation === "horizontal") {
    transform = rawTransform2D(
      1 / xSemiAxis,
      0,
      0,
      0,
      0,
      1,
      0,
      1 / ySemiAxis,
      0,
    );
  } else {
    transform = rawTransform2D(
      0,
      0,
      1,
      0,
      1 / ySemiAxis,
      0,
      1 / xSemiAxis,
      0,
      0,
    );
  }

  return new Conic2D(transform);
}

export function parabola(
  focusDistance: number,
  orientation: "vertical" | "horizontal" = "vertical",
) {
  let transform;

  if (orientation === "vertical") {
    // For a vertical parabola with focus at (0, p), the equation is y = x²/(4p)
    // Using the quadratic form T^T Q T where Q = diag(1, 1, -1)
    // The transformation matrix T produces the desired parabola
    // The resulting transform parameters are:
    transform = rawTransform2D(
      Math.sqrt(focusDistance),
      0,
      0,
      0,
      1 / 2,
      -1 / 2,
      0,
      1 / 2,
      1 / 2,
    );
  } else {
    // For a horizontal parabola with focus at (p, 0), the equation is x = y²/(4p)
    // After computing T^T Q T for this case
    // The resulting transform parameters are:
    transform = rawTransform2D(
      0.5,
      0,
      -0.5,
      0,
      Math.sqrt(focusDistance),
      0,
      0.5,
      0,
      0.5,
    );
  }

  return new Conic2D(transform);
}

export function arc(startPoint: Point2D, endPoint: Point2D, bulge: number) {
  const chord = startPoint.vecTo(endPoint);
  const chordPerp = chord.perp();

  const line = Line2D.fromPoints(startPoint, endPoint);

  if (bulge === 0) {
    const startLimit = Line2D.fromPointAndVec(startPoint, chordPerp);
    const endLimit = Line2D.fromPointAndVec(endPoint, chordPerp);

    return new Segment2D(line, [
      new HalfPlane2D(startLimit, true),
      new HalfPlane2D(endLimit, false),
    ]);
  }

  const bb = (bulge - 1 / bulge) / 4;

  const center = startPoint.midPoint(endPoint).sub(chordPerp.scale(bb));
  const radius = Math.abs((chord.magnitude() / 4) * (bulge + 1 / bulge));

  const baseCircle = translate(circle(radius), center);
  const halfPlane = new HalfPlane2D(line, bulge > 0);

  return new Segment2D(baseCircle, [halfPlane]);
}
