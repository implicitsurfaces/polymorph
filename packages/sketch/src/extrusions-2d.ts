import { Point } from "./geom";
import { Num, ONE, ZERO } from "./num";
import { clamp, hypot } from "./num-ops";
import { DistField } from "./types";

export function staticWidth(width: Num): (x: Num, t: Num) => Num {
  const halfWidth = width.div(2);
  return (x) => x.abs().sub(halfWidth);
}

export function linearWidthVariation(
  startWidth: Num,
  endWidth: Num,
): (x: Num, t: Num) => Num {
  const widthDiff = endWidth.sub(startWidth);
  return (x, t) => {
    const currentWidth = startWidth.add(widthDiff.mul(t));
    return x.abs().sub(currentWidth.div(2));
  };
}

export class LinearExtrusion2D implements DistField {
  constructor(
    public readonly height: Num,
    public readonly widthAtT: (x: Num, t: Num) => Num,
  ) {}

  distanceTo(p: Point): Num {
    // Note that we are by default in a standardized coordinate system where
    // the extrusion is along the positive y-axis.

    const halfHeight = this.height.div(2);
    const yDistance = p.y.sub(halfHeight).abs().sub(halfHeight);

    const t = clamp(p.y.div(this.height), ZERO, ONE);
    const xDistance = this.widthAtT(p.x, t);

    // This is non zero only  when both distance are negative (i.e. inside)
    // In that case we choose the closest distance (the max of two negative numbers)
    const insideDistance = xDistance.max(yDistance).min(0);

    // this is non zero only when at least one distance is positive (i.e. outside)
    // In that case we take the norm of the two distances as xy_dist is (x^2 + y^2)^0.5
    // norm(xy_distance, z_distance) is (x^2 + y^2 + z^2)^0.5
    const xOutsideDistance = xDistance.max(0);
    const yOutsideDistance = yDistance.max(0);
    const outsideDistance = hypot(xOutsideDistance, yOutsideDistance);

    return insideDistance.add(outsideDistance);
  }
}
