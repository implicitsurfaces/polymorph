import { Angle, Point } from "./geom";
import { Num, ONE, ZERO } from "./num";
import { clamp, hypot, ifTruthyElse, lessThan } from "./num-ops";
import { DistField } from "./types";

function distToWidth(width: Num, x: Num): Num {
  return x.abs().sub(width.div(2));
}

export function staticWidth(width: Num): (t: Num) => Num {
  return () => width;
}

export function linearWidthVariation(
  startWidth: Num,
  endWidth: Num,
): (t: Num) => Num {
  const widthDiff = endWidth.sub(startWidth);
  return (t) => {
    return startWidth.add(widthDiff.mul(t));
  };
}

export class LinearExtrusion2D implements DistField {
  constructor(
    public readonly height: Num,
    public readonly widthAtT: (t: Num) => Num,
  ) {}

  distanceTo(p: Point): Num {
    // Note that we are by default in a standardized coordinate system where
    // the extrusion is along the positive y-axis.

    const halfHeight = this.height.div(2);
    const yDistance = p.y.sub(halfHeight).abs().sub(halfHeight);

    const t = clamp(p.y.div(this.height), ZERO, ONE);
    const xDistance = distToWidth(this.widthAtT(t), p.x);

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

export class OrientedLine {
  constructor(public readonly length: Num) {}

  distanceTo(p: Point): Num {
    const xDist = p.x.abs().sub(this.length.div(2));

    const positiveDist = hypot(xDist.max(0), p.y);
    const negativeDist = hypot(xDist, p.y);
    return ifTruthyElse(lessThan(p.y, 0), negativeDist, positiveDist);
  }
}

function orientedEndCap(p: Point, length: Num): Num {
  const xDist = p.x.abs().sub(length.div(2));
  const positiveDist = hypot(xDist.max(0), p.y);
  const negativeDist = hypot(xDist, p.y);
  return ifTruthyElse(lessThan(p.y, 0), negativeDist, positiveDist);
}

export class ArcExtrusion2D implements DistField {
  private readonly arcLength: Num;
  constructor(
    public readonly radius: Num,
    public readonly angle: Angle,
    public readonly widthAtT: (t: Num) => Num,
  ) {
    this.arcLength = this.angle.asUnitArcLength();
  }

  distanceTo(p: Point): Num {
    const rotationCenter = new Point(this.radius.neg(), ZERO);
    const pointAngle = p.vecFrom(rotationCenter).asAngle();

    const withinAngle = lessThan(
      pointAngle.asSortValue(),
      this.angle.asSortValue(),
    );

    const unconstrainedParam = pointAngle.asUnitArcLength().div(this.arcLength);
    const param = clamp(unconstrainedParam, ZERO, ONE);

    const radiasPos = p.vecFrom(rotationCenter).norm().sub(this.radius);
    const width = this.widthAtT(param);

    const radialDistance = distToWidth(width, radiasPos);

    const widthAtStart = this.widthAtT(ZERO);
    const widthAtEnd = this.widthAtT(ONE);

    const invertedP = new Point(p.x, p.y.neg());
    const rotatedP = rotationCenter.add(
      p.vecFrom(rotationCenter).rotate(this.angle.neg()),
    );
    const outsideEndCapsDistance = orientedEndCap(invertedP, widthAtStart)
      .min(orientedEndCap(rotatedP, widthAtEnd))
      .mul(ONE.sub(withinAngle));

    const rotatedAndInvertedP = rotationCenter.add(
      invertedP.vecFrom(rotationCenter).rotate(this.angle),
    );
    const insideEndCapsDistance = orientedEndCap(p, widthAtStart)
      .min(orientedEndCap(rotatedAndInvertedP, widthAtEnd))
      .neg();

    return radialDistance
      .max(insideEndCapsDistance)
      .mul(withinAngle)
      .add(outsideEndCapsDistance);
  }
}
