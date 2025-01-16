import { Point } from "./geom";
import { embedPoint, Plane } from "./geom-3d";
import { closestPointOnEllipse } from "./geom-utils/closestPointOnEllipse";
import { Num, ONE, asNum } from "./num";
import { hypot, max, min } from "./num-ops";
import { Segment, SolidDistField } from "./types";

export class Circle {
  readonly radius: Num;
  constructor(radius: Num | number) {
    this.radius = asNum(radius);
  }

  distanceTo(point: Point): Num {
    return point.vecFromOrigin().norm().sub(this.radius);
  }
}

export class Ellipse {
  readonly majorRadius: Num;
  readonly minorRadius: Num;

  constructor(majorRadius: Num | number, minorRadius: Num | number) {
    this.majorRadius = asNum(majorRadius);
    this.minorRadius = asNum(minorRadius);
  }

  distanceTo(point: Point): Num {
    const closestPoint = closestPointOnEllipse(
      this.majorRadius,
      this.minorRadius,
      point,
    );
    const sign = point
      .vecFromOrigin()
      .norm()
      .sub(closestPoint.vecFromOrigin().norm())
      .sign();
    return point.vecFrom(closestPoint).norm().mul(sign);
  }
}

export class Box {
  readonly width: Num;
  readonly height: Num;
  constructor(width: Num | number, height: Num | number) {
    this.width = asNum(width);
    this.height = asNum(height);
  }

  distanceTo(point: Point): Num {
    const halfWidth = this.width.div(2);
    const halfHeight = this.height.div(2);

    const qX = point.x.smoothabs().sub(halfWidth);
    const qY = point.y.smoothabs().sub(halfHeight);

    return hypot(qX.softplus(), qY.softplus()).add(max(qX, qY).softminus());
  }
}

export class TopHalfPlane {
  distanceTo(point: Point): Num {
    return point.y;
  }
}

export class BottomHalfPlane {
  distanceTo(point: Point): Num {
    return point.y.neg();
  }
}

export class LeftHalfPlane {
  distanceTo(point: Point): Num {
    return point.x;
  }
}

export class RightHalfPlane {
  distanceTo(point: Point): Num {
    return point.x.neg();
  }
}

export class ClosedPath {
  readonly segments: Segment[];
  constructor(segments: Segment[]) {
    this.segments = segments;
  }

  distanceTo(point: Point): Num {
    const distances = this.segments.map((segment) => segment.distanceTo(point));
    const dist = min(distances[0], ...distances.slice(1));

    const windingNumber = this.segments
      .map((segment) => segment.solidAngle(point).turns)
      .reduce((a, b) => a.add(b), asNum(0));
    const insideSign = ONE.sub(windingNumber.abs().min(1).mul(2));

    return dist.mul(insideSign);
  }
}

export class OpenPath {
  readonly segments: Segment[];
  constructor(segments: Segment[]) {
    this.segments = segments;
  }

  distanceTo(point: Point): Num {
    const distances = this.segments.map((segment) => segment.distanceTo(point));
    return min(distances[0], ...distances.slice(1));
  }
}

export class SolidSlice {
  constructor(
    public readonly solid: SolidDistField,
    public readonly plane: Plane,
  ) {}

  distanceTo(point: Point): Num {
    return this.solid.valueAt(embedPoint(point, this.plane));
  }
}
