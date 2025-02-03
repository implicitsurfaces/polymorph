import { Angle, Point } from "./geom";
import { embedPoint, Plane } from "./geom-3d";
import {
  closestPointOnEllipse,
  closestPointsOnEllipseArc,
} from "./geom-utils/closestPointOnEllipse";
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

  private sign(point: Point): Num {
    const m = point.x.square().div(this.majorRadius.square());
    const n = point.y.square().div(this.minorRadius.square());

    return m.add(n).sub(ONE).sign();
  }

  distanceTo(point: Point): Num {
    const closestPoint = closestPointOnEllipse(
      this.majorRadius,
      this.minorRadius,
      point,
    );
    return point.vecFrom(closestPoint).norm().mul(this.sign(point));
  }
}

export class EllipseArc {
  readonly majorRadius: Num;
  readonly minorRadius: Num;
  readonly startAngle: Angle;
  readonly endAngle: Angle;
  readonly orientation: Num;

  private firstPoint: Point;
  private lastPoint: Point;

  constructor(
    majorRadius: Num | number,
    minorRadius: Num | number,
    startAngle: Angle,
    endAngle: Angle,
    orientation: Num | number,
  ) {
    this.majorRadius = asNum(majorRadius);
    this.minorRadius = asNum(minorRadius);
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.orientation = asNum(orientation);

    this.firstPoint = new Point(
      this.majorRadius.mul(this.startAngle.cos()),
      this.minorRadius.mul(this.startAngle.sin()),
    );

    this.lastPoint = new Point(
      this.majorRadius.mul(this.endAngle.cos()),
      this.minorRadius.mul(this.endAngle.sin()),
    );
  }

  distanceTo(point: Point): Num {
    const closestPoints = closestPointsOnEllipseArc(
      this.majorRadius,
      this.minorRadius,
      this.startAngle,
      this.endAngle,
      this.orientation,
      point,
    );

    const minDist = min(
      ...(closestPoints.map((closestPoint) =>
        point.vecFrom(closestPoint).norm(),
      ) as [Num, Num, Num, Num]),
    );

    const firstDist = point.vecFrom(this.firstPoint).norm();
    const lastDist = point.vecFrom(this.lastPoint).norm();

    return minDist.min(firstDist).min(lastDist);
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
