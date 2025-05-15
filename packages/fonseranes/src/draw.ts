import { Angle, angleFromDeg } from "./angle";
import { circleFromBulge } from "./conic-sections";
import {
  ClosedPath2D,
  Curve2D,
  Line2D,
  Point2D,
  ReversedConic2D,
  Vec2D,
} from "./primitives";

export type PointLike = Point2D | [number, number];

export function asPoint(point: PointLike): Point2D {
  if (Array.isArray(point)) {
    return new Point2D(point[0], point[1]);
  }
  return point;
}

export type AngleLike = number | Angle;

export function asAngle(angle: AngleLike): Angle {
  if (typeof angle === "number") {
    return angleFromDeg(angle);
  }
  return angle;
}

export class PointMaker {
  constructor(
    private currentPoint: Point2D,
    private done: (p: Point2D) => EdgeMaker,
    private end: (close: boolean) => ClosedPath2D,
  ) {}

  private returnPoint(p: Point2D): EdgeMaker {
    return this.done(p);
  }

  public goTo(x: number, y: number): EdgeMaker {
    return this.returnPoint(new Point2D(x, y));
  }

  public goToPolar(theta: AngleLike, radius: number): EdgeMaker {
    const angle = asAngle(theta);
    return this.returnPoint(
      new Point2D(radius * angle.cos(), radius * angle.sin()),
    );
  }

  public goToPoint(point: Point2D): EdgeMaker {
    return this.returnPoint(point);
  }

  public moveBy(x: number, y: number): EdgeMaker {
    const v = new Vec2D(x, y);
    return this.returnPoint(this.currentPoint.add(v));
  }

  public moveByPolar(theta: AngleLike, radius: number): EdgeMaker {
    const angle = asAngle(theta);
    const v = new Vec2D(angle.cos() * radius, angle.sin() * radius);
    return this.returnPoint(this.currentPoint.add(v));
  }

  public horizontalMoveBy(x: number): EdgeMaker {
    return this.moveBy(x, 0);
  }

  public horizontalMoveTo(x: number): EdgeMaker {
    const pos = new Point2D(x, this.currentPoint.y);
    return this.returnPoint(pos);
  }

  public verticalMoveTo(y: number): EdgeMaker {
    const pos = new Point2D(this.currentPoint.x, y);
    return this.returnPoint(pos);
  }

  public verticalMoveBy(y: number): EdgeMaker {
    return this.moveBy(0, y);
  }

  public close(): ClosedPath2D {
    return this.end(true);
  }
}

interface EdgeCreator {
  (p0: Point2D, p1: Point2D): Curve2D;
}

export class EdgeMaker {
  constructor(
    private done: (edgeCreator: EdgeCreator) => PointMaker,
    readonly currentPoint: Point2D,
  ) {}

  public line(): PointMaker {
    return this.done((p0, p1) => Line2D.fromPoints(p0, p1));
  }

  public arcFromBulge(bulge: number): PointMaker {
    return this.done((p0, p1) => {
      if (bulge === 0) {
        return Line2D.fromPoints(p0, p1);
      }
      const circle = circleFromBulge(p0, p1, bulge);
      if (bulge > 0) {
        return circle;
      }
      return new ReversedConic2D(circle.transformation);
    });
  }
}

export function draw(origin: PointLike = [0, 0]): EdgeMaker {
  let currentPoint = asPoint(origin);
  const firstPoint = currentPoint;

  const points = [firstPoint];
  const curves: (Curve2D | ReversedConic2D)[] = [];

  function lineDone(createEdge: EdgeCreator): PointMaker {
    function pointDone(point: Point2D): EdgeMaker {
      const previousPoint = currentPoint;
      currentPoint = point;

      const curve = createEdge(previousPoint, point);
      curves.push(curve);
      points.push(point);

      return new EdgeMaker((edge: EdgeCreator) => lineDone(edge), currentPoint);
    }

    function profileDone(): ClosedPath2D {
      const curve = createEdge(currentPoint, firstPoint);
      curves.push(curve);
      return new ClosedPath2D(points, curves);
    }

    return new PointMaker(currentPoint, pointDone, profileDone);
  }

  return new EdgeMaker((edge: EdgeCreator) => lineDone(edge), currentPoint);
}
