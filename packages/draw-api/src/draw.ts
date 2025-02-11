import {
  AngleNode,
  ArcFromEndControl,
  ArcFromStartControl,
  CCurve,
  SCurve,
  EdgeNode,
  Line,
  PathClose,
  PathEdge,
  PathNode,
  PathOpenEnd,
  PathStart,
  PointAsVectorFromOrigin,
  PointNode,
  PointVectorSum,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  EllipseArcNode,
} from "sketch";
import {
  AngleLike,
  asAngle,
  asDistance,
  asDistanceOrUndefined,
  asPoint,
  DistanceLike,
  PointLike,
} from "./convert";
import { point, angle, Point } from "./geom";
import { ProfileEditor } from "./ProfileEditor";

export class PointMaker {
  constructor(
    private currentPoint: PointNode,
    private done: (p: PointNode, cornerRadius?: number) => EdgeMaker,
    private end: (close: boolean) => ProfileEditor,
  ) {}

  private returnPoint(p: PointNode, cornerRadius?: number): EdgeMaker {
    return this.done(p, cornerRadius);
  }

  public to(
    p: PointLike | ((p: Point) => PointLike),
    cornerRadius?: number,
  ): EdgeMaker {
    if (typeof p === "function") {
      return this.returnPoint(
        asPoint(p(point(this.currentPoint))),
        cornerRadius,
      );
    }
    return this.returnPoint(asPoint(p), cornerRadius);
  }

  public goTo(x: number, y: number, cornerRadius?: number): EdgeMaker {
    const p = new VectorFromCartesianCoords(x, y);
    return this.returnPoint(new PointAsVectorFromOrigin(p), cornerRadius);
  }

  public goToPolar(
    theta: number,
    radius: number,
    cornerRadius?: number,
  ): EdgeMaker {
    const p = new VectorFromPolarCoods(asDistance(radius), asAngle(theta));
    return this.returnPoint(new PointAsVectorFromOrigin(p), cornerRadius);
  }

  public goToPoint(point: PointNode, cornerRadius?: number): EdgeMaker {
    return this.returnPoint(point, cornerRadius);
  }

  public moveBy(x: number, y: number, cornerRadius?: number): EdgeMaker {
    const v = new VectorFromCartesianCoords(x, y);
    return this.returnPoint(
      new PointVectorSum(this.currentPoint, v),
      cornerRadius,
    );
  }

  public moveByPolar(
    theta: number,
    radius: number,
    cornerRadius?: number,
  ): EdgeMaker {
    const p = new VectorFromPolarCoods(asDistance(radius), asAngle(theta));
    return this.returnPoint(
      new PointVectorSum(this.currentPoint, p),
      cornerRadius,
    );
  }

  public horizontalMoveBy(x: number, cornerRadius?: number): EdgeMaker {
    return this.moveBy(x, 0, cornerRadius);
  }

  public verticalMoveBy(y: number, cornerRadius?: number): EdgeMaker {
    return this.moveBy(0, y, cornerRadius);
  }

  public close(): ProfileEditor {
    return this.end(true);
  }

  public openEnd(): ProfileEditor {
    return this.end(false);
  }
}

interface EdgeCreator {
  (p0: PointNode, p1: PointNode): EdgeNode;
}

export class EdgeMaker {
  constructor(
    private done: (edgeCreator: EdgeCreator) => PointMaker,
    readonly currentPoint: PointNode,
  ) {}

  private returnEdge(edge: EdgeNode): PointMaker {
    return this.done(() => edge);
  }

  private withControl(
    EdgeFactory: new (control: PointNode) => EdgeNode,
    control: PointLike | ((p0: Point, p1: Point) => PointLike),
  ): PointMaker {
    if (typeof control === "function") {
      return this.done((p0, p1) => {
        return new EdgeFactory(asPoint(control(point(p0), point(p1))));
      });
    }
    return this.returnEdge(new EdgeFactory(asPoint(control)));
  }

  public line(): PointMaker {
    return this.returnEdge(new Line());
  }

  public arcFromStartControl(
    control: PointLike | ((p0: Point, p1: Point) => PointLike),
  ): PointMaker {
    return this.withControl(ArcFromStartControl, control);
  }

  public arcFromChordAngle(theta: number | AngleNode): PointMaker {
    return this.done((p0_, p1) => {
      const p0 = point(p0_);

      const chord = p0.vecTo(point(p1));
      const newAngle = angle(theta).add(chord.asAngle());

      return new ArcFromStartControl(asPoint(p0.translate(newAngle.asVec())));
    });
  }

  public arcFromEndControl(
    control: PointLike | ((p0: Point, p1: Point) => PointLike),
  ): PointMaker {
    return this.withControl(ArcFromEndControl, control);
  }

  public CCurve(
    control: PointLike | ((p0: Point, p1: Point) => PointLike),
  ): PointMaker {
    return this.withControl(CCurve, control);
  }

  public SCurve(
    control0: PointLike | ((p0: Point, p1: Point) => PointLike),
    control1: PointLike | ((p0: Point, p1: Point) => PointLike),
  ): PointMaker {
    return this.done((p0_, p1_) => {
      const p0 = point(p0_);
      const p1 = point(p1_);

      const c0 = typeof control0 === "function" ? control0(p0, p1) : control0;
      const c1 = typeof control1 === "function" ? control1(p0, p1) : control1;

      return new SCurve(asPoint(c0), asPoint(c1));
    });
  }

  public ellipseArc(
    majorRadius: DistanceLike,
    minorRadius: DistanceLike,
    rotation: AngleLike = 0,
    largeArc = false,
    sweep = false,
  ): PointMaker {
    return this.returnEdge(
      new EllipseArcNode(
        asDistance(majorRadius),
        asDistance(minorRadius),
        asAngle(rotation),
        largeArc,
        sweep,
      ),
    );
  }
}

export function draw(
  origin: PointLike = [0, 0],
  cornerRadius?: number,
): EdgeMaker {
  let currentPoint = asPoint(origin);
  const firstPoint = currentPoint;

  let path: PathNode = new PathStart(
    currentPoint,
    asDistanceOrUndefined(cornerRadius),
  );

  function lineDone(createEdge: EdgeCreator): PointMaker {
    function pointDone(point: PointNode, cornerRadius?: number): EdgeMaker {
      const previousPoint = currentPoint;
      currentPoint = point;

      const edge = createEdge(previousPoint, point);

      path = new PathEdge(
        path,
        edge,
        point,
        asDistanceOrUndefined(cornerRadius),
      );
      return new EdgeMaker((edge: EdgeCreator) => lineDone(edge), currentPoint);
    }

    function profileDone(close = true): ProfileEditor {
      const edge = createEdge(currentPoint, firstPoint);
      const Profile = close ? PathClose : PathOpenEnd;
      return new ProfileEditor(new Profile(path, edge));
    }

    return new PointMaker(currentPoint, pointDone, profileDone);
  }

  return new EdgeMaker((edge: EdgeCreator) => lineDone(edge), currentPoint);
}
