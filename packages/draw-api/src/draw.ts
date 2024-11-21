import {
  AngleNode,
  ArcFromEndControl,
  ArcFromStartControl,
  BiarcC,
  BiarcS,
  DistanceNode,
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
} from "sketch";
import { asAngle, asDistance, asPoint } from "./convert";
import { point, angle } from "./geom";
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

  public line(): PointMaker {
    return this.returnEdge(new Line());
  }

  public arcFromStartControl(
    control: PointNode | [number, number],
  ): PointMaker {
    return this.returnEdge(new ArcFromStartControl(asPoint(control)));
  }

  public arcFromChordAngle(theta: number | AngleNode): PointMaker {
    return this.done((p0_, p1) => {
      const p0 = point(p0_);

      const chord = p0.vecTo(point(p1));
      const newAngle = angle(theta).add(chord.asAngle());

      return new ArcFromStartControl(asPoint(p0.translate(newAngle.asVec())));
    });
  }

  public arcFromEndControl(control: PointNode | [number, number]): PointMaker {
    return this.returnEdge(new ArcFromEndControl(asPoint(control)));
  }

  public biarcC(control: PointNode | [number, number]): PointMaker {
    return this.returnEdge(new BiarcC(asPoint(control)));
  }

  public biarcS(
    control0: PointNode | [number, number],
    control1: PointNode | [number, number],
  ): PointMaker {
    return this.returnEdge(new BiarcS(asPoint(control0), asPoint(control1)));
  }
}

function parseCorner(
  cornerRadius: number | undefined,
): DistanceNode | undefined {
  return (cornerRadius || cornerRadius === 0) && asDistance(cornerRadius);
}

export function draw(origin = [0, 0], cornerRadius?: number): EdgeMaker {
  let currentPoint = asPoint(origin);
  const firstPoint = currentPoint;

  let path: PathNode = new PathStart(currentPoint, parseCorner(cornerRadius));

  function lineDone(createEdge: EdgeCreator): PointMaker {
    function pointDone(point: PointNode, cornerRadius?: number): EdgeMaker {
      const previousPoint = currentPoint;
      currentPoint = point;

      const edge = createEdge(previousPoint, point);

      path = new PathEdge(path, edge, point, parseCorner(cornerRadius));
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
