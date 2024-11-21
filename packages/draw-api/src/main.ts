import {
  AngleLiteral,
  AngleNode,
  ArcFromEndControl,
  ArcFromStartControl,
  BiarcC,
  BiarcS,
  Circle,
  Difference,
  DistanceLiteral,
  DistanceNode,
  EdgeNode,
  evalProfile,
  fidgetRender,
  Intersection,
  Line,
  Morph,
  PathClose,
  PathEdge,
  PathNode,
  PathOpenEnd,
  PathStart,
  PointAsVectorFromOrigin,
  PointNode,
  PointVectorSum,
  ProfileNode,
  Rotation,
  Scale,
  Shell,
  Translation,
  Union,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  VectorNode,
} from "sketch";
import { booleansToASCII, intArrayToImageData } from "./utils";

export function asDistance(distance: number | DistanceNode): DistanceNode {
  if (distance instanceof DistanceNode) {
    return distance;
  }
  return new DistanceLiteral(distance);
}

export function asAngle(angle: number | AngleNode): AngleNode {
  if (angle instanceof AngleNode) {
    return angle;
  }
  return new AngleLiteral(angle);
}

export function asVector(vector: [number, number] | VectorNode): VectorNode {
  if (vector instanceof VectorNode) {
    return vector;
  }
  const [x, y] = vector;
  return new VectorFromCartesianCoords(x, y);
}

export function asPolarVector(
  vector: [number | AngleNode, number | DistanceNode] | VectorNode,
): VectorNode {
  if (vector instanceof VectorNode) {
    return vector;
  }
  const [angle, radius] = vector;
  return new VectorFromPolarCoods(asDistance(radius), asAngle(angle));
}

export function asPoint(point: [number, number] | PointNode): PointNode {
  if (point instanceof PointNode) {
    return point;
  }
  return new PointAsVectorFromOrigin(asVector(point));
}

export function asPolarPoint(
  point: [number | AngleNode, number | DistanceNode] | PointNode,
): PointNode {
  if (point instanceof PointNode) {
    return point;
  }
  return new PointAsVectorFromOrigin(asPolarVector(point));
}

export class ProfileEditor {
  constructor(public shape: ProfileNode) {}

  public translate(vector: [number, number] | VectorNode): ProfileEditor {
    return new ProfileEditor(new Translation(this.shape, asVector(vector)));
  }

  public rotate(angle: number | AngleNode): ProfileEditor {
    return new ProfileEditor(new Rotation(this.shape, asAngle(angle)));
  }

  public union(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Union(this.shape, other.shape));
  }

  public intersect(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Intersection(this.shape, other.shape));
  }

  public diff(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Difference(this.shape, other.shape));
  }

  public shell(thickness: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Shell(this.shape, asDistance(thickness)));
  }

  public scale(factor: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Scale(this.shape, asDistance(factor)));
  }

  public morph(other: ProfileEditor, t: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Morph(this.shape, other.shape, asDistance(t)));
  }
  

  async debugRender(size = 50): Promise<string> {
    const distField = evalProfile(this.shape);
    const render = await fidgetRender(distField, size);
    return booleansToASCII(intArrayToImageData(render), true);
  }

  async render(size = 250): Promise<Uint8ClampedArray> {
    const distField = evalProfile(this.shape);
    const render = await fidgetRender(distField, size, true);
    return new Uint8ClampedArray(render);
  }
}

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
    angle: number,
    radius: number,
    cornerRadius?: number,
  ): EdgeMaker {
    const p = new VectorFromPolarCoods(asDistance(radius), asAngle(angle));
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
    angle: number,
    radius: number,
    cornerRadius?: number,
  ): EdgeMaker {
    const p = new VectorFromPolarCoods(asDistance(radius), asAngle(angle));
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

export class EdgeMaker {
  constructor(private done: (edge: EdgeNode) => PointMaker) {}

  public line(): PointMaker {
    return this.done(new Line());
  }

  public arcFromStartControl(
    control: PointNode | [number, number],
  ): PointMaker {
    return this.done(new ArcFromStartControl(asPoint(control)));
  }

  public arcFromEndControl(control: PointNode | [number, number]): PointMaker {
    return this.done(new ArcFromEndControl(asPoint(control)));
  }

  public biarcC(control: PointNode | [number, number]): PointMaker {
    return this.done(new BiarcC(asPoint(control)));
  }

  public biarcS(
    control0: PointNode | [number, number],
    control1: PointNode | [number, number],
  ): PointMaker {
    return this.done(new BiarcS(asPoint(control0), asPoint(control1)));
  }
}

function parseCorner(
  cornerRadius: number | undefined,
): DistanceNode | undefined {
  return (cornerRadius || cornerRadius === 0) && asDistance(cornerRadius);
}

export function drawCircle(
  radius: number | DistanceNode,
  center: [number, number] | PointNode | null = null,
): ProfileEditor {
  const circle = new ProfileEditor(new Circle(asDistance(radius)));
  if (!center) {
    return circle;
  }
  return circle.translate(center);
}

export function draw(origin = [0, 0], cornerRadius?: number): EdgeMaker {
  let currentPoint = asPoint(origin);
  let path: PathNode = new PathStart(currentPoint, parseCorner(cornerRadius));

  function lineDone(line: EdgeNode): PointMaker {
    function pointDone(point: PointNode, cornerRadius?: number): EdgeMaker {
      currentPoint = point;
      path = new PathEdge(path, line, point, parseCorner(cornerRadius));
      return new EdgeMaker((edge: EdgeNode) => lineDone(edge));
    }

    function profileDone(close = true): ProfileEditor {
      const Profile = close ? PathClose : PathOpenEnd;
      return new ProfileEditor(new Profile(path, line));
    }

    return new PointMaker(currentPoint, pointDone, profileDone);
  }

  return new EdgeMaker((edge: EdgeNode) => lineDone(edge));
}
