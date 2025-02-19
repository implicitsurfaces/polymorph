import { Vector2 } from "threejs-math";
import { PathStyle } from "./style";

/**
 * Represents the style to use to draw a shape.
 *
 * Note that a shape can be made of several sub-shapes (e.g., the arrow head
 * vs shaft).
 *
 * If all subshapes are to be drawn using the same path style, then you can
 * use a single `PathStyle`:
 *
 * ```
 * const style = new PathStyle({ lineWidth: 1, stroke: "black" });
 * const shape = new LineSegmentShape(p1, p2);
 * shape.draw(ctx, style);
 * ```
 *
 * If different subshapes are to be drawn using different path styles, then
 * you can specify which `PathStyle` to use for which subshape via the
 * subshape identifier:
 *
 * ```
 * const headStyle = new PathStyle({ lineWidth: 1, stroke: "black" });
 * const shaftStyle = new PathStyle({ fill: "black" });
 * const shape = new ArrowShape(p1, p2);
 * shape.draw(ctx, { head: headStyle, shaft: shaftStyle });
 * ```
 */
type ShapeStyle = undefined | PathStyle | { [key: string]: PathStyle };

export function getDefaultPathStyle(style: ShapeStyle): PathStyle | undefined {
  if (!style) {
    return undefined;
  }
  if (style instanceof PathStyle) {
    return style;
  } else {
    return style.default;
  }
}

/**
 * Represents the geometry of a shape that can be drawn in the Canvas.
 */
export abstract class Shape {
  constructor() {}

  abstract draw(ctx: CanvasRenderingContext2D, style: ShapeStyle): void;
}

function beginPath(ctx: CanvasRenderingContext2D) {
  ctx.beginPath();
}

function endPath(ctx: CanvasRenderingContext2D, style: PathStyle) {
  if (style.lineWidth > 0 && style.stroke) {
    ctx.lineWidth = style.lineWidth;
    ctx.strokeStyle = style.stroke;
    ctx.stroke();
  }
  if (style.fill) {
    ctx.fillStyle = style.fill;
    ctx.fill();
  }
}

export class LineSegmentShape extends Shape {
  constructor(
    readonly startPosition: Vector2,
    readonly endPosition: Vector2,
  ) {
    super();
  }

  draw(ctx: CanvasRenderingContext2D, style: ShapeStyle) {
    const pathStyle = getDefaultPathStyle(style);
    if (pathStyle) {
      beginPath(ctx);
      ctx.moveTo(this.startPosition.x, this.startPosition.y);
      ctx.lineTo(this.endPosition.x, this.endPosition.y);
      endPath(ctx, pathStyle);
    }
  }
}

export interface ArcShapeOptions {
  center: Vector2;
  radius: number;
  startAngle: number;
  endAngle: number;
  isCounterClockwise: boolean;
}

export class ArcShape extends Shape {
  readonly center: Vector2;
  readonly radius: number;
  readonly startAngle: number;
  readonly endAngle: number;
  readonly isCounterClockwise: boolean;

  constructor(options: ArcShapeOptions) {
    super();
    this.center = options.center;
    this.radius = options.radius;
    this.startAngle = options.startAngle;
    this.endAngle = options.endAngle;
    this.isCounterClockwise = options.isCounterClockwise;
  }

  draw(ctx: CanvasRenderingContext2D, style: ShapeStyle) {
    const pathStyle = getDefaultPathStyle(style);
    if (pathStyle) {
      beginPath(ctx);
      ctx.arc(
        this.center.x,
        this.center.y,
        this.radius,
        this.startAngle,
        this.endAngle,
        this.isCounterClockwise,
      );
      endPath(ctx, pathStyle);
    }
  }
}

export type GeneralizedArcShape = ArcShape | LineSegmentShape;
