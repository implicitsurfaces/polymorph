import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2";
import {
  getMeasureColor,
  getNodeStyleIndex,
  PathStyle,
  backgroundColor,
} from "./style";

import { Selection } from "../doc/Selection.ts";
import { MeasureNode, PointToPointDistance } from "../doc/Document";
import { LineSegmentShape } from "./Shapes.ts";

export function drawPointToPointDistance(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  measure: PointToPointDistance,
  selection: Selection,
) {
  const scale = 1 / camera.zoom;
  const w = 1 * scale;
  const styleIndex = getNodeStyleIndex(measure, selection);
  const color = getMeasureColor(styleIndex);
  const style = new PathStyle({ lineWidth: w, stroke: color });

  // p2                           q2
  //    |                       |
  // p1 |<--------------------->| q1       ^
  //    |                       |        v |
  //    o-----------------------o          o--->
  // p0                           q0         u
  //
  const p0 = measure.startPoint.position;
  const q0 = measure.endPoint.position;
  const u = q0.clone().sub(p0);
  const l = u.length();
  if (l == 0) {
    u.x = 1;
    u.y = 0;
  } else {
    const invL = 1 / l;
    u.multiplyScalar(invL);
  }
  const v = new Vector2(-u.y, u.x);
  const d1 = 20 * scale;
  const d2 = 30 * scale;
  const p1 = v.clone().multiplyScalar(d1).add(p0);
  const q1 = v.clone().multiplyScalar(d1).add(q0);
  const p2 = v.clone().multiplyScalar(d2).add(p0);
  const q2 = v.clone().multiplyScalar(d2).add(q0);

  // Arrow heads:
  //
  // p2 o
  //    |         o pb
  //    |        .
  //    |    p1 o  -------------
  //    |        .
  //    |         o pa
  //    |<----->
  //    | offset
  //    |
  //
  // The offset is important to compensate at least for line width, but can be
  // made larger for stylistic reasons.
  //
  const offset = u.clone().multiplyScalar(3 * scale);
  const da = 5 * scale;
  p1.add(offset);
  q1.sub(offset);
  const pa = u.clone().add(v).multiplyScalar(da).add(p1);
  const pb = u.clone().sub(v).multiplyScalar(da).add(p1);
  const qa = u.clone().add(v).multiplyScalar(-da).add(q1);
  const qb = u.clone().sub(v).multiplyScalar(-da).add(q1);

  const shapes = [
    new LineSegmentShape(p0, p2),
    new LineSegmentShape(q0, q2),
    new LineSegmentShape(p1, q1),
    new LineSegmentShape(pa, p1),
    new LineSegmentShape(p1, pb),
    new LineSegmentShape(qa, q1),
    new LineSegmentShape(q1, qb),
  ];
  for (const shape of shapes) {
    shape.draw(ctx, style);
  }

  // Text centered in the middle of the arrow shaft.
  //
  const valuePrecision = 2;
  const valueStr = measure.number.value.toFixed(valuePrecision);
  const fontSize = 12 * scale;
  const p = p1.clone().add(q1).multiplyScalar(0.5);
  ctx.font = `${fontSize}px Open Sans`;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = backgroundColor;
  ctx.lineWidth = 3 * scale;
  ctx.strokeText(valueStr, p.x, p.y);
  ctx.fillText(valueStr, p.x, p.y);
}

export function drawMeasures(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  nodes: Array<MeasureNode>,
  selection: Selection,
) {
  for (const measure of nodes) {
    if (measure instanceof PointToPointDistance) {
      drawPointToPointDistance(ctx, camera, measure, selection);
    }
  }
}
