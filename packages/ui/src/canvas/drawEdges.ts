import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import {
  StrokeStyle,
  edgeWidth,
  getNodeColor,
  getControlColor,
} from "./style.ts";

import { Selection } from "../Selection.ts";
import {
  Point,
  Document,
  NodeId,
  EdgeNode,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "../Document.ts";

export interface CanvasShapeBase {
  type: string;
}

export interface CanvasArc extends CanvasShapeBase {
  type: "Arc";
  center: Vector2;
  radius: number;
  startAngle: number;
  endAngle: number;
  isCounterClockwise: boolean;
}

export interface CanvasLineSegment extends CanvasShapeBase {
  type: "LineSegment";
  startPoint: Vector2;
  endPoint: Vector2;
}

export type CanvasGeneralizedArc = CanvasArc | CanvasLineSegment;
export type CanvasShape = CanvasGeneralizedArc;

interface EdgeStyle {
  lineWidth: number;
  strokeStyle: StrokeStyle;
}

function drawEdgeBegin(ctx: CanvasRenderingContext2D) {
  ctx.beginPath();
}

function drawEdgeEnd(ctx: CanvasRenderingContext2D, edgeStyle: EdgeStyle) {
  ctx.lineWidth = edgeStyle.lineWidth;
  ctx.strokeStyle = edgeStyle.strokeStyle;
  ctx.stroke();
}

function drawLineSegment(
  ctx: CanvasRenderingContext2D,
  lineSegment: CanvasLineSegment,
  edgeStyle: EdgeStyle,
) {
  drawEdgeBegin(ctx);
  ctx.moveTo(lineSegment.startPoint.x, lineSegment.startPoint.y);
  ctx.lineTo(lineSegment.endPoint.x, lineSegment.endPoint.y);
  drawEdgeEnd(ctx, edgeStyle);
}

function drawArc(
  ctx: CanvasRenderingContext2D,
  arc: CanvasArc,
  edgeStyle: EdgeStyle,
) {
  drawEdgeBegin(ctx);
  ctx.arc(
    arc.center.x,
    arc.center.y,
    arc.radius,
    arc.startAngle,
    arc.endAngle,
    arc.isCounterClockwise,
  );
  drawEdgeEnd(ctx, edgeStyle);
}

function drawShape(
  ctx: CanvasRenderingContext2D,
  shape: CanvasShape,
  edgeStyle: EdgeStyle,
) {
  switch (shape.type) {
    case "Arc":
      drawArc(ctx, shape, edgeStyle);
      break;
    case "LineSegment":
      drawLineSegment(ctx, shape, edgeStyle);
      break;
  }
}

function getLineSegment(
  startPoint: Vector2,
  endPoint: Vector2,
): CanvasLineSegment {
  return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
}

function getGeneralizedArcFromStartTangent(
  startPoint: Vector2,
  endPoint: Vector2,
  tangent: Vector2,
): CanvasGeneralizedArc {
  // Fallback to line segment in degenerate cases (collinear)
  const chord = endPoint.clone().sub(startPoint);
  const s_ = chord.cross(tangent);
  if (s_ === 0) {
    return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
  }

  // Compute center
  const tangentLength = tangent.length(); // guaranteed > 0 otherwise s_ == 0
  const s = s_ / tangentLength;
  const c = chord.dot(tangent) / tangentLength;
  const bulge = -s / (c + Math.sqrt(c * c + s * s));
  const chordPerp = new Vector2(-chord.y, chord.x);
  const bb = (bulge - 1 / bulge) / 4;
  const midPoint = startPoint.clone().add(endPoint).multiplyScalar(0.5);
  const center = midPoint.clone().sub(chordPerp.multiplyScalar(bb));

  return {
    type: "Arc",
    center: center,
    radius: center.distanceTo(startPoint),
    startAngle: startPoint.clone().sub(center).angle(),
    endAngle: endPoint.clone().sub(center).angle(),
    isCounterClockwise: s_ > 0,
  };
}

function getCCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  controlPoint: Vector2,
): Array<CanvasGeneralizedArc> {
  const w0 = controlPoint.distanceTo(endPoint);
  const w1 = controlPoint.distanceTo(startPoint);
  const w2 = startPoint.distanceTo(endPoint);

  const junction = startPoint
    .clone()
    .multiplyScalar(w0)
    .add(endPoint.clone().multiplyScalar(w1))
    .add(controlPoint.clone().multiplyScalar(w2))
    .divideScalar(w0 + w1 + w2);

  const startTangent = controlPoint.clone().sub(startPoint);
  const endTangent = controlPoint.clone().sub(endPoint);

  return [
    getGeneralizedArcFromStartTangent(startPoint, junction, startTangent),
    getGeneralizedArcFromStartTangent(endPoint, junction, endTangent),
  ];
}

function lineLineIntersection(
  p0: Vector2,
  v0: Vector2,
  p1: Vector2,
  v1: Vector2,
) {
  const crossDir = v0.cross(v1);
  const diffPoint = p1.clone().sub(p0);
  const param = diffPoint.cross(v1) / crossDir;
  return p0.clone().add(v0.clone().multiplyScalar(param));
}

function getMidpoint(a: Vector2, b: Vector2) {
  return a.clone().add(b).multiplyScalar(0.5);
}

function biarcLocusCenter(
  p1: Vector2,
  tangent1: Vector2,
  p2: Vector2,
  tangent2: Vector2,
) {
  const u1 = tangent1.clone().normalize();
  const u2 = tangent2.clone().normalize();

  const p3 = p1.clone().add(u1);
  const p4 = p2.clone().add(u2);

  const m1 = getMidpoint(p1, p2);
  const m2 = getMidpoint(p3, p4);

  const p1p2 = p2.clone().sub(p1);
  const p3p4 = p4.clone().sub(p3);

  const v1 = new Vector2(-p1p2.y, p1p2.x);
  const v2 = new Vector2(-p3p4.y, p3p4.x);

  return lineLineIntersection(m1, v1, m2, v2);
}

function getSCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  startControlPoint: Vector2,
  endControlPoint: Vector2,
): Array<CanvasGeneralizedArc> {
  const tangent1 = startControlPoint.clone().sub(startPoint);
  const tangent2 = endPoint.clone().sub(endControlPoint);

  const locusCenter = biarcLocusCenter(
    startPoint,
    tangent1,
    endPoint,
    tangent2,
  );

  // XXX: is it normal that positionRatio is always equal to 0.5
  // if the tangents are normalized?
  const positionRatio =
    tangent1.length() / (tangent1.length() + tangent2.length());
  const chord = endPoint.clone().sub(startPoint);

  const positionOnChord = startPoint
    .clone()
    .add(chord.clone().multiplyScalar(positionRatio));

  const junctionDir = positionOnChord.clone().sub(locusCenter).normalize();
  const junction = locusCenter
    .clone()
    .add(junctionDir.clone().multiplyScalar(locusCenter.distanceTo(endPoint)));

  return [
    getGeneralizedArcFromStartTangent(startPoint, junction, tangent1),
    getGeneralizedArcFromStartTangent(endPoint, junction, tangent2.negate()),
  ];
}

export interface EdgeShapesAndControls {
  shapes: Array<CanvasShape>;
  tangents: Array<CanvasLineSegment>;
}

export function getEdgeShapesAndControls(
  doc: Document,
  edge: EdgeNode,
): EdgeShapesAndControls {
  const res: EdgeShapesAndControls = {
    shapes: [],
    tangents: [],
  };

  const startPoint = doc.getNode(edge.startPoint, Point);
  const endPoint = doc.getNode(edge.endPoint, Point);
  if (!startPoint || !endPoint) {
    return res;
  }
  const startPos = startPoint.getPosition();
  const endPos = endPoint.getPosition();

  if (edge instanceof LineSegment) {
    res.shapes.push(getLineSegment(startPos, endPos));
  } else if (edge instanceof ArcFromStartTangent) {
    const cp = doc.getNode(edge.controlPoint, Point);
    if (!cp) {
      return res;
    }
    const tangent = cp.getPosition().clone().sub(startPos);
    res.shapes.push(
      getGeneralizedArcFromStartTangent(startPos, endPos, tangent),
    );
    res.tangents.push(getLineSegment(startPos, cp.getPosition()));
  } else if (edge instanceof CCurve) {
    const cp = doc.getNode(edge.controlPoint, Point);
    if (!cp) {
      return res;
    }
    const shapes_ = getCCurveShapes(startPos, endPos, cp.getPosition());
    for (const shape of shapes_) {
      res.shapes.push(shape);
    }
    res.tangents.push(getLineSegment(startPos, cp.getPosition()));
    res.tangents.push(getLineSegment(endPos, cp.getPosition()));
  } else if (edge instanceof SCurve) {
    const startCp = doc.getNode(edge.startControlPoint, Point);
    const endCp = doc.getNode(edge.endControlPoint, Point);
    if (!startCp || !endCp) {
      return res;
    }
    const shapes_ = getSCurveShapes(
      startPos,
      endPos,
      startCp.getPosition(),
      endCp.getPosition(),
    );
    for (const shape of shapes_) {
      res.shapes.push(shape);
    }
    res.tangents.push(getLineSegment(startPos, startCp.getPosition()));
    res.tangents.push(getLineSegment(endPos, endCp.getPosition()));
  }
  return res;
}

export function drawEdges(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  document: Document,
  nodes: Array<NodeId>,
  selection: Selection,
) {
  const edgeWidth_ = edgeWidth / camera.zoom;
  const edgeStyle = { lineWidth: edgeWidth_, strokeStyle: getNodeColor() };
  const tangentStyle = {
    lineWidth: edgeWidth_,
    strokeStyle: getControlColor(),
  };
  for (const id of nodes) {
    const edge = document.getNode(id, EdgeNode);
    if (edge) {
      const isEdgeHovered = selection.isHoveredNode(id);
      const isEdgeSelected = selection.isSelectedNode(id);
      edgeStyle.strokeStyle = getNodeColor(isEdgeHovered, isEdgeSelected);
      const sc = getEdgeShapesAndControls(document, edge);
      for (const shape of sc.shapes) {
        drawShape(ctx, shape, edgeStyle);
      }
      for (const lineSegment of sc.tangents) {
        drawLineSegment(ctx, lineSegment, tangentStyle);
      }
    }
  }
}
