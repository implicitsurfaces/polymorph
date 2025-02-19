import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2";
import {
  Shape,
  LineSegmentShape,
  ArcShape,
  GeneralizedArcShape,
} from "./Shapes";
import {
  PathStyle,
  edgeWidth,
  getNodeColor,
  getControlColor,
  getNodeStyleIndex,
} from "./style";

import { Selection } from "../Selection";
import {
  EdgeNode,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "../Document";

function getGeneralizedArcFromStartTangent(
  startPoint: Vector2,
  endPoint: Vector2,
  tangent: Vector2,
): GeneralizedArcShape {
  // Fallback to line segment in degenerate cases (collinear)
  const chord = endPoint.clone().sub(startPoint);
  const s_ = chord.cross(tangent);
  if (s_ === 0) {
    return new LineSegmentShape(startPoint, endPoint);
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

  return new ArcShape({
    center: center,
    radius: center.distanceTo(startPoint),
    startAngle: startPoint.clone().sub(center).angle(),
    endAngle: endPoint.clone().sub(center).angle(),
    isCounterClockwise: s_ > 0,
  });
}

function getCCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  controlPoint: Vector2,
): GeneralizedArcShape[] {
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
): GeneralizedArcShape[] {
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
  shapes: Shape[];
  tangents: LineSegmentShape[];
}

export function getEdgeShapesAndControls(
  edge: EdgeNode,
): EdgeShapesAndControls {
  const res: EdgeShapesAndControls = {
    shapes: [],
    tangents: [],
  };

  const startPos = edge.startPoint.position;
  const endPos = edge.endPoint.position;

  if (edge instanceof LineSegment) {
    res.shapes.push(new LineSegmentShape(startPos, endPos));
  } else if (edge instanceof ArcFromStartTangent) {
    const cpPos = edge.controlPoint.position;
    const tangent = cpPos.clone().sub(startPos);
    res.shapes.push(
      getGeneralizedArcFromStartTangent(startPos, endPos, tangent),
    );
    res.tangents.push(new LineSegmentShape(startPos, cpPos));
  } else if (edge instanceof CCurve) {
    const cpPos = edge.controlPoint.position;
    const shapes_ = getCCurveShapes(startPos, endPos, cpPos);
    for (const shape of shapes_) {
      res.shapes.push(shape);
    }
    res.tangents.push(new LineSegmentShape(startPos, cpPos));
    res.tangents.push(new LineSegmentShape(endPos, cpPos));
  } else if (edge instanceof SCurve) {
    const startCpPos = edge.startControlPoint.position;
    const endCpPos = edge.endControlPoint.position;
    const shapes_ = getSCurveShapes(startPos, endPos, startCpPos, endCpPos);
    for (const shape of shapes_) {
      res.shapes.push(shape);
    }
    res.tangents.push(new LineSegmentShape(startPos, startCpPos));
    res.tangents.push(new LineSegmentShape(endPos, endCpPos));
  }
  return res;
}

export function drawEdges(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  edges: EdgeNode[],
  selection: Selection,
) {
  const w = edgeWidth / camera.zoom;
  const tangentStyle = new PathStyle({
    lineWidth: w,
    stroke: getControlColor(0),
  });
  const shapeStyles = [
    new PathStyle({ lineWidth: w, stroke: getNodeColor(0) }),
    new PathStyle({ lineWidth: w, stroke: getNodeColor(1) }),
    new PathStyle({ lineWidth: w, stroke: getNodeColor(2) }),
    new PathStyle({ lineWidth: w, stroke: getNodeColor(3) }),
  ];
  for (const edge of edges) {
    const styleIndex = getNodeStyleIndex(edge, selection);
    const shapeStyle = shapeStyles[styleIndex];
    const sc = getEdgeShapesAndControls(edge);
    for (const shape of sc.shapes) {
      shape.draw(ctx, shapeStyle);
    }
    for (const lineSegment of sc.tangents) {
      lineSegment.draw(ctx, tangentStyle);
    }
  }
}
