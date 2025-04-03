import { Vector2 } from "threejs-math";

import { Document } from "../Document";
import { Layer } from "../Layer";
import { Point } from "../Point";

import { LineSegment } from "../edges/LineSegment";
import { ArcFromStartTangent } from "../edges/ArcFromStartTangent";
import { CCurve } from "../edges/CCurve";
import { SCurve } from "../edges/SCurve";

import { PointToPointDistance } from "../measures/PointToPointDistance";
import { LineToPointDistance } from "../measures/LineToPointDistance";
import { Angle } from "../measures/Angle";

export function createTestDocument() {
  const doc = new Document();
  const layer = doc.createNode(Layer, { name: "Layer 1" });
  doc.layers = [layer.id];
  const p1 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-100, 0),
  });
  const p2 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(0, 0),
  });
  const p3 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(100, 100),
  });
  const p4 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(200, 100),
  });
  const p5 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(200, 0),
  });
  const p6 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(100, -100),
  });
  const p7 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-100, -100),
  });
  const l1 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p1,
    endPoint: p2,
  });
  doc.createNode(PointToPointDistance, {
    layer: layer,
    startPoint: p1,
    endPoint: p2,
  });
  const cp1 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(50, 0),
  });
  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l1,
    point: cp1,
  });
  doc.createNode(ArcFromStartTangent, {
    layer: layer,
    startPoint: p2,
    endPoint: p3,
    controlPoint: cp1,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p3,
    endPoint: p4,
  });
  const cp2 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(150, 50),
  });
  doc.createNode(CCurve, {
    layer: layer,
    startPoint: p4,
    endPoint: p5,
    controlPoint: cp2,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p5,
    endPoint: p6,
  });
  const cp3 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(50, -150),
  });
  const cp4 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(-80, -60),
  });
  doc.createNode(SCurve, {
    layer: layer,
    startPoint: p6,
    endPoint: p7,
    startControlPoint: cp3,
    endControlPoint: cp4,
  });
  const l2 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7,
    endPoint: p1,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2,
    endPoint: p6,
  });
  doc.createNode(Angle, {
    layer: layer,
    line0: l2,
    line1: l1,
  });
  return doc;
}
