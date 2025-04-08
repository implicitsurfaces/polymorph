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
import { EdgeCycleProfile } from "../profiles/EdgeCycleProfile";
//import { Angle } from "../measures/Angle";

export function createTestDocument() {
  const doc = new Document();
  const layer = doc.createNode(Layer, { name: "Layer 1" });
  doc.layers = [layer.id];

  /////////////////////////// Skeleton ///////////////////////////

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

  const l2 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7,
    endPoint: p1,
  });

  const l3 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2,
    endPoint: p6,
  });

  const l4 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p3,
    endPoint: p4,
  });

  const l5 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p5,
    endPoint: p6,
  });

  const arcCP = doc.createNode(Point, {
    name: "Arc Control Point",
    layer: layer,
    role: "construction",
    position: new Vector2(50, 0),
  });

  doc.createNode(ArcFromStartTangent, {
    name: "Arc",
    layer: layer,
    startPoint: p2,
    endPoint: p3,
    controlPoint: arcCP,
  });

  const ccCP = doc.createNode(Point, {
    name: "C-Curve Control Point",
    layer: layer,
    role: "construction",
    position: new Vector2(300, 100),
  });

  doc.createNode(CCurve, {
    name: "C-Curve",
    layer: layer,
    startPoint: p4,
    endPoint: p5,
    controlPoint: ccCP,
  });

  const scStartCP = doc.createNode(Point, {
    name: "S-Curve Start Control Point",
    layer: layer,
    role: "construction",
    position: new Vector2(50, -150),
  });

  const scEndCP = doc.createNode(Point, {
    name: "S-Curve End Control Point",
    layer: layer,
    role: "construction",
    position: new Vector2(0, -50),
  });

  const sc = doc.createNode(SCurve, {
    name: "S-Curve",
    layer: layer,
    startPoint: p6,
    endPoint: p7,
    startControlPoint: scStartCP,
    endControlPoint: scEndCP,
  });

  /////////////////////////// Measures ///////////////////////////

  doc.createNode(PointToPointDistance, {
    layer: layer,
    startPoint: p1,
    endPoint: p2,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l1,
    point: arcCP,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l4,
    point: ccCP,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l5,
    point: ccCP,
  });

  // This control point tangent constraint is commented
  // to allow testing moving an unconstraint tangent, and
  // so that the S-Curve actually looks like an "S" shape
  // for demo purposes.
  //
  // doc.createNode(LineToPointDistance, {
  //   layer: layer,
  //   line: l2,
  //   point: scEndCP,
  // });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l5,
    point: scStartCP,
  });

  // doc.createNode(Angle, {
  //   layer: layer,
  //   line0: l2,
  //   line1: l1,
  // });

  /////////////////////////// Shapes ///////////////////////////

  doc.createNode(EdgeCycleProfile, {
    layer: layer,
    cycle: [l1, l3, sc, l2],
  });

  // Test JSON round-trip
  const json = doc.toJSON();
  return Document.fromJSON(json);
}
