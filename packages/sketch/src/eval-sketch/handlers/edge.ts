import { Point } from "../../geom";
import { Num, ONE, ZERO } from "../../num";
import { LineSegment } from "../../segments";
import {
  biarcC,
  biarcS,
  bulgingSegmentUsingEndControl,
  bulgingSegmentUsingStartControl,
  endpointsEllipticArc,
} from "../../segments-helpers";
import {
  AnyEdgeNode,
  ArcFromEndControl,
  ArcFromStartControl,
  CCurve,
  EllipseArcNode,
  Line,
  SCurve,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { Segment } from "../../types";
import { guardAngle, guardNum, guardPoint } from "../guards";

export interface EdgeHandler<T extends AnyEdgeNode> extends Handler<T> {
  eval: (edge: T, children: unknown[]) => (p0: Point, p1: Point) => Segment[];
}

const lineHandler: EdgeHandler<Line> = {
  category: "Edge",
  nodeType: "Line",
  children: () => [],
  eval: () => (p0: Point, p1: Point) => [new LineSegment(p0, p1)],
};

const arcFromStartControlHandler: EdgeHandler<ArcFromStartControl> = {
  category: "Edge",
  nodeType: "ArcFromStartControl",
  children: (edge) => [edge.control],
  eval: function evalArcFromStartControl(_, [control]) {
    return (p0: Point, p1: Point) => [
      bulgingSegmentUsingStartControl(p0, p1, guardPoint(control)),
    ];
  },
};

const arcFromEndControlHandler: EdgeHandler<ArcFromEndControl> = {
  category: "Edge",
  nodeType: "ArcFromEndControl",
  children: (edge) => [edge.control],
  eval: function evalArcFromEndControl(_, [control]) {
    return (p0: Point, p1: Point) => [
      bulgingSegmentUsingEndControl(p0, p1, guardPoint(control)),
    ];
  },
};

const cCurveHandler: EdgeHandler<CCurve> = {
  category: "Edge",
  nodeType: "CCurve",
  children: (edge) => [edge.control],
  eval: function evalCCurve(_edge, [control]) {
    return (p0: Point, p1: Point) => biarcC(p0, p1, guardPoint(control));
  },
};

const sCurveHandler: EdgeHandler<SCurve> = {
  category: "Edge",
  nodeType: "SCurve",
  children: (edge) => [edge.control0, edge.control1],
  eval: function evalSCurve(_, [control0, control1]) {
    return (p0: Point, p1: Point) =>
      biarcS(p0, p1, guardPoint(control0), guardPoint(control1));
  },
};

const numBool = (num: boolean): Num => (num ? ONE : ZERO);

const ellipseArcHandler: EdgeHandler<EllipseArcNode> = {
  category: "Edge",
  nodeType: "EllipseArc",
  children: (edge) => [edge.majorRadius, edge.minorRadius, edge.rotation],
  eval: function evalEllipseArc(edge, [majorRadius, minorRadius, rotation]) {
    return (p0: Point, p1: Point) => {
      return [
        endpointsEllipticArc(
          p0,
          p1,
          guardNum(majorRadius),
          guardNum(minorRadius),
          guardAngle(rotation),
          numBool(edge.largeArc),
          numBool(edge.sweep),
        ),
      ];
    };
  },
};

export const edgeHandlers = [
  lineHandler,
  arcFromStartControlHandler,
  arcFromEndControlHandler,
  cCurveHandler,
  sCurveHandler,
  ellipseArcHandler,
] as const;
