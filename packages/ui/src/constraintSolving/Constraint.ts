import {
  MeasureNode,
  PointToPointDistance,
  LineToPointDistance,
  Angle,
} from "../Document";

export type ParamId = string;
export type ParamValueMap = { [key: ParamId]: number };

export interface Constraint {
  type: "pointToPointDistance" | "lineToPointDistance" | "angle";
  params: ParamId[];
  value: number;
}

/**
 * Converts the given `node` to a `Constraint` for use in the constraint solver.
 */
// TODO: Maybe this should be an abstract method of MeasureNode instead?
//
export function getConstraint(node: MeasureNode): Constraint | undefined {
  if (node instanceof PointToPointDistance) {
    const startPoint = node.startPoint;
    const endPoint = node.endPoint;
    const params: string[] = [
      startPoint.x.id,
      startPoint.y.id,
      endPoint.x.id,
      endPoint.y.id,
    ];
    return {
      type: "pointToPointDistance",
      params: params,
      value: node.number.value,
    };
  } else if (node instanceof LineToPointDistance) {
    const line = node.line;
    const point = node.point;
    const params: string[] = [
      line.startPoint.x.id,
      line.startPoint.y.id,
      line.endPoint.x.id,
      line.endPoint.y.id,
      point.x.id,
      point.y.id,
    ];
    return {
      type: "lineToPointDistance",
      params: params,
      value: node.number.value,
    };
  } else if (node instanceof Angle) {
    const line0 = node.line0;
    const line1 = node.line1;
    const params: string[] = [
      line0.startPoint.x.id,
      line0.startPoint.y.id,
      line0.endPoint.x.id,
      line0.endPoint.y.id,
      line1.startPoint.x.id,
      line1.startPoint.y.id,
      line1.endPoint.x.id,
      line1.endPoint.y.id,
    ];
    return {
      type: "angle",
      params: params,
      value: node.number.value,
    };
  }
}
