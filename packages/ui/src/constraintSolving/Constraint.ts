import { MeasureNode, PointToPointDistance } from "../Document";

export type ParamId = string;
export type ParamValueMap = { [key: ParamId]: number };

export interface Constraint {
  type: "distance";
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
      type: "distance",
      params: params,
      value: node.number.value,
    };
  }
}
