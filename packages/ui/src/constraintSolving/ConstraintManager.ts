// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck

import { Document, MeasureNode, Number, Point } from "../doc/Document";
import { ParamValueMap, Constraint, getConstraint } from "./Constraint";

import { makeProgram } from "./makeProgram";
import { dogLeg } from "./optimizers/dogLeg";
import { evalJacobian } from "./evalJacobian";

export type RequestValues = {
  ptId: string;
  axis: string;
  param: string;
  value: number;
}[];

export class ConstraintManager {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public constraintFunction: (values: number[], ops: any) => any = null;
  public initValues: number[] = [];
  public doc: Document;

  constructor(doc: Document) {
    this.doc = doc;
  }

  setDocument(doc: Document): void {
    this.doc = doc;
    this.updateConstraintFunction();
  }

  updateConstraintFunction(): void {
    const { constraints, oldParamValues, points } = getData(this.doc);
    const program = makeProgram({
      currentPoints: points,
      paramValues: oldParamValues,
      constraints,
    });

    const { constraintFn, initValues } = program;

    // if (numerical) const fn = createConstraintFn(constraints, oldParamValues);

    this.constraintFunction = constraintFn;
    this.initValues = initValues;
  }

  evaluateConstraintFunction({
    requested = [],
  }: { requested?: RequestValues } = {}): void {
    if (!this.constraintFunction) {
      this.updateConstraintFunction();
    }

    const getValJacobian = (values) => {
      const result = this.constraintFunction(values, {
        requestedValues: requested,
      });

      const jacobian = evalJacobian(result.ad);
      result.jacobian = jacobian;
      return result;
    };

    const solution = dogLeg(getValJacobian, this.initValues);

    const result = this.constraintFunction(solution);
    this.initValues = solution;

    const resultPts = {};
    result.pts.forEach((pt) => {
      resultPts[pt.id] = pt;
    });

    // Set new param values.
    //
    // Keep in mind that this may change not only point positions, but also
    // measure values, since all of these are Number nodes.
    //
    for (const node of this.doc.nodes()) {
      if (node instanceof Point && node.id in resultPts) {
        node.x.value = resultPts[node.id].x.val;
        node.y.value = resultPts[node.id].y.val;
      }
    }

    // Update all unlocked measures.
    //
    // We need to do this last to take into account new positions
    // after solving the constraints.
    //
    for (const node of this.doc.nodes()) {
      if (node instanceof MeasureNode) {
        if (!node.isLocked) {
          node.updateMeasure();
        }
      }
    }
  }
}

function getData(doc: Document) {
  const constraints: Constraint[] = [];
  const allValues: ParamValueMap = {};
  const points: Point[] = [];
  for (const node of doc.nodes()) {
    if (node instanceof Number) {
      allValues[node.id] = node.value;
    } else if (node instanceof MeasureNode) {
      if (node.isLocked) {
        const constraint = getConstraint(node);
        if (constraint) {
          constraints.push(constraint);
        }
      }
    } else if (node instanceof Point) {
      points.push(node);
    }
  }

  const oldParamValues: ParamValueMap = {};
  points.forEach((point) => {
    oldParamValues[point.x.id] = allValues[point.x.id];
    oldParamValues[point.y.id] = allValues[point.y.id];
  });

  return { constraints, oldParamValues, points };
}
