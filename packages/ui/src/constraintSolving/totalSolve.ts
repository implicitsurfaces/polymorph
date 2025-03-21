// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck

import { steveTreeToStack } from "./compile/steveTreeToStack";
import { levenbergMarquardt } from "./optimizers/levenbergMarquardt";
import { ppDist, lpDist, angle } from "./steveConstraints.js";

export function totalSolve(constraints, currentParams) {
  const eqs = [];

  constraints.forEach((constraint) => {
    if (constraint.type === "angle") {
      eqs.push(angle(...constraint.params, constraint.value).n);
    } else if (constraint.type === "perpendicular") {
      eqs.push(angle(...constraint.params, 90).n);
    } else if (constraint.type === "parallel") {
      eqs.push(angle(...constraint.params, 0).n);
    } else if (constraint.type === "pointToPointDistance") {
      eqs.push(ppDist(...constraint.params, constraint.value).n);
    } else if (constraint.type === "lineToPointDistance") {
      eqs.push(lpDist(...constraint.params, constraint.value).n);
    } else {
      console.log("missing", constraint);
    }
  });

  const variableNames = [];
  const initialValues = [];

  for (const id in currentParams) {
    const val = currentParams[id];
    variableNames.push(id);
    initialValues.push(val);
  }

  const fn = createConstraintFn(eqs, variableNames);

  const solution = levenbergMarquardt(fn, initialValues);

  const finalValues = {};

  for (let i = 0; i < variableNames.length; i += 1) {
    const id = variableNames[i];

    finalValues[id] = solution[i];
  }

  return finalValues;
}

function createConstraintFn(constraintEquations, variableNames) {
  const compiled = constraintEquations.map((eq) =>
    steveTreeToStack(variableNames, eq),
  );

  const getValDers = (variableValues, onlyVals = false) => {
    const results = compiled.map((f) => f(variableValues, onlyVals));

    if (onlyVals) return results;

    const vals = [];
    const ders = [];

    results.forEach((valDer) => {
      const [val, der] = valDer;
      vals.push(val);
      ders.push(der);
    });

    return { vals, jacobian: ders };
  };

  return getValDers;
}
