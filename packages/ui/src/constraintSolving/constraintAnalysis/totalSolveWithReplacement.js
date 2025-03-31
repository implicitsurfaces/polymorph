import { createConstraintFn } from "../compile/createConstraintFn.js";
import { dogLeg } from "../optimizers/dogLeg.js";

import {
  angle,
  ppDist,
  equal,
  vertical,
  horizontal,
  lpDist,
  pointOnLine,
} from "./constraints.js";

export function totalSolveWithReplacement({
  currentPoints,
  paramValues,
  constraints,
  epsilon = 1e-12,
}) {
  const eqs = [];
  const forwardSubs = [];

  constraints.forEach((constraint) => {
    if (constraint.type === "angle") {
      eqs.push(angle(...constraint.params, constraint.value));
    } else if (constraint.type === "perpendicular") {
      eqs.push(angle(...constraint.params, 90));
    } else if (constraint.type === "parallel") {
      eqs.push(angle(...constraint.params, 0));
    } else if (constraint.type === "pointToPointDistance") {
      eqs.push(ppDist(...constraint.params, constraint.value));
    } else if (constraint.type === "equal") {
      eqs.push(equal(...constraint.params));
    } else if (constraint.type === "lineToPointDistance") {
      let val = constraint.value;
      // TODO: how to deal with 0 distance incidence constraints
      // if (Math.abs(val) < 1e-8) val += 1e-4;
      eqs.push(
        val === 0
          ? pointOnLine(...constraint.params)
          : lpDist(...constraint.params, val),
      );
    } else if (constraint.type === "vertical") {
      eqs.push(vertical(...constraint.params)); // need to include these for jacobian analysis
      // forwardSubs.push(constraint.params.map((p) => `${p}_x`));
    } else if (constraint.type === "horizontal") {
      eqs.push(horizontal(...constraint.params)); // need to include these for jacobian analysis
      // forwardSubs.push(constraint.params.map((p) => `${p}_y`));
    } else if (constraint.type === "coincident") {
      forwardSubs.push(constraint.params);
      forwardSubs.push(constraint.params);
    } else if (constraint.type === "fixed") {
      const [x, y] = constraint.params;
      eqs.push(`${x} - ${constraint.x}`);
      eqs.push(`${y} - ${constraint.y}`);
    } else {
      console.log("missing", constraint);
    }
  });

  // Build full list of variable names and their initial values.
  const variableNames = [];
  const initialValues = [];
  for (const id in paramValues) {
    const val = paramValues[id];
    variableNames.push(id);
    initialValues.push(val);
  }

  const mergedGroups = mergePairs(forwardSubs);
  const finalEqs = replaceMerged(eqs, mergedGroups);

  // Reduce the optimization variables by merging those equal
  const { reducedVariableNames, reducedInitialValues, indexMapping } =
    reduceVariables(variableNames, initialValues, mergedGroups);

  // Build and solve the reduced system.
  const fn = createConstraintFn(finalEqs, reducedVariableNames);
  const solutionReduced = dogLeg(fn, reducedInitialValues, {
    epsilon,
    maxSteps: 1000,
  });
  const valDers = fn(solutionReduced);
  const { jacobian } = valDers;

  // Expand the reduced solution back to a full vector using indexMapping.
  const fullSolution = variableNames.map(
    (_, i) => solutionReduced[indexMapping[i]],
  );

  const finalPts = {};
  // Assume variableNames are ordered as [pt_x, pt_y, pt_x, pt_y, ...]
  for (let i = 0; i < currentPoints.length; i += 1) {
    const id = currentPoints[i].id;
    finalPts[id] = { x: fullSolution[i * 2], y: fullSolution[i * 2 + 1] };
  }

  return {
    points: finalPts,
    fn,
    jacobian,
    idVec: reducedVariableNames.map((name, i) => {
      if (typeof name === "number") {
        // Look through the original variable names for a non-number
        // that was reduced to the same index.
        const replacement = variableNames.find(
          (origName, j) =>
            indexMapping[j] === i && typeof origName !== "number",
        );
        return replacement !== undefined ? replacement : name;
      }
      return name;
    }),
    valVec: solutionReduced,
    indexMapping, // maps each original variable index to its new index
  };
}

// Build groups of merged variables (or fixed values) from forwardSubs.
function mergePairs(pairs) {
  const adj = new Map();
  for (const [a, b] of pairs) {
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a).add(b);
    adj.get(b).add(a);
  }
  const visited = new Set();
  const result = [];
  for (const node of adj.keys()) {
    if (visited.has(node)) continue;
    const queue = [node];
    const component = new Set([node]);
    visited.add(node);
    while (queue.length) {
      const current = queue.shift();
      for (const neighbor of adj.get(current)) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          component.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    if (component.size) {
      const sorted = [...component].sort((x, y) => {
        const xIsNum = typeof x === "number";
        const yIsNum = typeof y === "number";
        if (xIsNum && yIsNum) return x - y;
        if (xIsNum && !yIsNum) return -1;
        if (!xIsNum && yIsNum) return 1;
        return x < y ? -1 : x > y ? 1 : 0;
      });
      result.push(sorted);
    }
  }
  return result;
}

// Replace merged variable names in the equations with their representative.
function replaceMerged(eqs, mergedGroups) {
  const lookup = {};
  mergedGroups.forEach((group) => {
    const [first] = group;
    group.slice(1).forEach((item) => {
      lookup[item] = first;
    });
  });
  return eqs.map((eq) => {
    let final = eq;
    for (const id in lookup) {
      const newId = lookup[id];
      final = final.replaceAll(id, `(${newId})`);
    }
    return final;
  });
}

// Create a reduced variable set by using only one representative per merged group.
// Returns an object with the new variable names, initial values, and an index mapping
// from the original variableNames array to the new (reduced) index.
function reduceVariables(variableNames, initialValues, mergedGroups) {
  const repMapping = {};
  mergedGroups.forEach((group) => {
    const rep = group[0];
    group.forEach((v) => {
      repMapping[v] = rep;
    });
  });
  const repToNewIndex = {};
  const reducedVariableNames = [];
  const reducedInitialValues = [];
  const indexMapping = new Array(variableNames.length);
  for (let i = 0; i < variableNames.length; i++) {
    const varName = variableNames[i];
    const rep = repMapping[varName] || varName;
    if (rep in repToNewIndex) {
      indexMapping[i] = repToNewIndex[rep];
    } else {
      const newIndex = reducedVariableNames.length;
      repToNewIndex[rep] = newIndex;
      reducedVariableNames.push(rep);
      reducedInitialValues.push(initialValues[i]);
      indexMapping[i] = newIndex;
    }
  }
  return { reducedVariableNames, reducedInitialValues, indexMapping };
}
