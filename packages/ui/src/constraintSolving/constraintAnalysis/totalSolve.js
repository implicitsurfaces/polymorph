import { createConstraintFn } from "../compile/createConstraintFn.js";
import { dogLegAI } from "../optimizers/dogLegAI.js";
import { levenbergMarquardt } from "../optimizers/levenbergMarquardt.js";
import { angle, ppDist, equal, vertical, horizontal } from "./constraints.js";

export function totalSolve({ currentPoints, constraints, epsilon }) {
  const eqs = [];

  constraints.forEach((constraint) => {
    switch (constraint.name) {
      case "angle":
        eqs.push(angle(...constraint.points, constraint.value));
        break;
      case "perpendicular":
        eqs.push(angle(...constraint.points, 90));
        break;
      case "parallel":
        eqs.push(angle(...constraint.points, 0));
        break;
      case "distance":
        eqs.push(ppDist(...constraint.points, constraint.value));
        break;
      case "equal":
        eqs.push(equal(...constraint.points));
        break;
      case "vertical":
        eqs.push(vertical(...constraint.points));
        break;
      case "horizontal":
        eqs.push(horizontal(...constraint.points));
        break;
      case "coincident":
        // "coincident" = same X and same Y
        eqs.push(vertical(...constraint.points));
        eqs.push(horizontal(...constraint.points));
        break;
      case "fixed":
        const [pt] = constraint.points;
        const x = `${pt}_x`;
        const y = `${pt}_y`;
        eqs.push(`${x} - ${constraint.x}`);
        eqs.push(`${y} - ${constraint.y}`);
        break;
      default:
        console.log("missing", constraint);
    }
  });

  // Build variableNames and initialValues
  const variableNames = [];
  const initialValues = [];
  const pointIndexMap = {}; // e.g. { "A": 0, "B": 1, ... }
  let idx = 0;

  for (const id in currentPoints) {
    const pt = currentPoints[id];
    // We'll remember that "id" corresponds to slot (idx)
    pointIndexMap[id] = idx;
    idx++;
    // So the x,y for that point become:
    variableNames.push(`${id}_x`);
    variableNames.push(`${id}_y`);
    initialValues.push(pt.x, pt.y);
  }

  const fn = createConstraintFn(eqs, variableNames);

  // Solve once
  const solution = dogLegAI(fn, initialValues, { epsilon: epsilon ?? 1e-12 });

  // Evaluate at solution to get final residuals + Jacobian
  const valDers = fn(solution);
  const { jacobian } = valDers;

  // Build a nice object for final point coords
  const finalPts = {};
  Object.keys(pointIndexMap).forEach((pid) => {
    const i = pointIndexMap[pid];
    finalPts[pid] = {
      x: solution[2 * i],
      y: solution[2 * i + 1],
    };
  });

  return {
    points: finalPts,
    idVec: variableNames,
    valVec: solution,
    jacobian,
    fn,
  };
}
