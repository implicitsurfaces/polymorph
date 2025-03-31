import { generateProg } from "../generateProg/generateProg.js";
import { totalSolveWithReplacement } from "./totalSolveWithReplacement.js";
import { checkOverconstrained } from "./checkOverconstrained.js";
import { findRigidSubSystemsKernel } from "./findRigidSubSystemsKernel.js";

export function perturbClusters({ constraints, currentPoints, paramValues }) {
  const {
    points: baselinePoints,
    jacobian,
    idVec,
    valVec,
    fn,
  } = totalSolveWithReplacement({
    currentPoints,
    paramValues,
    constraints,
    epsilon: 1e-10,
  });

  const overconstrained = checkOverconstrained(jacobian);
  if (overconstrained.R.length > 0) {
    console.log("System is overconstrained:", overconstrained);
  }

  let coincident = [];
  constraints.forEach((c, i) => {
    // this may be off
    if (c.type === "coincident") {
      coincident.push(c.points);
    }
  });

  const kernelMethod = findRigidSubSystemsKernel({
    ptIdVec: idVec,
    ptValVec: valVec,
    jacobian,
    coincident: coincident,
    points: currentPoints,
  });

  const clusters = kernelMethod.map((ptIds) => {
    const clusterObj = {};
    ptIds.forEach((ptId) => {
      clusterObj[ptId] = { ...baselinePoints[ptId] };
    });
    return clusterObj;
  });

  const processedCurrentPoints = {};
  currentPoints.forEach((pt) => {
    processedCurrentPoints[pt.id] = { x: pt.x.value, y: pt.y.value };
  });

  const prog = generateProg({
    clusters,
    currentPoints: processedCurrentPoints,
    constraints,
  });

  return prog;
}
