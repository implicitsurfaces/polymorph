import { levenbergMarquardt } from "./optimizers/levenbergMarquardt";

export function totalSolve(fn, currentParams) {
  const variableNames = [];
  const initialValues = [];

  for (const id in currentParams) {
    const val = currentParams[id];
    variableNames.push(id);
    initialValues.push(val);
  }

  const solution = levenbergMarquardt(fn, initialValues);

  const finalValues = {};

  for (let i = 0; i < variableNames.length; i += 1) {
    const id = variableNames[i];

    finalValues[id] = solution[i];
  }

  return finalValues;
}
