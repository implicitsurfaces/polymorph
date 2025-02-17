import { compile } from "./compile.js";

export function createConstraintFn(eqs, params) {
  const constraintEquations = eqs;

  const compiled = constraintEquations.map((eq) => compile(params, eq));

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
