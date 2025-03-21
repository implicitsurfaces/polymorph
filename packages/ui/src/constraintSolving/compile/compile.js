// Leo McElroy (c) 2024

import { evaluateTape } from "./evaluateTape.js";
import { evaluateTapeVal } from "./evaluateTapeVal.js";
import { parse, tokenize } from "./parserCombinator";

const compileHelperStackWithTape = (argList, rootNode) => {
  const tape = [];

  const buildTape = (node) => {
    if (node.type === "number") {
      tape.push({ type: "pushValue", value: parseFloat(node.value) });
    } else if (node.type === "symbol") {
      let variable = node.value;
      let index = argList.indexOf(variable);
      const jacobian = argList.map((x) => (x === variable ? 1 : 0));
      tape.push({ type: "pushSymbol", index, jacobian });
    } else if (node.type === "binary") {
      buildTape(node.left);
      buildTape(node.right);
      tape.push({ type: "evaluateBinary", operator: node.operator });
    } else if (node.type === "call") {
      node.args.forEach(buildTape);
      tape.push({
        type: "evaluateCall",
        func: node.value,
        argCount: node.args.length,
      });
    } else if (node.type === "exp") {
      buildTape(node.value);
    } else {
      throw new Error(`Unsupported node type: ${node.type}`);
    }
  };

  buildTape(rootNode);

  return (args, onlyVal = false) =>
    !onlyVal ? evaluateTape(tape, args) : evaluateTapeVal(tape, args);
};

export const compile = (args, expression) => {
  const toks = tokenize(expression);

  const result = parse(toks);

  const [ast, remainder] = result;

  if (remainder > 0) console.log(remainder);

  return compileHelperStackWithTape(args, ast);
};
