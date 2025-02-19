// Leo McElroy (c) 2024

import {
  valder,
  sin,
  cos,
  tan,
  asin,
  acos,
  atan,
  mul,
  div,
  neg,
  plus,
  minus,
  exp,
  sqrt,
  log,
  power,
  abs,
} from "./autodiff";

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

function evaluateTape(tape, args) {
  const valueStack = [];

  for (const instruction of tape) {
    if (instruction.type === "pushValue") {
      valueStack.push(instruction.value);
    } else if (instruction.type === "pushSymbol") {
      const { index, jacobian } = instruction;
      valueStack.push(valder(args[index], jacobian));
    } else if (instruction.type === "evaluateBinary") {
      const right = valueStack.pop();
      const left = valueStack.pop();
      switch (instruction.operator) {
        case "+":
          valueStack.push(plus(left, right));
          break;
        case "*":
          valueStack.push(mul(left, right));
          break;
        case "/":
          valueStack.push(div(left, right));
          break;
        case "-":
          valueStack.push(minus(left, right));
          break;
        case "^":
          valueStack.push(power(left, right));
          break;
      }
    } else if (instruction.type === "evaluateCall") {
      const newArgs = [];
      for (let i = 0; i < instruction.argCount; i++) {
        newArgs.unshift(valueStack.pop());
      }
      switch (instruction.func) {
        case "sin":
          valueStack.push(sin(...newArgs));
          break;
        case "cos":
          valueStack.push(cos(...newArgs));
          break;
        case "tan":
          valueStack.push(tan(...newArgs));
          break;
        case "asin":
          valueStack.push(asin(...newArgs));
          break;
        case "acos":
          valueStack.push(acos(...newArgs));
          break;
        case "atan":
          valueStack.push(atan(...newArgs));
          break;
        case "exp":
          valueStack.push(exp(...newArgs));
          break;
        case "sqrt":
          valueStack.push(sqrt(...newArgs));
          break;
        case "log":
          valueStack.push(log(...newArgs));
          break;
        case "neg":
          valueStack.push(neg(...newArgs));
          break;
        case "abs":
          valueStack.push(abs(...newArgs));
          break;
      }
    }
  }

  return valueStack.pop();
}

function evaluateTapeVal(tape, args) {
  const valueStack = [];

  for (const instruction of tape) {
    if (instruction.type === "pushValue") {
      valueStack.push(instruction.value);
    } else if (instruction.type === "pushSymbol") {
      const { index, jacobian } = instruction;
      valueStack.push(args[index]);
    } else if (instruction.type === "evaluateBinary") {
      let right = valueStack.pop();
      const left = valueStack.pop();
      switch (instruction.operator) {
        case "+":
          valueStack.push(left + right);
          break;
        case "*":
          valueStack.push(left * right);
          break;
        case "/":
          if (right === 0) right += 1e-10;
          valueStack.push(left / right);
          break;
        case "-":
          valueStack.push(left - right);
          break;
        case "^":
          valueStack.push(left ** right);
          break;
      }
    } else if (instruction.type === "evaluateCall") {
      const newArgs = [];
      for (let i = 0; i < instruction.argCount; i++) {
        newArgs.unshift(valueStack.pop());
      }
      switch (instruction.func) {
        case "sin":
          valueStack.push(Math.sin(...newArgs));
          break;
        case "cos":
          valueStack.push(Math.cos(...newArgs));
          break;
        case "tan":
          valueStack.push(Math.tan(...newArgs));
          break;
        case "asin":
          valueStack.push(Math.asin(...newArgs));
          break;
        case "acos":
          valueStack.push(Math.acos(...newArgs));
          break;
        case "atan":
          valueStack.push(Math.atan(...newArgs));
          break;
        case "exp":
          valueStack.push(Math.exp(...newArgs));
          break;
        case "sqrt":
          valueStack.push(Math.sqrt(...newArgs));
          break;
        case "log":
          valueStack.push(Math.log(...newArgs));
          break;
        case "neg":
          valueStack.push(-newArgs[0]);
          break;
        case "abs":
          valueStack.push(Math.abs(...newArgs));
          break;
      }
    }
  }

  return valueStack.pop();
}

export const compile = (args, expression) => {
  const toks = tokenize(expression);

  const result = parse(toks);

  const [ast, remainder] = result;

  if (remainder > 0) console.log(remainder);

  return compileHelperStackWithTape(args, ast);
};
