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

export function evaluateTape(tape, args) {
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
