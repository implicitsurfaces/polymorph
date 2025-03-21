export function evaluateTapeVal(tape, args) {
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
