import { evaluateTape } from "./evaluateTape.js";
import { evaluateTapeVal } from "./evaluateTapeVal.js";

const operatorMapping = {
  // Binary operators
  ADD: "+",
  SUB: "-",
  MUL: "*",
  DIV: "/",
  POWER: "^", // Not supported
  MOD: "%",
  ATAN2: "atan2",
  MIN: "min",
  MAX: "max",
  COMPARE: "compare",
  AND: "and",
  OR: "or",

  // Unary / function operators
  SIN: "sin",
  COS: "cos",
  TAN: "tan",
  ASIN: "asin",
  ACOS: "acos",
  ATAN: "atan",
  EXP: "exp",
  SQRT: "sqrt",
  LOG: "log",
  NEG: "neg",
  ABS: "abs",
  CBRT: "cbrt",
  SIGN: "sign",
  NOT: "not",
  TANH: "tanh",
  LOG1P: "log1p",
  DEBUG: "debug",
};

export function steveTreeToStack(argList, rootNode) {
  const tape = [];

  const buildTape = (node) => {
    switch (node.operation) {
      case "LITERAL":
        tape.push({ type: "pushValue", value: node.value });
        break;
      case "VAR":
        const variable = node.name;
        const index = argList.indexOf(variable);
        const jacobian = argList.map((x) => (x === variable ? 1 : 0));
        tape.push({ type: "pushSymbol", index, jacobian });
        break;
      case "ADD":
      case "SUB":
      case "MUL":
      case "DIV":
      case "MOD":
      case "ATAN2":
      case "MIN":
      case "MAX":
      case "COMPARE":
      case "AND":
      case "OR":
      case "POWER":
        buildTape(node.left);
        buildTape(node.right);
        tape.push({
          type: "evaluateBinary",
          operator: operatorMapping[node.operation],
        });
        break;
      case "SQRT":
      case "CBRT":
      case "COS":
      case "ACOS":
      case "ASIN":
      case "TAN":
      case "ATAN":
      case "LOG":
      case "EXP":
      case "ABS":
      case "NEG":
      case "SIN":
      case "SIGN":
      case "NOT":
      case "TANH":
      case "LOG1P":
      case "DEBUG":
        if (!node.original) {
          throw new Error(
            `Unary operation ${node.operation} missing "original" property`,
          );
        }
        buildTape(node.original);
        tape.push({
          type: "evaluateUnary",
          operator: operatorMapping[node.operation],
        });
        break;
      default:
        console.log("Error", node);
        throw new Error(`Unsupported node operation: ${node.operation}`);
    }
  };

  buildTape(rootNode);

  return (args, onlyVal = false) =>
    !onlyVal ? evaluateTape(tape, args) : evaluateTapeVal(tape, args);
}
