import { visitFromLeaves } from "./dag-tools/dag-traversal";
import { UnaryOperation, BinaryOperation } from "./types";
export class NumNode {
  readonly operation: string = "NONE";
}

export class Derivative extends NumNode {
  constructor(readonly variable: Variable) {
    super();
  }
}

export class UnaryOp extends NumNode {
  constructor(
    readonly operation: UnaryOperation,
    readonly original: NumNode,
  ) {
    super();
  }
}

export class DebugNode extends UnaryOp {
  constructor(
    readonly original: NumNode,
    readonly debug: string,
  ) {
    super("DEBUG", original);
    this.debug = debug;
  }
}

export class BinaryOp extends NumNode {
  constructor(
    readonly operation: BinaryOperation,
    readonly left: NumNode,
    readonly right: NumNode,
  ) {
    super();
  }
}

export class LiteralNum extends NumNode {
  readonly operation = "LITERAL";
  constructor(public value: number) {
    super();
  }
}

export class Variable extends NumNode {
  readonly operation = "VAR";
  constructor(public name: string) {
    super();
  }
}

export const ZERO_NODE = new LiteralNum(0);
export const ONE_NODE = new LiteralNum(1);
export const TWO_NODE = new LiteralNum(2);
export const NEG_ONE_NODE = new LiteralNum(-1);

const cloneNode = (node: NumNode): NumNode => {
  if (node instanceof LiteralNum) {
    return new LiteralNum(node.value);
  } else if (node instanceof Variable) {
    return new Variable(node.name);
  } else if (node instanceof Derivative) {
    return new Derivative(node.variable);
  } else if (node instanceof UnaryOp) {
    return new UnaryOp(node.operation, cloneNode(node.original));
  } else if (node instanceof BinaryOp) {
    return new BinaryOp(
      node.operation,
      cloneNode(node.left),
      cloneNode(node.right),
    );
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
};

export const replaceVariable = (
  node: NumNode,
  variables: Map<string, number | NumNode>,
): NumNode => {
  const modifiedNodes = new Map<NumNode, NumNode>();

  visitFromLeaves(node, childrenOfNumNode, (node) => {
    let outNode;

    if (node instanceof Variable) {
      const value = variables.get(node.name);
      if (value === undefined) {
        outNode = node;
      } else if (value instanceof NumNode) {
        outNode = value;
      } else {
        outNode = new LiteralNum(value);
      }
    } else if (node instanceof UnaryOp) {
      const newOriginal = modifiedNodes.get(node.original)!;
      outNode =
        newOriginal === node.original
          ? node
          : new UnaryOp(node.operation, newOriginal);
    } else if (node instanceof BinaryOp) {
      const newLeft = modifiedNodes.get(node.left)!;
      const newRight = modifiedNodes.get(node.right)!;

      outNode =
        newLeft === node.left && newRight === node.right
          ? node
          : new BinaryOp(node.operation, newLeft, newRight);
    } else {
      outNode = cloneNode(node);
    }

    modifiedNodes.set(node, outNode);
  });

  return modifiedNodes.get(node)!;
};

export const partialDerivative = (node: NumNode, variable: string): NumNode => {
  if (node instanceof Derivative) {
    if (node.variable.name === variable) {
      return ONE_NODE;
    }
    return ZERO_NODE;
  } else if (node instanceof UnaryOp) {
    const operand = partialDerivative(node.original, variable);
    return new UnaryOp(node.operation, operand);
  } else if (node instanceof BinaryOp) {
    const left = partialDerivative(node.left, variable);
    const right = partialDerivative(node.right, variable);
    return new BinaryOp(node.operation, left, right);
  }
  return node;
};

export function childrenOfNumNode(node: NumNode): NumNode[] {
  if (node instanceof UnaryOp) {
    return [node.original];
  } else if (node instanceof BinaryOp) {
    return [node.left, node.right];
  } else if (node instanceof Derivative) {
    return [node.variable];
  }
  return [];
}

const allVariables = function (node: NumNode): Set<string> {
  const variables = new Set<string>();

  visitFromLeaves(node, childrenOfNumNode, (node) => {
    if (node instanceof Variable) {
      console.log("found variable", node.name);
      variables.add(node.name);
    }
  });

  return variables;
};

export { allVariables };
