import {
  BinaryOp,
  childrenOfNumNode,
  DebugNode,
  Derivative,
  LiteralNum,
  NumNode,
  UnaryOp,
  Variable,
} from "../num-tree";
import { compress } from "../dag-tools/dag-compression";

function getValue(node: NumNode): string {
  if (node instanceof LiteralNum) {
    return `LIT:${node.value.toString()}`;
  } else if (node instanceof Variable) {
    return `VAR:${node.name}`;
  } else if (node instanceof UnaryOp) {
    return `UOP:${node.operation}`;
  } else if (node instanceof BinaryOp) {
    return `BOP:${node.operation}`;
  } else if (node instanceof Derivative) {
    return `DER`;
  }
  return "UNKNOWN";
}

function buildNode(originalNode: NumNode, children: NumNode[]): NumNode {
  if (originalNode instanceof LiteralNum) {
    return new LiteralNum(originalNode.value);
  } else if (originalNode instanceof Variable) {
    return new Variable(originalNode.name);
  } else if (originalNode instanceof UnaryOp) {
    if (originalNode instanceof DebugNode) {
      return new DebugNode(children[0], originalNode.debug);
    }
    return new UnaryOp(originalNode.operation, children[0]);
  } else if (originalNode instanceof BinaryOp) {
    return new BinaryOp(originalNode.operation, children[0], children[1]);
  } else if (originalNode instanceof Derivative) {
    return new Derivative(children[0] as Variable);
  }
  throw new Error("Unknown node type");
}

/**
 * Consolidates a tree to create a DAG with only one version of each semantically identical node
 * @param root The root node of the tree to consolidate
 * @returns The root node of the consolidated DAG
 */
export function compressNum(root: NumNode): NumNode {
  return compress(root, childrenOfNumNode, getValue, buildNode);
}
