import {
  NumNode,
  LiteralNum,
  Derivative,
  UnaryOp,
  BinaryOp,
  Variable,
} from "../num-tree";

export function renderNodeAsDot(root: NumNode): string {
  let nodeId = 0;
  const lines: string[] = [];

  function getNodeId(): number {
    return nodeId++;
  }

  function getNodeLabel(node: NumNode): string {
    if (node instanceof LiteralNum) {
      return `${node.value}`;
    } else if (node instanceof Variable) {
      return node.name;
    } else if (node instanceof Derivative) {
      return `d${node.variable.name}`;
    } else if (node instanceof UnaryOp) {
      return node.operation.toLowerCase();
    } else if (node instanceof BinaryOp) {
      return node.operation.toLowerCase();
    }
    return node.operation;
  }

  function processNode(node: NumNode): number {
    const currentId = getNodeId();
    const label = getNodeLabel(node);

    // Add node definition
    lines.push(`    node${currentId} [label="${label}"];`);

    // Process children based on node type
    if (node instanceof UnaryOp) {
      const childId = processNode(node.original);
      lines.push(`    node${currentId} -> node${childId};`);
    } else if (node instanceof BinaryOp) {
      const leftId = processNode(node.left);
      const rightId = processNode(node.right);
      lines.push(`    node${currentId} -> node${leftId};`);
      lines.push(`    node${currentId} -> node${rightId};`);
    }

    return currentId;
  }

  // Start DOT file
  lines.push("digraph ExpressionTree {");
  lines.push("    node [shape=circle, style=filled, fillcolor=lightblue];");

  // Process the entire tree
  processNode(root);

  // End DOT file
  lines.push("}");

  return lines.join("\n");
}
