import { AllSketchNode, AnySketchNode } from "../sketch-nodes/types";
import { visitFromLeaves } from "../dag-tools/dag-traversal";
import { asNum } from "../num";

import { getHandler, HandlerNodeType } from "./handlerMap";

export function childrenOfNode<T extends AnySketchNode>(
  node: T,
): AllSketchNode[] {
  if (typeof node === "number") {
    return [];
  }

  const handler = getHandler(node);

  // @ts-expect-error typescript is confused here
  return handler.children(node) as AllSketchNode[];
}

export function evalSketch<T extends AllSketchNode>(
  root: T,
): ReturnType<HandlerNodeType<T["nodeType"]>["eval"]> {
  const evaledNodes = new Map<unknown, unknown>();

  visitFromLeaves(root, childrenOfNode<AnySketchNode>, (node) => {
    if (typeof node === "number") {
      evaledNodes.set(node, asNum(node));
      return;
    }

    const handler = getHandler(node);

    // @ts-expect-error typescript is confused here
    const children = handler.children(node);
    const evaledChildren = children.map((child) => evaledNodes.get(child)!);

    // @ts-expect-error typescript is confused here
    const evaled = handler.eval(node, evaledChildren);
    evaledNodes.set(node, evaled);
  });

  return evaledNodes.get(root) as ReturnType<
    HandlerNodeType<T["nodeType"]>["eval"]
  >;
}
