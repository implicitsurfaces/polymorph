// This file contains helper functions to implement the `dataFromOptions` static
// method of each node type.

import { Node, NodeId, NodeData, NodeOptions, NodeType } from "./Node";
import { Document } from "./Document";

/**
 * This helper function returns the given `node` as is if not undefined,
 * otherwise it creates and returns a node of the given `type` and with the
 * given `options`.
 */
export function getOrCreateNode<
  T extends Node,
  Data extends NodeData,
  Options extends NodeOptions,
>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Data, Options>,
  options: Options,
): T {
  if (node) {
    // Note: if `node` is given, then it is trusted to be in the document
    // and returned as is. This is important when cloning a whole document,
    // as the node may not be cloned yet, but another node being
    // cloned now may reference its ID.
    return node;
  } else {
    return doc.createNode(type, options);
  }
}

/**
 * This helper function is the same as `getOrCreate()` except that it returns
 * the node ID instead of the node itself.
 */
export function getOrCreateNodeId<
  T extends Node,
  Data extends NodeData,
  Options extends NodeOptions,
>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Data, Options>,
  options: Options,
): NodeId {
  return getOrCreateNode(doc, node, type, options).id;
}
