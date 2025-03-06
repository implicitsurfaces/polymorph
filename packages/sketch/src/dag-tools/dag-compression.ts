import { combineHashes, hashValue } from "../utils/hash-fnv-1a";
import { stackTraversal } from "./dag-traversal";

type TempNode = {
  hash: number;
  children: TempNode[];
};

/**
 * Compresses a directed acyclic graph by removing duplicate nodes based on their structure and values.
 * @param root The root node of the DAG
 * @param childrenFcn Function that returns a node's children
 * @param valueFcn Function that returns a node's value for hashing
 * @param buildNode Function to construct a new node
 * @returns The root of the compressed graph
 */
export function compress<Node extends object>(
  root: Node,
  childrenFcn: (node: Node) => Node[],
  valueFcn: (node: Node) => number | string,
  buildNode: (originalNode: Node, children: Node[]) => Node,
) {
  if (!root) throw new Error("Node cannot be null or undefined");

  const hashToNode = new Map<number, Node>();
  const nodeToHash = new WeakMap<Node, number>();
  const hashStructure = new Map<number, number[]>();

  stackTraversal(root, childrenFcn, {
    postCallBack: (node) => {
      if (nodeToHash.has(node)) {
        throw new Error("Node has already been processed");
      }

      const childrenHashes = childrenFcn(node).map((child) => {
        return nodeToHash.get(child)!;
      });

      const valueHash = hashValue(valueFcn(node));
      const nodeHash = combineHashes(valueHash, childrenHashes);

      nodeToHash.set(node, nodeHash);
      hashToNode.set(nodeHash, node);
      hashStructure.set(nodeHash, childrenHashes);
    },
  });

  const destructuredNodes = new Map<number, TempNode>(
    Array.from(hashToNode.keys()).map((hash) => {
      return [
        hash,
        {
          hash,
          children: [],
        },
      ];
    }),
  );

  // For each unique node, resolve its children references
  for (const [hash, tempNode] of destructuredNodes.entries()) {
    const childrenHashes = hashStructure.get(hash)!;
    tempNode.children = childrenHashes.map((childHash) => {
      return destructuredNodes.get(childHash)!;
    });
  }

  const compressedTree = new Map<number, Node>();

  const rootHash = nodeToHash.get(root)!;
  stackTraversal(
    destructuredNodes.get(rootHash)!,
    (node: TempNode) => node.children,
    {
      postCallBack: (node) => {
        const originalNode = hashToNode.get(node.hash)!;
        const children = node.children!.map(
          (child) => compressedTree.get(child.hash)!,
        );

        const newNode = buildNode(originalNode, children);
        compressedTree.set(node.hash, newNode);
      },
    },
  );

  return compressedTree.get(rootHash)!;
}
