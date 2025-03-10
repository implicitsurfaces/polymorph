import { NumNode, childrenOfNumNode } from "../num-tree";
import { mapDAG } from "./dag-traversal";

export function treeInfo(node: NumNode): {
  depth: number;
  uniqueNodeCount: number;
} {
  let visited = 0;

  const treeDepths = mapDAG(
    node,
    childrenOfNumNode,
    (_, children: number[]): number => {
      visited++;
      if (!children.length) return 1;
      return Math.max(...children) + 1;
    },
  );

  return {
    uniqueNodeCount: visited,
    depth: treeDepths.get(node)!,
  };
}
