import { describe, test, expect } from "vitest";
import { compress } from "./dag-compression"; // Adjust the import path as needed

// Define a simple Node interface for testing
interface TestNode {
  id: string;
  value: number | string;
  children: TestNode[];
}

describe("DAG Compression", () => {
  // Utility function to create nodes for testing
  function createNode(
    id: string,
    value: number | string,
    children: TestNode[] = [],
  ): TestNode {
    return { id, value, children };
  }

  // Functions required by compress
  const getChildren = (node: TestNode) => node.children;
  const getValue = (node: TestNode) => node.value;
  const buildNode = (
    originalNode: TestNode,
    children: TestNode[],
  ): TestNode => {
    return {
      id: originalNode.id,
      value: originalNode.value,
      children,
    };
  };

  // Utility function to count nodes in a graph
  function countNodes(root: TestNode, visited = new Set<TestNode>()): number {
    if (visited.has(root)) return 0;
    visited.add(root);

    let count = 1;
    for (const child of root.children) {
      count += countNodes(child, visited);
    }
    return count;
  }

  test("should handle null/undefined input", () => {
    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compress(null as any, getChildren, getValue, buildNode),
    ).toThrow("Node cannot be null or undefined");

    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compress(undefined as any, getChildren, getValue, buildNode),
    ).toThrow("Node cannot be null or undefined");
  });

  test("should not change a tree with no duplicate subtrees", () => {
    // Create a simple tree: A -> B -> C
    const nodeC = createNode("C", 3);
    const nodeB = createNode("B", 2, [nodeC]);
    const nodeA = createNode("A", 1, [nodeB]);

    const result = compress(nodeA, getChildren, getValue, buildNode);

    // Verify structure remains the same
    expect(result.value).toBe(1);
    expect(result.children.length).toBe(1);
    expect(result.children[0].value).toBe(2);
    expect(result.children[0].children.length).toBe(1);
    expect(result.children[0].children[0].value).toBe(3);

    // Count should be the same
    expect(countNodes(result)).toBe(3);
  });

  test("should compress a tree with duplicate leaf nodes", () => {
    // Create a tree with duplicate leaf nodes:
    //      A
    //     / \
    //    B   C
    //   / \ / \
    //  D   E   D  <- D appears twice

    const nodeD1 = createNode("D1", 4);
    const nodeE = createNode("E", 5);
    const nodeD2 = createNode("D2", 4); // Same value as D1
    const nodeB = createNode("B", 2, [nodeD1, nodeE]);
    const nodeC = createNode("C", 3, [nodeE, nodeD2]); // Shares E and has another D
    const nodeA = createNode("A", 1, [nodeB, nodeC]);

    const result = compress(nodeA, getChildren, getValue, buildNode);

    // Original has 6 nodes (A, B, C, D1, E, D2)
    expect(countNodes(nodeA)).toBe(6);

    // Compressed should have 5 nodes (A, B, C, D, E)
    // D nodes should be consolidated
    expect(countNodes(result)).toBe(5);

    // Verify node E is shared
    expect(result.children[0].children[1]).toBe(result.children[1].children[0]);

    // Verify node D is shared
    expect(result.children[0].children[0]).toBe(result.children[1].children[1]);
  });

  test("should compress a complex DAG with shared subtrees", () => {
    // Create a more complex DAG with shared subtrees:
    //        A
    //       / \
    //      B   C
    //     / \ / \
    //    D   E   F
    //        |
    //        G   <- Subtree E->G appears twice
    //       / \
    //      H   I

    const nodeI = createNode("I", 9);
    const nodeH = createNode("H", 8);
    const nodeG = createNode("G", 7, [nodeH, nodeI]);
    const nodeE1 = createNode("E1", 5, [nodeG]);
    const nodeE2 = createNode("E2", 5, [nodeG]); // Duplicate subtree structure
    const nodeF = createNode("F", 6);
    const nodeD = createNode("D", 4);
    const nodeB = createNode("B", 2, [nodeD, nodeE1]);
    const nodeC = createNode("C", 3, [nodeE2, nodeF]);
    const nodeA = createNode("A", 1, [nodeB, nodeC]);

    const result = compress(nodeA, getChildren, getValue, buildNode);

    // Original has 10 nodes (A, B, C, D, E1, E2, F, G, H, I)
    expect(countNodes(nodeA)).toBe(10);

    // Compressed should have 9 nodes (A, B, C, D, E, F, G, H, I)
    // E nodes should be consolidated
    expect(countNodes(result)).toBe(9);

    // Verify node E is shared
    expect(result.children[0].children[1]).toBe(result.children[1].children[0]);
  });

  test("should handle complex shared structures correctly", () => {
    // Create a complex shared structure:
    //         A
    //        / \
    //       B   C
    //      / \ / \
    //     D   E   F
    //    / \ / \ / \
    //   G   H   I   J
    //  / \         / \
    // K   L       M   N

    // Build from the bottom up
    const nodeK = createNode("K", 11);
    const nodeL = createNode("L", 12);
    const nodeM = createNode("M", 13);
    const nodeN = createNode("N", 14);

    const nodeG = createNode("G", 7, [nodeK, nodeL]);
    const nodeH1 = createNode("H1", 8);
    const nodeH2 = createNode("H2", 8);
    const nodeI1 = createNode("I1", 9);
    const nodeI2 = createNode("I2", 9);
    const nodeJ = createNode("J", 10, [nodeM, nodeN]);

    const nodeD = createNode("D", 4, [nodeG, nodeH1]);
    const nodeE = createNode("E", 5, [nodeH2, nodeI1]); // Shares H with D
    const nodeF = createNode("F", 6, [nodeI2, nodeJ]); // Shares I with E

    const nodeB = createNode("B", 2, [nodeD, nodeE]);
    const nodeC = createNode("C", 3, [nodeE, nodeF]); // Shares E with B

    const nodeA = createNode("A", 1, [nodeB, nodeC]);

    const result = compress(nodeA, getChildren, getValue, buildNode);

    // Count nodes in original - should have 14 nodes with duplicates
    expect(countNodes(nodeA)).toBe(16);

    // Count nodes in compressed - should have 14 - 2 = 12 nodes (H and I appear twice)
    expect(countNodes(result)).toBe(14);

    // Verify shared nodes
    // H is shared between D and E
    expect(result.children[0].children[0].children[1]).toBe(
      result.children[0].children[1].children[0],
    );

    // I is shared between E and F
    expect(result.children[0].children[1].children[1]).toBe(
      result.children[1].children[1].children[0],
    );
  });
});
