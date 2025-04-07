import type { NodeCategory, NodeCategoryMap } from "sketch";

export interface NodeWrapper<T> {
  inner: T;
}

export function isNodeWrapper<T extends NodeCategory>(
  obj: unknown,
  categoryName: T,
): obj is NodeWrapper<NodeCategoryMap[T]> {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "inner" in obj &&
    isOfCategory(obj.inner, categoryName)
  );
}

export function isOfCategory<T extends NodeCategory>(
  obj: unknown,
  categoryName: T,
): obj is NodeCategoryMap[T] {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "category" in obj &&
    typeof obj.category === "string" &&
    obj.category === categoryName
  );
}
