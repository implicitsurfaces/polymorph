export interface NodeWrapper<T> {
  inner: T;
}

type ClassType<T> = new (...args: unknown[]) => T;

export function isNodeWrapper<T>(
  obj: unknown,
  klass: ClassType<T>,
): obj is NodeWrapper<T> {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "inner" in obj &&
    obj.inner instanceof klass
  );
}
